//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

// ToDo rename and move file

#define HLSL
#include "..\RaytracingHlslCompat.h"
#include "..\RaytracingShaderHelper.hlsli"

Texture2D<float> g_inLowResValue1 : register(t0);
Texture2D<float> g_inLowResValue2 : register(t1);
Texture2D<NormalDepthTexFormat> g_inLowResNormalDepth : register(t2);
Texture2D<float> g_inHiResValue : register(t3);
Texture2D<NormalDepthTexFormat> g_inHiResNormalDepth : register(t4);
Texture2D<float2> g_inHiResPartialDistanceDerivative : register(t5);
RWTexture2D<float> g_outValue : register(u0);

// ToDo Test and use conditional weights as in Upsample..hlsl?

// ToDo consider 3x3 tap upsample instead 2x2

// ToDo remove outNormal if not written to.
//RWTexture2D<float4> g_texOutNormal : register(u1);

// We essentially want 5 weights:  4 for each low-res pixel and 1 to blend in when none of the 4 really
// match.  The filter strength is 1 / DeltaZTolerance.  So a tolerance of 0.01 would yield a strength of 100.
// Note that a perfect match of low to high depths would yield a weight of 10^6, completely superceding any
// noise filtering.  The noise filter is intended to soften the effects of shimmering when the high-res depth
// buffer has a lot of small holes in it causing the low-res depth buffer to inaccurately represent it.
float BilateralUpsample(float ActualDistance, float4 SampleDistances, float4 SampleValues)
{
    // ToDo handle out of bounds values?
    // ToDo test using nearest-depth if not all 4 are within depth dist threshold
    float4 weights = float4(9, 3, 3, 1) / (abs(SampleDistances - ActualDistance) + 1e-6 * ActualDistance);
    return dot(weights, SampleValues) / dot(weights, 1);
}

float DepthThreshold(float distance, float2 ddxy)
{
    float maxPixelDistance = 2; // Scale to compensate for the fact that the downsampled depth value may come from up to two pixels away in the high-res texture scale.
    float ddx = maxPixelDistance * max(ddxy.x, ddxy.y);

#if 1
    // adjust the depth threshold based on slope angle.
    float slopeAngle = atan(ddx);
    float fovAngleY = 45;   // ToDO pass from the app
    float2 resolution = float2(960, 540);
    float pixelAngle = (fovAngleY / resolution.y) * PI / 180;
    return distance * ((sin(PI - slopeAngle) / sin(slopeAngle - pixelAngle)) - 1);
#else
    return ddx;
#endif

}

float BilateralUpsample(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float4 BilinearWeights, in float4 SampleValues, in uint2 hiResPixelIndex)
{
    // Depth weights.
    // Use ddxy as depth threshold.
    float2 ddxy = abs(g_inHiResPartialDistanceDerivative[hiResPixelIndex]);  // ToDo move to caller
    float depthThreshold = DepthThreshold(ActualDistance, ddxy);
    float4 depthWeights = min(depthThreshold / (abs(SampleDistances - ActualDistance) + FLT_EPSILON), 1);

    // Normal weights.
    const uint normalExponent = 32;
    float4 normalWeights =
        float4(
            pow(saturate(dot(ActualNormal, SampleNormals[0])), normalExponent),
            pow(saturate(dot(ActualNormal, SampleNormals[1])), normalExponent),
            pow(saturate(dot(ActualNormal, SampleNormals[2])), normalExponent),
            pow(saturate(dot(ActualNormal, SampleNormals[3])), normalExponent));

    // Ensure a non-zero weight in case none of the normals match.
    normalWeights += 0.001f;

    float4 weights = normalWeights * depthWeights * BilinearWeights;

    float totalWeight = dot(weights, 1);
    weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0;

    return InterpolateValidValues(weights, SampleValues);
}


// ToDo double check coorrect threadgroups used across shaders
[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    // ToDo comment

    // Process each 2x2 high res quad at a time, starting from [-1,-1] 
    // so that subsequent internal high-res pixel quads are within low-res quads.
    int2 topLeftHiResIndex = (DTid << 1) + int2(-1, -1);
    int2 topLeftLowResIndex = (topLeftHiResIndex + int2(-1, -1)) >> 1;
    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

    // ToDo combine loads into single loop - perf?
    float  hiResDepths[4];
    float3 hiResNormals[4];
    {
        for (int i = 0; i < 4; i++)
        {
            DecodeNormalDepth(g_inHiResNormalDepth[topLeftHiResIndex + srcIndexOffsets[i]], hiResNormals[i], hiResDepths[i]);
        }
    }
    float4 vHiResDepths = float4(hiResDepths[0], hiResDepths[1], hiResDepths[2], hiResDepths[3]);

    float  lowResDepths[4];
    float3 lowResNormals[4];
    {
        for (int i = 0; i < 4; i++)
        {
            DecodeNormalDepth(g_inLowResNormalDepth[topLeftLowResIndex + srcIndexOffsets[i]], lowResNormals[i], lowResDepths[i]);
        }
    }
    float4 vLowResDepths = float4(lowResDepths[0], lowResDepths[1], lowResDepths[2], lowResDepths[3]);


    // ToDo perf with gather()?
    // ToDo consider moving depth to x in normal if it makes a difference.
    float lowResValues1[4];
    {
        for (int i = 0; i < 4; i++)
        {
            lowResValues1[i] = g_inLowResValue1[topLeftLowResIndex + srcIndexOffsets[i]];
        }
    }

    float lowResValues2[4];
    {
        for (int i = 0; i < 4; i++)
        {
            lowResValues2[i] = g_inLowResValue2[topLeftLowResIndex + srcIndexOffsets[i]];
        }
    }

    float hiResValues[4];
    {
        for (int i = 0; i < 4; i++)
        {
            hiResValues[i] = g_inHiResValue[topLeftHiResIndex + srcIndexOffsets[i]];
        }
    }
#define WORKAROUND_FOR_VALUES_OVERFLOW_ON_MULTISCALE_COMBINE 1  // ToDo remove
    float lowResValues[4];
    {
        for (int i = 0; i < 4; i++)
        {
#if WORKAROUND_FOR_VALUES_OVERFLOW_ON_MULTISCALE_COMBINE
            lowResValues[i] = lowResValues2[i];
#else
            lowResValues[i] = -lowResValues1[i] + lowResValues2[i];
#endif
        }
    }
    float4 vLowResValues = float4(lowResValues[0], lowResValues[1], lowResValues[2], lowResValues[3]);

#if 1
    const float4 bilinearWeights[4] = {
        float4(9, 3, 3, 1),
        float4(3, 9, 1, 3),
        float4(3, 1, 9, 3),
        float4(1, 3, 3, 9)
    };

    for (int i = 0; i < 4; i++)
    {
        float actualDistance = hiResDepths[i];
        float3 actualNormal = hiResNormals[i];
        uint2 hiResPixelIndex = topLeftHiResIndex + srcIndexOffsets[i];

        // ToDo this should follow the Delbracio2012, but it computes incorrect values, i.e. 1 where it should be less. Upsampling issue?
#if WORKAROUND_FOR_VALUES_OVERFLOW_ON_MULTISCALE_COMBINE
        float outValue = BilateralUpsample(actualDistance, actualNormal, vLowResDepths, lowResNormals, bilinearWeights[i], vLowResValues, hiResPixelIndex);
#else
        float outValue = hiResValues[i] + BilateralUpsample(actualDistance, actualNormal, vLowResDepths, lowResNormals, bilinearWeights[i], vLowResValues, hiResPixelIndex);
#endif
        // ToDo revise
        g_outValue[hiResPixelIndex] = actualDistance < DISTANCE_ON_MISS ? outValue : hiResValues[i];
    }

#else
    float4 vHiResValues = float4(hiResValues[0], hiResValues[1], hiResValues[2], hiResValues[3]);

    g_outValue[topLeftHiResIndex] = vHiResValues.x + BilateralUpsample(vHiResDepths.x, vLowResDepths.xyzw, vLowResValues.xyzw);
    g_outValue[topLeftHiResIndex + int2(1, 0)] = vHiResValues.y + BilateralUpsample(vHiResDepths.y, vLowResDepths.yxwz, vLowResValues.yxwz);
    g_outValue[topLeftHiResIndex + int2(0, 1)] = vHiResValues.z + BilateralUpsample(vHiResDepths.z, vLowResDepths.zwxy, vLowResValues.zwxy);
    g_outValue[topLeftHiResIndex + int2(1, 1)] = vHiResValues.w + BilateralUpsample(vHiResDepths.w, vLowResDepths.wzyx, vLowResValues.wzyx);
#endif

}