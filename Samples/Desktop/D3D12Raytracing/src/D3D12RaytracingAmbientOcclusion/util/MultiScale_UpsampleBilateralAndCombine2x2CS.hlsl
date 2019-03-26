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
Texture2D<float4> g_inLowResNormalDepth : register(t2);
Texture2D<float> g_inHiResValue : register(t3);
Texture2D<float4> g_inHiResNormalDepth : register(t4);
RWTexture2D<float> g_outValue : register(u0);


ConstantBuffer<DownAndUpsampleFilterConstantBuffer> g_CB : register(b0);

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

float BilateralUpsample(float ActualDistance, float3 ActualNormal, float4 SampleDistances, float3 SampleNormals[4], float4 BilinearWeights, float4 SampleValues)
{
    float4 depthWeights = 1.0 / (abs(SampleDistances - ActualDistance) + 1e-6 * ActualDistance);

    float4 normalWeights = float4(
        pow(saturate(dot(ActualNormal, SampleNormals[0])), 32),
        pow(saturate(dot(ActualNormal, SampleNormals[1])), 32),
        pow(saturate(dot(ActualNormal, SampleNormals[2])), 32),
        pow(saturate(dot(ActualNormal, SampleNormals[3])), 32)
        );

    // Ensure a non-zero weight in case none of the normals match.
    normalWeights += 0.001f;

 //   BilinearWeights = g_CB.useBilinearWeights ? BilinearWeights : 1;
 //   depthWeights = g_CB.useDepthWeights ? depthWeights : 1;
 //   normalWeights = g_CB.useNormalWeights ? normalWeights : 1;

    float4 weights = normalWeights * depthWeights * BilinearWeights;

    float totalWeight = dot(weights, 1);
    weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0;

    return dot(weights, SampleValues) / dot(weights, 1);
}

void LoadDepthAndNormal(Texture2D<float4> inNormalDepthTexture, in uint2 texIndex, out float4 encodedNormalAndDepth, out float depth, out float3 normal)
{
    encodedNormalAndDepth = inNormalDepthTexture[texIndex];
    depth = encodedNormalAndDepth.z;
    normal = DecodeNormal(encodedNormalAndDepth.xy);

    // ToDo remove
#if !COMPRES_NORMALS || !PACK_NORMAL_AND_DEPTH
    Not supported
#endif
}

// ToDo double check coorrect threadgroups used across shaders
[numthreads(MultiScale_UpsampleBilateralFilterAndCombine::ThreadGroup::Width, MultiScale_UpsampleBilateralFilterAndCombine::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    // ToDo comment

    // Process each 2x2 high res quad at a time, starting from [-1,-1] 
    // so that subsequent internal high-res pixel quads are within low-res quads.
    int2 topLeftHiResIndex = (DTid << 1) + int2(-1, -1);
    int2 topLeftLowResIndex = (topLeftHiResIndex + int2(-1, -1)) >> 1;
    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

    // ToDo combine loads into single loop - perf?
    float4 hiResEncodedNormalsAndDepths[4];
    float  hiResDepths[4];
    float3 hiResNormals[4];
    for (int i = 0; i < 4; i++)
    {
        LoadDepthAndNormal(g_inHiResNormalDepth, topLeftHiResIndex + srcIndexOffsets[i], hiResEncodedNormalsAndDepths[i], hiResDepths[i], hiResNormals[i]);
    }
    float4 vHiResDepths = float4(hiResDepths[0], hiResDepths[1], hiResDepths[2], hiResDepths[3]);

    float4 lowResEncodedNormalsAndDepths[4];
    float  lowResDepths[4];
    float3 lowResNormals[4];
    for (int i = 0; i < 4; i++)
    {
        LoadDepthAndNormal(g_inLowResNormalDepth, topLeftLowResIndex + srcIndexOffsets[i], lowResEncodedNormalsAndDepths[i], lowResDepths[i], lowResNormals[i]);
    }
    float4 vLowResDepths = float4(lowResDepths[0], lowResDepths[1], lowResDepths[2], lowResDepths[3]);


    // ToDo perf with gather()?
    // ToDo consider moving depth to x in normal if it makes a difference.
    float lowResValues1[4];
    for (int i = 0; i < 4; i++)
    {
        lowResValues1[i] = g_inLowResValue1[topLeftLowResIndex + srcIndexOffsets[i]];
    }

    float lowResValues2[4];
    for (int i = 0; i < 4; i++)
    {
        lowResValues2[i] = g_inLowResValue2[topLeftLowResIndex + srcIndexOffsets[i]];
    }

    float hiResValues[4];
    for (int i = 0; i < 4; i++)
    {
        hiResValues[i] = g_inHiResValue[topLeftHiResIndex + srcIndexOffsets[i]];
    }
#define WORKAROUND_FOR_VALUES_OVERFLOW_ON_MULTISCALE_COMBINE 0  // ToDo remove
    float lowResValues[4];
    for (int i = 0; i < 4; i++)
    {
#if WORKAROUND_FOR_VALUES_OVERFLOW_ON_MULTISCALE_COMBINE
        lowResValues[i] = lowResValues2[i];
#else
        lowResValues[i] = -lowResValues1[i] + lowResValues2[i];
#endif
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

        // ToDo this should follow the Delbracio2012, but it computes incorrect values, i.e. 1 where it should be less. Upsampling issue?
#if WORKAROUND_FOR_VALUES_OVERFLOW_ON_MULTISCALE_COMBINE
        float outValue = 0.5f * (hiResValues[i] + BilateralUpsample(actualDistance, actualNormal, vLowResDepths, lowResNormals, bilinearWeights[i], vLowResValues));
#else
        float outValue = hiResValues[i] + BilateralUpsample(actualDistance, actualNormal, vLowResDepths, lowResNormals, bilinearWeights[i], vLowResValues);
#endif
        // ToDo revise
        g_outValue[topLeftHiResIndex + srcIndexOffsets[i]] = actualDistance < DISTANCE_ON_MISS ? outValue : hiResValues[i];
    }

#else
    float4 vHiResValues = float4(hiResValues[0], hiResValues[1], hiResValues[2], hiResValues[3]);

    g_outValue[topLeftHiResIndex] = vHiResValues.x + BilateralUpsample(vHiResDepths.x, vLowResDepths.xyzw, vLowResValues.xyzw);
    g_outValue[topLeftHiResIndex + int2(1, 0)] = vHiResValues.y + BilateralUpsample(vHiResDepths.y, vLowResDepths.yxwz, vLowResValues.yxwz);
    g_outValue[topLeftHiResIndex + int2(0, 1)] = vHiResValues.z + BilateralUpsample(vHiResDepths.z, vLowResDepths.zwxy, vLowResValues.zwxy);
    g_outValue[topLeftHiResIndex + int2(1, 1)] = vHiResValues.w + BilateralUpsample(vHiResDepths.w, vLowResDepths.wzyx, vLowResValues.wzyx);
#endif

}