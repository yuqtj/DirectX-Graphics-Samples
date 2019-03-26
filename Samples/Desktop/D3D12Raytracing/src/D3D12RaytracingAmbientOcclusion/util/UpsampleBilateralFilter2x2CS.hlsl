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

#define HLSL
#include "..\RaytracingHlslCompat.h"
#include "..\RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValue : register(t0);
Texture2D<float4> g_inLowResNormalDepth : register(t1);
Texture2D<float4> g_inHiResNormalDepth : register(t2);
RWTexture2D<float> g_outValue : register(u0);

// ToDo standardize cb vs g_CB
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


void LoadDepthAndNormal(Texture2D<float4> inNormalDepthTexture, in uint2 texIndex, out float depth, out float3 normal)
{
    float4 encodedNormalAndDepth = inNormalDepthTexture[texIndex];
    depth = encodedNormalAndDepth.z;
    normal = DecodeNormal(encodedNormalAndDepth.xy);

    // ToDo remove
#if !COMPRES_NORMALS || !PACK_NORMAL_AND_DEPTH
    Not supported
#endif
}

float BilateralUpsample(float ActualDistance, float3 ActualNormal, float4 SampleDistances, float3 SampleNormals[4], float4 BilinearWeights, float4 SampleValues)
{
    float4 depthWeights = 1;
    float4 normalWeights = 1;

    if (g_CB.useDepthWeights)
    {
        float depthThreshold = 1.f;
        float fScale = 1.f / depthThreshold;
        float fEpsilon = 1e-6 * ActualDistance;


        if (g_CB.useDynamicDepthThreshold)
        {

        }

        depthWeights = 1.0 / (fScale * abs(SampleDistances - ActualDistance) + fEpsilon);
    }

    if (g_CB.useNormalWeights)
    {
        normalWeights =  
            float4(
                pow(saturate(dot(ActualNormal, SampleNormals[0])), 32),
                pow(saturate(dot(ActualNormal, SampleNormals[1])), 32),
                pow(saturate(dot(ActualNormal, SampleNormals[2])), 32),
                pow(saturate(dot(ActualNormal, SampleNormals[3])), 32));
    }

       
    // Ensure a non-zero weight in case none of the normals match.
    normalWeights += 0.001f;

    BilinearWeights = g_CB.useBilinearWeights ? BilinearWeights : 1;

    float4 weights = normalWeights * depthWeights * BilinearWeights;

    float totalWeight = dot(weights, 1);
    weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0;

    return dot(weights, SampleValues) / dot(weights, 1);
}

[numthreads(UpsampleBilateralFilter::ThreadGroup::Width, UpsampleBilateralFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    // Process each 2x2 high res quad at a time, starting from [-1,-1] 
    // so that subsequent internal high-res pixel quads are within low-res quads.
    int2 topLeftHiResIndex = (DTid << 1) + int2(-1, -1);
    int2 topLeftLowResIndex = (topLeftHiResIndex + int2(-1, -1)) >> 1;
    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

#if 1
    float  hiResDepths[4];
    float3 hiResNormals[4];
    for (int i = 0; i < 4; i++)
    {
        if (g_CB.useDepthWeights || g_CB.useNormalWeights)
            LoadDepthAndNormal(g_inHiResNormalDepth, topLeftHiResIndex + srcIndexOffsets[i], hiResDepths[i], hiResNormals[i]);
    }
    float4 vHiResDepths = float4(hiResDepths[0], hiResDepths[1], hiResDepths[2], hiResDepths[3]);

    float  lowResDepths[4];
    float3 lowResNormals[4];
    for (int i = 0; i < 4; i++)
    {
        if (g_CB.useDepthWeights || g_CB.useNormalWeights)
            LoadDepthAndNormal(g_inLowResNormalDepth, topLeftLowResIndex + srcIndexOffsets[i], lowResDepths[i], lowResNormals[i]);
    }
    float4 vLowResDepths = float4(lowResDepths[0], lowResDepths[1], lowResDepths[2], lowResDepths[3]);

    float4 vLowResValues = float4(
        g_inValue[topLeftLowResIndex + srcIndexOffsets[0]],
        g_inValue[topLeftLowResIndex + srcIndexOffsets[1]],
        g_inValue[topLeftLowResIndex + srcIndexOffsets[2]],
        g_inValue[topLeftLowResIndex + srcIndexOffsets[3]]);

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

        float outValue = BilateralUpsample(actualDistance, actualNormal, vLowResDepths, lowResNormals, bilinearWeights[i], vLowResValues);
        // ToDo revise
        g_outValue[topLeftHiResIndex + srcIndexOffsets[i]] = actualDistance < DISTANCE_ON_MISS ? outValue : vLowResValues[i];
    }
#else

    float4 hiResDepths = float4(
        g_inHiResNormalDepth[topLeftHiResIndex].z,
        g_inHiResNormalDepth[topLeftHiResIndex + int2(1, 0)].z,
        g_inHiResNormalDepth[topLeftHiResIndex + int2(0, 1)].z,
        g_inHiResNormalDepth[topLeftHiResIndex + int2(1, 1)].z);

    float4  lowResDepths = float4(
        g_inLowResNormalDepth[topLeftLowResIndex].z,
        g_inLowResNormalDepth[topLeftLowResIndex + int2(1, 0)].z,
        g_inLowResNormalDepth[topLeftLowResIndex + int2(0, 1)].z,
        g_inLowResNormalDepth[topLeftLowResIndex + int2(1, 1)].z);

    // ToDo perf with gather()?
    // ToDo consider moving depth to x in normal if it makes a difference.
    float4 values = float4(
        g_inValue[topLeftLowResIndex],
        g_inValue[topLeftLowResIndex + int2(1, 0)],
        g_inValue[topLeftLowResIndex + int2(0, 1)],
        g_inValue[topLeftLowResIndex + int2(1, 1)]);

    g_outValue[topLeftHiResIndex] = BilateralUpsample(hiResDepths.x, lowResDepths.xyzw, values.xyzw);
    g_outValue[topLeftHiResIndex + int2(1, 0)] = BilateralUpsample(hiResDepths.y, lowResDepths.yxwz, values.yxwz);
    g_outValue[topLeftHiResIndex + int2(0, 1)] = BilateralUpsample(hiResDepths.z, lowResDepths.zwxy, values.zwxy);
    g_outValue[topLeftHiResIndex + int2(1, 1)] = BilateralUpsample(hiResDepths.w, lowResDepths.wzyx, values.wzyx);
#endif
}