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

Texture2D<float> g_texInput : register(t0);
Texture2D<float4> g_inLowResNormal : register(t1);
Texture2D<float4> g_inHiResNormal : register(t2);
RWTexture2D<float> g_texOutput : register(u0);

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
    float4 weights = float4(9, 3, 3, 1) / (abs(ActualDistance - SampleDistances) + 1e-6 * ActualDistance);
    return dot(weights, SampleValues) / dot(weights, 1);
}

[numthreads(UpsampleBilateralFilter::ThreadGroup::Width, UpsampleBilateralFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    // Process each 2x2 high res quad at a time, starting from [-1,-1] 
    // so that subsequent internal high-res pixel quads are within low-res quads.
    int2 topLeftHiResIndex = (DTid << 1) + int2(-1, -1);
    int2 topLeftLowResIndex = (topLeftHiResIndex + int2(-1, -1)) >> 1;

    float4 hiResDepths = float4(
        g_inHiResNormal[topLeftHiResIndex].z,
        g_inHiResNormal[topLeftHiResIndex + int2(1, 0)].z,
        g_inHiResNormal[topLeftHiResIndex + int2(0, 1)].z,
        g_inHiResNormal[topLeftHiResIndex + int2(1, 1)].z);

    float4  lowResDepths = float4(
        g_inLowResNormal[topLeftLowResIndex].z,
        g_inLowResNormal[topLeftLowResIndex + int2(1, 0)].z,
        g_inLowResNormal[topLeftLowResIndex + int2(0, 1)].z,
        g_inLowResNormal[topLeftLowResIndex + int2(1, 1)].z);

    // ToDo perf with gather()?
    // ToDo consider moving depth to x in normal if it makes a difference.
    float4 values = float4(
        g_texInput[topLeftLowResIndex],
        g_texInput[topLeftLowResIndex + int2(1, 0)],
        g_texInput[topLeftLowResIndex + int2(0, 1)],
        g_texInput[topLeftLowResIndex + int2(1, 1)]);

    g_texOutput[topLeftHiResIndex] = BilateralUpsample(hiResDepths.x, lowResDepths.xyzw, values.xyzw);
    g_texOutput[topLeftHiResIndex + int2(1, 0)] = BilateralUpsample(hiResDepths.y, lowResDepths.yxwz, values.yxwz);
    g_texOutput[topLeftHiResIndex + int2(0, 1)] = BilateralUpsample(hiResDepths.z, lowResDepths.zwxy, values.zwxy);
    g_texOutput[topLeftHiResIndex + int2(1, 1)] = BilateralUpsample(hiResDepths.w, lowResDepths.wzyx, values.wzyx);
}