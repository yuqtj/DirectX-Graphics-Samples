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

// ToDO pack value and depth beforehand?
Texture2D<float> g_texInputCachedValue : register(t0);
Texture2D<float> g_texInputCurrentFrameValue : register(t1);
Texture2D<float> g_texInputCachedDepth : register(t2);
Texture2D<float> g_texInputCurrentFrameDepth : register(t3);

RWTexture2D<float> g_texOutputCachedValue : register(u0);
ConstantBuffer<RTAO_TemporalCache_ReverseReprojectConstantBuffer> cb : register(b0);

SamplerState LinearSampler : register(s0);

#if 0
// Retrieves pixel's position in world space.
// linearDepth - linear depth in [0, 1] range   // ToDo
float3 CalculateWorldPositionFromLinearDepth(in uint2 DTid, in float linearDepth)
{
    // Convert to non-linear depth.
#if USE_NORMALIZED_Z
    float linearDistance = linearDepth * (cb.zFar - cb.zNear) + cb.zNear;
#else
    float linearDistance = linearDepth;
#endif
    float logDepth = cb.zFar / (cb.zFar - cb.zNear) - cb.zFar * cb.zNear

    // Calculate Normalized Device Coordinates xyz = { [-1,1], [-1,1], [0,-1] }
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 screenPos = 2 * xy * cb.invTextureDim - 1;   // Convert to [-1, 1]
    //screenPos.y = -screenPos.y;                         // Invert Y for DirectX-style coordinates.
    float3 ndc = float3(screenPos, logDepth);  

    float4 viewPosition = mul(float4(ndc, 1), cb.invProj);
    //viewPosition /= viewPosition.w; // Perspective division
    float4 worldPosition = mul(viewPosition, cb.invView);
    
    return worldPosition.xyz;
}
#else
// Retrieves pixel's position in clip space.
// linearDepth - linear depth in [0, 1] range   // ToDo
float4 GetClipSpacePosition(in uint2 DTid, in float linearDepth)
{
    // Convert to non-linear depth.
#if USE_NORMALIZED_Z
    ToDo
    float linearDistance = linearDepth * (cb.zFar - cb.zNear) + cb.zNear;
#else
    float linearDistance = linearDepth;
#endif

    // Calculate Normalized Device Coordinates xyz = {[-1,1], [-1,1], [0,-1]}
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 screenPos = 2 * xy * cb.invTextureDim - 1;   // Convert to [-1, 1]
    screenPos.y = -screenPos.y;                         // Invert Y for DirectX-style coordinates.
    float logDepth = ViewToLogDepth(linearDepth, cb.zNear, cb.zFar);
    float3 ndc = float3(screenPos, logDepth);

    float A = cb.zFar / (cb.zFar - cb.zNear);
    float B = -cb.zNear * cb.zFar / (cb.zFar - cb.zNear);
    float w = B / (logDepth - A);

    float4 projPos = float4(ndc, 1) * w; // Reverse perspective division.
#if 0 
    float4 viewPos = mul(projPos, cb.invProj);
    float4 worldPos = mul(viewPos, cb.invView);

    return worldPos.xyz;
#else
    return projPos;
#endif
}
#endif


float4 BilateralResampleWeights(in float ActualDistance, in float4 SampleDistances, in float2 offset)
{
#if 0
  // ToDo remove - blurs more/incorrectly
    float4 depthDiffs = abs(SampleDistances - ActualDistance);
    float4 depthWeightsMask = depthDiffs < cb.depthTolerance * ActualDistance;

    // Weight the depths based of smallest difference.
    float minDepthDiff = min(min(min(depthDiffs.x, depthDiffs.y), depthDiffs.z), depthDiffs.w);
    depthWeights = min(depthWeightsMask * (minDepthDiff + 1e-6) / (depthDiffs + 1e-6), 1);
#endif
    float4 depthMask = abs(SampleDistances - ActualDistance) / ActualDistance < cb.depthTolerance;
  
    float4 bilinearWeights = float4(
        (1 - offset.x) * (1 - offset.y),
        offset.x * (1 - offset.y),
        (1 - offset.x) * offset.y,
        offset.x * offset.y);

    float4 weights = bilinearWeights * depthMask;    // ToDo invalidate samples too pixel offcenter? <0.1

    weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0; // ToDo?

    return weights / (dot(weights, 1) + FLT_EPSILON);
}



[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{

    float linearDepth = g_texInputCurrentFrameDepth[DTid];

    float4 clipSpacePos = GetClipSpacePosition(DTid, linearDepth);

    // Reverse project into previous frame
    float4 cacheClipSpacePos = mul(clipSpacePos, cb.reverseProjectionTransform);

    float3 cacheNDCpos = cacheClipSpacePos.xyz;
    cacheNDCpos /= cacheClipSpacePos.w;                             // Perspective division.
    cacheNDCpos.y = -cacheNDCpos.y;                                 // Invert Y for DirectX-style coordinates.
    float cacheLinearDepth = LogToViewDepth(cacheNDCpos.z, cb.zNear, cb.zFar);
    float2 cacheFrameTexturePos = (cacheNDCpos.xy + 1) * 0.5f;      // [-1,1] -> [0, 1]

    int2 topLeftCacheFrameIndex = int2(cacheFrameTexturePos * cb.textureDim - 0.5);
    float2 cachePixelOffset = cacheFrameTexturePos * cb.textureDim - topLeftCacheFrameIndex - 0.5;

    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    // ToDo conditional loads if really needed?
    // ToDo use gather
    float4 vCacheDepths = float4(
        g_texInputCachedDepth[topLeftCacheFrameIndex + srcIndexOffsets[0]],
        g_texInputCachedDepth[topLeftCacheFrameIndex + srcIndexOffsets[1]],
        g_texInputCachedDepth[topLeftCacheFrameIndex + srcIndexOffsets[2]],
        g_texInputCachedDepth[topLeftCacheFrameIndex + srcIndexOffsets[3]]);


    float4 vCacheValues = float4(
        g_texInputCachedValue[topLeftCacheFrameIndex + srcIndexOffsets[0]],
        g_texInputCachedValue[topLeftCacheFrameIndex + srcIndexOffsets[1]],
        g_texInputCachedValue[topLeftCacheFrameIndex + srcIndexOffsets[2]],
        g_texInputCachedValue[topLeftCacheFrameIndex + srcIndexOffsets[3]]);
    

    // ToDo should some tests be inclusive?
    bool isNotOutOfBounds = cacheFrameTexturePos.x > 0 && cacheFrameTexturePos.y > 0 && cacheFrameTexturePos.x < 1 && cacheFrameTexturePos.y < 1;
    bool isWithinDepthTolerance = abs(cacheLinearDepth - linearDepth) / linearDepth < 0.1;
    
    bool isCacheValueValid = isNotOutOfBounds;// && isWithinDepthTolerance;

    float value = g_texInputCurrentFrameValue[DTid];
    float mergedValue;
    
    if (isCacheValueValid)
    {
        float4 weights = BilateralResampleWeights(linearDepth, vCacheDepths, cachePixelOffset);
        float weightSum = dot(1, weights);
        if (weightSum > 0)
        {
            float cachedValue = dot(weights, vCacheValues);
            float a = max(cb.invCacheFrameAge, cb.minSmoothingFactor);
            mergedValue = lerp(cachedValue, value, a);
        }
        else
        {
            mergedValue = value;
        }
    }
    else
    {
        mergedValue = value;
    }
   
    g_texOutputCachedValue[DTid] =  mergedValue;
    //g_texOutputCachedValue[DTid] = cacheFrameTexturePos.x;
}