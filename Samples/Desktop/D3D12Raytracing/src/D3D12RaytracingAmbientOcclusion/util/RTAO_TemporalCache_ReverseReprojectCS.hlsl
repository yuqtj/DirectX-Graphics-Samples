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
Texture2D<float4> g_texInputCachedNormal : register(t4);
Texture2D<float4> g_texInputCurrentFrameNormal : register(t5);
Texture2D<uint> g_texInputCacheFrameAge : register(t6);

RWTexture2D<float> g_texOutputCachedValue : register(u0);
RWTexture2D<uint> g_texOutputCacheFrameAge : register(u1);

ConstantBuffer<RTAO_TemporalCache_ReverseReprojectConstantBuffer> cb : register(b0);

SamplerState LinearSampler : register(s0);


// ToDo
// - Fix heavy disocclusion on min/magnifaction. Use bilateraly downsampled mip maps?
//   - this happens only when starting to move not on smooth move.

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
    //screenPos.y = -screenPos.y;                       // Invert Y for DirectX-style coordinates.
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



float4 BilateralResampleWeights(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float2 offset, in uint2 indices[4])
{
#if 0
  // ToDo remove - blurs more/incorrectly
    float4 depthDiffs = abs(SampleDistances - ActualDistance);
    float4 depthWeightsMask = depthDiffs < cb.depthTolerance * ActualDistance;

    // Weight the depths based of smallest difference.
    float minDepthDiff = min(min(min(depthDiffs.x, depthDiffs.y), depthDiffs.z), depthDiffs.w);
    depthWeights = min(depthWeightsMask * (minDepthDiff + 1e-6) / (depthDiffs + 1e-6), 1);
#endif
    uint4 isWithinBounds = uint4(
        IsWithinBounds(indices[0], cb.textureDim),
        IsWithinBounds(indices[1], cb.textureDim),
        IsWithinBounds(indices[2], cb.textureDim),
        IsWithinBounds(indices[3], cb.textureDim));

    float4 depthMask = isWithinBounds &&
                       abs(SampleDistances - ActualDistance) / ActualDistance < cb.depthTolerance;
  
    float4 normalWeights = 1;
    if (cb.useNormalWeights)
    {
        const uint normalExponent = 32;
        const float minNormalWeight = 1e-3f; // ToDo pass as parameter
        normalWeights =
            float4(
                pow(saturate(dot(ActualNormal, SampleNormals[0])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[1])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[2])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[3])), normalExponent) >= minNormalWeight);
    }

    float4 bilinearWeights = float4(
        (1 - offset.x) * (1 - offset.y),
        offset.x * (1 - offset.y),
        (1 - offset.x) * offset.y,
        offset.x * offset.y);

    // ToDo can we prevent diffusion across plane?
    float4 weights = bilinearWeights * depthMask * normalWeights;    // ToDo invalidate samples too pixel offcenter? <0.1

    weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0; // ToDo?

    float weightSum = dot(weights, 1);

    float minWeightSum = 1e-3f; 
    return weightSum >= minWeightSum ? weights / (dot(weights, 1) + FLT_EPSILON) : 0;
}


// ToDo use common helper
void LoadDepthAndNormal(Texture2D<float4> inNormalDepthTexture, in uint2 texIndex, out float depth, out float3 normal)
{
    if (cb.useNormalWeights)
    {
        float4 encodedNormalAndDepth = inNormalDepthTexture[texIndex];
        depth = encodedNormalAndDepth.z;
        normal = DecodeNormal(encodedNormalAndDepth.xy);
    }
    // ToDo remove
#if !COMPRES_NORMALS || !PACK_NORMAL_AND_DEPTH
    Not supported
#endif
}


[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{

    float linearDepth = g_texInputCurrentFrameDepth[DTid];
    float3 normal;
    float dummy;
    LoadDepthAndNormal(g_texInputCurrentFrameNormal, DTid, dummy, normal);

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

    uint2 indices[4] = {
        topLeftCacheFrameIndex + srcIndexOffsets[0],
        topLeftCacheFrameIndex + srcIndexOffsets[1],
        topLeftCacheFrameIndex + srcIndexOffsets[2],
        topLeftCacheFrameIndex + srcIndexOffsets[3] };
    // ToDo conditional loads if really needed?
    // ToDo use gather
    float3 cacheNormals[4];
    {
        for (int i = 0; i < 4; i++)
        {
            float dummy;
            LoadDepthAndNormal(g_texInputCachedNormal, indices[i], dummy, cacheNormals[i]);
        }
    }

    float4 vCacheDepths = float4(
        g_texInputCachedDepth[indices[0]],
        g_texInputCachedDepth[indices[1]],
        g_texInputCachedDepth[indices[2]],
        g_texInputCachedDepth[indices[3]]);


    float4 vCacheValues = float4(
        g_texInputCachedValue[indices[0]],
        g_texInputCachedValue[indices[1]],
        g_texInputCachedValue[indices[2]],
        g_texInputCachedValue[indices[3]]);

    uint4 vCacheFrameAge = uint4(
        g_texInputCacheFrameAge[indices[0]],
        g_texInputCacheFrameAge[indices[1]],
        g_texInputCacheFrameAge[indices[2]],
        g_texInputCacheFrameAge[indices[3]]);

    
    

    float value = g_texInputCurrentFrameValue[DTid];
    float mergedValue;
   
    float4 weights = BilateralResampleWeights(linearDepth, normal, vCacheDepths, cacheNormals, cachePixelOffset, indices);
    float weightSum = dot(1, weights);

    bool isCacheValueValid = weightSum > FLT_EPSILON;
    uint isDisoccluded;
    uint frameAge;
    if (isCacheValueValid)
    {
        isDisoccluded = false;
        float cachedValue = dot(weights, vCacheValues);
        
        float cacheFrameAge = dot(weights, vCacheFrameAge);       
        frameAge = uint(cacheFrameAge + 0.5f);              // HLSL rounds down when converting float to uint, bump up the value by 0.5 before rounding.
        float invFrameAge = 1.f / (frameAge + 1);

        float a = cb.forceUseMinSmoothingFactor ? cb.minSmoothingFactor : max(invFrameAge, cb.minSmoothingFactor);
        mergedValue = lerp(cachedValue, value, a);
    }
    else
    {
        isDisoccluded = true;
        mergedValue = value;
        frameAge = 0;
    }
   
    g_texOutputCachedValue[DTid] =  mergedValue;
    g_texOutputCacheFrameAge[DTid] = frameAge + 1;
    //g_texOutputCachedValue[DTid] = cacheClipSpacePos.x;
}