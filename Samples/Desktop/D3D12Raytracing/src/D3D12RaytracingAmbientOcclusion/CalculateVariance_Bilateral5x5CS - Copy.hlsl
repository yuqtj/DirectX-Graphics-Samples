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
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValues : register(t0);
Texture2D<float4> g_inNormal : register(t1);
Texture2D<float> g_inDepth : register(t2);
Texture2D<float> g_inDepth : register(t2);
RWTexture2D<float> g_outVariance : register(u0);

// ToDo use custom CB
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);


void AddFilterContribution(inout float weightedValueSum, inout float weightedSquaredValueSum, inout float weightSum, inout UINT numWeights, in float value, in float depth, in float3 normal, float obliqueness, in uint row, in uint col, in uint2 DTid)
{
    const float normalSigma = cb.normalSigma;
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
    const float depthSigma = 0.5;
#else
    const float depthSigma = cb.depthSigma;
#endif
    int2 id = int2(DTid) + (int2(row - 2, col - 2) );
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        float iValue = g_inValues[id];
#if COMPRES_NORMALS
        float4 normalBufValue = g_inNormal[id];
        float4 normal4 = float4(DecodeNormal(normalBufValue.xy), normalBufValue.z);
#else
        float4 normal4 = g_inNormal[id];
#endif 
        float3 iNormal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
        float iDepth = normal4.w;
#else
        float  iDepth =  g_inDepth[id];
#endif

        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
        float w = w_n * w_d;

        float weightedValue = w * iValue;
        weightedValueSum += weightedValue;
        weightedSquaredValueSum += weightedValue * iValue;
        weightSum += w;
        numWeights += w > 0.0001f ? 1 : 0;
    }
}

void AddFilterContribution2(inout float minVal, inout float maxVal, in float value, in float depth, in float3 normal, float obliqueness, in uint row, in uint col, in uint2 DTid)
{
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    int2 id = int2(DTid)+(int2(row - 2, col - 2));
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        float iValue = g_inValues[id];
#if COMPRES_NORMALS
        float4 normalBufValue = g_inNormal[id];
        float4 normal4 = float4(DecodeNormal(normalBufValue.xy), normalBufValue.z);
#else
        float4 normal4 = g_inNormal[id];
#endif 
        float3 iNormal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
        float iDepth = normal4.w;
#else
        float  iDepth = g_inDepth[id];
#endif

        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
        float w = w_n * w_d;

        minVal = (w > 0.01) ? min(iValue, minVal) : minVal;
        maxVal = (w > 0.01) ? max(iValue, maxVal) : maxVal;
    }
}

// Calculates local per-pixel variance ~ Sum(X^2)/N - mean^2;
[numthreads(CalculateVariance_Bilateral::ThreadGroup::Width, CalculateVariance_Bilateral::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
#if COMPRES_NORMALS
    float4 normalBufValue = g_inNormal[DTid];
    float4 normal4 = float4(DecodeNormal(normalBufValue.xy), normalBufValue.z);
    float obliqueness = max(0.0001f, pow(normalBufValue.w, 10));
#else
    float4 normal4 = g_inNormal[DTid];
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
    float obliqueness = 1;    // ToDO
#else
    float obliqueness = max(0.0001f, pow(normal4.w, 10));
#endif
#endif 
    float3 normal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
    float depth = normal4.w;
#else
    float  depth = g_inDepth[DTid];
#endif 

    float  value = g_inValues[DTid];

    float variance;
    if (cb.useApproximateVariance)
    {

        float minVal = value;
        float maxVal = value;

        [unroll]
        for (UINT r = 0; r < 5; r++)
            [unroll]
            for (UINT c = 0; c < 5; c++)
                if (r != 2 || c != 2)
                    AddFilterContribution2(minVal, maxVal, value, depth, normal, obliqueness, r, c, DTid);

        // ToDo
        // Approximate variance as max(Value) - min(Value)
        // Ref: Bavoil2009, Multi-Layer Dual-Resolution Screen-Space Ambient Occlusion
        variance = maxVal - minVal;
    }
    else
    {
        UINT numWeights = 1;
        float weightedValueSum = value;
        float weightedSquaredValueSum = value * value;
        float weightSum = 1.f;  // ToDo handle missing value. or skip if miss hit.

        [unroll]
        for (UINT r = 0; r < 5; r++)
            [unroll]
            for (UINT c = 0; c < 5; c++)
                if (r != 2 || c != 2)
                    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, r, c, DTid);

        if (numWeights > 1)
        {
            float invWeightSum = weightSum > 0.0001f ? 1 / weightSum : 0.f;
            float mean = invWeightSum * weightedValueSum;

            // ToDo 0 variance causes acne.
            const float minVar = 0.01f;
            variance = max(minVar, (numWeights / float(numWeights - 1)) * (invWeightSum * weightedSquaredValueSum - mean * mean));
        }
        else
        {
            variance = 0;
        }
    }
    g_outVariance[DTid] = variance;
}