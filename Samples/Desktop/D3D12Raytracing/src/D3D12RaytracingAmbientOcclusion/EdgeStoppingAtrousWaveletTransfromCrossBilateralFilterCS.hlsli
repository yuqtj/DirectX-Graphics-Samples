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
#include "Kernels.hlsli"

Texture2D<float> g_inValues : register(t0); // ToDo input is 3841x2161 instead of 2160p..

Texture2D<float4> g_inNormal : register(t1);
Texture2D<float> g_inDepth : register(t2);
Texture2D<float> g_inVariance : register(t4);   // ToDo remove
Texture2D<float> g_inSmoothedVariance : register(t5);   // ToDo rename
Texture2D<float> g_inHitDistance : register(t6);   // ToDo remove?
Texture2D<float2> g_inPartialDistanceDerivatives : register(t7);   // ToDo remove?

RWTexture2D<float> g_outFilteredValues : register(u0);
RWTexture2D<float> g_outFilteredVariance : register(u1);
#if !WORKAROUND_ATROUS_VARYING_OUTPUTS 
RWTexture2D<float> g_outFilterWeightSum : register(u2);
#endif
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> g_CB: register(b0);


void AddFilterContribution(
    inout float weightedValueSum, 
    inout float weightedVarianceSum, 
    inout float weightSum, 
    in float value, 
    in float stdDeviation,
    in float depth, 
    in float3 normal, 
    in float obliqueness, 
    in float2 ddxy,
    in uint row, 
    in uint col, 
    in uint2 DTid)
{
    const float valueSigma = g_CB.valueSigma;
    const float normalSigma = g_CB.normalSigma;
    const float depthSigma = g_CB.depthSigma;
 
    int2 pixelOffset = int2(row - FilterKernel::Radius, col - FilterKernel::Radius) << g_CB.kernelStepShift;
    int2 id = int2(DTid) + pixelOffset;
    if (id.x >= 0 && id.y >= 0 && id.x < g_CB.textureDim.x && id.y < g_CB.textureDim.y)
    {
        float iValue = 0.f;
        float iVariance;
        float e_x = 0;

        if (g_CB.outputFilteredValue)
        {
            iValue = g_inValues[id];
            iVariance = g_inSmoothedVariance[id];

            if (g_CB.useCalculatedVariance)
            {
                const float errorOffset = 0.005f;
                e_x = valueSigma > 0.01f ? -abs(value - iValue) / (valueSigma * stdDeviation + errorOffset) : 0;
            }
            else
            {
                e_x = valueSigma > 0.01f ? g_CB.kernelStepShift > 0 ? exp(-abs(value - iValue) / (valueSigma * valueSigma)) : 0 : 0;
            }
        }

        // ToDo standardize index vs id
#if COMPRES_NORMALS
        float4 normalBufValue = g_inNormal[id];
        
        float4 normal4 = float4(DecodeNormal(normalBufValue.xy), normalBufValue.z);
        float iObliqueness = normalBufValue.w;
#else
        float4 normal4 = g_inNormal[id];
#endif 
        float3 iNormal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
        float iDepth = normal4.w;
#else
        float iDepth = g_inDepth[id];
#endif

        // Ref: SVGF
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1;

#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
        float surfaceDistance = obliqueness;
        float iSurfaceDistance = iObliqueness;

        float e_surfaceDistance = w_n > 0.8 && depthSigma > 0.01f ? -abs(surfaceDistance - iSurfaceDistance) / (depthSigma) : 0;
        float e_d = -abs(depth - iDepth) / (0.5 * 0.5);
        e_d += e_surfaceDistance;
#else

#define USE_PARTIAL_DERIVATIVES 1
#if USE_PARTIAL_DERIVATIVES
        float fEpsilon = depth * 0.001f;// depth * 0.005f;     // ToDo finalize the value
        float e_d = depthSigma > 0.01f ? -abs(depth - iDepth) / (depthSigma * length(ddxy * pixelOffset) + fEpsilon) : 0;
#else
        float e_d = depthSigma > 0.01f ? -abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma) : 0;
#endif
#endif
        float w_h = FilterKernel::Kernel[row][col];

        // ToDo apply exp combination where applicable 
        float w_xd = exp(e_x + e_d);        // exp(x) * exp(y) == exp(x + y)
        float w = w_h * w_n * w_xd;

        if (g_CB.outputFilteredValue)
        {
            weightedValueSum += w * iValue;
        }

        weightSum += w;

        // ToDo standardize g_CB naming
        if (g_CB.outputFilteredVariance)
        {
            weightedVarianceSum += w * w * iVariance;   // ToDo rename to sqWeight...
        }
    }
}

// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilterCS::ThreadGroup::Width, AtrousWaveletTransformFilterCS::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint2 Gid : SV_GroupID)
{
    // ToDo double check all CS for out of bounds.
    if (DTid.x >= g_CB.textureDim.x || DTid.y >= g_CB.textureDim.y)
        return;

    // Initialize values to the current pixel / center filter kernel value.
#if COMPRES_NORMALS
    float4 normalBufValue = g_inNormal[DTid];
    float4 normal4 = float4(DecodeNormal(normalBufValue.xy), normalBufValue.z);
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
    float obliqueness = normalBufValue.w;    // ToDO review
#else
    float obliqueness = max(0.0001f, pow(normalBufValue.w, 10));    // ToDO review
#endif
#else
    float4 normal4 = g_inNormal[DTid];
    #if PACK_NORMAL_AND_DEPTH
        float obliqueness = 1;
    #else
        float obliqueness = max(0.0001f, pow(normal4.w, 10));
    #endif
#endif
    float3 normal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
    float depth = normal4.w;
#else
    float depth = g_inDepth[DTid];
#endif

    float value = 0;
    if (g_CB.outputFilteredValue)
    {
        value = g_inValues[DTid];
    }

    float2 ddxy = g_inPartialDistanceDerivatives[DTid];

    float weightSum = FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius];
    float weightedValueSum = weightSum * value;
    float weightedVarianceSum = 0;
    float variance = 0;
    float stdDeviation = 0;
  
    if (g_CB.useCalculatedVariance)
    {
        variance = g_inSmoothedVariance[DTid];
        weightedVarianceSum = FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius] * FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius]
                              * variance;
        stdDeviation = sqrt(variance);
    }

    // Add contributions from the neighborhood.
    [unroll]
    for (UINT r = 0; r < FilterKernel::Width; r++)
    [unroll]
    for (UINT c = 0; c < FilterKernel::Width; c++)
        if (r != FilterKernel::Radius || c != FilterKernel::Radius)
             AddFilterContribution(weightedValueSum, weightedVarianceSum, weightSum, value, stdDeviation, depth, normal, obliqueness, ddxy, r, c, DTid);

#if WORKAROUND_ATROUS_VARYING_OUTPUTS
    float outputValue = (g_CB.outputFilterWeigthSum) ? weightSum : weightedValueSum / weightSum;
    g_outFilteredValues[DTid] = outputValue;
#else
    // ToDo why the resource doesnt get picked up in PIX if its written to under condition?
    if (g_CB.outputFilterWeigthSum)
    {
        g_outFilterWeightSum[DTid] = weightSum;
    }

    // ToDo separate output filtered value and weight sum into two shaders?
    if (g_CB.outputFilteredValue)
    {
        g_outFilteredValues[DTid] = weightedValueSum / weightSum;
    }
#endif
    if (g_CB.outputFilteredVariance)
    {
        g_outFilteredVariance[DTid] = weightedVarianceSum / (weightSum * weightSum);
    }

}