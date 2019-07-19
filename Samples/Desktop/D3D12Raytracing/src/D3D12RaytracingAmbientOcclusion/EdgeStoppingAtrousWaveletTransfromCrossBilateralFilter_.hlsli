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
Texture2D<uint> g_inNormalOct : register(t3);
Texture2D<float> g_inVariance : register(t4);   // ToDo remove
Texture2D<float> g_inSmoothedVariance : register(t5);   // ToDo rename
RWTexture2D<float> g_outFilteredValues : register(u0);
RWTexture2D<float> g_outFilteredVariance : register(u1);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);


void AddFilterContribution(inout float weightedValueSum, inout float weightedVarianceSum, inout float weightSum, in float value, in float depth, in float3 normal, float obliqueness, in uint row, in uint col, in float w_h, in uint2 DTid)
{
    const float valueSigma = cb.valueSigma;
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    int2 id = int2(DTid) + (int2(row - 1, col - 1) << cb.kernelStepShift);
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        float iValue = g_inValues[id];

        float iVariance;
        float e_x;
        if (cb.useCalculatedVariance)
        {
            iVariance = g_inSmoothedVariance[id];
            const float errorOffset = 0.005f;
            e_x = valueSigma > 0.01f ? -abs(value - iValue) / (valueSigma * sqrt(max(iVariance, 0)) + errorOffset) : 0;
        }
        else
        {
            e_x = valueSigma > 0.01f ? cb.kernelStepShift > 0 ? exp(-abs(value - iValue) / (valueSigma * valueSigma)) : 0 : 0;
        }


        // ToDo standardize index vs id
#if COMPRES_NORMALS
        float4 normalBufValue = g_inNormal[id];
        float4 normal4 = float4(Decode(normalBufValue.xy), normalBufValue.z);
#else
        float4 normal4 = g_inNormal[id];
#endif 
        float3 iNormal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
        float iDepth = normal4.w;
#else
        float  iDepth = g_inDepth[id];
#endif
        float e_d = depthSigma > 0.01f ? -abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma) : 0;
        // Ref: SVGF
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1;

        // ToDo apply exp combination where applicable 
        float w_xd = exp(e_x + e_d);
        float w = w_h * w_n * w_xd;

        weightedValueSum += w * iValue;
        weightSum += w;

        if (cb.useCalculatedVariance)
        {
            weightedVarianceSum += w * w * iVariance;
        }
    }
}

static const float kernel1D[3] = { 0.27901, 0.44198, 0.27901 };
static const float kernel[3][3] =
{
    { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2] },
    { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2] },
    { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2] },
};

// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilterCS::ThreadGroup::Width, AtrousWaveletTransformFilterCS::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint2 Gid : SV_GroupID)
{
#if COMPRES_NORMALS
    float4 normalBufValue = g_inNormal[DTid];
    float4 normal4 = float4(Decode(normalBufValue.xy), normalBufValue.z);
    float obliqueness = max(0.0001f, pow(normalBufValue.w, 10));
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
    float  depth = g_inDepth[DTid];
#endif

    float  value = g_inValues[DTid];

    float weightedValueSum = kernel[1][1] * value;
    float weightSum = kernel[1][1];
    float weightedVarianceSum = 0;
  
    if (cb.useCalculatedVariance)
    {
        weightedVarianceSum = kernel[1][1] * kernel[1][1] * g_inSmoothedVariance[DTid];
    }

    [unroll]
    for (UINT r = 0; r < 3; r++)
    [unroll]
    for (UINT c = 0; c < 3; c++)
        if (r != 1 || c != 1)
             AddFilterContribution(weightedValueSum, weightedVarianceSum, weightSum, value, depth, normal, obliqueness, r, c, kernel[r][c], DTid);


    g_outFilteredValues[DTid] = weightSum > 0.0001f ? weightedValueSum / weightSum : 0.f;

    if (cb.useCalculatedVariance)
    {
        g_outFilteredVariance[DTid] = weightSum > 0.0001f ? weightedVarianceSum / (weightSum * weightSum) : 0.f;
    }
}