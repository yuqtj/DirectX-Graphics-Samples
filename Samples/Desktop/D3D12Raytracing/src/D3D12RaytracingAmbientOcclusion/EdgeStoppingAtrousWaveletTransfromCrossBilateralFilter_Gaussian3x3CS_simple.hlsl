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
RWTexture2D<float> g_outFilteredValues : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

void AddFilterContribution(inout float weightedValueSum, inout float weightSum, in float value, in float depth, in float3 normal, float obliqueness, in uint row, in uint col, in float w_h, in uint2 DTid)
{
    const float valueSigma = cb.valueSigma;
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    int2 id = int2(DTid)+(int2(row - 1, col - 1) << cb.kernelStepShift);
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        float iValue = g_inValues[id];
#if COMPRES_NORMALS
        uint normalOct = g_inNormalOct[id];
        float3 iNormal = i_octahedral_32(normalOct, 16u);
#elif 1
        float3 iNormal = g_inNormal[id].xyz;
#else
        float3 iNormal = float3(1, 1, 1);
#endif 
        float  iDepth = g_inDepth[id];
#if 0
        float w_d = depth - iDepth;
        float w_x = value - iValue;
        float w_n = dot(normal, iNormal);
#else
        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
        float w_x = valueSigma > 0.01f ? cb.kernelStepShift > 0 ? exp(-abs(value - iValue) / (valueSigma * valueSigma)) : 1.f : 1.f;

        // Ref: SVGF
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
#endif
        float w = w_h * w_x * w_n * w_d;

        weightedValueSum += w * iValue;
        weightSum += w;
    }
}

// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    const uint N = 3;
    const float kernel1D[N] = { 0.27901, 0.44198, 0.27901 };
    const float kernel[N][N] =
    {
        { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2] },
        { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2] },
        { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2] },
    };
#if COMPRES_NORMALS
    uint normalOct = g_inNormalOct[DTid];
    float3 normal = i_octahedral_32(normalOct, 16u);
    float obliqueness = 1.f;
#elif 1
    float4 normal4 = g_inNormal[DTid];
    float3 normal = normal4.xyz;
    float obliqueness = max(0.0001f, pow(normal4.w, 10));
#else
    float3 iNormal = float3(1, 1, 1);
    float obliqueness = 1.f;
#endif 

    float  depth = g_inDepth[DTid];
    float  value = g_inValues[DTid];

    float weightedValueSum = value * kernel[1][1];
    float weightSum = kernel[1][1];

#if 1
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 0, 0, kernel[0][0], DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 1, 0, kernel[0][1], DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 2, 0, kernel[0][2], DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 0, 1, kernel[1][0], DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 2, 1, kernel[1][2], DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 0, 2, kernel[0][0], DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 1, 2, kernel[1][1], DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, 2, 2, kernel[2][2], DTid);
#endif

#if 0
    g_outFilteredValues[DTid] = weightSum > 0.0001f ? weightedValueSum / weightSum : 0.f;
#else
    g_outFilteredValues[DTid] = weightedValueSum / weightSum ;
#endif
}