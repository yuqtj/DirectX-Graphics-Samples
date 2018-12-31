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

Texture2D<float> g_inValues : register(t0);
Texture2D<float4> g_inNormal : register(t1);
Texture2D<float> g_inDepth : register(t2);
Texture2D<uint> g_inNormalOct : register(t3);
RWTexture2D<float> g_outFilteredValues : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

void AddFilterContribution(inout float weightedValueSum, inout float weightSum, in float value, in float depth, in float3 normal, in uint row, in uint col, in uint2 DTid)
{
    const float valueSigma = cb.valueSigma;
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    int2 id = int2(DTid)+(int2(row - 1, col - 1) << cb.kernelStepShift);
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        float iValue = g_inValues[id];
        float3 iNormal = g_inNormal[id].xyz;
        float  iDepth = g_inDepth[id];

        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) / (depthSigma * depthSigma)) : 1.f;
        float w_x = valueSigma > 0.01f ? cb.kernelStepShift > 0 ? exp(-abs(value - iValue) / (valueSigma * valueSigma)) : 1.f : 1.f;

        // Ref: SVGF
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
        float w = w_x * w_n * w_d;

        weightedValueSum += w * iValue;
        weightSum += w;
    }
}
// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    float3 normal = g_inNormal[DTid].xyz;
    float  depth = g_inDepth[DTid];
    float  value = g_inValues[DTid];

    float weightedValueSum = value;
    float weightSum = 1.f;

    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 0, 0, DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 0, 1, DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 0, 2, DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 1, 0, DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 1, 2, DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 2, 0, DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 2, 1, DTid);
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, 2, 2, DTid);

    g_outFilteredValues[DTid] = weightSum > 0.0001f ? weightedValueSum / weightSum : 0.f;
}