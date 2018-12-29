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
RWTexture2D<float> g_outFilteredValues : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

groupshared uint gShared[ReduceSumCS::ThreadGroup::Size];

// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    const uint N = 5;
    const float kernel1D[N] = { 1.f / 16, 1.f / 4, 3.f / 8, 1.f / 4, 1.f / 16 };
    const float kernel[N][N] =
    {
        { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2], kernel1D[0] * kernel1D[3], kernel1D[0] * kernel1D[4] },
        { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2], kernel1D[1] * kernel1D[3], kernel1D[1] * kernel1D[4] },
        { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2], kernel1D[2] * kernel1D[3], kernel1D[2] * kernel1D[4] },
        { kernel1D[3] * kernel1D[0], kernel1D[3] * kernel1D[1], kernel1D[3] * kernel1D[2], kernel1D[3] * kernel1D[3], kernel1D[3] * kernel1D[4] },
        { kernel1D[4] * kernel1D[0], kernel1D[4] * kernel1D[1], kernel1D[4] * kernel1D[2], kernel1D[4] * kernel1D[3], kernel1D[4] * kernel1D[4] },
    };
    float3 normal = g_inNormal[DTid].xyz;
    float  depth = g_inDepth[DTid];
    float  value = g_inValues[DTid];

    // Ref: SVGF
    const float valueSigma = cb.valueSigma;
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    float sum = 0.f;
    float sumWeight = 0.f;
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
        {
            int2 id = int2(DTid) + (int2(row - 2, col - 2) << cb.kernelStepShift);
            float val = g_inValues[id];
            float w = 0.f;
            if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
            {
                float3 iNormal = g_inNormal[id].xyz;
                float  iDepth = g_inDepth[id];

                float w_h = kernel[row][col];
                float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) / (depthSigma * depthSigma)) : 1.f;
                float w_x = valueSigma > 0.01f ? cb.kernelStepShift > 0 ? exp(-abs(value - val) / (valueSigma * valueSigma)) : 1.f : 1.f;

                // Ref: SVGF
                float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
                w = w_h * w_x * w_n * w_d;
            }
            sum += w * val;
            sumWeight += w;
        }

    g_outFilteredValues[DTid] = sumWeight > 0.0001f ? sum / sumWeight : 0.f;
}