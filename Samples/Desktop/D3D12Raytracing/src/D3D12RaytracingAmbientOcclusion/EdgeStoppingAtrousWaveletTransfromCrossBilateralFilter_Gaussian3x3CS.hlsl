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
    const uint N = 3;
    const float kernel1D[N] = { 0.27901, 0.44198, 0.27901 };
    const float kernel[N][N] =
    {
        { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2] },
        { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2] },
        { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2] },
    };
    float3 normal = g_inNormal[DTid].xyz;
    float  depth = g_inDepth[DTid];

    // Ref: SVGF
    const UINT sigmaValue = 4;
    const UINT sigmaNormal = 128;
    const UINT sigmaDepth = 5;
    
    float sum = 0.f;
    float sumWeight = 0.f;
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
        {
            int2 id = int2(DTid) + (int2(row - 1, col - 1) << cb.kernelStepShift);
            float val = g_inValues[id];
            float w = 0.f;
            if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
            {
                float3 iNormal = g_inNormal[id].xyz;
                float  iDepth = g_inDepth[id];

                float w_h = 1;// kernel[row][col];
                float w_d =  exp(-abs(depth - iDepth) / (sigmaDepth * sigmaDepth));

                // Ref: SVGF
                float w_n = pow(max(0, dot(normal, iNormal)), sigmaNormal);
                w = w_h * w_n * w_d;
            }
            sum += w * val;
            sumWeight += w;
        }

    g_outFilteredValues[DTid] = sumWeight > 0.0001f ? sum / sumWeight : 0.f;
}