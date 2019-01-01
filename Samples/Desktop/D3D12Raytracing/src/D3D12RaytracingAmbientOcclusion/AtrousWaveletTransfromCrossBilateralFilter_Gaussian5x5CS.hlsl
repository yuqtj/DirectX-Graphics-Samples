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

// Atrous Wavelet Transform Filter
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
    
    float sum = 0.f;
    float sumWeight = 0.f;
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
        {
            int2 id = int2(DTid) + (int2(row - 2, col - 2) << cb.kernelStepShift);
            float val = g_inValues[id];
            float w_h = (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y) ?
                kernel[row][col] :
                0.f;
            sum += w_h * val;
            sumWeight += w_h;
        }

    g_outFilteredValues[DTid] = sumWeight > 0.0001f ? sum / sumWeight : 0.f;
}