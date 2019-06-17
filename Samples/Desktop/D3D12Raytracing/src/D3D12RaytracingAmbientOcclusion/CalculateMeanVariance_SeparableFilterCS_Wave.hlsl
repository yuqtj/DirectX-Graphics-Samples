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

// Desc: Calculate Variance via Separable kernel.
// Pitfalls: // ToDo rename drawback 
//  - it is not edge aware.
// Performance: 
// 0.126ms for 7x7 kernel at 1080p on TitanXP.
// 0.368ms for 7x7 kernel at 4K on 2080Ti.

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValues : register(t0);
RWTexture2D<float2> g_outMeanVariance : register(u0);

ConstantBuffer<CalculateMeanVarianceConstantBuffer> cb: register(b0);

// Group shared memory caches.
// Spaced at 4 Byte element widths to avoid bank conflicts on access.
// Trade precision for speed and pack floats to 16bit.
// 0.4ms -> 0.31ms for 7x7 kernel at 4K on TitanXp. // ToDo test
groupshared UINT PackedResultCache[16][8];            // 16bit float valueSum, squaredValueSum.

// ToDo rename blur
// Load up to 16x16 pixels and blur them horizontally.
void BlurHorizontally(in uint2 Gid, in uint2 GTid)
{
    const uint2 KernelResultDim = uint2(8, 8);
    const uint NumRowsToLoadPerThread = 4;
    const uint NumValuesToLoadPerRowOrColumn = 8 + (cb.kernelWidth - 1);

    if (GTid.x >= NumValuesToLoadPerRowOrColumn)
    {
        return;
    }

    // 16x4 Group dim.
    int2 KernelBasePixel = Gid * KernelResultDim - int2(cb.kernelRadius, cb.kernelRadius);

    // Load row of data for each 16 lanes.
    // Load up to 4 pixels in a column per thread, totalling up to 256 values for 16x4 Thread Group.
    for (UINT i = 0; i < NumRowsToLoadPerThread; i++)
    {
        // ToDo test interleave rows.
        uint rowID = i + GTid.y * NumRowsToLoadPerThread;
        if (rowID >= NumValuesToLoadPerRowOrColumn 
        {
            break;
        }

        int2 pixel = KernelBasePixel + int2(GTid.x, GTid.y * NumRowsToLoadPerThread);
        float value = g_inValues[pixel];

        if (GTid.x < KernelResultDim.x)
        {
            // Accumulate for the whole kernel width.
            float valueSum = value;
            float squaredValueSum = value * value;
            for (UINT c = 1; c < cb.kernelWidth; c++)
            {
                // Broadcast values across the wave.
                float xValue = WaveReadLaneAt(value, GTid.x + c);
                valueSum += xValue;
                squaredValueSum += xValue * xValue;
            }
            // ToDo test transposing - trade bank collision on writes vs reads
            // ToDo offset row start by rowIndex to avoid bank conflicts on read
            PackedResultCache[rowID][GTid.x] = Float2ToHalf(float2(valueSum, squaredValueSum));
        }
    }
}

// ToDo handle OOB and inactive pixels
void BlurVertically(uint2 DTid, in uint GI)
{
    float valueSum = 0;
    float squaredValueSum = 0;

    // Accumulate for the whole kernel.
    for (UINT c = 0; c < cb.kernelWidth; c++)
    {
        uint2 ID = uint2(GI % 8, GI / 8);

        float2 unpackedValue = HalfToFloat2(PackedResultCache[ID.y][ID.x]);
        valueSum += unpackedValue.x;
        squaredValueSum += unpackedValue.y;
    }
       
    // Calculate mean and variance
    UINT N = cb.kernelWidth * cb.kernelWidth;
    float invN = 1.f / N;
    float mean = invN * valueSum;

    // Apply Bessel's correction to the estimated variance, divide by N-1, 
    // since the true population mean is not known. It is only estimated as the sample mean.
    float besselCorrection = N / float(N - 1);
    float variance = besselCorrection * (invN * squaredValueSum - mean * mean);

    variance = max(0, variance);    // Ensure variance doesn't go negative due to imprecision.
    
    g_outMeanVariance[DTid] = float2(variance, mean);
}


[numthreads(CalculateMeanVarianceFilter16x4::ThreadGroup::Width, CalculateMeanVarianceFilter16x4::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex, uint2 DTid : SV_DispatchThreadID)
{
    BlurHorizontally(Gid, GTid);
    GroupMemoryBarrierWithGroupSync();

    BlurVertically(DTid, GI);
}
