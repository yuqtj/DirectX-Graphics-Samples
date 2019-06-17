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
// 0.4ms -> 0.31ms for 7x7 kernel at 4K on TitanXp.
#define PACK_OPTIMIZATION 1
// ToDo pack mean and variance ouputs to 2x16bit

#if PACK_OPTIMIZATION
groupshared float VCache[256];
groupshared UINT PackedResultCache[256];            // 16bit float valueSum, squaredValueSum.

#else
groupshared float VCache[256];
groupshared float ValueSumCache[256];
groupshared float SquaredValueSumCache[256];
#endif

void LoadToSharedMemory(UINT smemIndex, int2 pixel)
{
    VCache[smemIndex] = g_inValues[pixel];
}

void PrefetchData(uint index, int2 ST)
{
    // ToDo use gather for values
    LoadToSharedMemory(index, ST + int2(-1, -1));
    LoadToSharedMemory(index + 1, ST + int2(0, -1));
    LoadToSharedMemory(index + 16, ST + int2(-1, 0));
    LoadToSharedMemory(index + 17, ST + int2(0, 0));
}


void BlurHorizontally(uint leftMostIndex)
{
    // Load the reference values for the current pixel.
    UINT Cid = leftMostIndex + cb.kernelRadius;
    UINT numWeights = 0;
    float valueSum = 0;
    float squaredValueSum = 0;

    // Accumulate for the whole kernel.
    for (UINT c = 0; c < cb.kernelWidth; c++)
    {
        UINT ID = leftMostIndex + c;
        float value = VCache[ID];

        valueSum += value;
        squaredValueSum += value * value;
    }

#if PACK_OPTIMIZATION
    PackedResultCache[leftMostIndex] = Float2ToHalf(float2(valueSum, squaredValueSum));
#else
    SquaredValueSumCache[leftMostIndex] = squaredValueSum;
    ValueSumCache[leftMostIndex] = valueSum;
#endif
}

// ToDo handle OOB and inactive pixels

void BlurVertically(uint2 DTid, uint topMostIndex)
{
    // Load the reference values for the current pixel.
    UINT Cid = topMostIndex + cb.kernelRadius * 16;

    float valueSum = 0;
    float squaredValueSum = 0;

    // Accumulate for the whole kernel.
    for (UINT c = 0; c < cb.kernelWidth; c++)
    {
        UINT ID = topMostIndex + c * 16;
#if PACK_OPTIMIZATION
        float2 unpackedValue = HalfToFloat2(PackedResultCache[ID]);
        valueSum += unpackedValue.x;
        squaredValueSum += unpackedValue.y;
 #else
        squaredValueSum += SquaredValueSumCache[ID];
        valueSum += ValueSumCache[ID];
#endif
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


[numthreads(CalculateVariance_Bilateral::ThreadGroup::Width, CalculateVariance_Bilateral::ThreadGroup::Height, 1)]
void main(uint GI : SV_GroupIndex, uint2 GTid : SV_GroupThreadID, uint2 DTid : SV_DispatchThreadID)
{
    //
    // Load 4 pixels per thread into LDS to fill the 16x16 LDS cache
    // Goal: Load 16x16: <[-4,-4], [11,11]> block from input texture for 8x8 group.
    //
    PrefetchData(GTid.x << 1 | GTid.y << 5, int2(DTid + GTid) - 3);
    GroupMemoryBarrierWithGroupSync();

    // Goal:  End up with a 9x9 patch that is blurred.

    //
    // Horizontally blur the pixels.	16x16 -> 16x8
    // Blur two columns per thread ID and one full row of 8 wide per 4 threads.
    //
    BlurHorizontally((GI / 4) * 16 + (GI % 4) * 2 + (4 - cb.kernelRadius)); // Even columns.
    BlurHorizontally(1 + (GI / 4) * 16 + (GI % 4) * 2 + (4 - cb.kernelRadius)); // Odd columns.
    GroupMemoryBarrierWithGroupSync();

    //
    // Vertically blur the pixels.	    16x8 -> 8x8	
    //
    BlurVertically(DTid, GTid.y * 16 + GTid.x + 17 * (4 - cb.kernelRadius));
}
