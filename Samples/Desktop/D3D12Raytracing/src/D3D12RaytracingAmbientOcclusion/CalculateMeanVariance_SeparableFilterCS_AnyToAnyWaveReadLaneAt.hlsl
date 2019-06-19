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

// ToDo
// Desc: Calculate Variance and Mean via Separable kernel.
// This shader utilizes any-to-any WaveReadLaneAt
// Assumes the GPUs wave size is 16 or higher.
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

// Group shared memory cache for the row filtered results.
groupshared uint PackedRowResultCache[16][8];            // 16bit float valueSum, squaredValueSum.

// Load up to 16x16 pixels and filter them horizontally.
// The output is cached in Shared Memory and contains NumRows x 8 results.
void FilterHorizontally(in uint2 Gid, in uint GI)
{
    const uint NumValuesToLoadPerRowOrColumn = 8 + (cb.kernelWidth - 1);

    // Process the thread group as row-major 16x4, where each sub group of 16 threads processes one row.
    // Each thread loads up to 4 values, with the sub groups loading rows interleaved.
    // Loads up to 16x4x4 == 256 input values.
    uint2 GTid16x4_row0 = uint2(GI % 16, GI / 16);
    if (GTid16x4_row0.x >= NumValuesToLoadPerRowOrColumn)
    {
        return;
    }

    const uint2 GroupDim = uint2(8, 8);
    int2 KernelBasePixel = Gid * GroupDim - int2(cb.kernelRadius, cb.kernelRadius);
    const uint NumRowsToLoadPerThread = 4;
    const uint Row_BaseWaveLaneIndex = (WaveGetLaneIndex() / 16) * 16;
    [unroll]
    for (uint i = 0; i < NumRowsToLoadPerThread; i++)
    {
        uint2 GTid16x4 = GTid16x4_row0 + uint2(0, i * 4);
        if (GTid16x4.y >= NumValuesToLoadPerRowOrColumn)
        {
            break;
        }

        // Load all the contributing columns for each row.
        int2 pixel = KernelBasePixel + GTid16x4;
        float value = g_inValues[pixel];

        // Filter the values for the first GroupDim columns.
        {
            // Accumulate for the whole kernel width.
#if 1
            float valueSum = 0;
            float squaredValueSum = 0;

            // Since we are using 2x or more threads than the kernel width is for each row in a 16 thread lane groups,
            // and only the first half of those lanes output actual results below,
            // split the kernel wide aggregation among the first 8 and the second 8 lanes.
            
            // Initialize the first 8 lanes to the first cell contribution of the kernel. 
            // This covers the remainder of 1 in cb.kernelWidth / 2 used in the loop below. 
            if (GTid16x4.x < GroupDim.x)
            {
                valueSum = value;
                squaredValueSum = value * value;
            }

            for (uint c = 0; c < cb.kernelRadius; c++)
            {
                uint laneToReadFrom = Row_BaseWaveLaneIndex + 
                                     GTid16x4.x < GroupDim.x ? GTid16x4.x + 1 + c : (GTid16x4.x - GroupDim.x) + 1 + c + cb.kernelRadius;
                float cValue = WaveReadLaneAt(value, laneToReadFrom);
                valueSum += cValue;
                squaredValueSum += cValue * cValue;
            }
            
            // Combine the sub-results.
            // Make sure not to index outside the warp, as otherwise the results get incorrect, 
            // even though the results from such lanes are ignored below.
            uint laneToReadFrom = max(WaveGetLaneCount() - 1, Row_BaseWaveLaneIndex + GTid16x4.x + 8);
            valueSum += WaveReadLaneAt(valueSum, laneToReadFrom);
            squaredValueSum += WaveReadLaneAt(squaredValueSum, laneToReadFrom);
#else
            float valueSum = value;
            float squaredValueSum = value * value;

            for (uint c = 1; c < cb.kernelWidth; c++)
            {
                // Retrieve the loaded values for the row.
                float xValue = WaveReadLaneAt(value, Row_BaseWaveLaneIndex + GTid16x4.x + c);
                valueSum += xValue;
                squaredValueSum += xValue * xValue;
            }
#endif
            // Store only the valid results, i.e. first GroupDim columns.
            if (GTid16x4.x < GroupDim.x)
            {
                // ToDo offset row start by rowIndex to avoid bank conflicts on read
                PackedRowResultCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(float2(valueSum, squaredValueSum));
            }
        }
    }
}


// ToDo handle OOB and inactive pixels
void FilterVertically(uint2 DTid, in uint2 GTid)
{
    float valueSum = 0;
    float squaredValueSum = 0;

    // Accumulate for the whole kernel.
    for (uint c = 0; c < cb.kernelWidth; c++)
    {
        uint rowID = GTid.y + c;
        float2 unpackedRowSum = HalfToFloat2(PackedRowResultCache[rowID][GTid.x]);

        valueSum += unpackedRowSum.x;
        squaredValueSum += unpackedRowSum.y;
    }
       
    // Calculate mean and variance
    uint N = cb.kernelWidth * cb.kernelWidth;
    float invN = 1.f / N;
    float mean = invN * valueSum;

    // Apply Bessel's correction to the estimated variance, divide by N-1, 
    // since the true population mean is not known. It is only estimated as the sample mean.
    float besselCorrection = N / float(N - 1);
    float variance = besselCorrection * (invN * squaredValueSum - mean * mean);

    variance = max(0, variance);    // Ensure variance doesn't go negative due to imprecision.
    
    g_outMeanVariance[DTid] = float2(variance, mean);
}


[numthreads(CalculateMeanVarianceFilter::ThreadGroup::Width, CalculateMeanVarianceFilter::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex, uint2 DTid : SV_DispatchThreadID)
{
    FilterHorizontally(Gid, GI);
    GroupMemoryBarrierWithGroupSync();

    FilterVertically(DTid, GTid);
}
