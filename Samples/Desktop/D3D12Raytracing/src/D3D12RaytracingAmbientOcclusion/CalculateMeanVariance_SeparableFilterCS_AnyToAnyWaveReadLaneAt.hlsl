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

// Desc: Calculate Variance and Mean via a separable kernel.
// Supports up to 9x9 kernels.
// Requirements:
// - wave lane size 16 or higher.
// Performance: 
// 0.235ms for 7x7 kernel at 4K on 2080Ti.

// ToDo handle inactive pixels
// ToDo check WaveLaneCountMin cap to be 16 or higher and fail or disable using this shader.

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValues : register(t0);
RWTexture2D<float2> g_outMeanVariance : register(u0);

ConstantBuffer<CalculateMeanVarianceConstantBuffer> cb: register(b0);

// Group shared memory cache for the row aggregated results.
groupshared uint PackedRowResultCache[16][8];            // 16bit float valueSum, squaredValueSum.
groupshared uint NumValuesCache[16][8]; 

// ToDo use a commonly defined value.
#define INVALID_VALUE -1

// Load up to 16x16 pixels and filter them horizontally.
// The output is cached in Shared Memory and contains NumRows x 8 results.
void FilterHorizontally(in uint2 Gid, in uint GI)
{
    const uint2 GroupDim = uint2(8, 8);
    const uint NumValuesToLoadPerRowOrColumn = GroupDim.x + (cb.kernelWidth - 1);

    // Process the thread group as row-major 16x4, where each sub group of 16 threads processes one row.
    // Each thread loads up to 4 values, with the sub groups loading rows interleaved.
    // Loads up to 16x4x4 == 256 input values.
    uint2 GTid16x4_row0 = uint2(GI % 16, GI / 16);
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
        float value = INVALID_VALUE;

        // The lane is out of bounds of the GroupDim + kernel, 
        // but could be within bounds of the input texture,
        // so don't read it from the texture.
        // but need to keep it as an active lane for a below split sum.
        if (GTid16x4.x < NumValuesToLoadPerRowOrColumn)
        {
            value = g_inValues[pixel];
        }

        // Filter the values for the first GroupDim columns.
        {
            // Accumulate for the whole kernel width.
            float valueSum = 0;
            float squaredValueSum = 0;
            uint numValues = 0;

            // Since a row uses 16 lanes, but we only need to calculate the aggregate for the first half (8) lanes,
            // split the kernel wide aggregation among the first 8 and the second 8 lanes, and then combine them.
            
            // Initialize the first 8 lanes to the first cell contribution of the kernel. 
            // This covers the remainder of 1 in cb.kernelWidth / 2 used in the loop below. 
            if (GTid16x4.x < GroupDim.x && value != INVALID_VALUE)
            {
                valueSum = value;
                squaredValueSum = value * value;
                numValues++;
            }

            for (uint c = 0; c < cb.kernelRadius; c++)
            {
                uint laneToReadFrom = Row_BaseWaveLaneIndex + 1 + c +
                                     (GTid16x4.x < GroupDim.x ? GTid16x4.x : (GTid16x4.x - GroupDim.x) + cb.kernelRadius);
                float cValue = WaveReadLaneAt(value, laneToReadFrom);
                if (cValue != INVALID_VALUE)
                {
                    valueSum += cValue;
                    squaredValueSum += cValue * cValue;
                    numValues++;
                }
            }
            
            // Combine the sub-results.
            uint laneToReadFrom = min(WaveGetLaneCount() - 1, Row_BaseWaveLaneIndex + GTid16x4.x + 8);
            valueSum += WaveReadLaneAt(valueSum, laneToReadFrom);
            squaredValueSum += WaveReadLaneAt(squaredValueSum, laneToReadFrom);

            // Store only the valid results, i.e. first GroupDim columns.
            if (GTid16x4.x < GroupDim.x)
            {
                // ToDo offset row start by rowIndex to avoid bank conflicts on read
                PackedRowResultCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(float2(valueSum, squaredValueSum));
                NumValuesCache[GTid16x4.y][GTid16x4.x] = numValues;
            }
        }
    }
}

void FilterVertically(uint2 DTid, in uint2 GTid)
{
    float valueSum = 0;
    float squaredValueSum = 0;
    uint numValues = 0;

    // Accumulate for the whole kernel.
    for (uint c = 0; c < cb.kernelWidth; c++)
    {
        uint rowID = GTid.y + c;
        float2 unpackedRowSum = HalfToFloat2(PackedRowResultCache[rowID][GTid.x]);

        valueSum += unpackedRowSum.x;
        squaredValueSum += unpackedRowSum.y;

        numValues += NumValuesCache[rowID][GTid.x];
    }
       
    // Calculate mean and variance.
    // Adjust the kernel size for the valid pixels. 
    // Out of texture bound reads return 0 and thus have no impact on the aggregates.
    uint leftMostIndex = max(0, int(DTid.x) - int(cb.kernelRadius));
    uint rightMostIndex = min(cb.textureDim.x - 1, DTid.x + cb.kernelRadius);
    uint kernelWidthX = rightMostIndex - leftMostIndex + 1;

    uint topMostIndex = max(0, int(DTid.y) - int(cb.kernelRadius));
    uint bottomMostIndex = min(cb.textureDim.y - 1, DTid.y + cb.kernelRadius);
    uint kernelWidthY = bottomMostIndex - topMostIndex + 1;

    float invN = 1.f / max(numValues, 1);
    float mean = invN * valueSum;

    // Apply Bessel's correction to the estimated variance, divide by N-1, 
    // since the true population mean is not known. It is only estimated as the sample mean.
    float besselCorrection = numValues / float(max(numValues, 2) - 1);
    float variance = besselCorrection * (invN * squaredValueSum - mean * mean);

    variance = max(0, variance);    // Ensure variance doesn't go negative due to imprecision.
    
    g_outMeanVariance[DTid] = numValues > 0 ? float2(mean, variance) : INVALID_VALUE;
}


[numthreads(CalculateMeanVarianceFilter::ThreadGroup::Width, CalculateMeanVarianceFilter::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex, uint2 DTid : SV_DispatchThreadID)
{
    FilterHorizontally(Gid, GI);
    GroupMemoryBarrierWithGroupSync();

    FilterVertically(DTid, GTid);
}
