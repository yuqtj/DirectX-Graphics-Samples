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
// Desc: Filters invalid values from neighborhood via gaussian filter.
// Supports up to 9x9 kernels.
// Requirements:
// - wave lane size 16 or higher.
// Performance: 
// ToDo:

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "RTAO/Shaders/RTAO.hlsli"

#define GAUSSIAN_KERNEL_3X3
#include "Kernels.hlsli"

Texture2D<float> g_inValues : register(t0);
Texture2D<NormalDepthTexFormat> g_inNormalDepth : register(t1);
Texture2D<float> g_inBlurStrength: register(t2);
RWTexture2D<float> g_outValues : register(u0);

RWTexture2D<float4> g_outDebug1 : register(u3);
RWTexture2D<float4> g_outDebug2 : register(u4);


ConstantBuffer<BilateralFilterConstantBuffer> cb: register(b0);

// Group shared memory cache for the row aggregated results.
// ToDo parameterize SMEM based on kernel dims.
groupshared uint PackedValueDepthCache[16][8];         // 16bit float value, depth.
groupshared uint PackedRowResultCache[16][8];            // 16bit float weightedValueSum, weightSum.
groupshared uint PackedEncodedNormalCache[16][8];        // 16bit float encodedNormal X and Y.

uint2 GetPixelIndex(in uint2 Gid, in uint2 GTid)
{
    // Find a DTID with steps in between the group threads and groups interleaved to cover all pixels.
    uint2 GroupDim = uint2(8, 8);
    uint2 groupBase = (Gid / cb.step) * GroupDim * cb.step + Gid % cb.step;
    uint2 groupThreadOffset = GTid * cb.step;
    uint2 sDTid = groupBase + groupThreadOffset;

    return sDTid;
}

float ReadValue(in uint2 pixel)
{
    if (cb.readWriteUAV_and_skipPassthrough)
    {
        return g_outValues[pixel];
    }
    else
    {
        return g_inValues[pixel];
    }
}

// Load up to 16x16 pixels and filter them horizontally.
// The output is cached in Shared Memory and contains NumRows x 8 results.
void FilterHorizontally(in uint2 Gid, in uint GI)
{
    const uint2 GroupDim = uint2(8, 8);
    const uint NumValuesToLoadPerRowOrColumn = GroupDim.x + (FilterKernel::Width - 1);

    // Process the thread group as row-major 16x4, where each sub group of 16 threads processes one row.
    // Each thread loads up to 4 values, with the sub groups loading rows interleaved.
    // Loads up to 16x4x4 == 256 input values.
    // ToDo rename to 4x16
    uint2 GTid16x4_row0 = uint2(GI % 16, GI / 16);
    int2 GroupKernelBasePixel = GetPixelIndex(Gid, 0) - int(FilterKernel::Radius * cb.step);
    const uint NumRowsToLoadPerThread = 4;
    const uint Row_BaseWaveLaneIndex = (WaveGetLaneIndex() / 16) * 16;

    // ToDo load 8x8 center values to cache and skip if none of the values are missing.
    // ToDo blend low frame age values too with a falloff?

    [unroll]
    for (uint i = 0; i < NumRowsToLoadPerThread; i++)
    {
        uint2 GTid16x4 = GTid16x4_row0 + uint2(0, i * 4);
        if (GTid16x4.y >= NumValuesToLoadPerRowOrColumn)
        {
            break;
        }

        // Load all the contributing columns for each row.
        int2 pixel = GroupKernelBasePixel + GTid16x4 * cb.step;
        float value = RTAO::InvalidAOValue;
        float depth = 0;
        float3 normal = 0;

        // The lane is out of bounds of the GroupDim + kernel, 
        // but could be within bounds of the input texture,
        // so don't read it from the texture.
        // However, we need to keep it as an active lane for a below split sum.
        if (GTid16x4.x < NumValuesToLoadPerRowOrColumn && IsWithinBounds(pixel, cb.textureDim))
        {
            value = ReadValue(pixel);

            float2 encodedNormal;
            UnpackEncodedNormalDepth(g_inNormalDepth[pixel], encodedNormal, depth);
            
            normal = DecodeNormal(encodedNormal);
            PackedEncodedNormalCache[GTid16x4.y][GTid16x4.x - FilterKernel::Radius] = Float2ToHalf(encodedNormal);
        }

        // Cache the kernel center values.
        if (IsInRange(GTid16x4.x, FilterKernel::Radius, FilterKernel::Radius + GroupDim.x - 1))
        {
            PackedValueDepthCache[GTid16x4.y][GTid16x4.x - FilterKernel::Radius] = Float2ToHalf(float2(value, depth));
        }

        // Filter the values for the first GroupDim columns.
        {
            // Accumulate for the whole kernel width.
            float weightedValueSum = 0;
            float weightSum = 0;

            // Since a row uses 16 lanes, but we only need to calculate the aggregate for the first half (8) lanes,
            // split the kernel wide aggregation among the first 8 and the second 8 lanes, and then combine them.


            // Get the lane index that has the first value for a kernel in this lane.
            uint Row_KernelStartLaneIndex =
                (Row_BaseWaveLaneIndex + GTid16x4.x)
                - (GTid16x4.x < GroupDim.x
                    ? 0
                    : GroupDim.x);

            // Get values for the kernel center.
            uint kcLaneIndex = Row_KernelStartLaneIndex + FilterKernel::Radius;
            float kcValue = WaveReadLaneAt(value, kcLaneIndex);
            float kcDepth = WaveReadLaneAt(depth, kcLaneIndex);
            float3 kcNormal = WaveReadLaneAt(normal, kcLaneIndex);

            // Initialize the first 8 lanes to the center cell contribution of the kernel. 
            // This covers the remainder of 1 in FilterKernel::Width / 2 used in the loop below. 
            if (GTid16x4.x < GroupDim.x && kcValue != RTAO::InvalidAOValue && kcDepth != 0)
            {
                float w = FilterKernel::Kernel1D[FilterKernel::Radius];
                weightedValueSum = w * kcValue;
                weightSum = w;
            }

            // Second 8 lanes start just past the kernel center.
            uint KernelCellIndexOffset =
                GTid16x4.x < GroupDim.x
                ? 0
                : (FilterKernel::Radius + 1); // Skip over the already accumulated center cell of the kernel.


            // For all columns in the kernel.
            for (uint c = 0; c < FilterKernel::Radius; c++)
            {
                uint kernelCellIndex = KernelCellIndexOffset + c;

                uint laneToReadFrom = Row_KernelStartLaneIndex + kernelCellIndex;
                float cValue = WaveReadLaneAt(value, laneToReadFrom);
                float cDepth = WaveReadLaneAt(depth, laneToReadFrom);
                float3 cNormal = WaveReadLaneAt(normal, laneToReadFrom);

                if (cValue != RTAO::InvalidAOValue && kcDepth != 0 && cDepth != 0)
                {
                    float w = FilterKernel::Kernel1D[kernelCellIndex];

                    float depthThreshold = 0.01 + cb.step * 0.001 * abs(int(FilterKernel::Radius) - c);
                    float w_d = abs(kcDepth - cDepth) <= depthThreshold * kcDepth;

                    float w_n = max(pow(dot(kcNormal, cNormal), cb.normalWeightExponent), cb.minNormalWeightStrength);

                    w *= w_d * w_n;

                    weightedValueSum += w * cValue;
                    weightSum += w;
                }
            }

            // Combine the sub-results.
            uint laneToReadFrom = min(WaveGetLaneCount() - 1, Row_BaseWaveLaneIndex + GTid16x4.x + GroupDim.x);
            weightedValueSum += WaveReadLaneAt(weightedValueSum, laneToReadFrom);
            weightSum += WaveReadLaneAt(weightSum, laneToReadFrom);

            // Store only the valid results, i.e. first GroupDim columns.
            if (GTid16x4.x < GroupDim.x)
            {
                // ToDo offset row start by rowIndex to avoid bank conflicts on read
                PackedRowResultCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(float2(weightedValueSum, weightSum));
            }
        }
    }
}

float BilateralWeight_DepthNormalKernelAware(
    in float ActualDistance,
    float3 ActualNormal,
    float4 SampleDistances,
    float3 SampleNormals[4],
    float4 BilinearWeights,
    float4 SampleValues)
{
    float4 depthWeights = 1.0 / (abs(SampleDistances - ActualDistance) + 1e-6 * ActualDistance);
    float4 normalWeights = float4(
        pow(dot(ActualNormal, SampleNormals[0]), 32),
        pow(dot(ActualNormal, SampleNormals[1]), 32),
        pow(dot(ActualNormal, SampleNormals[2]), 32),
        pow(dot(ActualNormal, SampleNormals[3]), 32)
        );
    float4 weights = normalWeights * depthWeights * BilinearWeights;

    return InterpolateValidValues(weights, SampleValues);
}

void FilterVertically(uint2 DTid, in uint2 GTid, in float blurStrength)
{
    float2 kcValueDepth = HalfToFloat2(PackedValueDepthCache[GTid.y + FilterKernel::Radius][GTid.x]);
    float kcValue = kcValueDepth.x;
    float kcDepth = kcValueDepth.y;

    float2 kcEncodedNormal = HalfToFloat2(PackedEncodedNormalCache[GTid.y + FilterKernel::Radius][GTid.x]);
    float3 kcNormal = DecodeNormal(kcEncodedNormal);

    float filteredValue = kcValue;
    if (kcDepth != 0)
    {
        float weightedValueSum = 0;
        float weightSum = 0;

        // For all rows in the kernel.
        // ToDo Unroll
        for (uint r = 0; r < FilterKernel::Width; r++)
            // ToDo test with skipping center value
        {
            uint rowID = GTid.y + r;

            float2 rUnpackedValueDepth = HalfToFloat2(PackedValueDepthCache[rowID][GTid.x]);
            float rDepth = rUnpackedValueDepth.y;

            float2 rEncodedNormal = HalfToFloat2(PackedEncodedNormalCache[rowID][GTid.x]);
            float3 rNormal = DecodeNormal(rEncodedNormal);

            if (rDepth != 0)
            {
                float2 rUnpackedRowResult = HalfToFloat2(PackedRowResultCache[rowID][GTid.x]);
                float rWeightedValueSum = rUnpackedRowResult.x;
                float rWeightSum = rUnpackedRowResult.y;

                float w = FilterKernel::Kernel1D[r];
                float depthThreshold = 0.01 + cb.step * 0.001 * abs(int(FilterKernel::Radius) - int(r));
                float w_d = abs(kcDepth - rDepth) <= depthThreshold * kcDepth;

                float w_n = max(pow(dot(kcNormal, rNormal), cb.normalWeightExponent), cb.minNormalWeightStrength);

                w *= w_d * w_n;

                weightedValueSum += w * rWeightedValueSum;
                weightSum += w * rWeightSum;
            }
        }
        filteredValue = weightSum > 1e-9 ? weightedValueSum / weightSum : RTAO::InvalidAOValue;
    }

    g_outValues[DTid] = filteredValue != RTAO::InvalidAOValue ? lerp(kcValue, filteredValue, blurStrength) : RTAO::InvalidAOValue;
}


[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    uint2 sDTid = GetPixelIndex(Gid, GTid);
    // Pass through if all pixels have 0 blur strength set.
    float blurStrength;
    {
        if (GI == 0)
            PackedRowResultCache[0][0] = 0;
        GroupMemoryBarrierWithGroupSync();

        blurStrength = g_inBlurStrength[sDTid];

        float MinBlurStrength = 0.01;
        bool valueNeedsFiltering = blurStrength >= MinBlurStrength;
        if (valueNeedsFiltering)
            PackedRowResultCache[0][0] = 1;

        GroupMemoryBarrierWithGroupSync();

        if (PackedRowResultCache[0][0] == 0)
        {
            if (!cb.readWriteUAV_and_skipPassthrough)
            {
                g_outValues[sDTid] = g_inValues[sDTid];
            }
            return;
        }
    }


    FilterHorizontally(Gid, GI);
    GroupMemoryBarrierWithGroupSync();

    FilterVertically(sDTid, GTid, blurStrength);
}
