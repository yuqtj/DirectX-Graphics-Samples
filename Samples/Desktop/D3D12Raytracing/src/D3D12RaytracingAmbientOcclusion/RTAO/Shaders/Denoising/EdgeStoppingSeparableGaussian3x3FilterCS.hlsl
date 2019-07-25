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
// Desc: Calculate Variance and Mean via a separable kernel.
// Supports up to 9x9 kernels.
// Requirements:
// - wave lane size 16 or higher.
// Performance: 
// 0.235ms for 7x7 kernel at 4K on 2080Ti.

// ToDo handle inactive pixels
// ToDo check WaveLaneCountMin cap to be 16 or higher and fail or disable using this shader.

#define HLSL
#define GAUSSIAN_KERNEL_3X3
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "Kernels.hlsli"

Texture2D<float> g_inValues : register(t0); // ToDo input is 3841x2161 instead of 2160p..

Texture2D<float4> g_inNormalDepth : register(t1);
Texture2D<float> g_inVariance : register(t4);   // ToDo remove
Texture2D<float2> g_inSmoothedMeanVariance : register(t5);   // ToDo rename
Texture2D<float> g_inHitDistance : register(t6);   // ToDo remove?
Texture2D<float2> g_inPartialDistanceDerivatives : register(t7);   // ToDo remove?

RWTexture2D<float> g_outFilteredValues : register(u0);
RWTexture2D<float> g_outFilteredVariance : register(u1);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> g_CB: register(b0);

#define OUTPUT_FILTERED_VARIANCE 0

float DepthThreshold(float distance, float2 ddxy, float2 pixelOffset, float depthDelta)
{
    float depthThreshold;

    // Todo rename ddxy to dxdy?
    // ToDo use a common helper
    // ToDo rename to: Perspective correct interpolation
    // Pespective correction for the non-linear interpolation
    if (g_CB.perspectiveCorrectDepthInterpolation)
    {
        // Calculate depth via interpolation with perspective correction.
        // Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/visibility-problem-depth-buffer-depth-interpolation
        // Given depth buffer interpolation for finding z at offset q along z0 to z1
        //      z =  1 / (1 / z0 * (1 - q) + 1 / z1 * q)
        // and z1 = z0 + ddxy, where z1 is at a unit pixel offset [1, 1]
        // z can be calculated via ddxy as
        //
        //      z = (z0 + ddxy) / (1 + (1-q) / z0 * ddxy) 

        float z0 = distance;
        float2 zxy = (z0 + ddxy) / (1 + ((1 - pixelOffset) / z0) * ddxy);
        depthThreshold = dot(1, abs(zxy - z0)); // ToDo this should be sqrt(dot(zxy - z0, zxy - z0))
    }
    else
    {
        depthThreshold = dot(1, abs(pixelOffset * ddxy));
    }

    return depthThreshold;
}

void UnpackNormalDepth(in float3 encodedNormalDepth, out float3 normal, out float depth)
{
    depth = encodedNormalDepth.z;
    normal = DecodeNormal(encodedNormalDepth.xy);
}

void CalculateFilterWeight(
    in float value,
    in float stdDeviation,
    in float depth,
    in float3 normal,
    in float2 ddxy,
    in uint2 pixelOffset,
    in float iValue,
    in float iDepth,
    in float3 iNormal,
    in float iVariance,
    in float iFilterKernelWeight)
{
    const float valueSigma = g_CB.valueSigma;
    const float normalSigma = g_CB.normalSigma;
    const float depthSigma = g_CB.depthSigma;

    // Calculate normal difference based weight.
    float w_n;
    {
        w_n = pow(max(0, dot(normal, iNormal)), normalSigma);
    }

    // Calculate depth difference based weight.
    float w_d;
    {
        // Account for sample offset in bilateral downsampled partial depth derivative buffer.
        if (g_CB.usingBilateralDownsampledBuffers)
        {
            pixelOffset += float2(0.5, 0.5);
        }
        float depthTolerance = depthSigma * DepthThreshold(depth, ddxy, pixelOffset, depth - iDepth);

        // Account for input resource value precision.
        // ToDo why is 2x needed to get rid of banding?
        float depthFloatPrecision = 2.0f * FloatPrecision(max(depth, iDepth), g_CB.DepthNumMantissaBits);
        depthTolerance += depthFloatPrecision;

        // ToDo compare to exp version from SVGF.
        w_d = min(depthTolerance / (abs(depth - iDepth) + FLT_EPSILON), 1);
    }

    // Calculate value difference based weight.
    float w_x;
    {
        const float errorOffset = 0.005f;
        float e_x = -abs(value - iValue) / (valueSigma * stdDeviation + errorOffset);
        w_x = exp(e_x);
    }

    // Filter kernel weight.
    float w_h = iFilterKernelWeight;

    return w_h * w_n * w_x * w_d;
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
    uint2 GTid16x4_row0 = uint2(GI % 16, GI / 16);
    int2 KernelBasePixel = Gid * GroupDim - int2(FilterKernel::Radius, FilterKernel::Radius);
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

        float value = 0;
        float3 normal = 0;
        float depth = 0;
        float2 ddxy = 0;
        float variance = 0;
        float stdDeviation = 0;

        // The lane is out of bounds of the GroupDim + kernel, 
        // but could be within bounds of the input texture,
        // so don't read from the texture.
        // However it needs to be kept as an active lane for a below split filter.
        if (GTid16x4.x < NumValuesToLoadPerRowOrColumn)
        {
            value = g_inValues[pixel];
            float3 packedNormalDepth = g_inNormalDepth[pixel].xyz;
            UnpackNormalDepth(packedNormalDepth, normal, depth);
            float smoothedVariance = g_inSmoothedMeanVariance[pixel].y;
            float variance = g_inVariance[pixel];
            stdDeviation = sqrt(smoothedVariance);
        }

        // Filter the values for the first GroupDim columns.
        {
            // Accumulate for the whole kernel width.
            float weightSum = 0;
            float weightedValueSum = 0;
            float weightedVarianceSum = 0;

            // Since a row uses 16 lanes, but we only need to calculate the aggregate for the first half (8) lanes,
            // split the kernel wide aggregation among the first 8 and the second 8 lanes, and then combine them.
            
            // Initialize the first 8 lanes to the first cell contribution of the kernel. 
            // This covers the remainder of 1 in FilterKernel::Width / 2 used in the loop below. 
            if (GTid16x4.x < GroupDim.x)
            {
                float w_h = FilterKernel::Kernel1D[FilterKernel::Radius];
                weightSum = w_h;
                weightedValueSum = w_h * value;
#if OUTPUT_FILTERED_VARIANCE
                weightedVarianceSum = w_h * w_h * variance;
#endif
            }

            // ToDo unroll?
            for (uint i = 0; i < FilterKernel::Radius; i++)
            {
                uint laneToReadFrom = Row_BaseWaveLaneIndex + 1 + i +
                                     (GTid16x4.x < GroupDim.x ? GTid16x4.x : (GTid16x4.x - GroupDim.x) + FilterKernel::Radius);
                float iValue = WaveReadLaneAt(value, laneToReadFrom);
                float iDepth = WaveReadLaneAt(depth, laneToReadFrom);
                float3 iNormal = WaveReadLaneAt(normal, laneToReadFrom);
                float iVariance = WaveReadLaneAt(variance, laneToReadFrom);
                uint2 pixelOffset = uint2(i + 1, 0);
                float iFilterKernelWeight = FilterKernel::Kernel1D[FilterKernel::Radius - (i + 1)];

                float weight = CalculateFilterWeight(
                    value,
                    stdDeviation,
                    depth,
                    normal,
                    ddxy,
                    pixelOffset,
                    iValue,
                    iDepth,
                    iNormal,
                    iVariance,
                    iFilterKernelWeight);


                weightedValueSum += weight * iValue;
                weightSum += weight;

#if OUTPUT_FILTERED_VARIANCE
                if (g_CB.outputFilteredVariance)
                {
                    weightedVarianceSum += weight * weight * iVariance;   // ToDo rename to sqWeight...
                }
#endif
            }
            
            // Combine the sub-results.
            uint laneToReadFrom = min(WaveGetLaneCount() - 1, Row_BaseWaveLaneIndex + GTid16x4.x + 8);
            weightSum += WaveReadLaneAt(weightSum, laneToReadFrom);
            weightedValueSum += WaveReadLaneAt(weightedValueSum, laneToReadFrom);
#if OUTPUT_FILTERED_VARIANCE
            weightedVarianceSum += WaveReadLaneAt(weightedVarianceSum, laneToReadFrom);
#endif

            // Store the filtered results.
#if 0       
            if (GTid16x4.x < GroupDim.x)
            {
                uint2 GTid16x4 = GTid16x4_row0 + uint2(0, i * 4);
                if (GTid16x4.y >= NumValuesToLoadPerRowOrColumn)
                {
                    break;
                }

                // Load all the contributing columns for each row.
                int2 pixel = KernelBasePixel + GTid16x4;
                g_outFilteredValues[DTid] = weightedValueSum / weightSum;

#if OUTPUT_FILTERED_VARIANCE
                if (g_CB.outputFilteredVariance)
                {
                    g_outFilteredVariance[DTid] = weightedVarianceSum / (weightSum * weightSum);
                }
            }
#endif
#else
            // Store only the valid results, i.e. first GroupDim columns.
            if (GTid16x4.x < GroupDim.x)
            {
                // ToDo move this up?
                PackedInputValueStdDeviationCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(float2(value, standardDeviation));
                PackedInputDepthNormalXCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(float2(depth, normal.x));
                PackedInputNormalYZCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(float2(normal.y, normal.z));
                PackedInputDdxyCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(ddxy);

                float filteredValue = weightedValueSum / weightSum;
#if OUTPUT_FILTERED_VARIANCE
                float filteredVariance = weightedVarianceSum / (weightSum * weightSum);
#else
                float filteredVariance = 0;
#endif
                PackedRowResultCache[GTid16x4.y][GTid16x4.x] = Float2ToHalf(float2(filteredValue, filteredVariance));
            }
        }
    }
}


[numthreads(CalculateMeanVarianceFilter::ThreadGroup::Width, CalculateMeanVarianceFilter::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex, uint2 DTid : SV_DispatchThreadID)
{
    FilterHorizontally(Gid, GI);
    GroupMemoryBarrierWithGroupSync();

    FilterVertically(DTid, GTid);
}
