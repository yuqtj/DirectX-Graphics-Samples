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

// ToDo add desc for each kernel
// Desc: Calculate Variance via Separable Bilateral kernel.
// Uses normal and depth weights.
// Pitfalls: 
//  - normal weights may limit number of samples for small round objects
//  - depth weights may limit number of samples for thin objects (i.e. grass).
// Performance: 0.313 ms for 7x7 kernel at 1080p on TitanXP.

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"


Texture2D<float> g_inValues : register(t0);
Texture2D<float> g_inDepth : register(t1);  // ToDo use from normal tex directly
Texture2D<float4> g_inNormalDepthObliqueness : register(t2);

RWTexture2D<float> g_outVariance : register(u0);
RWTexture2D<float> g_outMean : register(u1);


// Group shared memory caches.
// Spaced at 4 Byte element widths to avoid bank conflicts on access.
// Trade precision for speed and pack floats to 16bit.
// 0.48ms -> 0.313 ms for 7x7 kernel on TitanXp at 1080p.
#define PACK_OPTIMIZATION 1   

#if PACK_OPTIMIZATION
groupshared UINT PackedValueObliquenessCache[256];  // 16bit float value and obliqueness.
groupshared UINT PackedNormalXYCache[256];          // 16bit float normal X and Y.
groupshared UINT PackedNormalZDepthCache[256];      // 16bit float normal Z and depth.

groupshared UINT PackedResultCache[256];          // 16bit float weightedValueSum, weightSum.
groupshared UINT PackedResultCache2[256];         // 16bit float weightedSqauredValueSum, numWeights.

#else
groupshared float VCache[256];
groupshared float NCacheX[256];
groupshared float NCacheY[256];
groupshared float DCache[256];
groupshared float NCacheZ[256];
groupshared float OCache[256];

groupshared float WeightedValueSumCache[256];
groupshared float WeightSumCache[256];
groupshared float WeightedSquaredValueSumCache[256];
groupshared UINT NumWeightsCache[256];
#endif

ConstantBuffer<CalculateVariance_BilateralFilterConstantBuffer> cb: register(b0);

void LoadToSharedMemory(UINT smemIndex, int2 pixel)
{
    float value = g_inValues[pixel];
    float4 packedValue = g_inNormalDepthObliqueness[pixel];
    float3 normal = DecodeNormal(packedValue.xy);
    float obliqueness = max(0.0001f, pow(packedValue.w, 10));
#if PACK_OPTIMIZATION
    PackedValueObliquenessCache[smemIndex] = Float2ToHalf(float2(value, obliqueness));
    PackedNormalXYCache[smemIndex] = Float2ToHalf(normal.xy);
    PackedNormalZDepthCache[smemIndex] = Float2ToHalf(float2(normal.z, packedValue.z));
#else
    VCache[smemIndex] = value;
    NCacheX[smemIndex] = normal.x;
    NCacheY[smemIndex] = normal.y;
    NCacheZ[smemIndex] = normal.z;
    DCache[smemIndex] = packedValue.z;
    OCache[smemIndex] = obliqueness;
#endif
}

void PrefetchData(uint index, int2 ST)
{
    // ToDo use gather for values
    LoadToSharedMemory(index, ST + int2(-1, -1));
    LoadToSharedMemory(index + 1, ST + int2(0, -1));
    LoadToSharedMemory(index + 16, ST + int2(-1, 0));
    LoadToSharedMemory(index + 17, ST + int2(0, 0));
}


float SampleInfluence(in float3 n1, in float3 n2, in float d1, in float d2, in float obliqueness)
{
    // todo remove check if weights used
    // Don't use normal weights as small round objects or edges along pixel diagonal will get undersampled
    // via separable filter.
    float w_n = cb.useNormalWeights ? pow(max(0, dot(n1, n2)), cb.normalSigma) : 1;
    float w_d = cb.useDepthWeights ? exp(-abs(d1 - d2) / (cb.depthSigma)) : 1;
    return w_n * w_d;
}

void BlurHorizontally(uint leftMostIndex)
{
    // Load the reference values for the current pixel.
    UINT Cid = leftMostIndex + cb.kernelRadius;
#if PACK_OPTIMIZATION
    float2 refUnpackedNormalZDepthValue = HalfToFloat2(PackedNormalZDepthCache[Cid]);
    float3 refNormal = float3(HalfToFloat2(PackedNormalXYCache[Cid]), refUnpackedNormalZDepthValue.x);
    float refDepth = refUnpackedNormalZDepthValue.y;

    float2 refUnpackedValueObliquenessCache = HalfToFloat2(PackedValueObliquenessCache[Cid]);
    float obliqueness = refUnpackedValueObliquenessCache.y;
#else
    float3 refNormal = float3(NCacheX[Cid], NCacheY[Cid], NCacheZ[Cid]);
    float refDepth = DCache[Cid];
    float obliqueness = OCache[Cid];
#endif

    UINT numWeights = 0;
    float weightedValueSum = 0;
    float weightedSquaredValueSum = 0;
    float weightSum = 0;  // ToDo check for missing value

    // Accumulate for the whole kernel.
    for (UINT c = 0; c < cb.kernelWidth; c++)
    {
        UINT ID = leftMostIndex + c;
#if PACK_OPTIMIZATION
        float2 unpackedNormalZDepthValue = HalfToFloat2(PackedNormalZDepthCache[ID]);
        float3 normal = float3(HalfToFloat2(PackedNormalXYCache[ID]), unpackedNormalZDepthValue.x);
        float depth = unpackedNormalZDepthValue.y;

        float2 unpackedValueObliquenessCache = HalfToFloat2(PackedValueObliquenessCache[ID]);
        float value = unpackedValueObliquenessCache.x;
#else
        float3 normal = float3(NCacheX[ID], NCacheY[ID], NCacheZ[ID]);
        float depth = DCache[ID];
        float value = VCache[ID];
#endif
        float w = SampleInfluence(refNormal, normal, refDepth, depth, obliqueness);

        const float SmallValue = 0.001f;
        if (w > SmallValue)
        {
            float weightedValue = w * value;
            weightedValueSum += weightedValue;
            weightedSquaredValueSum += weightedValue * value;
            weightSum += w;
            numWeights += 1;
        }
    }

#if PACK_OPTIMIZATION
    PackedResultCache[leftMostIndex] = Float2ToHalf(float2(weightedValueSum, weightSum));
    PackedResultCache2[leftMostIndex] = Float2ToHalf(float2(weightedSquaredValueSum, numWeights));
#else
    WeightedSquaredValueSumCache[leftMostIndex] = weightedSquaredValueSum;
    WeightedValueSumCache[leftMostIndex] = weightedValueSum;
    WeightSumCache[leftMostIndex] = weightSum;
    NumWeightsCache[leftMostIndex] = numWeights;
#endif
}

void BlurVertically(uint2 DTid, uint topMostIndex)
{
    // Load the reference values for the current pixel.
    UINT Cid = topMostIndex + cb.kernelRadius * 16;
#if PACK_OPTIMIZATION
    float2 refUnpackedNormalZDepthValue = HalfToFloat2(PackedNormalZDepthCache[Cid]);
    float3 refNormal = float3(HalfToFloat2(PackedNormalXYCache[Cid]), refUnpackedNormalZDepthValue.x);
    float refDepth = refUnpackedNormalZDepthValue.y;

    float2 refUnpackedValueObliquenessCache = HalfToFloat2(PackedValueObliquenessCache[Cid]);
    float obliqueness = refUnpackedValueObliquenessCache.y;
#else
    float3 refNormal = float3(NCacheX[Cid], NCacheY[Cid], NCacheZ[Cid]);
    float refDepth = DCache[Cid];
    float obliqueness = OCache[Cid];
#endif

    float weightedValueSum = 0;
    float weightedSquaredValueSum = 0;
    float weightSum = 0;  // ToDo check for missing value
    UINT numWeights = 0;

    // Accumulate for the whole kernel.
    for (UINT c = 0; c < cb.kernelWidth; c++)
    {
        UINT ID = topMostIndex + c * 16;

#if PACK_OPTIMIZATION
        float2 unpackedNormalZDepthValue = HalfToFloat2(PackedNormalZDepthCache[ID]);
        float3 normal = float3(HalfToFloat2(PackedNormalXYCache[ID]), unpackedNormalZDepthValue.x);
        float depth = unpackedNormalZDepthValue.y;
#else
        float3 normal = float3(NCacheX[ID], NCacheY[ID], NCacheZ[ID]);
        float depth = DCache[ID];
#endif
        float w = SampleInfluence(refNormal, normal, refDepth, depth, obliqueness);

        const float SmallValue = 0.001f;
        if (w > SmallValue)
        {
#if PACK_OPTIMIZATION
            float2 unpackedValue = HalfToFloat2(PackedResultCache[ID]);
            weightedValueSum += w * unpackedValue.x;
            weightSum += unpackedValue.y;

            float2 unpackedValue2 = HalfToFloat2(PackedResultCache2[ID]);
            weightedSquaredValueSum += w * unpackedValue2.x;
            numWeights += (unpackedValue.y + 0.1f); // Bump the value a bit to avoid imprecision rounding down one further.
#else
            weightedSquaredValueSum += w * WeightedSquaredValueSumCache[ID];
            weightedValueSum += w * WeightedValueSumCache[ID];
            weightSum += WeightSumCache[ID];
            numWeights += NumWeightsCache[ID];
#endif
        }
    }

    float variance = 0;
    float mean = 0;
    if (numWeights > 1)
    {
        float invWeightSum = 1 / weightSum;
        mean = invWeightSum * weightedValueSum;

        // Apply Bessel's correction to the estimated variance, divide by N-1, 
        // since the true population mean is not known. It is only estimated as the sample mean.
        float besselCorrection = numWeights / float(numWeights - 1);
        variance = besselCorrection * (invWeightSum * weightedSquaredValueSum - mean * mean);

        variance = max(0, variance);    // Ensure variance doesn't go negative due to imprecision.
    }
    if (cb.outputMean)
    {
        g_outMean[DTid] = mean;
    }
    g_outVariance[DTid] = variance;
}


// Do a separable bilateral accumulation of mean and variance
// Supports kernel radius of up to 4, i.e. kernel widths 9x9.
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
