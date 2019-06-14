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

// Desc: Counting Sort of rays based on their origin depth and direction.
// Supports:
// - Up to {8K rays per ray group + 12 bit hash key} | {4K rays per ray group + 13 bit hash key}
// - Max Ray Group dimensions: 64x256
// Rays can be disabled by setting their depth to less than 0. Such rays will get moved
// to the end of the sorted ray group and have a source index offset of [0xff, 0xff]. 
// Performance:
//  2080 Ti, 1spp@4K, 128x64 ray groups, 10 bit hash key: 0.235ms
//  2080 Ti, 1spp@4K, 128x64 ray groups, 12 bit hash key: 0.275ms


// Algorithm: Counting Sort
// - Load ray origin depths and directions into SMem.
// - Calculate min max origin depth per ray group.
// - Generate hash keys from ray directions and origin depths.
// - Calculates histograms for the key hashes.
// - Calculates a prefix sum of the histograms.
// - Scatter write the ray source index offsets based on their hash and the prefix sum for the hash key into SMem cache.
// - Linearly spill sorted ray source index offsets from SMem cache into VRAM.

// The ray hash is calculated from ray direction and origin depth
// Ref: Ray Reordering Techniques for GPU Ray-Cast Ambient Occlusion

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "RaySorting.hlsli"

Texture2D<float4> g_inRayDirectionOriginDepth : register(t0);    // R11G11B10 texture. Note this format doesn't store negative values.

// Sorted Ray Group index offsets. 
// This lists ray index offsets of the sorted rays from the Ray Group start.
RWTexture2D<uint2> g_outSortedRayGroupIndexOffsets : register(u0);   

// Inverted sorted Ray Group index offsets. 
// This lists sorted ray index offsets of the non-sorted rays from the Ray Group start.
// This can be used to find which ray group index (based on sorting) processed ray for a given index.
// In other words, ray tracing pass for the sorted rays can output their results sequentially according to their dispatch thread indices 
// and then another post-process pass can use this list to gather those results, instead of doing a more costly scatter write during ray tracing pass.
// RWTexture2D<uint2> g_outInvertedSortedRayGroupIndexOffsets : register(u0);

//RWTexture2D<uint2> g_outInvertedSortedRayGroupIndexOffsets : register(u1);  // Thread group per-pixel index offsets within each 128x64 pixel group.
RWTexture2D<float4> g_outDebug : register(u1);  // Thread group per-pixel index offsets within each 128x64 pixel group.

ConstantBuffer<SortRaysConstantBuffer> CB: register(b0);

namespace HashKey {
    enum { 
        RayDirectionKeyBits1D = 4, 
        RayOriginDepthKeyBits = 2,
        NumBits = 2 * RayDirectionKeyBits1D + RayOriginDepthKeyBits  // <= 12
    };
}

#define MIN_WAVE_LANE_COUNT 16
#define MAX_WAVES ((MAX_RAYS + MIN_WAVE_LANE_COUNT - 1) / MIN_WAVE_LANE_COUNT)

namespace SMem
{
    namespace Size
    {
        enum {
            Histogram = 1 << (2 * RAY_DIRECTION_HASH_KEY_BITS_1D + DEPTH_HASH_KEY_BITS),
            Key8b = 4096,
            Key16b = 8192,
        };
    }
    namespace Offset
    {
        enum {
            Histogram = 0,
            HistogramTemp = Size::Histogram,    // <= 4096
            Key8b = 4096,
            Key16b = 8192,
            Depth16b = 8192,
            WaveDepthMin = 0,
            WaveDepthMax = MAX_WAVES,           // <= 512
            WaveDepthSum = 2 * MAX_WAVES,       // <= 1024
            WaveValidRayCount = 3 * MAX_WAVES,  // <= 1536
            RayIndex = Size::Histogram
        };
    }
}


//********************************************************************
// Hash Key
//  - a hash calculated from ray direction and origin depth
//  - max values:
//      12 bits(4096) for 8K rays.
//      13 bits(8192) for 4K rays.
// The 15th and 16th bits are reserved:
// - 15th bit == (1) - invalid ray. These rays will get sorted to the end.
// - 16th bit == (1) - invalid key. To handle when a key is replaced by Source Ray index in SMEM.

#if (KEY_NUM_BITS > 13) || (KEY_NUM_BITS > 12 && MAX_RAYS > 4096)
Key bit size is out of supported limits.
#endif

#define INVALID_16BIT_KEY_VALUE 0x8000      // A value used to denote if the SMEM entry is a 16bit key value or not.
#define INVALID_RAY_ORIGIN_DEPTH 0
//********************************************************************


//********************************************************************
// Ray Count SMem cache.
// Supports up to 16 bit (64K) ray counts per bin.
// Used for:
//  - to store number of binned rays. 
//  - as an intermediate cache for prefix sum computations.
// Stores 16bit values, with two values per entry.
// Stored as two ping-pong buffers.
// - Hi bits: odd ping-pong buffer ID
// - Lo bits: even ping-pong buffer ID
//********************************************************************

// ToDo use single cache array;

//********************************************************************
// SMEM stores 16 bit values, two 16bit values per 32bit entry:
// - Hi bits: odd indices
// - Lo bits: even indices
// SMEM is used for two mutually exclusive and temporally overlapping purposes
// so as to fit all caching within Shared Memory limits:
// - First it caches the hashed key per pixel - 15 bits
// - Second it caches the source index offset for a given sorted pixel - 2D 7+8bit index.
//   The values for the two purposes overlap in the cache during the shader execution. 
//   The key is generated first, but the source indices overwrite it later, while
//   the key may still be needer by another thread. 
//   Therefore the most significant bit is used
//   to denote whether the stored hashed key is still valid. 
//   If the key is no longer valid, it is regenerated again. 
//   To lower the collision and keep as many cached keys, the cache
//   is extended to the remaining shader memory limit and the keys
//   are stored at an offset. This way, the last 2 * offset
//   keys won't be invalidated.
//
//  PERFORMANCE tip:
//   Use as little rays and as small hash key bit size to leave 
//   as much room as possible for the hashed keys.
#define SMCACHE_SIZE 8192
groupshared uint SMEM[SMCACHE_SIZE];
//********************************************************************


//********************************************************************
// Store a 16 bit value in the Shared Memory.
// The 16 bit value is stored in 32bit value range <0, 8K)
// in layered fashion to avoid bank conflicts on subsequent 
// index accesses among subsequent threads. 
// It is stored in 16bit layers at index16b starting from indexOffset32b at first row
// which is the row of least significant 16 bits.
//  index16B - 32bit offset up to (16K - 1).
//  indexOffset32b - 32bit offset up to (8K - 1). 
//  index16b / 2 + indexOffset32b must be less than 8K.
// For example: 
//  - index16b == {0 - 4}, 
//  - indexOffset32b == 6
//  Shared memory {8x32b}:
//  | - - - - - 0 1 2 |  Least significant bits
//  | - - - - - 0 1 2 |
//  | 3 4 - - - - - - |
//  | 3 4 - - - - - - |  Most significant bits
void Store16bitUintInSMem(in uint index16b, in uint value, in uint indexOffset32b)
{
    uint offsetIndex = indexOffset32b + index16b;
    bool useHi16Bits = offsetIndex >= SMCACHE_SIZE;
    uint smemIndex = offsetIndex - useHi16Bits * SMCACHE_SIZE;

    uint packedValue = (value & 0xffff) << (useHi16Bits * 16);
    // To set a value via XOR, the target bits need to be zeroed out first.
    uint bitsToZeroOut = useHi16Bits ? 0x0000ffff : 0xffff0000;
    InterlockedAnd(SMEM[smemIndex], bitsToZeroOut);
    InterlockedXor(SMEM[smemIndex], packedValue);
}

void Store16bitFloatInSMem(in uint index16b, in float value, in uint indexOffset32b)
{
    uint encoded16bitFloat = f32tof16(value);
    Store16bitUintInSMem(index16b, encoded16bitFloat, indexOffset32b);
}

uint Load16bitUintFromSMem(in uint index16b, in uint indexOffset32b)
{
    uint offsetIndex = indexOffset32b + index16b;
    bool useHi16Bits = offsetIndex >= SMCACHE_SIZE;
    uint smemIndex = offsetIndex - useHi16Bits * SMCACHE_SIZE;
    uint packedValue = SMEM[smemIndex];
    return (packedValue >> (useHi16Bits * 16)) & 0xffff;
}

float Load16bitFloatFromSMem(in uint index16b, in uint indexOffset32b)
{
    uint encoded16bitFloat = Load16bitUintFromSMem(index16b, indexOffset32b);
    return f16tof32(encoded16bitFloat);
}

uint AddTo16bitValueInSMem(in uint index16b, in uint value, in uint indexOffset32b)
{
    uint offsetIndex = indexOffset32b + index16b;
    bool useHi16Bits = offsetIndex >= SMCACHE_SIZE;
    uint smemIndex = offsetIndex - useHi16Bits * SMCACHE_SIZE;
    uint packedValue = (value & 0xffff) << (useHi16Bits * 16);
    uint newValue;
    InterlockedAdd(SMEM[smemIndex], packedValue, newValue);

    return (newValue >> (useHi16Bits * 16)) & 0xffff;
}
//********************************************************************


//********************************************************************
// Store an 8 bit value in the Shared Memory.
// The 8bit value is stored in 32bit value range <indexOffset32b, 8K)
// in layered fashion to avoid bank conflicts on subsequent 
// index accesses among subsequent threads. 
// It is stored in the least significant 16 bit layer at index8b 
// starting from indexOffset32b at each row.
//  index8b - 32bit offset up to (16K - 1).
//  indexOffset32b - 32bit offset up to (8K - 1). 
//  index8b / 2 + indexOffset32b must be less than 8K.
// For example: 
//  - index8b == {0 - 4}, 
//  - indexOffset32b == 6
//  Shared memory {8x32b}:
//  | - - - - - 0 1 2 |  Least significant 8 bits
//  | - - - - - 3 4 - |
//  | - - - - - - - - |
//  | - - - - - - - - |  Most significant 8 bits
void Store8bitUintInLow16bitSMem(in uint index8b, in uint value, in uint indexOffset32b)
{
    uint offsetIndex = indexOffset32b + index8b;
    bool useHi8Bits = offsetIndex >= SMCACHE_SIZE;
    uint smemIndex = offsetIndex - useHi8Bits * (SMCACHE_SIZE - indexOffset32b);
    uint packedValue = (value & 0xff) << (useHi8Bits * 8);

    // To set a value via XOR, the target bits need to be zeroed out first.
    uint bitsToZeroOut = useHi8Bits ? 0xffff00ff : 0xffffff00;
    InterlockedAnd(SMEM[smemIndex], bitsToZeroOut);
    InterlockedXor(SMEM[smemIndex], packedValue);
}

uint Load8bitUintFromLow16bitSMem(in uint index8b, in uint indexOffset32b)
{
    uint offsetIndex = indexOffset32b + index8b;
    bool useHi8Bits = offsetIndex >= SMCACHE_SIZE;
    uint smemIndex = offsetIndex - useHi8Bits * (SMCACHE_SIZE - indexOffset32b);
    uint packedValue = SMEM[smemIndex];
    return (packedValue >> (useHi8Bits * 8)) & 0xff;
}
//********************************************************************


void InitializeSharedMemory(in uint GI)
{
    for (uint i = GI; i < SMCACHE_SIZE; i += NUM_THREADS)
    {
        SMEM[i] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

// Create a hash key from a ray direction. 
uint CreateRayDirectionHashKey(in float3 rayDirection)
{
    //ToDo test quantization of octnormal instead.

    // Convert the vector from cartesian to spherical coordinates.
    float azimuthAngle = atan2(rayDirection.y, rayDirection.x);
    float polarAngle = acos(rayDirection.z);

    // Normalize to <0,1>.
    float2 rayDirectionKey = float2((azimuthAngle / (2 * PI)),
        polarAngle / PI);

    // Calculate hashes.
    const uint NormalHashKeyBins1D = 1 << RAY_DIRECTION_HASH_KEY_BITS_1D;
    uint2 rayDirectionHashKey = min(NormalHashKeyBins1D * rayDirectionKey, NormalHashKeyBins1D - 1);

    return   (rayDirectionHashKey.y << RAY_DIRECTION_HASH_KEY_BITS_1D)
            + rayDirectionHashKey.x;
}

// Create a hash key from a ray origin depth. 
uint CreateDepthHashKey(in float rayOriginDepth, in float2 rayGroupMinMaxDepth)
{
    // ToDo test log depth quantization
    float relativeDepth = rayOriginDepth - rayGroupMinMaxDepth.x;
    float rayGroupDepthRange = rayGroupMinMaxDepth.y - rayGroupMinMaxDepth.x;
    const uint DepthHashKeyBins = 1 << DEPTH_HASH_KEY_BITS;
    float binDepthSize = max(rayGroupDepthRange / DepthHashKeyBins, CB.binDepthSize);
    uint depthHashKey = min(rayOriginDepth / binDepthSize, DepthHashKeyBins - 1);

    return depthHashKey;
}

uint CreateRayHashKey(in float3 rayDirection, in float rayOriginDepth, in float2 rayGroupMinMaxDepth)
{
    uint rayDirectionHashKey = CreateRayDirectionHashKey(rayDirection);
    uint rayOriginDepthHashKey = CreateDepthHashKey(rayOriginDepth, rayGroupMinMaxDepth);
    return  rayDirectionHashKey +
            (rayOriginDepthHashKey << (2 * RAY_DIRECTION_HASH_KEY_BITS_1D));
}


// Calculate ray direction hash keys and cache depths.
void CalculatePartialRayDirectionHashKeyAndCacheDepth(in uint2 Gid, in uint GI)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;

    for (uint ray = GI; ray < NUM_RAYS; ray += NUM_THREADS)
    {
        uint2 pixel = GroupStart + uint2(ray % SortRays::RayGroup::Width, ray / SortRays::RayGroup::Width);
        if (IsWithinBounds(pixel, CB.dim))
        {
            float3 rayDirectionOriginDepth = g_inRayDirectionOriginDepth[pixel].xyz;
            float rayOriginDepth = rayDirectionOriginDepth.z;
            bool isRayValid = rayOriginDepth != INVALID_RAY_ORIGIN_DEPTH; 

            // The ray direction hash key doesn't need to store if the ray value is valid for now, 
            // as there's no space when storing it as 8bit. The reason the hash from 
            // ray direction needs to be stored as 8 bit is to leave room in Shared Memory 
            // for Ray Origin Depth Min Max reduction values. Once that is computed, 
            // ray origin depth hash can be computed and the final 16bit hash key will reserve 
            // bits for invalid rays. Until then the ray origin depth is used to identify
            // invalid rays.        
            uint rayDirectionHashKey = 0;
            if (isRayValid)     // ToDo test remove perf
            {
                float3 rayDirection = DecodeNormal(rayDirectionOriginDepth.xy);
                rayDirectionHashKey = CreateRayDirectionHashKey(rayDirection);

            }

            // Cache the depth.
            Store16bitFloatInSMem(ray, rayOriginDepth, SMem::Offset::Depth16b);

            // Cache the key.
            Store8bitUintInLow16bitSMem(ray, rayDirectionHashKey, SMem::Offset::Key8b);
#if 0
            uint2 pixel = GroupStart + uint2(ray % SortRays::RayGroup::Width, ray / SortRays::RayGroup::Width);
            g_outDebug[pixel] = float4(pixel, rayOriginDepth, rayDirectionHashKey);
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// Calculate depth min max range of all rays within the ray group.
float2 CalculateRayGroupMinMaxDepth(in uint GI, uint2 Gid)
{
    // Reduce all ray values to per wave values.
    for (uint ray = GI; ray < NUM_RAYS; ray += NUM_THREADS)
    {
        float rayOriginDepth = Load16bitFloatFromSMem(ray, SMem::Offset::Depth16b);
        bool isValidRay = rayOriginDepth != INVALID_RAY_ORIGIN_DEPTH;
        float waveDepthMin = WaveActiveMin(isValidRay ? rayOriginDepth : FLT_MAX);
        float waveDepthMax = WaveActiveMax(rayOriginDepth);
        float waveDepthSum = WaveActiveSum(rayOriginDepth);
        uint waveValidRayCount = WaveActiveSum(isValidRay ? 1 : 0);

        // Store the per-wave result once per wave.
        if (WaveGetLaneIndex() == 0)    // ToDo remove?
        {
            uint waveIndex = ray / WaveGetLaneCount();
            Store16bitFloatInSMem(waveIndex, waveDepthMin, 7000); //SMem::Offset::WaveDepthMin);
            //Store16bitFloatInSMem(waveIndex, waveDepthMax, SMem::Offset::WaveDepthMax);
           // Store16bitFloatInSMem(waveIndex, waveDepthSum, SMem::Offset::WaveDepthSum);
            //Store16bitUintInSMem(waveIndex, waveValidRayCount, SMem::Offset::WaveValidRayCount);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    return float2(0, 4);
    // Reduce all per-wave values into a single value.
    uint NumValuesToReduce = (NUM_RAYS + WaveGetLaneCount() - 1) / WaveGetLaneCount();
    for (UINT s = 1; s < NumValuesToReduce; s *= WaveGetLaneCount())
    {
        uint numLanesToProcess = (NumValuesToReduce + s - 1) / s;
        
        // Set the defaults for lanes as they contribute even tho
        float depthMin = FLT_MAX;
        float depthMax = 0;
        float depthSum = 0;
        uint validRayCount = 0;
        if (GI < numLanesToProcess)
        {
            // Load the per-wave results from the previous iteration.
            uint index = GI * s;
            depthMin = Load16bitFloatFromSMem(index, SMem::Offset::WaveDepthMin);
            depthMax = Load16bitFloatFromSMem(index, SMem::Offset::WaveDepthMax);
            depthSum = Load16bitFloatFromSMem(index, SMem::Offset::WaveDepthSum);
            validRayCount = Load16bitUintFromSMem(index, SMem::Offset::WaveValidRayCount);

            float waveDepthMin = WaveActiveMin(depthMin);
            float waveDepthMax = WaveActiveMax(depthMax);
            float waveDepthSum = WaveActiveSum(depthSum);
            uint waveValidRayCount = WaveActiveSum(validRayCount);

            if (WaveGetLaneIndex() == 0) // ToDo remove?
            {
                Store16bitFloatInSMem(GI, waveDepthMin, SMem::Offset::WaveDepthMin);
                Store16bitFloatInSMem(GI, waveDepthMax, SMem::Offset::WaveDepthMax);
                Store16bitFloatInSMem(GI, waveDepthSum, SMem::Offset::WaveDepthSum);
                Store16bitUintInSMem(GI, waveValidRayCount, SMem::Offset::WaveValidRayCount);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // ToDo calc the result here and then broadcast?

    // ToDo test perf if only first thread computes the value and stores it and waits
    float depthMin = Load16bitFloatFromSMem(0, SMem::Offset::WaveDepthMin);
    float depthMax = Load16bitFloatFromSMem(0, SMem::Offset::WaveDepthMax);
    float depthSum = Load16bitFloatFromSMem(0, SMem::Offset::WaveDepthSum);
    uint validRayCount = Load16bitUintFromSMem(0, SMem::Offset::WaveValidRayCount);

    // If validRayCount is ever 0, and thus producing bad value, 
    // it's ok as the depth values won't end up being used.
    // ToDo double check this
    float depthMean = depthSum / validRayCount;

    // Calculate conservative min/max depth range.
    // Given the fact this will be used to bucketize the ray origin depth
    // and we have a very small number of bins to work with
    // we want to give more finer bucketization to ray origin depth ranges with higher
    // occurence and wider depth range buckets for areas with fewer ray origins.
    // This is done by taking a smallest union of true Min, Max and reflected
    // Min, Max around the mean. The reflected Min, Max variants will compress the 
    // depth range if the rays are non-uniform distributed along the depth.
    // Any ray origin depths outside the calculated range will get clamped to this range.
    float reflectedDepthMin = depthMean + (depthMean - depthMin);
    float reflectedDepthMax = depthMean - (depthMax - depthMean);
    float conservativeDepthMin = max(reflectedDepthMax, depthMin);
    float conservativeDepthMax = min(reflectedDepthMin, depthMax);

#if 1
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;
    for (uint i = GI; i < NUM_RAYS; i += NUM_THREADS)
    {
        uint2 pixel = GroupStart + uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
        g_outDebug[pixel] = float4(depthMin, depthMax, depthSum, validRayCount);
    }
#endif
    return float2(depthMin, depthMax);
}

// Combine the depth hash key with the ray direction hash keys and update the key histograms.
void FinalizeHashKeyAndCalculateKeyHistogram(in uint GI, in uint2 Gid, in float2 rayGroupMinMaxDepth)
{
#if 0
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;
#endif
    for (uint ray = GI; ray < NUM_RAYS; ray += NUM_THREADS)
    {
        float rayOriginDepth = Load16bitFloatFromSMem(ray, SMem::Offset::Depth16b);
        bool isValidRay = rayOriginDepth != INVALID_RAY_ORIGIN_DEPTH;
        uint rayDirectionHashKey = Load8bitUintFromLow16bitSMem(ray, SMem::Offset::Key8b);
        uint hashKey = INACTIVE_RAY_KEY;
#if 0
        uint2 pixel = GroupStart + uint2(ray % SortRays::RayGroup::Width, ray / SortRays::RayGroup::Width);
        g_outDebug[pixel] = float4(pixel, rayOriginDepth, rayDirectionHashKey);
#endif
        if (isValidRay)
        {
            uint depthHashKey = CreateDepthHashKey(rayOriginDepth, rayGroupMinMaxDepth);
            hashKey = rayDirectionHashKey + 
                     (depthHashKey << (2 * RAY_DIRECTION_HASH_KEY_BITS_1D));

            // Increase histogram bin count.
            AddTo16bitValueInSMem(hashKey, 1, SMem::Offset::Histogram);
        }

        // Cache the key.
        Store16bitUintInSMem(ray, hashKey, SMem::Offset::Key16b);


    }
    GroupMemoryBarrierWithGroupSync();
}

void GenerateHashKeysAndKeyHistogram(in uint2 Gid, in uint GI, out float2 rayGroupMinMaxDepth)
{
    CalculatePartialRayDirectionHashKeyAndCacheDepth(Gid, GI);
    rayGroupMinMaxDepth = CalculateRayGroupMinMaxDepth(GI, Gid);
    FinalizeHashKeyAndCalculateKeyHistogram(GI, Gid, rayGroupMinMaxDepth);
}

// Prefix sum.
// Requirements: 
//  - NUM_KEYS <= WaveGetLaneCount() ^ 2
//    ToDo support up to 4K key bins
void PrefixSum(in uint GI)
{
    // PrefixSum at the lane level.
    for (uint key = GI; key < NUM_KEYS; key += NUM_THREADS)
    {
        uint count = Load16bitUintFromSMem(key, SMem::Offset::Histogram);
        uint prefixSum = WavePrefixSum(count);

        Store16bitUintInSMem(key, prefixSum, SMem::Offset::HistogramTemp);
    }
    GroupMemoryBarrierWithGroupSync();

    // PostfixSum for last lane at each wave, except the last.
    uint NumWaves = NUM_KEYS / WaveGetLaneCount();
    for (uint i = GI; i < NumWaves - 1; i += NUM_THREADS)
    {
        uint lastPrevWaveLaneIndex = (i + 1) * WaveGetLaneCount() - 1;
        uint count = Load16bitUintFromSMem(lastPrevWaveLaneIndex, SMem::Offset::HistogramTemp);
        uint postfixSum = count + WavePrefixSum(count);

        Store16bitUintInSMem(i, postfixSum, SMem::Offset::Histogram);
    }
    GroupMemoryBarrierWithGroupSync();

    // Add the previous lane wave level postfix sum to lane values in waves 2nd and higher.
    for (uint j = GI + WaveGetLaneCount(); j < NUM_KEYS; j += NUM_THREADS)
    {
        uint index = j / WaveGetLaneCount() - 1;
        uint prevWavePostfixSum = Load16bitUintFromSMem(index, SMem::Offset::Histogram);

        AddTo16bitValueInSMem(j, prevWavePostfixSum, SMem::Offset::HistogramTemp);
    }
    GroupMemoryBarrierWithGroupSync();

    // Copy the final prefixSums to the primary histogram cache 
    // as the HistogramTemp cache memory area gets repurposed after this.
    for (key = GI; key < NUM_KEYS; key += NUM_THREADS)
    {
        uint prefixSum = Load16bitUintFromSMem(key, SMem::Offset::HistogramTemp);
        Store16bitUintInSMem(key, prefixSum, SMem::Offset::Histogram);
    }
    GroupMemoryBarrierWithGroupSync();
}

// Write the sorted indices to shared memory to avoid costly scatter write to VRAM.
// These can then be later linearly spilled to VRAM.
void ScatterWriteSortedIndicesToSharedMemory(in uint2 Gid, in uint GI, in float2 rayGroupMinMaxDepth)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;
    uint2 RayGroupEnd = min(GroupStart + RayGroupDim, CB.dim);
    RayGroupDim = RayGroupEnd - GroupStart;
    
    for (uint ray = GI; ray < NUM_RAYS; ray += NUM_THREADS)
    {
        uint2 rayIndex = uint2(ray % SortRays::RayGroup::Width, ray / SortRays::RayGroup::Width);
        uint2 pixel = GroupStart + rayIndex;
        if (IsWithinBounds(pixel, CB.dim))
        {
            // Get the key for the corresponding pixel.
            uint key;
            bool isValidRay;
            {
                // First, see if the cached key is stil valid.
                uint cacheValue = Load16bitUintFromSMem(ray, SMem::Offset::Key16b);
                bool isHashKeyEntry = !(cacheValue & INVALID_16BIT_KEY_VALUE);

                if (isHashKeyEntry)
                {
                    isValidRay = cacheValue != INACTIVE_RAY_KEY;
                    key = cacheValue;
                }
                else  // The cached key has been already replaced with the ray's source index. Regenerate the key.
                {
                    float3 rayDirectionOriginDepth = g_inRayDirectionOriginDepth[pixel].xyz;
                    float rayOriginDepth = rayDirectionOriginDepth.z;
                    isValidRay = rayOriginDepth != INVALID_RAY_ORIGIN_DEPTH;

                    if (isValidRay)
                    {
                        float3 rayDirection = DecodeNormal(rayDirectionOriginDepth.xy);
                        key = CreateRayHashKey(rayDirection, rayOriginDepth, rayGroupMinMaxDepth);
                    }
                    else
                    {
                        key = INACTIVE_RAY_KEY;
                    }
                }
            }
            
            uint index = AddTo16bitValueInSMem(key, 1, SMem::Offset::Histogram);

            // To avoid costly scatter writes to VRAM, cache indices into SMem here instead.
            
            uint encodedRayIndex = isValidRay ? (rayIndex.x << 8) + rayIndex.y
                                              : INACTIVE_RAY_KEY;
            encodedRayIndex |= INVALID_16BIT_KEY_VALUE;     // Denote the target cache entry doesn't store a key no more.
            Store16bitUintInSMem(index, encodedRayIndex, SMem::Offset::RayIndex);
#if 0
            g_outDebug[outPixel] = float4(index, 0, encodedRayIndex);
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
}


// Spill cached sorted indices to VRAM
// Also scatter write the inversion of the indices.
// The inverted table will be used to find sorted ray index given a ray index and 
// instead of doing expensive scatter write after ray tracing, the subsequent pass
// reading ray tracing results will do gather read.
void SpillCachedIndicesToVRAMAndCacheInvertedSortedIndices(in uint2 Gid, in uint GI)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;

    // Sequentially spill cached indices into VRAM.
    for (uint index = GI; index < NUM_RAYS; index += NUM_THREADS)
    {
        uint packedSrcIndex = Load16bitUintFromSMem(index, SMem::Offset::RayIndex);
        packedSrcIndex &= ~INVALID_16BIT_KEY_VALUE;     // Strip the helper bit

        uint2 srcIndex = uint2(packedSrcIndex >> 8, packedSrcIndex & 0xff);
        uint2 outPixel = GroupStart + uint2(index % SortRays::RayGroup::Width, index / SortRays::RayGroup::Width);
        g_outSortedRayGroupIndexOffsets[outPixel] = srcIndex;
    }
}

void SpillCInvertedSortedIndicesToVRAM(in uint2 Gid, in uint GI)
{
}

[numthreads(SortRays::ThreadGroup::Width, SortRays::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    InitializeSharedMemory(GI);

    float2 rayGroupMinMaxDepth;
    GenerateHashKeysAndKeyHistogram(Gid, GI, rayGroupMinMaxDepth);
    PrefixSum(GI);
    ScatterWriteSortedIndicesToSharedMemory(Gid, GI, rayGroupMinMaxDepth);
    SpillCachedIndicesToVRAMAndCacheInvertedSortedIndices(Gid, GI);
    SpillCInvertedSortedIndicesToVRAM(Gid, GI);
}
