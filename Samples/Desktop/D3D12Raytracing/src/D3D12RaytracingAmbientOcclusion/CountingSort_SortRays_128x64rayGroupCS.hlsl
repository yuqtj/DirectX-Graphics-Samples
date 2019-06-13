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

// Desc: Counting Sort of rays based on their linear depth and direction.
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

Texture2D<float4> g_inRayDirectionOriginDepthHit : register(t0);
RWTexture2D<uint2> g_outSortedThreadGroupIndices : register(u0);  // Thread group per-pixel index offsets within each 128x64 pixel group.
RWTexture2D<uint4> g_outDebug : register(u1);  // Thread group per-pixel index offsets within each 128x64 pixel group.

ConstantBuffer<SortRaysConstantBuffer> CB: register(b0);

#define RAY_DIRECTION_KEY_HASH_BITS_1D 4
#define DEPTH_KEY_HASH_BITS 2
#define NUM_RAYS SortRays::RayGroup::Size
#define NUM_THREADS SortRays::ThreadGroup::Size

#define MIN_VALID_RAY_DEPTH FLT_10BIT_MIN
#define MAX_RAYS 8192
#if MAX_RAYS > 8192
The shader supports up to 8192 input rays.
#endif

namespace HashKey {
    enum { 
        RayDirectionKeyBits1D = 4, 
        RayOriginDepthKeyBits = 2,
        NumBits = 2 * RayDirectionKeyBits1D + RayOriginDepthKeyBits  // <= 12
    };
}

namespace SMem
{
    namespace Size
    {
        enum {
            Histogram = 1 << (2 * RAY_DIRECTION_KEY_HASH_BITS_1D + DEPTH_KEY_HASH_BITS),
            Key8b = 4096,
            Key16b = 8192,
        };
    }
    namespace Offset
    {
        enum {
            Histogram = 0,
            Key8b = 4096,
            Key16b = 8192,
            Depth16b = 0,
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
#define KEY_NUM_BITS (DEPTH_KEY_HASH_BITS + 2*RAY_DIRECTION_KEY_HASH_BITS_1D)
#if (KEY_NUM_BITS > 13) || (KEY_NUM_BITS > 12 && MAX_RAYS > 4096)
Key bit size is out of supported limits.
#endif

// Hash key for an invalid/disabled ray. 
// These rays will get sorted to the end and 
// have the same value set for source index.
#define INVALID_RAY_KEY 0x7fff 
#define NUMBER_OF_INVALID_RAY_KEYS 1
#define INVALID_RAY_SOURCE_INDEX_OFFSET 0xffff
#define NUM_KEYS ((1 << KEY_NUM_BITS) + NUMBER_OF_INVALID_RAY_KEYS)
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
#define NUM_KEY_COUNT_ARRAY_ELEMENTS ((NUM_KEYS + 1) / 2)
groupshared uint KeyCounts[NUM_KEYS];
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
#define SMCACHE_SIZE (8192 - NUM_KEYS)
#define SMCACHE_INTERLEAVE_EVEN_ODD_INDICES 0   // Faster by 3.5%
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
    #define SMCACHE_KEY_OFFSET (SMCACHE_SIZE - (MAX_RAYS + 1) / 2)
#else
    #define SMCACHE_KEY_OFFSET (2 * SMCACHE_SIZE - NUM_RAYS)
#endif
#define SMCACHE_WAVE_MIN_MAX_DEPTH_OFFSET MAX_RAYS
#define MIN_WAVE_LANE_SIZE 16
#define SMCACHE_WAVE_MIN_MAX_DEPTH_SIZE (MAX_RAYS / MIN_WAVE_LANE_SIZE)

// ToDo use enum instead of defines
#define SMCACHE_RAY_DEPTH_OFFSET (SMCACHE_MIN_MAX_DEPTH_OFFSET + SMCACHE_WAVE_MIN_MAX_DEPTH_SIZE)
groupshared uint SMEM[SMCACHE_SIZE];
//********************************************************************

uint AddToKeyCounts(in uint bufferID, in uint key, in uint countToAdd)
{
    uint useHiBits = bufferID & 1;
    uint packedCount = countToAdd << (useHiBits * 16);
    uint newValue;
    InterlockedAdd(KeyCounts[key], packedCount, newValue);

    return (newValue >> (useHiBits * 16)) & 0xffff;
}

void SetKeyCount(in uint bufferID, in uint key, in uint countToSet)
{
    uint useHiBits = bufferID & 1;

    // Zero out the key so that XOR cam next set the prefix sum.
    uint countBitsToZeroOut = useHiBits ? 0x0000ffff : 0xffff0000;
    uint packedCount = countToSet << (useHiBits * 16);
    InterlockedAnd(KeyCounts[key], countBitsToZeroOut);
    InterlockedXor(KeyCounts[key], packedCount);
}

uint GetKeyCount(in uint bufferID, in uint key)
{
    bool useHiBits = bufferID & 1;
    uint packedCount = KeyCounts[key];
    return (packedCount >> (useHiBits * 16)) & 0xffff;
}

// Store a 16 bit value in the Shared Memory.
// The 16 bit value is stored in 32bit value range <0, 8K)
// in layered fashion to avoid bank conflicts on subsequent 
// index accesses among subsequent threads. 
// It is stored in 16bit layers at index8b starting from indexOffset32b at first row
// which is the row of least significant 16 bits.
//  index16B - 32bit offset up to (16K - 1).
//  indexOffset32b - 32bit offset up to (8K - 1). 
//  index16b + indexOffset32b must be less than 16K.
// For example: 
//  - index16b == {0 - 4}, 
//  - indexOffset32b == 6
//  Shared memory {8x32b}:
//  | - - - - - 0 1 2 |  Least significant bits
//  | - - - - - 0 1 2 |
//  | 3 4 - - - - - - |
//  | 3 4 - - - - - - |  Most significant bits
void Store16bitUintInSMem(in uint index16b, in uint value, in uint indexOffset32b = 0)
{
    uint offsetIndex = indexOffset32b + index16b;
    bool useHi16Bits = offsetIndex >= SMCACHE_SIZE;
    uint cacheIndex = offsetIndex - useHi16Bits * (SMCACHE_SIZE - offsetIndex);
    uint packedValue = (value & 0xffff) << (useHi16Bits * 16);

    // To set a value via XOR, the target bits need to be zeroed out first.
    uint bitsToZeroOut = useHi16Bits ? 0x0000ffff : 0xffff0000;
    InterlockedAnd(SMEM[cacheIndex], bitsToZeroOut);
    InterlockedXor(SMEM[cacheIndex], packedValue);
}

void Store16bitFloatInSMem(in uint index16b, in float value, in uint indexOffset32b = 0)
{
    uint encoded16bitFloat = f32tof16(value);
    Store16bitValueInSMem(index16b, encoded16bitFloat, indexOffset32b);
}

uint Load16bitUintFromSMem(in uint index, in uint indexOffset = 0)
{
    uint offsetIndex = indexOffset32b + index16b;
    bool useHi16Bits = offsetIndex >= SMCACHE_SIZE;
    uint cacheIndex = offsetIndex - useHi16Bits * (SMCACHE_SIZE - offsetIndex);
    uint packedValue = SMEM[cacheIndex];
    return (packedValue >> (useHiBits * 16)) & 0xffff; 
}

float Load16bitFloatFromSMem(in uint index, in uint indexOffset = 0)
{
    uint encoded16bitFloat = Load16bitUintFromSMem(index, indexOffset);
    return f16tof32(encoded16bitFloat);
}

uint AddTo16bitValueInSMem(in uint index16b, in uint value, in uint indexOffset32b = 0)
{
    uint offsetIndex = indexOffset32b + index16b;
    bool useHi16Bits = offsetIndex >= SMCACHE_SIZE;
    uint cacheIndex = offsetIndex - useHi16Bits * (SMCACHE_SIZE - offsetIndex);
    uint packedValue = (value & 0xffff) << (useHiBits * 16);
    uint newValue;
    InterlockedAdd(KeyCounts[key], packedValue, newValue);

    return (newValue >> (useHiBits * 16)) & 0xffff;
}

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
void Store8bitUintInLow16bitSMem(in uint index8b, in uint value, in uint indexOffset32b = 0)
{
    uint offsetIndex = indexOffset32b + index8b;
    bool useHi8Bits = offsetIndex >= SMCACHE_SIZE;
    uint cacheIndex = offsetIndex - useHi8Bits * (SMCACHE_SIZE - offsetIndex);
    uint packedValue = (value & 0xff) << (useHi8Bits * 8);

    // To set a value via XOR, the target bits need to be zeroed out first.
    uint bitsToZeroOut = useHi8Bits ? 0xffff00ff : 0xffffff00);
    InterlockedAnd(SMEM[cacheIndex], bitsToZeroOut);
    InterlockedXor(SMEM[cacheIndex], packedValue);
}

uint Load8bitUintFromLow16bitSMem(in uint index, in uint indexOffset32b = 0)
{
    uint offsetIndex = indexOffset32b + index8b;
    bool useHi8Bits = offsetIndex >= SMCACHE_SIZE;
    uint cacheIndex = offsetIndex - useHi8Bits * (SMCACHE_SIZE - offsetIndex);
    uint packedValue = SMEM[cacheIndex];
    return (packedValue >> (useHi8Bits * 8)) & 0xff;
}

void SetCacheDepthValue(in uint rayIndex, in float depth)
{
    uint encodedDepth = f32tof16(depth);
    Store16bitValueInSMem(i, in uint value, in uint indexOffset32b = 0)

    // Store first NUM_KEYS rays in KeyCounts Buffer IS 1 cache
    if (rayIndex < NUM_KEYS)
    {
        SetKeyCount(1, rayIndex, encodedDepth);
    }
    else // Store the rest in SMEM.
    {
        Store16bitValueInSMem(rayIndex - NUM_KEYS, encodedDepth);
    }
}

float GetCachedDepthValue(in uint rayIndex)
{
    uint encodedDepth;
    // Store first NUM_KEYS rays in KeyCounts Buffer IS 1 cache
    if (rayIndex < NUM_KEYS)
    {
        encodedDepth = GetKeyCount(1, rayIndex);
    }
    else // Store the rest in SMEM.
    {
        encodedDepth = Get16bitValueFromSMem(rayIndex - NUM_KEYS);
    }
    return f16tof32(encodedDepth);
}

void InitializeSharedMemory(uint GI)
{
    for (uint i = GI; i < SMCACHE_SIZE; i += NUM_THREADS)
    {
        SMEM[key] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

// Create a hash key from a normal . 
uint CreateNormalHashKey(in float3 normal)
{
    //ToDo test quantization of octnormal instead.

    // Convert cartesian coordinate normal to spherical coordinates.
    float azimuthAngle = atan2(normal.y, normal.x);
    float polarAngle = acos(normal.z);

    // Normalize to <0,1>.
    float2 normalKey = float2(
        (azimuthAngle / (2 * PI)),
        polarAngle / PI);

    // Calculate hashes.
    const uint NormalKeyHashBins1D = 1 << NORMAL_KEY_HASH_BITS_1D;
    uint2 normalKeyHash = min(NormalKeyHashBins1D * normalKey, NormalKeyHashBins1D - 1);

    return (normalKeyHash.y << (NORMAL_KEY_HASH_BITS_1D))
        + normalKeyHash.x;
}

uint CreateDepthHashKey(in float linearDepth, in float2 rayGroupMinMaxDepth)
{
    float relativeDepth = linearDepth - rayGroupMinMaxDepth.x;
    float rayGroupDepthRange = rayGroupMinMaxDepth.y - rayGroupMinMaxDepth.x;
    const uint DepthKeyHashBins = 1 << DEPTH_KEY_HASH_BITS;
    float binDepthSize = max(rayGroupDepthRange / DepthKeyHashBins, CB.binDepthSize);
    uint depthKeyHash = min(linearDepth / binDepthSize, DepthKeyHashBins - 1);

    return depthKeyHash;
}


// Calculate ray direction hash keys and cache depths.
void CalculatePartialRayDirectionHashKeyAndCacheDepth(uint2 Gid, uint GI)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;

    for (uint i = GI; i < NUM_RAYS; i += NUM_THREADS)
    {
        uint2 pixel = GroupStart + uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
        if (IsWithinBounds(pixel, CB.dim))
        {
            float4 rayDirectionDepthHit = g_inRayDirectionOriginDepthHit[pixel];
            float depth = rayDirectionDepthHit.z;
            bool isRayValid = depth >= MIN_VALID_RAY_DEPTH;

            // The ray direction hash key doesn't need to store if the ray value is valid for now, 
            // as there's no space when storing it as 8bit. The reason the hash from 
            // ray direction needs to be stored as 8 bit is to leave room in Shared Memory 
            // for Ray Origin Depth Min Max reduction values. Once that is computed, 
            // depth hash can be computed and the final 16bit hash key will reserve 
            // bits for invalid rays. Until then the ray origin depth is used to identify
            // invalid rays.        
            uint rayDirectionHashKey = 0;
            if (isRayValid)
            {
                float3 rayDirection = DecodeNormal(rayDirectionDepthHit.xy);
                rayDirectionHashKey = CreateRayDirectionHashKey(rayDirection);

            }

            // Cache the depth.
            Store16bitFloatInSMem(i, depth, SMem::Offset:Depth16b);

            // Cache the key.
            Store8bitUintInLow16bitSMem(i, key, SMem::Offset:Key8b);
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

// Depth Min Max reduction.
float2 GetRayGroupMinMaxDepth(uint GI)
{
    // Reduce all ray values to per wave values.
    for (uint i = GI; i < NUM_RAYS; i += NUM_THREADS)
    {
        uint key = Load8bitValueFromLow16bitSMem(i, SMem::Offset:Key8b);
        float depth = Load16bitFloatFromSMem(i, SMem::Offset:Depth16b);
        bool isValidRay = depth != INVALID_RAY_DEPTH;
        
        float waveDepthMin = WaveActiveMin(isValidRay ? depth : FLT_MAX);
        float waveDepthSum = WaveActiveSum(isValidRay ? depth : 0);
        uint waveValidRayCount = WaveActiveSum(isValidRay);

        if (WaveGetLaneIndex() == 0)
        {
            uint waveIndex = i / WaveGetLaneCount();

            Store16bitFloatInSMem(waveIndex, waveDepthMin, SMem::Offset:WaveDepthMin);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Reduce all per-wave values into a single value.
    uint RayGroupSize = SortRays::RayGroup::Size;
    for (UINT s = WaveGetLaneCount(); s < RayGroupSize; s *= WaveGetLaneCount())
    {
        uint numLanesToProcess = (RayGroupSize + s - 1) / s;
        if (GI >= numLanesToProcess)
        {
            break;
        }

        uint encodedWaveMinMaxDepth = Get16bitValueFromSMem(GI + SMCACHE_WAVE_MIN_MAX_DEPTH_OFFSET);
        float2 waveMinMaxDepth = HalfToFloat2(encodedWaveMinMaxDepth);

        waveMinMaxDepth.x = WaveActiveMin(waveMinMaxDepth.x);
        waveMinMaxDepth.y = WaveActiveMax(waveMinMaxDepth.y);

        if (WaveGetLaneIndex() == 0)
        {
            uint encodedWaveMinMaxDepth = Float2ToHalf(waveMinMaxDepth);
            uint waveIndex = i / WaveGetLaneCount();

            Store16bitValueInSMem(waveIndex + SMCACHE_WAVE_MIN_MAX_DEPTH_OFFSET, encodedWaveMinMaxDepth);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint encodedWaveMinMaxDepth = Get16bitValueFromSMem(0 + SMCACHE_WAVE_MIN_MAX_DEPTH_OFFSET);
    return HalfToFloat2(encodedWaveMinMaxDepth);
}

// Combine the depth hash key with the ray direction hash keys and update the key histograms.
FinalizeHashKeyAndCalculateKeyHistogram(uint GI)
{
    for (uint i = GI; i < NUM_RAYS; i += NUM_THREADS)
    {
        bool useHi8Bits = i >= 4096;
        uint encodedKey = Get8bitValueFromSMem(i + SMCACHE_KEY_OFFSET, useHi8Bits);
        bool isValidKey = encodedKey & 0x8000;

        if (isValidKey)
        {
            float depth = GetCachedDepthValue(i);
            uint depthHashKey = CreateDepthHashKey(depth, rayGroupMinMaxDepth);

            key += depthHashKey << (2 * RAY_DIRECTION_HASH_KEY_BITS_1D)

                // Update the key histograms.
                AddTo16bitValueInSMem()
                AddToKeyCounts(bufferID, encodedKey, 1);
        }
        // Cache the key.
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
        bool useHiBits = index & 1;
        uint packedKey = encodedKey << (useHiBits * 16);
        InterlockedXor(SMEM[index / 2 + SMCACHE_KEY_OFFSET], packedKey);
#else
        Store16bitValueInSMem(i + SMCACHE_KEY_OFFSET, encodedKey);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
}

void GenerateHashKeysAndKeyHistogram(uint2 Gid, uint GI)
{
    CalculatePartialRayDirectionHashKeyAndCacheDepth(Gid, GI);
    float2 minMaxDepth = GetRayGroupMinMaxDepth(GI);
    FinalizeHashKeyAndCalculateKeyHistogram(GI);
}

// Prefix sum.
// Requirements: 
//  - NUM_KEYS <= WaveGetLaneCount() ^ 2
//    To support smaller waves or more keys, the implementation needs 
//    to be extended to have more iterations levels.
void PrefixSum(uint GI)
{
    // PrefixSum at the lane level.
    for (uint key = GI; key < NUM_KEYS; key += NUM_THREADS)
    {
        uint count = GetKeyCount(0, key);
        uint prefixSum = WavePrefixSum(count);

        SetKeyCount(1, key, prefixSum);
    }
    GroupMemoryBarrierWithGroupSync();

    // PostfixSum for last lane at each wave, except the last.
    uint NumWaves = NUM_KEYS / WaveGetLaneCount();
    for (uint i = GI; i < NumWaves - 1; i += NUM_THREADS)
    {
        uint lastPrevWaveIndex = (i + 1) * WaveGetLaneCount() - 1;
        uint count = GetKeyCount(1, lastPrevWaveIndex);
        uint postfixSum = count + WavePrefixSum(count);
        
        SetKeyCount(0, i, postfixSum);
    }
    GroupMemoryBarrierWithGroupSync();

    // Add the last lane wave level postfix sum to lane values of subsequent wave.
    for (uint j = GI + WaveGetLaneCount(); j < NUM_KEYS; j += NUM_THREADS)
    {
        uint index = j / WaveGetLaneCount() - 1; // Get the last value of the previous wave segment.
        uint prefixSum = GetKeyCount(0, index);
        
        AddToKeyCounts(1, j, prefixSum);
    }
    GroupMemoryBarrierWithGroupSync();
}

// Write the sorted indices to shared memory to avoid costly scatter write to VRAM.
// These can then be later linearly spilled to VRAM.
void ScatterWriteSortedIndicesToSharedMemory(uint2 Gid, uint GI)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;
    uint2 RayGroupEnd = min(GroupStart + RayGroupDim, CB.dim);
    RayGroupDim = RayGroupEnd - GroupStart;

    uint p = 1;   // index of a buffer last written to in prefix sum.

    for (uint i = GI; i < NUM_RAYS; i += NUM_THREADS)
    {
        uint2 srcIndex = uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
        uint2 pixel = GroupStart + srcIndex;
        if (IsWithinBounds(pixel, CB.dim))
        {
            // Get the key for the corresponding pixel.
            uint key;
            {
                // First, see if the cached key is stil valid.
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
                bool useHiBits = i & 1;
                uint packedKey = SMEM[i / 2 + SMCACHE_KEY_OFFSET];
                uint encodedKey = (packedKey >> (useHiBits * 16)) & 0xffff;
#else
                uint encodedKey = Get16bitValueFromSMem(i + SMCACHE_KEY_OFFSET);
#endif
                bool isValidKey = encodedKey & 0x8000;

                if (isValidKey)
                {
                    key = encodedKey & 0x7fff;
                }
                else  // The cached key has been already replaced with the ray's source index. Regenerate the key.
                {
                    float4 rayDirectionDepthHit = g_inRayDirectionOriginDepthHit[pixel];
                    bool isValidRay = rayDirectionDepthHit.z >= 0;

                    if (isValidRay)
                    {
                        float3 rayDirection = DecodeNormal(rayDirectionDepthHit.xy);
                        float linearDepth = rayDirectionDepthHit.z;
                        key = CreateHashKey(rayDirection, linearDepth);
                    }
                    else
                    {
                        key = INVALID_RAY_KEY;
                    }
                }
            }
            
            uint index = AddToKeyCounts(p, key, 1);
            // To avoid costly scatter writes to VRAM, cache indices into SMem here instead.
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
            uint useHiBits = index & 1;
            uint packedSrcIndex = (srcIndex.y << (useHiBits * 16))
                + (srcIndex.x << (useHiBits * 16 + 8));

            // Zero out the key so that XOR cam next set the srcIndex.
            uint keyBitsToZeroOut = useHiBits ? 0x0000ffff : 0xffff0000;
            InterlockedAnd(SMEM[index / 2], keyBitsToZeroOut);
            InterlockedXor(SMEM[index / 2], packedSrcIndex);
#else
            uint packedSrcIndex = srcIndex.y + (srcIndex.x << 8);
            Store16bitValueInSMem(index, packedSrcIndex);
#endif
#if 0
            g_outDebug[outPixel] = uint4(index, 0, srcIndex);
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
}


void SpillCachedIndicesToVRAM(uint2 Gid, uint GI)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;

    // Sequentially spill cached indices into VRAM.
    for (uint index = GI; index < NUM_RAYS; index += NUM_THREADS)
    {
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
        uint packedSrcIndex = SMEM[index / 2];
        bool useHiBits = index & 1;
        packedSrcIndex = useHiBits ? (packedSrcIndex >> 16) : (packedSrcIndex & 0xffff);
#else
        uint packedSrcIndex = Get16bitValueFromSMem(index);
#endif

        uint2 srcIndex = uint2(packedSrcIndex >> 8, packedSrcIndex & 0xff);
        uint2 outPixel = GroupStart + uint2(index % SortRays::RayGroup::Width, index / SortRays::RayGroup::Width);
        g_outSortedThreadGroupIndices[outPixel] = srcIndex;
    }
}

[numthreads(SortRays::ThreadGroup::Width, SortRays::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    InitializeSharedMemory(GI);
    GenerateHashKeysAndKeyHistogram(Gid, GI);
    PrefixSum(GI);
    ScatterWriteSortedIndicesToSharedMemory(Gid, GI);
    SpillCachedIndicesToVRAM(Gid, GI);
}
