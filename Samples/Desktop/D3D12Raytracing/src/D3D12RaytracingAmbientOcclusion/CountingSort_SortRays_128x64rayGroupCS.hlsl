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


#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float4> g_inRayDirectionOriginDepthHit : register(t0);
RWTexture2D<uint2> g_outSortedThreadGroupIndices : register(u0);  // Thread group per-pixel index offsets within each 128x64 pixel group.
RWTexture2D<uint4> g_outDebug : register(u1);  // Thread group per-pixel index offsets within each 128x64 pixel group.

ConstantBuffer<SortRaysConstantBuffer> CB: register(b0);

#define NORMAL_KEY_HASH_BITS_1D 4
#define DEPTH_KEY_HASH_BITS 2
#define NUM_RAYS SortRays::RayGroup::Size
#define NUM_THREADS SortRays::ThreadGroup::Size

#define MAX_RAYS 8192
#if MAX_RAYS > 8192
The shader supports up to 8192 input rays.
#endif

//********************************************************************
// Hash Key
//  - a hash calculated from normal and depth of a ray
//  - max values:
//      12 bits(4096) for 8K rays.
//      13 bits(8192) for 4K rays.
// The 15th and 16th bits are reserved:
// - 15th bit - key for invalid ray. These rays will get sorted to the end.
// - 16th bit - invalid key. To handle when a key is replaced by Source Ray index in SMCache.
#define KEY_NUM_BITS (DEPTH_KEY_HASH_BITS + 2*NORMAL_KEY_HASH_BITS_1D)
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
// Ray Count SMEM cache.
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


//********************************************************************
// SMCache stores 16 bit values, two 16bit values per 32bit entry:
// - Hi bits: odd indices
// - Lo bits: even indices
// SMCache is used for two mutually exclusive and temporally overlapping purposes
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
#define SMCACHE_INTERLEAVE_EVEN_ODD_INDICES 1   // Faster by 3.5%
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
    #define SMCACHE_KEY_OFFSET (SMCACHE_SIZE - (MAX_RAYS + 1) / 2)
#else
    #define SMCACHE_KEY_OFFSET (2 * SMCACHE_SIZE - NUM_RAYS)
#endif
groupshared uint SMCache[SMCACHE_SIZE];
//********************************************************************

// Create a hash key from a normal and depth. 
// Ref: Ray Reordering Techniques for GPU Ray-Cast Ambient Occlusion
uint CreateHashKey(in float3 normal, in float linearDepth)
{
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
  
    const uint DepthKeyHashBins = 1 << DEPTH_KEY_HASH_BITS;
    uint depthKeyHash = min(linearDepth / CB.binDepthSize, DepthKeyHashBins - 1);

    // Create a combined hash key:
    //  - Hi bits: depthKeyHash
    //  - Low bits: normalKeyHashes Y and X
    return   (depthKeyHash << (2 * NORMAL_KEY_HASH_BITS_1D))
            + (normalKeyHash.y << (NORMAL_KEY_HASH_BITS_1D ))
            + normalKeyHash.x;
}

void InitializeSharedMemory(uint GI)
{
    for (uint key = GI; key < NUM_KEYS; key += NUM_THREADS)
    {
        KeyCounts[key] = 0;
    }

    for (uint index = GI; index < SMCACHE_SIZE; index += NUM_THREADS)
    {
        SMCache[index] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

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

#if !SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
void SetSMCacheValue(in uint index, in uint value)
{
    bool useHiBits = index >= SMCACHE_SIZE;
    uint cacheIndex = index - useHiBits * SMCACHE_SIZE;
    uint packedValue = value << (useHiBits * 16);
    uint keyBitsToZeroOut = useHiBits ? 0x0000ffff : 0xffff0000;
    InterlockedAnd(SMCache[cacheIndex], keyBitsToZeroOut);
    InterlockedXor(SMCache[cacheIndex], packedValue);
}

uint GetSMCacheValue(in uint index)
{
    bool useHiBits = index >= SMCACHE_SIZE;
    uint cacheIndex = index - useHiBits * SMCACHE_SIZE;
    uint packedValue = SMCache[cacheIndex];
    return (packedValue >> (useHiBits * 16)) & 0xffff;
}
#endif

void AddKeyCount(uint bufferID, uint2 pixel, uint index)
{
    if (IsWithinBounds(pixel, CB.dim))
    {
        float4 normalDepthHit = g_inRayDirectionOriginDepthHit[pixel];
        bool isRayValid = normalDepthHit.z >= 0;

        uint key;
        if (isRayValid)
        {
            float3 normal = DecodeNormal(normalDepthHit.xy);
            float linearDepth = normalDepthHit.z;
            key = CreateHashKey(normal, linearDepth);
        }
        else
        {
            key = INVALID_RAY_KEY;
        }

        // Cache the key.
        uint encodedKey = key | 0x8000;   // Denote this being valid key entry with the most significant bit being 1.
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
        bool useHiBits = index & 1;
        uint packedKey = encodedKey << (useHiBits * 16);
        InterlockedXor(SMCache[index / 2 + SMCACHE_KEY_OFFSET], packedKey);
#else
        SetSMCacheValue(index + SMCACHE_KEY_OFFSET, encodedKey);
#endif
        AddToKeyCounts(bufferID, key, 1);
    }
}

// Calculate histograms of input occurrence per key and stores them in buffer 0.
void CalculateKeyCountHistograms(uint2 Gid, uint GI)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;
    uint outBufferID = 0;

    for (uint i = GI; i < NUM_RAYS; i += NUM_THREADS)
    {
        uint2 pixel = GroupStart + uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
        AddKeyCount(outBufferID, pixel, i);
    }
    GroupMemoryBarrierWithGroupSync();
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
                uint packedKey = SMCache[i / 2 + SMCACHE_KEY_OFFSET];
                uint encodedKey = (packedKey >> (useHiBits * 16)) & 0xffff;
#else
                uint encodedKey = GetSMCacheValue(i + SMCACHE_KEY_OFFSET);
#endif
                bool isValidKey = encodedKey & 0x8000;

                if (isValidKey)
                {
                    key = encodedKey & 0x7fff;
                }
                else  // The cached key has been already replaced with the ray's source index. Regenerate the key.
                {
                    float4 normalDepthHit = g_inRayDirectionOriginDepthHit[pixel];
                    bool isValidRay = normalDepthHit.z >= 0;

                    if (isValidRay)
                    {
                        float3 normal = DecodeNormal(normalDepthHit.xy);
                        float linearDepth = normalDepthHit.z;
                        key = CreateHashKey(normal, linearDepth);
                    }
                    else
                    {
                        key = INVALID_RAY_KEY;
                    }
                }
            }
            
            uint index = AddToKeyCounts(p, key, 1);
            // To avoid costly scatter writes to VRAM, cache indices into SMEM here instead.
#if SMCACHE_INTERLEAVE_EVEN_ODD_INDICES
            uint useHiBits = index & 1;
            uint packedSrcIndex = (srcIndex.y << (useHiBits * 16))
                + (srcIndex.x << (useHiBits * 16 + 8));

            // Zero out the key so that XOR cam next set the srcIndex.
            uint keyBitsToZeroOut = useHiBits ? 0x0000ffff : 0xffff0000;
            InterlockedAnd(SMCache[index / 2], keyBitsToZeroOut);
            InterlockedXor(SMCache[index / 2], packedSrcIndex);
#else
            uint packedSrcIndex = srcIndex.y + (srcIndex.x << 8);
            SetSMCacheValue(index, packedSrcIndex);
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
        uint packedSrcIndex = SMCache[index / 2];
        bool useHiBits = index & 1;
        packedSrcIndex = useHiBits ? (packedSrcIndex >> 16) : (packedSrcIndex & 0xffff);
#else
        uint packedSrcIndex = GetSMCacheValue(index);
#endif

        uint2 srcIndex = uint2(packedSrcIndex >> 8, packedSrcIndex & 0xff);
        uint2 outPixel = GroupStart + uint2(index % SortRays::RayGroup::Width, index / SortRays::RayGroup::Width);
        g_outSortedThreadGroupIndices[outPixel] = srcIndex;
    }
}

[numthreads(SortRays::ThreadGroup::Width, SortRays::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    // Algorithm: Counting Sort
    // - Calculates histograms for the key hashes
    // - Calculates a prefix sum of the histograms
    // - Scatter writes the ray source index offsets based on its hash and the prefix sum for the hash key into SMEM cache
    // - Linearly spills source index offsets from SMEM cache into VRAM

    InitializeSharedMemory(GI);
    CalculateKeyCountHistograms(Gid, GI);
    PrefixSum(GI);
    ScatterWriteSortedIndicesToSharedMemory(Gid, GI);
    SpillCachedIndicesToVRAM(Gid, GI);
}
