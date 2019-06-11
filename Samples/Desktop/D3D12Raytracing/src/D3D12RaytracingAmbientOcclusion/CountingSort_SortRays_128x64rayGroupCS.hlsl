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

// Desc: Counting Sort.
// Algorithm:
// - Calculates histograms for the key hashes
// - Calculates a prefix sum of the histograms
// - Scatter writes the ray source index offsets based on its hash and the prefix sum for the hash key into SMEM cache
// - Linearly spills source index offsets from SMEM cache into VRAM
// Supports RayGroups of up to 8K rays with Ray Group XY dimensions up to {64, 128}.


#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float4> g_inRayDirectionOriginDepthHit : register(t0);
RWTexture2D<uint2> g_outSortedThreadGroupIndices : register(u0);  // Thread group per-pixel index offsets within each 128x64 pixel group.
RWTexture2D<uint4> g_outDebug : register(u1);  // Thread group per-pixel index offsets within each 128x64 pixel group.

ConstantBuffer<SortRaysConstantBuffer> CB: register(b0);

#define NORMAL_KEY_HASH_BITS_1D 4
#define DEPTH_KEY_HASH_BITS 2
#define NUM_ELEMENTS SortRays::RayGroup::Size
#define NUM_THREADS SortRays::ThreadGroup::Size

#define KEY_NUM_BITS (DEPTH_KEY_HASH_BITS + 2*NORMAL_KEY_HASH_BITS_1D)

#if KEY_NUM_BITS > 14
Key can have only up to 14 bits as the 15th bit is used to signify
the cached key value is a valid key.
#endif

#define NUM_KEYS (1 << KEY_NUM_BITS)

#define CACHE_KEYS 1

// Counts of inputs for each key.
// Stored as two ping-pong buffers.
#define NUM_PING_PONG_BUFFERS 2
groupshared uint KeyCounts[NUM_PING_PONG_BUFFERS][NUM_KEYS];

// SMCache stores 16 bit values, two values per 32bit entry:
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
//   to denote whether the stored hashed key is still valid. If the key is no longer
//   valid, it is regenerated again. 
//   To lower the collision and keep as many cached keys, the cache
//   is extended to the remaining shader memory limit and the keys
//   are stored at an offset. This way, only the first 2*OFFSET keys
//   will be replaced by the source indices.
#define SMCACHE_KEY_OFFSET 2048
#define SMCACHE_SIZE  (NUM_ELEMENTS / 2 + SMCACHE_KEY_OFFSET)
// Sorted source indices for each ray, packed as two 2D 8bit indices
groupshared uint SMCache[SMCACHE_SIZE];

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
        KeyCounts[0][key] = 0;
    }

    for (uint index = GI; index < SMCACHE_SIZE; index += NUM_THREADS)
    {
        SMCache[index] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

void AddKeyCount(uint bufferID, uint2 pixel, uint index)
{
    if (IsWithinBounds(pixel, CB.dim))
    {
        float4 normalDepthHit = g_inRayDirectionOriginDepthHit[pixel];
        //bool hit = normalDepthHit.z;

        //if (hit)
        {
            float3 normal = DecodeNormal(normalDepthHit.xy);
            float linearDepth = normalDepthHit.z;
            uint key = CreateHashKey(normal, linearDepth);
#if CACHE_KEYS
            bool useHiBits = index & 1;
            uint encodedKey = key | 0x8000;   // Denote this being valid key entry with the most significant bit.
            uint packedKey = encodedKey << (useHiBits * 16);
            InterlockedXor(SMCache[index / 2 + SMCACHE_KEY_OFFSET], packedKey);
#endif
            InterlockedAdd(KeyCounts[bufferID][key], 1);
        }
    }
}

// Calculate histograms of input occurrence per key and stores them in buffer 0.
void CalculateKeyCountHistograms(uint2 Gid, uint GI)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;
    uint outBufferID = 0;

    for (uint i = GI; i < NUM_ELEMENTS; i += NUM_THREADS)
    {
        uint2 pixel = GroupStart + uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
        AddKeyCount(outBufferID, pixel, i);
    }
    GroupMemoryBarrierWithGroupSync();
}

void ExclusivePrefixSum(uint GI)
{
    // For exclusive sum, initial value is equal to that of preceding neighbor.
    for (int key = GI; key < NUM_KEYS; key += NUM_THREADS)
    {
        KeyCounts[1][key] = key > 0 ? KeyCounts[0][key - 1] : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Prefix sum.
    // ToDo support non-pow 2 NUM_KEYS
    for (uint p = 1, k = 1; k < NUM_KEYS; k <<= 1, p ^= 1)
    {
        for (uint key = GI; key < NUM_KEYS; key += NUM_THREADS)
        {
            uint sum = KeyCounts[p][key];
            if (key >= k)
            {
                sum += KeyCounts[p][key - k];
            }
            KeyCounts[p ^ 1][key] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// Prefix sum.
// Requirements: 
//  - NUM_KEYS <= 1024
//  - WaveGetLaneCount() >= 32
void PrefixSum(uint GI)
{
    // PrefixSum
    for (uint key = GI; key < NUM_KEYS; key += NUM_THREADS)
    {
        uint count = KeyCounts[0][key];
        uint prefixSum = WavePrefixSum(count);

        KeyCounts[1][key] = prefixSum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Calculate PostfixSum across waves
    uint NumWaves = NUM_KEYS / WaveGetLaneCount() - 1;
    for (uint i = GI; i < NumWaves; i += NUM_THREADS)
    {
        uint lastPrevWaveIndex = (i + 1) * WaveGetLaneCount() - 1;

        uint count = KeyCounts[1][lastPrevWaveIndex];
        uint postfixSum = count + WavePrefixSum(count);

        KeyCounts[0][i] = postfixSum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Add the wave level postfix sums to lanes.
    for (uint j = GI + WaveGetLaneCount(); j < NUM_KEYS; j += NUM_THREADS)
    {
        uint prefixSum = KeyCounts[0][j / WaveGetLaneCount() - 1];
        KeyCounts[1][j] += prefixSum;
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

    uint p = 1;// (uint(log2(NUM_KEYS)) & 1) ^ 1;    // index of a buffer last written to in prefix sum.

    for (uint i = GI; i < NUM_ELEMENTS; i += NUM_THREADS)
    {
        uint2 srcIndex = uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
        uint2 pixel = GroupStart + srcIndex;
        if (IsWithinBounds(pixel, CB.dim))
        {
            // Get the key for the corresponding pixel.
            uint key;
            {
                // First, see if the cached key is stil valid.
                bool useHiBits = i & 1;
                uint packedKey = SMCache[i / 2 + SMCACHE_KEY_OFFSET];
                uint encodedKey = useHiBits ? (packedKey >> 16) : (packedKey & 0xffff);
                bool isValidKey = encodedKey & 0x8000;

                if (isValidKey)
                {
                    key = encodedKey & 0x7fff;// ~0x8000;
                }
                else  // The cached key has been already replaced with srcIndex. Regenerate it.
                {
                    float4 normalDepthHit = g_inRayDirectionOriginDepthHit[pixel];
                    //bool hit = normalDepthHit.w;

                    //if (hit)
                    {
                        float3 normal = DecodeNormal(normalDepthHit.xy);
                        float linearDepth = normalDepthHit.z;
                        key = CreateHashKey(normal, linearDepth);
                    }
                }
            }

            uint index;
            InterlockedAdd(KeyCounts[p][key], 1, index);

            // To avoid costly scatter writes to VRAM, cache indices into SMEM here instead.
            uint useHiBits = index & 1;
            uint packedSrcIndex = (srcIndex.y << (useHiBits * 16))
                + (srcIndex.x << (useHiBits * 16 + 8));

            // Zero out the key so that XOR cam next set the srcIndex.
            uint keyBitsToZeroOut = useHiBits ? 0x0000ffff : 0xffff0000;
            InterlockedAnd(SMCache[index / 2], keyBitsToZeroOut);
            InterlockedXor(SMCache[index / 2], packedSrcIndex);
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
    for (uint index = GI; index < NUM_ELEMENTS; index += NUM_THREADS)
    {
        uint packedSrcIndex = SMCache[index / 2];
        bool useHiBits = index & 1;
        packedSrcIndex = useHiBits ? (packedSrcIndex >> 16) : (packedSrcIndex & 0xffff);

        uint2 srcIndex = uint2(packedSrcIndex >> 8, packedSrcIndex & 0xff);
        uint2 outPixel = GroupStart + uint2(index % SortRays::RayGroup::Width, index / SortRays::RayGroup::Width);
        g_outSortedThreadGroupIndices[outPixel] = srcIndex;
    }
}

[numthreads(SortRays::ThreadGroup::Width, SortRays::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    InitializeSharedMemory(GI);
    CalculateKeyCountHistograms(Gid, GI);
    //ExclusivePrefixSum(GI); 
    PrefixSum(GI);
    ScatterWriteSortedIndicesToSharedMemory(Gid, GI);
    SpillCachedIndicesToVRAM(Gid, GI);
}
