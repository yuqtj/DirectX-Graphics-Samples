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

// Desc: Counting Sort. ToDo
// Algorithm:
// - Calculates histograms for the key hashes
// - Calculates a prefix sum of the histograms
// - Scatter writes the ray source index offsets based on its hash and the prefix sum for the hash key into SMEM cache
// - Linearly spills source index offsets from SMEM cache into VRAM


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

#define NUM_KEYS (1 << (DEPTH_KEY_HASH_BITS + 2*NORMAL_KEY_HASH_BITS_1D))


// Counts of inputs for each key.
// Stored as two ping-pong buffers.
#define NUM_PING_PONG_BUFFERS 2
groupshared uint KeyCounts[NUM_PING_PONG_BUFFERS][NUM_KEYS];

// Sorted source indices for each ray, packed as two 2D 8bit indices
// Hi bits: odd indices
// Lo bits: even indices
groupshared uint SrcIndices[NUM_ELEMENTS / 2];


// Create a hash key from a normal and depth. 
// The hash key also embeds source pixel thread group index to lower SM cost.
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

// Zeroes out counts for keys in buffer 0.
void ZeroOutKeys(uint GI)
{
    for (uint key = GI; key < NUM_KEYS; key += NUM_THREADS)
    {
        KeyCounts[0][key] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

void AddKeyCount(uint bufferID, uint2 pixel)
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
        AddKeyCount(outBufferID, pixel);
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

void DebugPrintOutHistograms(uint2 Gid, uint GI, uint bufferId)
{
    uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 GroupStart = Gid * RayGroupDim;

    for (uint i = GI; i < NUM_KEYS; i += NUM_THREADS)
    {
        uint2 pixel = GroupStart + uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
        g_outDebug[pixel].xyz = uint3(KeyCounts[0][i], KeyCounts[1][i], KeyCounts[bufferId][i]);
    }
}

[numthreads(SortRays::ThreadGroup::Width, SortRays::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    ZeroOutKeys(GI);

    CalculateKeyCountHistograms(Gid, GI);

    ExclusivePrefixSum(GI);
    
    // Zero out the source indices caches as we're using XOR to update it.
    for (uint index = GI; index < NUM_ELEMENTS / 2; index += NUM_THREADS)
    {
        SrcIndices[index] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Write the sorted indices to memory.
    {
        uint2 RayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
        uint2 GroupStart = Gid * RayGroupDim;
        uint2 RayGroupEnd = min(GroupStart + RayGroupDim, CB.dim);
        RayGroupDim = RayGroupEnd - GroupStart;

        uint p = (uint(log2(NUM_KEYS)) & 1) ^ 1;    // index of a buffer last written to in prefix sum.

        for (uint i = GI; i < NUM_ELEMENTS; i += NUM_THREADS)
        {
            uint2 srcIndex = uint2(i % SortRays::RayGroup::Width, i / SortRays::RayGroup::Width);
            uint2 pixel = GroupStart + srcIndex;
            if (IsWithinBounds(pixel, CB.dim))
            {
                // ToDo double check ray groups overflowing dimensions are properly handled
                float4 normalDepthHit = g_inRayDirectionOriginDepthHit[pixel];
                //bool hit = normalDepthHit.w;

                //if (hit)
                {
                    float3 normal = DecodeNormal(normalDepthHit.xy);
                    float linearDepth = normalDepthHit.z;
                    uint key = CreateHashKey(normal, linearDepth);

                    uint index;
                    InterlockedAdd(KeyCounts[p][key], 1, index);
                    
                    // To avoid costly scatter writes to VRAM, cache indices into SMEM here instead.
                    bool useHiBits = index & 1;
                    uint packedSrcIndex = (srcIndex.x << (useHiBits * 16)) 
                                        + (srcIndex.y << (useHiBits * 16 + 8));
                    InterlockedXor(SrcIndices[index / 2], packedSrcIndex);
#if 0
                    g_outDebug[outPixel] = uint4(index, 0, srcIndex);
#endif

                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Sequentially spill cached indices into VRAM.
        for (uint index = GI; index < NUM_ELEMENTS; index += NUM_THREADS)
        {
            uint packedSrcIndex = SrcIndices[index/2];
            bool useHiBits = index & 1;
            packedSrcIndex = useHiBits ? (packedSrcIndex >> 16) : (packedSrcIndex & 0xffff);

            uint2 srcIndex = uint2(packedSrcIndex & 0xff, (packedSrcIndex >> 8));
            uint2 outPixel = GroupStart + uint2(index % SortRays::RayGroup::Width, index / SortRays::RayGroup::Width);
            g_outSortedThreadGroupIndices[outPixel] = srcIndex;

        }
    }
}
