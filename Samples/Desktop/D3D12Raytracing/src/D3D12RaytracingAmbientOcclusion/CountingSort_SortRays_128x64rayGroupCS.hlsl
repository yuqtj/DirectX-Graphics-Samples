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

// Desc: Sort rays indices based on their origin and normals.
// Performs a bitonic sort of 128x64 (8192) rays per thread group.

// Description:  A bitonic sort must eventually sort the power-of-two
// ceiling of items.  E.g. 391 items -> 512 items.  Because of this
// "null items" must be used as padding at the end of the list so that
// they can participate in the sort but remain at the end of the list.
//
// The sort does two things.  It appends null items as need, and
// it does the initial sort for k values up to 8192.  This is because
// we can run 4096 threads, each of of which can compare and swap two
// elements without contention.  And because we can always fit 8192
// keys & indices in LDS with occupancy greater than one.  (A single
// thread group can use as much as 32KB of LDS.)
//
// The sort is done by comparing a 32bit hash key. Hash key is generated
// from depth, normal, source thread group index, in order of importance.
// The thread group index is packed within hash key to save on group shared 
// memory and be able to refer to original pixel index they refer to. It's contribution
// to the sort efficiency is minimal and secondary, it will just order same 
// depth/normal signature rays in the sort order.

// ToDo test hard coded ascending w/o CB.nullItem

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float4> g_inRayDirectionOriginDepthHit : register(t0);
RWTexture2D<uint2> g_outSortedThreadGroupIndices : register(u0);  // Thread group per-pixel index offsets within each 128x64 pixel group.
RWTexture2D<uint4> g_outDebug : register(u1);  // Thread group per-pixel index offsets within each 128x64 pixel group.

ConstantBuffer<SortRaysConstantBuffer> CB: register(b0);

#define INDEX_HASH_BITS 13          // ~ Up to 8192 indices
#define NORMAL_KEY_HASH_BITS_1D 4
#define DEPTH_KEY_HASH_BITS 2
#define NUM_ELEMENTS SortRays::RayGroup::Size
#define NUM_THREADS SortRays::ThreadGroup::Size
#define NUM_ELEMENTS_PER_THREAD (2 * SortRays::RayGroup::NumElementPairsPerThread)

#define NUM_KEYS (1 << (DEPTH_KEY_HASH_BITS + 2*NORMAL_KEY_HASH_BITS_1D))


// Counts of inputs for each key.
// Stored as two ping-pong buffers.
#define NUM_PING_PONG_BUFFERS 2
#if RTAO_RAY_SORT_PING_PONG_MASKED
groupshared uint KeyCounts[NUM_KEYS];
#else
groupshared uint KeyCounts[NUM_PING_PONG_BUFFERS][NUM_KEYS];
#endif
#if RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM

#if RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM_PACK_INDICES
groupshared uint SrcIndices[NUM_ELEMENTS / 2];
#else
groupshared uint SrcIndices[NUM_ELEMENTS];
#endif
#endif

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
    //  - Mid bits: normalKeyHashes Y and X
    //  - Low bits: source data thread group index
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
#if RTAO_RAY_SORT_PING_PONG_MASKED
            InterlockedAdd(KeyCounts[key], 1 << (bufferID * 16));
#else
            InterlockedAdd(KeyCounts[bufferID][key], 1);
#endif
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


//
//uint ReadKeyCount(in uint index, in uint iteration)
//{
//#if 0
//    // Ping pong between low and high 16 bits
//    uint mask = 0xffff << (iteration * 16);
//    uint shiftRight = iteration * 16;
//    return (KeyCounts[index] & readMask) >> shiftRight;
//#else
//    if (iteration == 0)
//        return KeyCounts[index] & 0xffff;
//    else
//        return KeyCounts[index] >> 16;
//#endif
//}
//
//uint WriteKeyCount(in uint index, in uint iteration)
//{
//    // Ping pong between low and high 16 bits
//    uint mask = 0xffff << ((iteration + 1) * 16);
//    uint shiftRight = (iteration + 1) * 16;
//    return (KeyCounts[index] & readMask) >> shiftRight;
//}

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

    //uint p = (uint(log2(NUM_KEYS)) & 1) ^ 1;
    //DebugPrintOutHistograms(Gid, GI, p);
    //return;

#if RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM && RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM_PACK_INDICES
    for (uint index = GI; index < NUM_ELEMENTS / 2; index += NUM_THREADS)
    {
        SrcIndices[index] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
#endif
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
                float4 normalDepthHit = g_inRayDirectionOriginDepthHit[pixel];
                //bool hit = normalDepthHit.w;

                //if (hit)
                {
                    float3 normal = DecodeNormal(normalDepthHit.xy);
                    float linearDepth = normalDepthHit.z;
                    uint key = CreateHashKey(normal, linearDepth);

                    uint index;
                    InterlockedAdd(KeyCounts[p][key], 1, index);
                    
#if RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM
                    // To avoid costly scatter writes to VRAM, cache indices into SMEM instead.
#if RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM_PACK_INDICES
                    bool useHiBits = index & 1;
                    uint packedSrcIndex = (srcIndex.x << (useHiBits * 16)) 
                                        + (srcIndex.y << (useHiBits * 16 + 8));
                    InterlockedXor(SrcIndices[index / 2], packedSrcIndex);
#else
                    SrcIndices[index] = srcIndex.x + (srcIndex.y << 8);
#endif
#else
                    uint2 destIndex = uint2(index % SortRays::RayGroup::Width, index / SortRays::RayGroup::Width);

                    // ToDo clip RayGroups if they go out of bounds.

                    // Flip order so that buckets are countinuous when reading the sorted rays
                    // row by row within ray group and then row by row ray groups
                    // ToDo destIndex = (Gid.x & 1) ? (RayGroupDim - 1) - destIndex : destIndex;
                    uint2 outPixel = GroupStart + destIndex;
                    g_outSortedThreadGroupIndices[outPixel] = srcIndex;

#if 0
                    g_outDebug[outPixel] = uint4(index, 0, srcIndex);
#endif
#endif

                }
            }
        }
#if RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM
        GroupMemoryBarrierWithGroupSync();

        // Sequentially spill cached indices into VRAM.
        for (uint index = GI; index < NUM_ELEMENTS; index += NUM_THREADS)
        {
#if RTAO_RAY_SORT_LINEARIZE_WRITE_TO_VRAM_PACK_INDICES
            uint packedSrcIndex = SrcIndices[index/2];
            bool useHiBits = index & 1;
            packedSrcIndex = useHiBits ? (packedSrcIndex >> 16) : (packedSrcIndex & 0xffff);
#else
            uint packedSrcIndex = SrcIndices[index];
#endif
            uint2 srcIndex = uint2(packedSrcIndex & 0xff, (packedSrcIndex >> 8));
            uint2 outPixel = GroupStart + uint2(index % SortRays::RayGroup::Width, index / SortRays::RayGroup::Width);
            g_outSortedThreadGroupIndices[outPixel] = srcIndex;

#if 0
            g_outDebug[outPixel] = uint4(index, 0, srcIndex);
#endif
        }
#endif
    }
}
