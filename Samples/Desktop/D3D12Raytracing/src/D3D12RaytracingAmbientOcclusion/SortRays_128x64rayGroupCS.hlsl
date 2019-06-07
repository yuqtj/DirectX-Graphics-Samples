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

// ToDo test 64KB available on NV Pascal & Turing
// ToDo test hard coded ascending w/o CB.nullItem

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float4> g_inRayDirectionOriginDepthHit : register(t0);
RWTexture2D<uint2> g_outSortedThreadGroupIndices : register(u0);  // Thread group per-pixel index offsets within each 128x64 pixel group.
RWTexture2D<uint4> g_outDebug : register(u1);  // Thread group per-pixel index offsets within each 128x64 pixel group.

ConstantBuffer<SortRaysConstantBuffer> CB: register(b0);

#define INDEX_HASH_BITS 14          // ~ Up to 16384 indices
#define NORMAL_KEY_HASH_BITS_1D 3
#define DEPTH_KEY_HASH_BITS (32 - (INDEX_HASH_BITS + 2 * NORMAL_KEY_HASH_BITS_1D))
#define NUM_ELEMENTS SortRays::RayGroup::Size
#define NUM_THREADS SortRays::ThreadGroup::Size
#define NUM_ELEMENT_PAIRS_PER_THREAD SortRays::RayGroup::NumElementPairsPerThread

// The keys consist of the ray origin and normal hash in the high bits and index value in the low bits.
groupshared uint SortKeys[NUM_ELEMENTS];

// Create a hash key from a normal and depth. 
// The hash key also embeds source pixel thread group index to lower SM cost.
// Ref: Ray Reordering Techniques for GPU Ray-Cast Ambient Occlusion
uint CreateHashKey(in float3 normal, in float linearDepth, in uint GI)
{
    // Convert cartesian coordinate normal to spherical coordinates.
    float azimuthAngle = atan2(normal.y, normal.x);
    float polarAngle = acos(normal.z);

    // Normalize to <0,1>.
    float2 normalKey = float2(
        azimuthAngle / (2 * PI),
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
    return    (depthKeyHash << (2 * NORMAL_KEY_HASH_BITS_1D + INDEX_HASH_BITS))
            + (normalKeyHash.y << (NORMAL_KEY_HASH_BITS_1D + INDEX_HASH_BITS))
            + (normalKeyHash.x << INDEX_HASH_BITS)
            + GI;
}

void FillSortKey(uint2 pixel, uint GI)
{
    // Invalid ray entries must sort to the end.
    if (IsWithinBounds(pixel, CB.dim))
    {
        float4 normalDepthHit = g_inRayDirectionOriginDepthHit[pixel];
        float3 normal = DecodeNormal(normalDepthHit.xy);
        float linearDepth = normalDepthHit.z;
        bool hit = normalDepthHit.w;

        if (hit)
        {
            SortKeys[GI] = CreateHashKey(normal, linearDepth, GI);
        }
        else
        {
            SortKeys[GI] =  CB.nullItem;
        }
    }
    else
    {
        SortKeys[GI] = CB.nullItem;
    }
}

// Takes Value and widens it by one bit at the location of the bit
// in the mask.  A one is inserted in the space.  OneBitMask must
// have one and only one bit set.
uint InsertOneBit(uint Value, uint OneBitMask)
{
    uint Mask = OneBitMask - 1;
    return (Value & ~Mask) << 1 | (Value & Mask) | OneBitMask;
}

uint InsertZeroBit(uint Value, uint BitIdx)
{
    uint Mask = BitIdx - 1;
    return (Value & ~Mask) << 1 | (Value & Mask);
}

// Determines if two sort keys should be swapped in the list.  NullItem is
// either 0 or 0xffffffff.  XOR with the NullItem will either invert the bits
// (effectively a negation) or leave the bits alone.  When the the NullItem is
// 0, we are sorting descending, so when A < B, they should swap.  For an
// ascending sort, ~A < ~B should swap.
bool ShouldSwap(uint A, uint B)
{
    return (A ^ CB.nullItem) < (B ^ CB.nullItem);
}

void BitonicSort(uint GI)
{
    // ToDo DXC fails to unroll
    // This is better unrolled because it reduces ALU and because some
    // architectures can load/store two LDS items in a single instruction
    // as long as their separation is a compile-time constant.
    // [unroll]
    for (uint k = 2; k <= NUM_ELEMENTS; k <<= 1)
    {
        //[unroll]
        for (uint j = k / 2; j > 0; j /= 2)
        {
            // Loop over all N/2 unique element pairs
            for (uint i = GI; i < NUM_ELEMENTS / 2; i += NUM_THREADS)
            {
                uint Index1 = InsertZeroBit(i, j);
                uint Index2 = Index1 | j;

                uint A = SortKeys[Index1];
                uint B = SortKeys[Index2];

                if ((A < B) != ((Index1 & k) == 0))
                {
                    // Swap the keys
                    SortKeys[Index1] = B;
                    SortKeys[Index2] = A;
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }
}

void StoreKeyIndex(in uint2 pixel, in uint element)
{
    if (IsWithinBounds(pixel, CB.dim))
    {
        uint sortKey = SortKeys[element];

        uint srcGI = (sortKey & ((1 << INDEX_HASH_BITS) - 1));
        g_outSortedThreadGroupIndices[pixel] = uint2(srcGI % SortRays::RayGroup::Width, srcGI / SortRays::RayGroup::Width);

#if 1
        const uint RayDirectionMask = (1 << NORMAL_KEY_HASH_BITS_1D) - 1;
        uint2 rayDir = uint2(
            (sortKey >> INDEX_HASH_BITS) & RayDirectionMask,
            (sortKey >> (INDEX_HASH_BITS + NORMAL_KEY_HASH_BITS_1D)) & RayDirectionMask);

        g_outDebug[pixel] = uint4(rayDir, 0, 0);
#endif
    }
}


[numthreads(SortRays::ThreadGroup::Width, SortRays::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex)
{
    // Item index of the start of this group.
    uint2 GroupStart = Gid * uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    
    // Load inputs.
    {
        uint2 pixel = GroupStart + GTid;
        uint element = GI;
        for (uint i = 0; i < 2 * NUM_ELEMENT_PAIRS_PER_THREAD; i++)
        {
            FillSortKey(pixel, element);
            pixel.y += SortRays::ThreadGroup::Height;
            element += SortRays::ThreadGroup::Size;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Sort the particles from front to back.
    BitonicSort(GI);

    // Write the sorted indices to memory.
    {
        // ToDo morton code swizzle SM before outputting
        uint2 pixel = GroupStart + GTid;
        uint element = GI;
        for (uint i = 0; i < 2 * NUM_ELEMENT_PAIRS_PER_THREAD; i++)
        {
            StoreKeyIndex(pixel, element);
            pixel.y += SortRays::ThreadGroup::Height;
            element += SortRays::ThreadGroup::Size;
        }
    }
}
