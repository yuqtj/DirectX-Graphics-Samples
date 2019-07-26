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

#define RAY_DIRECTION_HASH_KEY_BITS_1D 4    // Max 4 bits
#define DEPTH_HASH_KEY_BITS 2  // ToDo test using 2 bits for RayGroupQuadrant
#define INDEX_HASH_KEY_BITS 0 // (4 - DEPTH_HASH_KEY_BITS)   // ToDo test using 2 bits for RayGroupQuadrant
#define NUM_RAYS SortRays::RayGroup::Size
#define NUM_THREADS SortRays::ThreadGroup::Size
#define MIN_VALID_RAY_DEPTH FLT_10BIT_MIN
#define MAX_RAYS 8192

#define AVOID_SCATTER_WRITES_FOR_SORTED_RAY_RESULTS 0

#define KEY_NUM_BITS (DEPTH_HASH_KEY_BITS + 2*RAY_DIRECTION_HASH_KEY_BITS_1D + INDEX_HASH_KEY_BITS)
#define NUM_KEYS (1 << KEY_NUM_BITS)        // Largest key is reserved for an invalid key.

// ToDo move this to a const?
// ToDo enumerate all reserved bits in one place
// INACTIVE_RAY_KEY must be greater than the max valid hash key but fit within 16bits.
// It must be exclusive to INACTIVE_RAY_INDEX_BIT. - todo cleanup/combine them?
#define INACTIVE_RAY_KEY NUM_KEYS     // Hash key for an invalid/disabled ray. These rays will get sorted to the end and are not to be raytraced.
// ToDo using ACTIVE instead of inactive may be easier to understand the code.
#define INACTIVE_RAY_INDEX_BIT 0x2000
#define INACTIVE_RAY_INDEX_BIT_Y 0x80

#define INVALID_16BIT_KEY_BIT 0x8000      // A value used to denote if the SMEM entry is a 16bit key value or not.
#define INVALID_RAY_ORIGIN_DEPTH 0

// If the ray is inactive, the ray index offset contains INACTIVE_RAY_INDEX_BIT
#define IsActiveRay(RayGroupRayIndexOffset) (!(RayGroupRayIndexOffset.y & INACTIVE_RAY_INDEX_BIT_Y))
#define GetRawRayIndexOffset(RayGroupRayIndexOffset) uint2(RayGroupRayIndexOffset.x, RayGroupRayIndexOffset.y & ~(INACTIVE_RAY_INDEX_BIT_Y))