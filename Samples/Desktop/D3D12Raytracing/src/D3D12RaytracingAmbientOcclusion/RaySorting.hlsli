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

#define RAY_DIRECTION_HASH_KEY_BITS_1D 4
#define DEPTH_HASH_KEY_BITS 2   // ToDo test using 2 bits for RayGroupQuadrant
#define NUM_RAYS SortRays::RayGroup::Size
#define NUM_THREADS SortRays::ThreadGroup::Size
#define MIN_VALID_RAY_DEPTH FLT_10BIT_MIN
#define MAX_RAYS 8192
#if MAX_RAYS > 8192
The shader supports up to 8192 input rays.
#endif

#define AVOID_SCATTER_WRITES_FOR_SORTED_RAY_RESULTS 1

#define SRC_INDEX_HASH_KEY_BITS 0

#define KEY_NUM_BITS (DEPTH_HASH_KEY_BITS + 2*RAY_DIRECTION_HASH_KEY_BITS_1D + SRC_INDEX_HASH_KEY_BITS)
#define NUM_KEYS (1 << KEY_NUM_BITS)        // Largest key is reserved for an invalid key.
#define INACTIVE_RAY_KEY (NUM_KEYS - 1)     // Hash key for an invalid/disabled ray. These rays will get sorted to the end and are not to be ray traced.
#define INACTIVE_RAY_INDEX_BIT 0x2000
#define IsActiveRay(RayGroupRayIndexOffset) (((RayGroupRayIndexOffset.y << 6) & INACTIVE_RAY_INDEX_BIT) == 0)