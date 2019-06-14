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

#define KEY_NUM_BITS (DEPTH_HASH_KEY_BITS + 2*RAY_DIRECTION_HASH_KEY_BITS_1D)
#define NUM_KEYS (1 << KEY_NUM_BITS)        // Largest key is reserved for an invalid key.
#define INACTIVE_RAY_KEY (NUM_KEYS - 1)      // Hash key for an invalid/disabled ray. These rays will get sorted to the end and are not to be ray traced.
#define IsActiveRay(RayGroupRayIndexOffset) (((RayGroupRayIndexOffset.x << 8) + RayGroupRayIndexOffset.y) != INACTIVE_RAY_KEY)