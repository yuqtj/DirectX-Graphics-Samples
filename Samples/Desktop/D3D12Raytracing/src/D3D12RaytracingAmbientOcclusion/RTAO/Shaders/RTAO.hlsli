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
#ifndef RTAO_HLSLI
#define RTAO_HLSLI
// ToDo use defines or namespace?
namespace RTAO {
    static const float RayHitDistanceOnMiss = -1;// ToDo unify with DISTANCE_ON_MISS - should be 0 as we're using non-negative low precision formats
    static const float InvalidAOValue = -2; // ToDo - can't be -1, as Temporal uses hack to negate AO so that denoiser knows which values are new and which stale.
    bool HasAORayHitAnyGeometry(in float tHit)
    {
        return tHit != RayHitDistanceOnMiss;
    }
}

#endif // RTAO_HLSLI