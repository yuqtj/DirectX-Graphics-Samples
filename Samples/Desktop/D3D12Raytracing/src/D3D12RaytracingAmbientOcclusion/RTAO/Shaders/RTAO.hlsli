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

namespace RTAO {
    static const float RayHitDistanceOnMiss = -1;// ToDo unify with DISTANCE_ON_MISS - should be 0 as we're using non-negative low precision formats
    bool HasAORayHitAnyGeometry(in float tHit)
    {
        return tHit != RayHitDistanceOnMiss;
    }
}