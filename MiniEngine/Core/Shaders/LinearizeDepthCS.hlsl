//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author:  James Stanard 
//

#include "SSAORS.hlsli"

#ifndef ENABLE_CHECKERBOARD
#define ENABLE_CHECKERBOARD 0
#endif

#if ENABLE_CHECKERBOARD
Texture2DMS<float> Depth : register(t0);
#else
Texture2D<float> Depth : register(t0);
#endif
ByteAddressBuffer HTile : register(t5);
RWTexture2D<float> LinearZ : register(u0);
RWTexture2D<float> _Unused1 : register(u1);
RWTexture2D<uint> _Unused2 : register(u2);

cbuffer CB0 : register(b0)
{
    float ZMagic;				// (zFar - zNear) / zNear
}

[RootSignature(SSAO_RootSig)]
[numthreads( 16, 16, 1 )]
void main( uint3 Gid : SV_GroupID, uint GI : SV_GroupIndex, uint3 GTid : SV_GroupThreadID, uint3 DTid : SV_DispatchThreadID )
{
#if ENABLE_CHECKERBOARD
    LinearZ[DTid.xy] = 1.0 / (ZMagic * Depth.Load(DTid.xy / 2, DTid.x & 1) + 1.0);
#else
    LinearZ[DTid.xy] = 1.0 / (ZMagic * Depth[DTid.xy] + 1.0);
#endif
}
