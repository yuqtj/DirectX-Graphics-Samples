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

RWTexture2D<float> AoResult : register(u0);
Texture2D<float4> Normal : register(t0);
Texture2D<float> Distance : register(t1);
Texture2D<float> InputAO : register(t2);
SamplerState LinearSampler : register(s0);

cbuffer g_aoBlurCB : register(b0)
{
    float2 kRcpBufferDim;
	float kNormalTolerance;
    float kDistanceTolerance;
}

groupshared float4 NCache[256]; // Normal+Z
groupshared float ZCache[256]; // 1/Distance
groupshared float DCache[256]; // Distance
groupshared float AOCache1[256];
groupshared float AOCache2[256];

void PrefetchData( uint index, int2 ST )
{
    float2 uv = ST * kRcpBufferDim;
    float4 AO = InputAO.Gather( LinearSampler, uv );

    AOCache1[index   ] = AO.w;
    AOCache1[index+ 1] = AO.z;
    AOCache1[index+16] = AO.x;
    AOCache1[index+17] = AO.y;

    NCache[index   ] = Normal[ST + int2(-1, -1)];
    NCache[index+ 1] = Normal[ST + int2( 0, -1)];
    NCache[index+16] = Normal[ST + int2(-1,  0)];
    NCache[index+17] = Normal[ST + int2( 0,  0)];

    float4 ID = 1.0 / Distance.Gather( LinearSampler, uv );
    DCache[index   ] = ID.w;
    DCache[index+ 1] = ID.z;
    DCache[index+16] = ID.x;
    DCache[index+17] = ID.y;

    ZCache[index   ] = 1.0 / DCache[index   ];
    ZCache[index+ 1] = 1.0 / DCache[index+ 1];
    ZCache[index+16] = 1.0 / DCache[index+16];
    ZCache[index+17] = 1.0 / DCache[index+17];
}

float SmartBlur(
    float aoA, float aoB, float aoC, float aoD, float aoE,
    float4 nA, float4 nB, float4 nC, float4 nD, float4 nE,
    float zA, float zB, float zC, float zD, float zE)
{
    const float ZTol = kDistanceTolerance * nC.w;
    float wA = 0.5 * dot(nA.xyz, nC.xyz) * (1 - smoothstep(0, 2.0*ZTol, abs(zA - zC)));
    float wB = 1.0 * dot(nB.xyz, nC.xyz) * (1 - smoothstep(0, 1.0*ZTol, abs(zB - zC)));
    float wC = 1.0;
    float wD = 1.0 * dot(nD.xyz, nC.xyz) * (1 - smoothstep(0, 1.0*ZTol, abs(zD - zC)));
    float wE = 0.5 * dot(nE.xyz, nC.xyz) * (1 - smoothstep(0, 2.0*ZTol, abs(zE - zC)));

    return (wA*aoA + wB*aoB + wC*aoC + wD*aoD + wE*aoE) * rcp(wA + wB + wC + wD + wE);
}

void BlurHorizontally( uint leftMostIndex )
{
    float ao0 = AOCache1[leftMostIndex  ];
    float ao1 = AOCache1[leftMostIndex+1];
    float ao2 = AOCache1[leftMostIndex+2];
    float ao3 = AOCache1[leftMostIndex+3];
    float ao4 = AOCache1[leftMostIndex+4];
    float ao5 = AOCache1[leftMostIndex+5];
    float ao6 = AOCache1[leftMostIndex+6];

    float4 n0 = NCache[leftMostIndex  ];
    float4 n1 = NCache[leftMostIndex+1];
    float4 n2 = NCache[leftMostIndex+2];
    float4 n3 = NCache[leftMostIndex+3];
    float4 n4 = NCache[leftMostIndex+4];
    float4 n5 = NCache[leftMostIndex+5];
    float4 n6 = NCache[leftMostIndex+6];

    float z0 = DCache[leftMostIndex  ];
    float z1 = DCache[leftMostIndex+1];
    float z2 = DCache[leftMostIndex+2];
    float z3 = DCache[leftMostIndex+3];
    float z4 = DCache[leftMostIndex+4];
    float z5 = DCache[leftMostIndex+5];
    float z6 = DCache[leftMostIndex+6];

    AOCache2[leftMostIndex  ] = SmartBlur( ao0, ao1, ao2, ao3, ao4, n0, n1, n2, n3, n4, z0, z1, z2, z3, z4 );
    AOCache2[leftMostIndex+1] = SmartBlur( ao1, ao2, ao3, ao4, ao5, n1, n2, n3, n4, n5, z1, z2, z3, z4, z5 );
    AOCache2[leftMostIndex+2] = SmartBlur( ao2, ao3, ao4, ao5, ao6, n2, n3, n4, n5, n6, z2, z3, z4, z5, z6 );
}

void BlurVertically( uint topMostIndex )
{
    float ao0 = AOCache2[topMostIndex   ];
    float ao1 = AOCache2[topMostIndex+16];
    float ao2 = AOCache2[topMostIndex+32];
    float ao3 = AOCache2[topMostIndex+48];
    float ao4 = AOCache2[topMostIndex+64];
    float ao5 = AOCache2[topMostIndex+80];

    float4 n0 = NCache[topMostIndex+ 2];
    float4 n1 = NCache[topMostIndex+18];
    float4 n2 = NCache[topMostIndex+34];
    float4 n3 = NCache[topMostIndex+50];
    float4 n4 = NCache[topMostIndex+66];
    float4 n5 = NCache[topMostIndex+82];

    float z0 = DCache[topMostIndex+ 2];
    float z1 = DCache[topMostIndex+18];
    float z2 = DCache[topMostIndex+34];
    float z3 = DCache[topMostIndex+50];
    float z4 = DCache[topMostIndex+66];
    float z5 = DCache[topMostIndex+82];

    AOCache1[topMostIndex   ] = SmartBlur( ao0, ao1, ao2, ao3, ao4, n0, n1, n2, n3, n4, z0, z1, z2, z3, z4 );
    AOCache1[topMostIndex+16] = SmartBlur( ao1, ao2, ao3, ao4, ao5, n1, n2, n3, n4, n5, z1, z2, z3, z4, z5 );
}

[numthreads( 8, 8, 1 )]
void main( uint GI : SV_GroupIndex, uint2 GTid : SV_GroupThreadID, uint2 DTid : SV_DispatchThreadID )
{
    //
    // Load 4 pixels per thread into LDS to fill the 16x16 LDS cache with depth and AO
    //
    PrefetchData( GTid.x << 1 | GTid.y << 5, int2(DTid + GTid) - 1 );
    GroupMemoryBarrierWithGroupSync();

    // Goal:  End up with a 9x9 patch that is blurred.

    //
    // Horizontally blur the pixels.	13x13 -> 9x13
    //
    if (GI < 39)
        BlurHorizontally((GI / 3) * 16 + (GI % 3) * 3);
    GroupMemoryBarrierWithGroupSync();

    //
    // Vertically blur the pixels.		9x13 -> 9x9
    //
    if (GI < 45)
        BlurVertically((GI / 9) * 32 + GI % 9);
    GroupMemoryBarrierWithGroupSync();

    AoResult[DTid] = AOCache1[GTid.x + GTid.y * 16];
}
