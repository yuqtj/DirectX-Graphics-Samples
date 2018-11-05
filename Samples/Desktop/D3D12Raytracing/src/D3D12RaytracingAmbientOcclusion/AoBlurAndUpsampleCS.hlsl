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
    float kDistanceTolerance;
}

groupshared float4 NCache[256]; // Normal+Z
groupshared float DCache[256]; // Distance
groupshared float AOCache1[256];
groupshared float AOCache2[256];

void PrefetchData( uint index, int2 ST )
{
    float4 AO = InputAO.Gather( LinearSampler, ST * kRcpBufferDim );

    AOCache1[index   ] = AO.w;
    AOCache1[index+ 1] = AO.z;
    AOCache1[index+16] = AO.x;
    AOCache1[index+17] = AO.y;

    NCache[index   ] = Normal[2*ST + int2(-2, -2)];
    NCache[index+ 1] = Normal[2*ST + int2( 0, -2)];
    NCache[index+16] = Normal[2*ST + int2(-2,  0)];
    NCache[index+17] = Normal[2*ST + int2( 0,  0)];

    DCache[index   ] = Distance[2*ST + int2(-2, -2)];
    DCache[index+ 1] = Distance[2*ST + int2( 0, -2)];
    DCache[index+16] = Distance[2*ST + int2(-2,  0)];
    DCache[index+17] = Distance[2*ST + int2( 0,  0)];
}

float SampleInfluence(float3 N1, float3 N2, float DPTol, float deltaZ, float ZTol)
{
    return step(DPTol, dot(N1, N2)) * smoothstep(ZTol, 0.0, deltaZ);
}

float SmartBlur(
    float aoA, float aoB, float aoC, float aoD, float aoE,
    float4 nA, float4 nB, float4 nC, float4 nD, float4 nE,
    float zA, float zB, float zC, float zD, float zE)
{
    float ZTol = kDistanceTolerance * nC.w * zC;
    float wA = 0.5 * SampleInfluence(nA.xyz, nC.xyz, 0.6, 0.25*abs(zA-zC), ZTol);
    float wB = 1.0 * SampleInfluence(nB.xyz, nC.xyz, 0.8, 0.50*abs(zB-zC), ZTol);
    float wD = 1.0 * SampleInfluence(nD.xyz, nC.xyz, 0.8, 0.50*abs(zD-zC), ZTol);
    float wE = 0.5 * SampleInfluence(nE.xyz, nC.xyz, 0.6, 0.25*abs(zE-zC), ZTol);

    return (wA*aoA + wB*aoB + aoC + wD*aoD + wE*aoE) / (wA + wB + 1.0 + wD + wE);
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

// We essentially want 5 weights:  4 for each low-res pixel and 1 to blend in when none of the 4 really
// match.  The filter strength is 1 / DeltaZTolerance.  So a tolerance of 0.01 would yield a strength of 100.
// Note that a perfect match of low to high depths would yield a weight of 10^6, completely superceding any
// noise filtering.  The noise filter is intended to soften the effects of shimmering when the high-res depth
// buffer has a lot of small holes in it causing the low-res depth buffer to inaccurately represent it.
float BilateralUpsample( float ActualDistance, float4 SampleDistances, float4 SampleAOs )
{
    float4 weights = float4(9, 3, 1, 3) / (abs(ActualDistance - SampleDistances) + 1e-6 * ActualDistance);
    return dot(weights, SampleAOs) / dot(weights, 1);
}

[numthreads( 8, 8, 1 )]
void main( uint GI : SV_GroupIndex, uint2 GTid : SV_GroupThreadID, uint2 DTid : SV_DispatchThreadID )
{
    //
    // Load 4 pixels per thread into LDS to fill the 16x16 LDS cache with depth and AO
    //
    PrefetchData( GTid.x << 1 | GTid.y << 5, int2(DTid + GTid - 2) );
    GroupMemoryBarrierWithGroupSync();

    // Goal:  End up with a 9x9 patch that is blurred so we can upsample.  Blur radius is 2 pixels, so start with 13x13 area.

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

    //
    // Bilateral upsample
    //
    uint Idx0 = GTid.x + GTid.y * 16;
    float4 AOs = float4(AOCache1[Idx0+16], AOCache1[Idx0+17], AOCache1[Idx0+1], AOCache1[Idx0]);

    uint Idx1 = Idx0 + 17 * 2;
    float4 SampleDistances = float4(DCache[Idx1+16], DCache[Idx1+17], DCache[Idx1+1], DCache[Idx1]);

    int2 OutST = DTid * 2;
    float4 TargetDistances = Distance.Gather(LinearSampler, OutST * kRcpBufferDim);

    AoResult[OutST + int2(-1,  0)] = BilateralUpsample(TargetDistances.x, SampleDistances.xyzw, AOs.xyzw);
    AoResult[OutST + int2( 0,  0)] = BilateralUpsample(TargetDistances.y, SampleDistances.yzwx, AOs.yzwx);
    AoResult[OutST + int2( 0, -1)] = BilateralUpsample(TargetDistances.z, SampleDistances.zwxy, AOs.zwxy);
    AoResult[OutST + int2(-1, -1)] = BilateralUpsample(TargetDistances.w, SampleDistances.wxyz, AOs.wxyz);
}
