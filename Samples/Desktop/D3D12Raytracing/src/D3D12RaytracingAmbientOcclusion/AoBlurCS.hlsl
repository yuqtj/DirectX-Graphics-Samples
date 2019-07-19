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

// Full kernel width is BLUR_RADIUS*2+1
// Width 5 -> 4x4 flat weighted aggregation
// Width 7 or 9 -> Gaussian approximation
#ifndef BLUR_RADIUS
#  define BLUR_RADIUS 2
#endif

cbuffer g_aoBlurCB : register(b0)
{
    float2 kRcpBufferDim;
    float kDistanceTolerance;
}

groupshared float4 NCache[256]; // Normal+Obliqueness
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

    float4 Distances = Distance.Gather( LinearSampler, uv );
    DCache[index   ] = Distances.w;
    DCache[index+ 1] = Distances.z;
    DCache[index+16] = Distances.x;
    DCache[index+17] = Distances.y;
}

float SampleInfluence(float3 N1, float3 N2, float DPTol, float deltaZ, float ZTol)
{
    return step(DPTol, dot(N1, N2)) * smoothstep(ZTol, 0.0, deltaZ);
}

#if BLUR_RADIUS == 2
float SmartBlur(
    float aoA, float aoB, float aoC, float aoD, float aoE,
    float4 nA, float4 nB, float4 nC, float4 nD, float4 nE,
    float zA, float zB, float zC, float zD, float zE)
{
    float ZTol = kDistanceTolerance * nC.w * zC;
    float wA = 0.5 * SampleInfluence(nA.xyz, nC.xyz, 0.6, 0.5*abs(zA-zC), ZTol);
    float wB = 1.0 * SampleInfluence(nB.xyz, nC.xyz, 0.8, 1.0*abs(zB-zC), ZTol);
    float wD = 1.0 * SampleInfluence(nD.xyz, nC.xyz, 0.8, 1.0*abs(zD-zC), ZTol);
    float wE = 0.5 * SampleInfluence(nE.xyz, nC.xyz, 0.6, 0.5*abs(zE-zC), ZTol);

    return (wA*aoA + wB*aoB + aoC + wD*aoD + wE*aoE) / (wA + wB + 1.0 + wD + wE);
}
#elif BLUR_RADIUS == 3
float SmartBlur(
    float aoA, float aoB, float aoC, float aoD, float aoE, float aoF, float aoG,
    float4 nA, float4 nB, float4 nC, float4 nD, float4 nE, float4 nF, float4 nG,
    float zA, float zB, float zC, float zD, float zE, float zF, float zG)
{
    float ZTol = kDistanceTolerance * nD.w * zD;

    float wA = 0.05 * SampleInfluence(nA.xyz, nD.xyz, 0.4, abs(zA-zD)/3.0, ZTol);
    float wB = 0.30 * SampleInfluence(nB.xyz, nD.xyz, 0.6, abs(zB-zD)/2.0, ZTol);
    float wC = 0.75 * SampleInfluence(nC.xyz, nD.xyz, 0.8, abs(zC-zD)/1.0, ZTol);

    float wE = 0.75 * SampleInfluence(nE.xyz, nD.xyz, 0.8, abs(zE-zD)/1.0, ZTol);
    float wF = 0.30 * SampleInfluence(nF.xyz, nD.xyz, 0.6, abs(zF-zD)/2.0, ZTol);
    float wG = 0.05 * SampleInfluence(nG.xyz, nD.xyz, 0.4, abs(zG-zD)/3.0, ZTol);

    return (wA*aoA + wB*aoB + wC*aoC + aoD + wE*aoE + wF*aoF + wG*aoG) / (wA + wB + wC + 1.0 + wE + wF + wG);
}
#else // BLUR_RADIUS == 4
float SmartBlur(
    float aoA, float aoB, float aoC, float aoD, float aoE, float aoF, float aoG, float aoH, float aoI,
    float4 nA, float4 nB, float4 nC, float4 nD, float4 nE, float4 nF, float4 nG, float4 nH, float4 nI,
    float zA, float zB, float zC, float zD, float zE, float zF, float zG, float zH, float zI)
{
    float ZTol = kDistanceTolerance * nD.w * zD;

    float wA = SampleInfluence(nA.xyz, nE.xyz, 0.2, abs(zA-zE)/4.0, ZTol) *  1.0 / 70.0;
    float wB = SampleInfluence(nB.xyz, nE.xyz, 0.4, abs(zB-zE)/3.0, ZTol) *  8.0 / 70.0;
    float wC = SampleInfluence(nC.xyz, nE.xyz, 0.6, abs(zC-zE)/2.0, ZTol) * 28.0 / 70.0;
    float wD = SampleInfluence(nD.xyz, nE.xyz, 0.8, abs(zD-zE)/1.0, ZTol) * 56.0 / 70.0;

    float wF = SampleInfluence(nF.xyz, nE.xyz, 0.8, abs(zF-zE)/1.0, ZTol) * 56.0 / 70.0;
    float wG = SampleInfluence(nG.xyz, nE.xyz, 0.6, abs(zG-zE)/2.0, ZTol) * 28.0 / 70.0;
    float wH = SampleInfluence(nH.xyz, nE.xyz, 0.4, abs(zH-zE)/3.0, ZTol) *  8.0 / 70.0;
    float wI = SampleInfluence(nI.xyz, nE.xyz, 0.2, abs(zI-zE)/4.0, ZTol) *  1.0 / 70.0;

    return (wA*aoA + wB*aoB + wC*aoC + wD*aoD + aoE + wF*aoF + wG*aoG + wH*aoH + wI*aoI) / (wA + wB + wC + wD + 1.0 + wF + wG + wH + wI);
}
#endif

void BlurHorizontally( uint leftMostIndex )
{
    float ao0 = AOCache1[leftMostIndex  ];
    float ao1 = AOCache1[leftMostIndex+1];
    float ao2 = AOCache1[leftMostIndex+2];
    float ao3 = AOCache1[leftMostIndex+3];
    float ao4 = AOCache1[leftMostIndex+4];
    float ao5 = AOCache1[leftMostIndex+5];
    float ao6 = AOCache1[leftMostIndex+6];
    float ao7 = AOCache1[leftMostIndex+7];
    float ao8 = AOCache1[leftMostIndex+8];
    float ao9 = AOCache1[leftMostIndex+9];

    float4 n0 = NCache[leftMostIndex  ];
    float4 n1 = NCache[leftMostIndex+1];
    float4 n2 = NCache[leftMostIndex+2];
    float4 n3 = NCache[leftMostIndex+3];
    float4 n4 = NCache[leftMostIndex+4];
    float4 n5 = NCache[leftMostIndex+5];
    float4 n6 = NCache[leftMostIndex+6];
    float4 n7 = NCache[leftMostIndex+7];
    float4 n8 = NCache[leftMostIndex+8];
    float4 n9 = NCache[leftMostIndex+9];

    float z0 = DCache[leftMostIndex  ];
    float z1 = DCache[leftMostIndex+1];
    float z2 = DCache[leftMostIndex+2];
    float z3 = DCache[leftMostIndex+3];
    float z4 = DCache[leftMostIndex+4];
    float z5 = DCache[leftMostIndex+5];
    float z6 = DCache[leftMostIndex+6];
    float z7 = DCache[leftMostIndex+7];
    float z8 = DCache[leftMostIndex+8];
    float z9 = DCache[leftMostIndex+9];

#if BLUR_RADIUS == 2
    AOCache2[leftMostIndex  ] = SmartBlur( ao0, ao1, ao2, ao3, ao4, n0, n1, n2, n3, n4, z0, z1, z2, z3, z4 );
    AOCache2[leftMostIndex+1] = SmartBlur( ao1, ao2, ao3, ao4, ao5, n1, n2, n3, n4, n5, z1, z2, z3, z4, z5 );
#elif BLUR_RADIUS == 3
    AOCache2[leftMostIndex  ] = SmartBlur( ao0, ao1, ao2, ao3, ao4, ao5, ao6, n0, n1, n2, n3, n4, n5, n6, z0, z1, z2, z3, z4, z5, z6 );
    AOCache2[leftMostIndex+1] = SmartBlur( ao1, ao2, ao3, ao4, ao5, ao6, ao7, n1, n2, n3, n4, n5, n6, n7, z1, z2, z3, z4, z5, z6, z7 );
#else // BLUR_RADIUS == 4
    AOCache2[leftMostIndex  ] = SmartBlur( ao0, ao1, ao2, ao3, ao4, ao5, ao6, ao7, ao8, n0, n1, n2, n3, n4, n5, n6, n7, n8, z0, z1, z2, z3, z4, z5, z6, z7, z8 );
    AOCache2[leftMostIndex+1] = SmartBlur( ao1, ao2, ao3, ao4, ao5, ao6, ao7, ao8, ao9, n1, n2, n3, n4, n5, n6, n7, n8, n9, z1, z2, z3, z4, z5, z6, z7, z8, z9 );
#endif
}

float BlurVertically( uint topMostIndex )
{
    float ao0 = AOCache2[topMostIndex     ];
    float ao1 = AOCache2[topMostIndex + 16];
    float ao2 = AOCache2[topMostIndex + 32];
    float ao3 = AOCache2[topMostIndex + 48];
    float ao4 = AOCache2[topMostIndex + 64];
    float ao5 = AOCache2[topMostIndex + 80];
    float ao6 = AOCache2[topMostIndex + 96];
    float ao7 = AOCache2[topMostIndex + 112];
    float ao8 = AOCache2[topMostIndex + 128];

    float4 n0 = NCache[topMostIndex + BLUR_RADIUS     ];
    float4 n1 = NCache[topMostIndex + BLUR_RADIUS + 16];
    float4 n2 = NCache[topMostIndex + BLUR_RADIUS + 32];
    float4 n3 = NCache[topMostIndex + BLUR_RADIUS + 48];
    float4 n4 = NCache[topMostIndex + BLUR_RADIUS + 64];
    float4 n5 = NCache[topMostIndex + BLUR_RADIUS + 80];
    float4 n6 = NCache[topMostIndex + BLUR_RADIUS + 96];
    float4 n7 = NCache[topMostIndex + BLUR_RADIUS + 112];
    float4 n8 = NCache[topMostIndex + BLUR_RADIUS + 128];

    float z0 = DCache[topMostIndex + BLUR_RADIUS     ];
    float z1 = DCache[topMostIndex + BLUR_RADIUS + 16];
    float z2 = DCache[topMostIndex + BLUR_RADIUS + 32];
    float z3 = DCache[topMostIndex + BLUR_RADIUS + 48];
    float z4 = DCache[topMostIndex + BLUR_RADIUS + 64];
    float z5 = DCache[topMostIndex + BLUR_RADIUS + 80];
    float z6 = DCache[topMostIndex + BLUR_RADIUS + 96];
    float z7 = DCache[topMostIndex + BLUR_RADIUS + 112];
    float z8 = DCache[topMostIndex + BLUR_RADIUS + 128];

#if BLUR_RADIUS == 2
    return SmartBlur( ao0, ao1, ao2, ao3, ao4, n0, n1, n2, n3, n4, z0, z1, z2, z3, z4 );
#elif BLUR_RADIUS == 3
    return SmartBlur( ao0, ao1, ao2, ao3, ao4, ao5, ao6, n0, n1, n2, n3, n4, n5, n6, z0, z1, z2, z3, z4, z5, z6 );
#else // BLUR_RADIUS == 4
    return SmartBlur( ao0, ao1, ao2, ao3, ao4, ao5, ao6, ao7, ao8, n0, n1, n2, n3, n4, n5, n6, n7, n8, z0, z1, z2, z3, z4, z5, z6, z7, z8 );
#endif
}

[numthreads( 8, 8, 1 )]
void main( uint GI : SV_GroupIndex, uint2 GTid : SV_GroupThreadID, uint2 DTid : SV_DispatchThreadID )
{
    //
    // Load 4 pixels per thread into LDS to fill the 16x16 LDS cache with depth and AO
    //
    PrefetchData( GTid.x << 1 | GTid.y << 5, int2(DTid + GTid) - 3 );
    GroupMemoryBarrierWithGroupSync();

    // Goal:  End up with a 9x9 patch that is blurred.

    //
    // Horizontally blur the pixels.	16x16 -> 8x16
    //

    BlurHorizontally((GI / 4) * 16 + (GI % 4) * 2 + (4 - BLUR_RADIUS));
    GroupMemoryBarrierWithGroupSync();

    //
    // Vertically blur the pixels.		8x16 -> 8x8
    //
    AoResult[DTid] = BlurVertically(GTid.y * 16 + GTid.x + 17 * (4 - BLUR_RADIUS));
}
