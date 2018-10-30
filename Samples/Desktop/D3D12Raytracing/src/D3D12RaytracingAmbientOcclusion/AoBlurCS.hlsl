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

#ifndef AOBLURCS_HLSL
#define AOBLURCS_HLSL

#define HLSL
#include "RaytracingHlslCompat.h"

RWTexture2D<float> AoResult : register(u0);
Texture2D<float> InputDepth : register(t0);
Texture2D<float> InputAO : register(t1);
SamplerState LinearSampler : register(s0);
ConstantBuffer<AoBlurConstantBuffer> g_aoBlurCB : register(b0);

groupshared float DepthCache[256];
groupshared float AOCache1[256];
groupshared float AOCache2[256];

void PrefetchData( uint index, float2 uv )
{
    float4 AO1 = InputAO.Gather( LinearSampler, uv );

    AOCache1[index   ] = AO1.w;
    AOCache1[index+ 1] = AO1.z;
    AOCache1[index+16] = AO1.x;
    AOCache1[index+17] = AO1.y;

    float4 ID = 1.0 / InputDepth.Gather( LinearSampler, uv );
    DepthCache[index   ] = ID.w;
    DepthCache[index+ 1] = ID.z;
    DepthCache[index+16] = ID.x;
    DepthCache[index+17] = ID.y;
}

float SmartBlur( float a, float b, float c, float d, float e, bool Left, bool Middle, bool Right )
{
    b = Left | Middle ? b : c;
    a = Left ? a : b;
    d = Right | Middle ? d : c;
    e = Right ? e : d;
    //return ((a + e) / 2.0 + b + c + d) / 4.0;
    return (a + e + (b + d) * 4.0 + c * 6.0) / 16.0;
}

bool CompareDeltas( float d1, float d2, float l1, float l2 )
{
    float temp = d1 * d2 + g_aoBlurCB.kStepSize;
    return temp * temp > l1 * l2 * g_aoBlurCB.kBlurTolerance;
}

void BlurHorizontally( uint leftMostIndex )
{
    float a0 = AOCache1[leftMostIndex  ];
    float a1 = AOCache1[leftMostIndex+1];
    float a2 = AOCache1[leftMostIndex+2];
    float a3 = AOCache1[leftMostIndex+3];
    float a4 = AOCache1[leftMostIndex+4];
    float a5 = AOCache1[leftMostIndex+5];
    float a6 = AOCache1[leftMostIndex+6];

    float d0 = DepthCache[leftMostIndex  ];
    float d1 = DepthCache[leftMostIndex+1];
    float d2 = DepthCache[leftMostIndex+2];
    float d3 = DepthCache[leftMostIndex+3];
    float d4 = DepthCache[leftMostIndex+4];
    float d5 = DepthCache[leftMostIndex+5];
    float d6 = DepthCache[leftMostIndex+6];

    float d01 = d1 - d0;
    float d12 = d2 - d1;
    float d23 = d3 - d2;
    float d34 = d4 - d3;
    float d45 = d5 - d4;
    float d56 = d6 - d5;

    float l01 = d01 * d01 + g_aoBlurCB.kStepSize;
    float l12 = d12 * d12 + g_aoBlurCB.kStepSize;
    float l23 = d23 * d23 + g_aoBlurCB.kStepSize;
    float l34 = d34 * d34 + g_aoBlurCB.kStepSize;
    float l45 = d45 * d45 + g_aoBlurCB.kStepSize;
    float l56 = d56 * d56 + g_aoBlurCB.kStepSize;

    bool c02 = CompareDeltas( d01, d12, l01, l12 );
    bool c13 = CompareDeltas( d12, d23, l12, l23 );
    bool c24 = CompareDeltas( d23, d34, l23, l34 );
    bool c35 = CompareDeltas( d34, d45, l34, l45 );
    bool c46 = CompareDeltas( d45, d56, l45, l56 );

    AOCache2[leftMostIndex  ] = SmartBlur( a0, a1, a2, a3, a4, c02, c13, c24 );
    AOCache2[leftMostIndex+1] = SmartBlur( a1, a2, a3, a4, a5, c13, c24, c35 );
    AOCache2[leftMostIndex+2] = SmartBlur( a2, a3, a4, a5, a6, c24, c35, c46 );
}

void BlurVertically( uint topMostIndex )
{
    float a0 = AOCache2[topMostIndex   ];
    float a1 = AOCache2[topMostIndex+16];
    float a2 = AOCache2[topMostIndex+32];
    float a3 = AOCache2[topMostIndex+48];
    float a4 = AOCache2[topMostIndex+64];
    float a5 = AOCache2[topMostIndex+80];

    float d0 = DepthCache[topMostIndex+ 2];
    float d1 = DepthCache[topMostIndex+18];
    float d2 = DepthCache[topMostIndex+34];
    float d3 = DepthCache[topMostIndex+50];
    float d4 = DepthCache[topMostIndex+66];
    float d5 = DepthCache[topMostIndex+82];

    float d01 = d1 - d0;
    float d12 = d2 - d1;
    float d23 = d3 - d2;
    float d34 = d4 - d3;
    float d45 = d5 - d4;

    float l01 = d01 * d01 + g_aoBlurCB.kStepSize;
    float l12 = d12 * d12 + g_aoBlurCB.kStepSize;
    float l23 = d23 * d23 + g_aoBlurCB.kStepSize;
    float l34 = d34 * d34 + g_aoBlurCB.kStepSize;
    float l45 = d45 * d45 + g_aoBlurCB.kStepSize;

    bool c02 = CompareDeltas( d01, d12, l01, l12 );
    bool c13 = CompareDeltas( d12, d23, l12, l23 );
    bool c24 = CompareDeltas( d23, d34, l23, l34 );
    bool c35 = CompareDeltas( d34, d45, l34, l45 );

    float aoResult1 = SmartBlur( a0, a1, a2, a3, a4, c02, c13, c24 );
    float aoResult2 = SmartBlur( a1, a2, a3, a4, a5, c13, c24, c35 );

    AOCache1[topMostIndex   ] = aoResult1;
    AOCache1[topMostIndex+16] = aoResult2;
}

[numthreads( 8, 8, 1 )]
void main( uint GI : SV_GroupIndex, uint2 GTid : SV_GroupThreadID, uint2 DTid : SV_DispatchThreadID )
{
    //
    // Load 4 pixels per thread into LDS to fill the 16x16 LDS cache with depth and AO
    //
    PrefetchData( GTid.x << 1 | GTid.y << 5, int2(DTid + GTid - 2) * g_aoBlurCB.kRcpBufferDim );
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

#endif // AOBLURCS_HLSL