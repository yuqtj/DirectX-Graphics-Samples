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
Texture2D<float> InputDepth : register(t0);
Texture2D<float> InputAO : register(t1);
SamplerState LinearSampler : register(s0);

cbuffer g_aoBlurCB : register(b0)
{
    float2 kRcpBufferDim;
	float kStepSize;
	float kBlurTolerance;
    float kUpsampleTolerance;
}

groupshared float DCache[256];
groupshared float ZCache[256];
groupshared float AOCache1[256];
groupshared float AOCache2[256];

void PrefetchData( uint index, int2 ST )
{
    float4 AO = InputAO.Gather( LinearSampler, ST * kRcpBufferDim );

    AOCache1[index   ] = AO.w;
    AOCache1[index+ 1] = AO.z;
    AOCache1[index+16] = AO.x;
    AOCache1[index+17] = AO.y;

    DCache[index   ] = InputDepth[2*ST + int2(-2, -2)];
    DCache[index+ 1] = InputDepth[2*ST + int2( 0, -2)];
    DCache[index+16] = InputDepth[2*ST + int2(-2,  0)];
    DCache[index+17] = InputDepth[2*ST + int2( 0,  0)];

    ZCache[index   ] = 1.0 / DCache[index   ];
    ZCache[index+ 1] = 1.0 / DCache[index+ 1];
    ZCache[index+16] = 1.0 / DCache[index+16];
    ZCache[index+17] = 1.0 / DCache[index+17];
}

float SmartBlur( float a, float b, float c, float d, float e, bool Left, bool Middle, bool Right )
{
    b = Left | Middle ? b : c;
    a = Left ? a : b;
    d = Right | Middle ? d : c;
    e = Right ? e : d;
    return ((a + e) / 2.0 + b + c + d) / 4.0;
}

bool CompareDeltas( float d1, float d2, float l1, float l2 )
{
    float temp = d1 * d2 + kStepSize;
    return temp * temp > l1 * l2 * kBlurTolerance;
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

    float d0 = ZCache[leftMostIndex  ];
    float d1 = ZCache[leftMostIndex+1];
    float d2 = ZCache[leftMostIndex+2];
    float d3 = ZCache[leftMostIndex+3];
    float d4 = ZCache[leftMostIndex+4];
    float d5 = ZCache[leftMostIndex+5];
    float d6 = ZCache[leftMostIndex+6];

    float d01 = d1 - d0;
    float d12 = d2 - d1;
    float d23 = d3 - d2;
    float d34 = d4 - d3;
    float d45 = d5 - d4;
    float d56 = d6 - d5;

    float l01 = d01 * d01 + kStepSize;
    float l12 = d12 * d12 + kStepSize;
    float l23 = d23 * d23 + kStepSize;
    float l34 = d34 * d34 + kStepSize;
    float l45 = d45 * d45 + kStepSize;
    float l56 = d56 * d56 + kStepSize;

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

    float d0 = ZCache[topMostIndex+ 2];
    float d1 = ZCache[topMostIndex+18];
    float d2 = ZCache[topMostIndex+34];
    float d3 = ZCache[topMostIndex+50];
    float d4 = ZCache[topMostIndex+66];
    float d5 = ZCache[topMostIndex+82];

    float d01 = d1 - d0;
    float d12 = d2 - d1;
    float d23 = d3 - d2;
    float d34 = d4 - d3;
    float d45 = d5 - d4;

    float l01 = d01 * d01 + kStepSize;
    float l12 = d12 * d12 + kStepSize;
    float l23 = d23 * d23 + kStepSize;
    float l34 = d34 * d34 + kStepSize;
    float l45 = d45 * d45 + kStepSize;

    bool c02 = CompareDeltas( d01, d12, l01, l12 );
    bool c13 = CompareDeltas( d12, d23, l12, l23 );
    bool c24 = CompareDeltas( d23, d34, l23, l34 );
    bool c35 = CompareDeltas( d34, d45, l34, l45 );

    float aoResult1 = SmartBlur( a0, a1, a2, a3, a4, c02, c13, c24 );
    float aoResult2 = SmartBlur( a1, a2, a3, a4, a5, c13, c24, c35 );

    AOCache1[topMostIndex   ] = aoResult1;
    AOCache1[topMostIndex+16] = aoResult2;
}

// We essentially want 5 weights:  4 for each low-res pixel and 1 to blend in when none of the 4 really
// match.  The filter strength is 1 / DeltaZTolerance.  So a tolerance of 0.01 would yield a strength of 100.
// Note that a perfect match of low to high depths would yield a weight of 10^6, completely superceding any
// noise filtering.  The noise filter is intended to soften the effects of shimmering when the high-res depth
// buffer has a lot of small holes in it causing the low-res depth buffer to inaccurately represent it.
float BilateralUpsample( float ActualDistance, float4 SampleDistances, float4 SampleAOs )
{
    float4 weights = float4(9, 3, 1, 3) / (abs(ActualDistance - SampleDistances) + kUpsampleTolerance);
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

    uint Idx1 = Idx0 + 32 + 2;
    float4 SampleDistances = float4(DCache[Idx1+16], DCache[Idx1+17], DCache[Idx1+1], DCache[Idx1]);

    int2 OutST = DTid * 2;
    float4 TargetDistances = InputDepth.Gather(LinearSampler, OutST * kRcpBufferDim);

    AoResult[OutST + int2(-1,  0)] = BilateralUpsample(TargetDistances.x, SampleDistances.xyzw, AOs.xyzw);
    AoResult[OutST + int2( 0,  0)] = BilateralUpsample(TargetDistances.y, SampleDistances.yzwx, AOs.yzwx);
    AoResult[OutST + int2( 0, -1)] = BilateralUpsample(TargetDistances.z, SampleDistances.zwxy, AOs.zwxy);
    AoResult[OutST + int2(-1, -1)] = BilateralUpsample(TargetDistances.w, SampleDistances.wxyz, AOs.wxyz);
}
