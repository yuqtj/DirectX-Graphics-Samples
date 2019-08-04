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

#define HLSL
#include "..\RaytracingHlslCompat.h"
#include "..\RaytracingShaderHelper.hlsli"

// ToDo update the name or split file for different resources?
// ToDo strip _tex from names
// ToDo remove unused input/ouput
Texture2D<float> g_texInput : register(t0);
Texture2D<float4> g_inNormal : register(t1);        // ToDo rename to normal and depth everywhere
Texture2D<float4> g_inHitPosition : register(t2);
Texture2D<uint> g_inGeometryHit : register(t3);
Texture2D<float2> g_inPartialDistanceDerivatives : register(t4);  // update file name to include ddxy
Texture2D<float> g_inDepth : register(t5);
Texture2D<float2> g_inMotionVector : register(t6);
Texture2D<NormalDepthTexFormat> g_inReprojectedNormalDepth : register(t7);
RWTexture2D<float> g_texOutput : register(u0);
RWTexture2D<float4> g_outNormal : register(u1);
RWTexture2D<float4> g_outHitPosition : register(u2);
RWTexture2D<uint> g_outGeometryHit : register(u3);   // ToDo rename hits to Geometryits everywhere
RWTexture2D<float2> g_outPartialDistanceDerivatives : register(u4);   // ToDo rename hits to Geometryits everywhere
RWTexture2D<float> g_outDepth : register(u5);   // ToDo remove or we need hi bit depth?
RWTexture2D<float2> g_outMotionVector : register(u6);
RWTexture2D<NormalDepthTexFormat> g_outReprojectedNormalDepth : register(u7);

SamplerState ClampSampler : register(s0);

ConstantBuffer<TextureDimConstantBuffer> cb : register(b0);

// ToDo remove duplicate downsampling with the other ValudeDepthNormal

void LoadDepthAndEncodedNormal(in uint2 texIndex, out float4 encodedNormalDepth, out float depth)
{
    // ToDo this is confusing as it doesn't decode the normal
    encodedNormalDepth = g_inNormal[texIndex];
    depth = encodedNormalDepth.z;
}

// ToDo dedupe - already in DownsampleBilateralFilterCS.hlsli
// Returns a selected depth index when bilateral downsapling.
uint GetIndexFromDepthAwareBilateralDownsample2x2(in float4 vDepths, in uint2 DTid)
{
    // Choose a alternate min max depth sample in a checkerboard 2x2 pattern to improve depth correlations for bilateral 2x2 upsampling.
    // Ref: http://c0de517e.blogspot.com/2016/02/downsampled-effects-with-depth-aware.html
    bool checkerboardTakeMin = ((DTid.x + DTid.y) & 1) == 0;

    float lowResDepth = checkerboardTakeMin ? min4(vDepths) : max4(vDepths);

    // Find the corresponding sample index to the the selected sample depth.
    return GetIndexOfValueClosestToTheReference(lowResDepth, vDepths);
}

// ToDo split into multiple per texture downsample (and use a mask input?)
[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint2 topLeftSrcIndex = DTid << 1;
    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    
    float2 centerTexCoord = (topLeftSrcIndex + 0.5) * cb.invTextureDim;
    float4 vDepths = g_inDepth.Gather(ClampSampler, centerTexCoord).wzxy;

    uint selectedOffset = GetIndexFromDepthAwareBilateralDownsample2x2(vDepths, DTid);
    uint2 selectedDTid = topLeftSrcIndex + srcIndexOffsets[selectedOffset];

    g_outDepth[DTid] = vDepths[selectedOffset];
    g_outNormal[DTid] = g_inNormal[selectedDTid];


    // Since we're reducing the resolution by 2, multiple the partial derivatives by 2. Either that or the multiplier should be applied when calculating weights.
    // ToDo it would be cleaner to apply that multiplier at weights calculation. Or recompute the partial derivatives on downsample?
    // ToDo use perspective correct
    g_outPartialDistanceDerivatives[DTid] = 2 * g_inPartialDistanceDerivatives[selectedDTid];

    g_outMotionVector[DTid] = g_inMotionVector[selectedDTid];
    g_outReprojectedNormalDepth[DTid] = g_inReprojectedNormalDepth[selectedDTid];
    g_outHitPosition[DTid] = g_inHitPosition[selectedDTid];
    g_outGeometryHit[DTid] = g_inGeometryHit[selectedDTid];
}