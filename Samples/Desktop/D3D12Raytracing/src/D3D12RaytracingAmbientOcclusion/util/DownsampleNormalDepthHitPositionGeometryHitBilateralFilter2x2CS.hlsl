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
Texture2D<float4> g_inReprojectedHitPosition : register(t7);
RWTexture2D<float> g_texOutput : register(u0);
RWTexture2D<float4> g_outNormal : register(u1);
RWTexture2D<float4> g_outHitPosition : register(u2);
RWTexture2D<uint> g_outGeometryHit : register(u3);   // ToDo rename hits to Geometryits everywhere
RWTexture2D<float2> g_outPartialDistanceDerivatives : register(u4);   // ToDo rename hits to Geometryits everywhere
RWTexture2D<float> g_outDepth : register(u5);
RWTexture2D<float2> g_outMotionVector : register(u6);
RWTexture2D<float4> g_outReprojectedHitPosition : register(u7);
RWTexture2D<float4> g_outNormalLowPrecisionNormal : register(u8); // ToDo cleanup - pass two normal inputs instead?

// ToDo remove duplicate downsampling with the other ValudeDepthNormal

// ToDo rename to DownsampleGBuffer

void LoadDepthAndEncodedNormal(in uint2 texIndex, out float4 normal, out float depth)
{
    // ToDo this is confusing as it doesn't decode the normal
    normal = g_inNormal[texIndex];
    depth = normal.z;
}

//ToDo
// ToDo dedupe - already in DownsampleBilateralFilterCS.hlsli
// ToDo strip Bilateral from the name?
// Returns a selected depth index when bilateral downsapling.
void GetDepthIndexFromDownsampleDepthBilateral2x2(out UINT outDepthIndex, in float depths[4], in uint2 DTid)
{
    // Choose a alternate min max depth sample in a checkerboard 2x2 pattern to improve depth correlations for bilateral 2x2 upsampling.
    // Ref: http://c0de517e.blogspot.com/2016/02/downsampled-effects-with-depth-aware.html
    bool checkerboardTakeMin = ((DTid.x + DTid.y) & 1) == 0;

    float lowResDepth = checkerboardTakeMin
        ? min(min(min(depths[0], depths[1]), depths[2]), depths[3])
        : max(max(max(depths[0], depths[1]), depths[2]), depths[3]);

    // Find the corresponding sample index to the the selected sample depth.
    float4 vDepths = float4(depths[0], depths[1], depths[2], depths[3]);
    float4 depthDelta = abs(lowResDepth - vDepths);

    outDepthIndex = depthDelta[0] < depthDelta[1] ? 0 : 1;
    outDepthIndex = depthDelta[2] < depthDelta[outDepthIndex] ? 2 : outDepthIndex;
    outDepthIndex = depthDelta[3] < depthDelta[outDepthIndex] ? 3 : outDepthIndex;
}

[numthreads(DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::ThreadGroup::Width, DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint2 topLeftSrcIndex = DTid << 1;

    float4 encodedNormals[4];
    float  depths[4];

    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    LoadDepthAndEncodedNormal(topLeftSrcIndex, encodedNormals[0], depths[0]);
    LoadDepthAndEncodedNormal(topLeftSrcIndex + srcIndexOffsets[1], encodedNormals[1], depths[1]);
    LoadDepthAndEncodedNormal(topLeftSrcIndex + srcIndexOffsets[2], encodedNormals[2], depths[2]);
    LoadDepthAndEncodedNormal(topLeftSrcIndex + srcIndexOffsets[3], encodedNormals[3], depths[3]);

    UINT outDepthIndex;
    GetDepthIndexFromDownsampleDepthBilateral2x2(outDepthIndex, depths, DTid);

    g_outNormalLowPrecisionNormal[DTid] = g_outNormal[DTid] = encodedNormals[outDepthIndex];
    g_outHitPosition[DTid] = g_inHitPosition[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
    g_outGeometryHit[DTid] = g_inGeometryHit[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
    g_outDepth[DTid] = g_inDepth[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];

    // Since we're reducing the resolution by 2, multiple the partial derivatives by 2. Either that or the multiplier should be applied when calculating weights.
    // ToDo it would be cleaner to apply that multiplier at weights calculation. Or recompute the partial derivatives on downsample?
    g_outPartialDistanceDerivatives[DTid] = 2 * g_inPartialDistanceDerivatives[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];

    g_outMotionVector[DTid] = g_inMotionVector[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
    g_outReprojectedHitPosition[DTid] = g_inReprojectedHitPosition[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
}