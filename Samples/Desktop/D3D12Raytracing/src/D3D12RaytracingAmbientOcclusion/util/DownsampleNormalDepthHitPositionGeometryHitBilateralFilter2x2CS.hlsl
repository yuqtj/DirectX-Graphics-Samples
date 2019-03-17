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
#include "DownsampleBilateralFilterCS.hlsli"

// ToDo strip _tex from names
// ToDo remove unused input/ouput
Texture2D<float> g_texInput : register(t0);
Texture2D<float4> g_inNormal : register(t1);        // ToDo rename to normal and depth everywhere
Texture2D<float4> g_inHitPosition : register(t2);
Texture2D<uint> g_inGeometryHit : register(t3);
RWTexture2D<float> g_texOutput : register(u0);
RWTexture2D<float4> g_outNormal : register(u1);
RWTexture2D<float4> g_outHitPosition : register(u2);
RWTexture2D<uint> g_outGeometryHit : register(u3);   // ToDo rename hits to Geometryits everywhere

// ToDo rename to DownsampleGBuffer

void LoadDepthAndEncodedNormal(in uint2 texIndex, out float4 normal, out float depth)
{
    // ToDo this is confusing as it doesn't decode the normal
    normal = g_inNormal[texIndex];
    depth = normal.z;
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

#if 1
    // ToDo cleanup
    //float4 vNormals = float4(encodedNormals[0], encodedNormals[1], encodedNormals[2], encodedNormals[3]);
    //float4 vDepths = float4(depths[0], depths[1], depths[2], depths[3]);

    float outWeigths[4];
    UINT outDepthIndex;
    GetWeightsForDownsampleDepthBilateral2x2(outWeigths, outDepthIndex, depths);

    g_outNormal[DTid] = encodedNormals[outDepthIndex];
    g_outHitPosition[DTid] = g_inHitPosition[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
    g_outGeometryHit[DTid] = g_inGeometryHit[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];

#elif 1
    // ToDo optimize min/max and remove this path
    // ToDo min/max 1.2ms, just index 0 0.36ms for 4K
    int lowResSubIndex = 0;


    //uint2 lowResSrcIndex = topLeftSrcIndex + uint2(lowResSubIndex & 1, lowResSubIndex > 1 ? 1 : 0);

    g_outNormal[DTid] = encodedNormals[lowResSubIndex];
    g_outHitPosition[DTid] = g_inHitPosition[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];
    g_outGeometryHit[DTid] = g_inGeometryHit[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];

#else

    // Choose a sample to maximize depth correlation for bilateral upsampling, do checkerboard min/max selection.
    // Ref: http://c0de517e.blogspot.com/2016/02/downsampled-effects-with-depth-aware.html
    bool checkerboardMin = ((DTid.x + DTid.y) & 1) == 0;

    // ToDo consider encodedNormals but decode first.
    // ToDo consider reflections.
#if 1
    float lowResDepth = max(max(depths[0], depths[1]), max(depths[2], depths[3]));
#else
    float lowResDepth = checkerboardMin ? min(min(depths[0], depths[1]), min(depths[2], depths[3]))
                                        : max(max(depths[0], depths[1]), max(depths[2], depths[3]));
#endif
    // ToDo prefer those with geometry hit for AO?

    // ToDo optimize
    float sampleDistances[4] = {
        abs(lowResDepth - depths[0]),
        abs(lowResDepth - depths[1]),
        abs(lowResDepth - depths[2]),
        abs(lowResDepth - depths[3]) 
    };

#if 0
    int lowResSubIndex = 0;// lowResDepth == depths[0] ? 1 : 0;
#else
    int lowResSubIndex;
    if (sampleDistances[0] > sampleDistances[1])
    {
        if (sampleDistances[1] > sampleDistances[2])
        {
            if (sampleDistances[2] > sampleDistances[3])
            {
                lowResSubIndex = 3;
            }
            else
            {
                lowResSubIndex = 2;
            }
        }
        else
        {
            if (sampleDistances[1] > sampleDistances[3])
            {
                lowResSubIndex = 3;
            }
            else
            {
                lowResSubIndex = 1;
            }
        }
    }
    else
    {
        if (sampleDistances[0] > sampleDistances[2])
        {
            if (sampleDistances[2] > sampleDistances[3])
            {
                lowResSubIndex = 3;
            }
            else
            {
                lowResSubIndex = 2;
            }
        }
        else
        {
            if (sampleDistances[0] > sampleDistances[3])
            {
                lowResSubIndex = 3;
            }
            else
            {
                lowResSubIndex = 0;
            }
        }
    }
#endif
    //uint2 lowResSrcIndex = topLeftSrcIndex + uint2(lowResSubIndex & 1, lowResSubIndex > 1 ? 1 : 0);

    g_outNormal[DTid] = encodedNormals[lowResSubIndex];
    g_outHitPosition[DTid] = g_inHitPosition[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];
    g_outGeometryHit[DTid] = g_inGeometryHit[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];
#endif
}