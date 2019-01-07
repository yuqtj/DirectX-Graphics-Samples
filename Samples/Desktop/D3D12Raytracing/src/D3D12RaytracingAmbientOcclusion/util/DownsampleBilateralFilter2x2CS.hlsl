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

// ToDo strip _tex from names
// ToDo remove unused input/ouput
Texture2D<float> g_texInput : register(t0);
Texture2D<float4> g_inNormal : register(t1);
Texture2D<float4> g_inHitPosition : register(t2);
Texture2D<uint> g_inGeometryHit : register(t3);
RWTexture2D<float> g_texOutput : register(u0);
RWTexture2D<float4> g_outNormal : register(u1);
RWTexture2D<float4> g_outHitPosition : register(u2);
RWTexture2D<uint> g_outGeometryHit : register(u3);   // ToDo rename hits to Geometryits everywhere

void LoadDepthAndNormal(in uint2 texIndex, out float4 normal, out float depth)
{
    normal = g_inNormal[texIndex];
    depth = normal.z;
}

[numthreads(DownsampleBilateralFilter::ThreadGroup::Width, DownsampleBilateralFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint2 topLeftSrcIndex = DTid << 1;

    float4 normals[4];
    float  depths[4];

    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    LoadDepthAndNormal(topLeftSrcIndex, normals[0], depths[0]);
    LoadDepthAndNormal(topLeftSrcIndex + srcIndexOffsets[1], normals[1], depths[1]);
    LoadDepthAndNormal(topLeftSrcIndex + srcIndexOffsets[2], normals[2], depths[2]);
    LoadDepthAndNormal(topLeftSrcIndex + srcIndexOffsets[3], normals[3], depths[3]);

    // Choose a sample to maximize depth correlation for bilateral upsampling, do checkerboard min/max selection.
    // Ref: http://c0de517e.blogspot.com/2016/02/downsampled-effects-with-depth-aware.html
    bool checkerboardMin = ((DTid.x + DTid.y) & 1) == 0;

    // ToDo consider normals but decode first.
    // ToDo consider reflections.
    float lowResDepth = checkerboardMin ? min(min(depths[0], depths[1]), min(depths[2], depths[3])) 
                                        : max(max(depths[0], depths[1]), max(depths[2], depths[3]));
    
    // ToDo prefer those with geometry hit for AO?

    // ToDo optimize
    float sampleDistances[4] = {
        abs(lowResDepth - depths[0]),
        abs(lowResDepth - depths[1]),
        abs(lowResDepth - depths[2]),
        abs(lowResDepth - depths[3]) 
    };

    uint lowResSubIndex;
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

    uint2 lowResSrcIndex = topLeftSrcIndex + uint2(lowResSubIndex & 1, lowResSubIndex > 1 ? 1 : 0);

    g_outNormal[DTid] = normals[lowResSubIndex];
    g_outHitPosition[DTid] = g_inHitPosition[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];
    g_outGeometryHit[DTid] = g_inGeometryHit[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];
}