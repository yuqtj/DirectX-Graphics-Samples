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


// ToDo strip Bilateral from the name?
// Computes downsampling weights for 2x2 depth input
void GetWeightsForDownsampleDepthBilateral2x2(out float outWeights[4], out UINT outDepthIndex, in float depths[4])
{
#if 1
    // ToDo optimize min/max and remove this path
    // ToDo min/max 1.2ms, just index 0 0.36ms for 4K
    int lowResSubIndex = 0;
    outWeights[0] = 1;
    outWeights[1] = outWeights[2] = outWeights[3] = 0;

    // ToDo comment on not good idea to blend depths. select one.
    outDepthIndex = 0;
#else

    // Choose a sample to maximize depth correlation for bilateral upsampling, do checkerboard min/max selection.
    // Ref: http://c0de517e.blogspot.com/2016/02/downsampled-effects-with-depth-aware.html
    bool checkerboardMin = ((DTid.x + DTid.y) & 1) == 0;

    // ToDo consider normals but decode first.
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

    g_outNormal[DTid] = normals[lowResSubIndex];
    g_outHitPosition[DTid] = g_inHitPosition[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];
    g_outGeometryHit[DTid] = g_inGeometryHit[topLeftSrcIndex + srcIndexOffsets[lowResSubIndex]];
#endif
}

// ToDo move to common helper
float Blend(float weights[4], float values[4])
{
    return  weights[0] * values[0]
        +   weights[1] * values[1]
        +   weights[2] * values[2]
        +   weights[3] * values[3];
}