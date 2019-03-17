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

Texture2D<float> g_inValue : register(t0);
Texture2D<float4> g_inNormalAndDepth : register(t1);
RWTexture2D<float> g_outValue : register(u0);
RWTexture2D<float4> g_outNormalAndDepth : register(u1);

void LoadDepthAndEncodedNormal(in uint2 texIndex, out float4 encodedNormalAndDepth, out float depth)
{
    encodedNormalAndDepth = g_inNormalAndDepth[texIndex];
    depth = encodedNormalAndDepth.z;
}

// ToDo remove _DepthAware from the name?

[numthreads(DownsampleValueNormalDepthBilateralFilter::ThreadGroup::Width, DownsampleValueNormalDepthBilateralFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint2 topLeftSrcIndex = DTid << 1;

    float4 encodedNormalsAndDepths[4];
    float  depths[4];

    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    LoadDepthAndEncodedNormal(topLeftSrcIndex, encodedNormalsAndDepths[0], depths[0]);
    LoadDepthAndEncodedNormal(topLeftSrcIndex + srcIndexOffsets[1], encodedNormalsAndDepths[1], depths[1]);
    LoadDepthAndEncodedNormal(topLeftSrcIndex + srcIndexOffsets[2], encodedNormalsAndDepths[2], depths[2]);
    LoadDepthAndEncodedNormal(topLeftSrcIndex + srcIndexOffsets[3], encodedNormalsAndDepths[3], depths[3]);

    float outWeigths[4];
    UINT outDepthIndex;
    // ToDo make it depth aware instead of defaulting to index 0
    GetWeightsForDownsampleDepthBilateral2x2(outWeigths, outDepthIndex, depths);

    g_outNormalAndDepth[DTid] = encodedNormalsAndDepths[outDepthIndex];
    g_outValue[DTid] = g_inValue[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
}