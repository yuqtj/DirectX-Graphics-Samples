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

Texture2D<float> g_inValue : register(t0);
RWTexture2D<float2> g_outValue : register(u0);

ConstantBuffer<CalculatePartialDerivativesConstantBuffer> g_CB : register(b0);

[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    //                x
    //        ----------------->
    //    |    x     [top]     x
    // y  |  [left]   DTiD   [right]
    //    v    x    [bottom]   x
    //
    uint2 top = clamp(DTid.xy + uint2(0, -1), 0, g_CB.textureDim - 1);
    uint2 bottom = clamp(DTid.xy + uint2(0, 1), 0, g_CB.textureDim - 1);
    uint2 left = clamp(DTid.xy + uint2(-1, 0), 0, g_CB.textureDim - 1);
    uint2 right = clamp(DTid.xy + uint2(1, 0), 0, g_CB.textureDim - 1);

    float centerValue = g_inValue[DTid.xy];
    float2 backwardDifferences = centerValue - float2(g_inValue[left], g_inValue[top]);
    float2 forwardDifferences = float2(g_inValue[right], g_inValue[bottom]) - centerValue;

    // Calculates partial derivatives as the min of absolute backward and forward differences. 

#if SIGNED_DDXY
    // Find the absolute minimum of the backward and foward differences in each axis
    // while preserving the sign of the difference.
    float2 ddx = float2(backwardDifferences.x, forwardDifferences.x);
    float2 ddy = float2(backwardDifferences.y, forwardDifferences.y);

    uint2 minIndex = {
        GetIndexOfValueClosestToTheReference(0, ddx),
        GetIndexOfValueClosestToTheReference(0, ddy)
    };

    float2 ddxy = float2(ddx[minIndex.x], ddy[minIndex.y]);


#if HACK_CLAMP_DDXY_TO_BE_SMALL
    float2 _sign = sign(ddxy);
    float maxDdxy = 1;
    ddxy = _sign * min(abs(ddxy), maxDdxy);
#endif
#else

    // The min is taken to handle edges when calculating partial distance derivatives.
    // The min avoids the distance derivative slope being to that of another surface behind/in front of it on surface edges.
    // ToDo dont strip the sign?
    float2 ddxy = min(abs(backwardDifferences), abs(forwardDifferences));
#if HACK_CLAMP_DDXY_TO_BE_SMALL
    float maxDdxy = 1;
    ddxy = min(ddxy, maxDdxy);
#endif
#endif

    g_outValue[DTid] = ddxy;
}