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
    //    x    [top]     x
    // [left]   DTiD   [right]
    //    x   [bottom]   x
    uint2 top = clamp(DTid.xy + uint2(0, -1), 0, g_CB.textureDim - 1);
    uint2 bottom = clamp(DTid.xy + uint2(0, 1), 0, g_CB.textureDim - 1);
    uint2 left = clamp(DTid.xy + uint2(-1, 0), 0, g_CB.textureDim - 1);
    uint2 right = clamp(DTid.xy + uint2(1, 0), 0, g_CB.textureDim - 1);

    // ToDo Pick 1 and update the file name
    // ToDo use shader intrinsics?
#if 1
    // Calculates partial derivatives as the min of absolute backward and forward differences. 
    // The min is taken to handle edges when calculating partial distance derivatives.
    // The min avoids the distance derivative slope being to that of another surface behind/in front of it on surface edges.
    float centerValue = g_inValue[DTid.xy];                                                 // deltaY
    // // ToDo use gather
    float2 backwardDifferences = centerValue - float2(g_inValue[left], g_inValue[top]);    //-0.000011252239
    float2 forwardDifferences = float2(g_inValue[right], g_inValue[bottom]) - centerValue;// -0.000012516976

    // ToDO pick both dimensions from one or the other?
    // ToDo retain pos/negative sides?


    // ToDo dont strip the sign
    float2 ddxy = min(abs(backwardDifferences), abs(forwardDifferences));

#if HACK_CLAMP_DDXY_TO_BE_SMALL
    float maxDdxy = 1;
    ddxy = min(ddxy, maxDdxy);
#endif
    g_outValue[DTid] = ddxy;

#else
    // Calculates central differences: (f(x+1, y) - f(x-1, y), f(x, y+1) - f(x, y-1))

    // Scale down to [1,1] pixel to pixel distance.
    g_outValue[DTid] = (1.f / 2) * float2(g_inValue[right] - g_inValue[left], g_inValue[bottom] - g_inValue[top]);
#endif
}