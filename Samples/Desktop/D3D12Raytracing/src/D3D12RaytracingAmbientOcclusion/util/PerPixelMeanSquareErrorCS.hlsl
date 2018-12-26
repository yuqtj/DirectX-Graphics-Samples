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

Texture2D<float> g_texReference : register(t0);
Texture2D<float> g_texValues : register(t1);
RWTexture2D<float> g_texOutput : register(u0);

groupshared uint gShared[PerPixelMeanSquareError::ThreadGroup::Size];

// PerPixelMeanSquareError
// - Calculates per pixel mean square error: (x - x_r)^2
[numthreads(ReduceSumCS::ThreadGroup::Width, ReduceSumCS::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    float delta = g_texValues[DTid] - g_texReference[DTid];
    g_texOutput[DTid] = delta * delta;
}