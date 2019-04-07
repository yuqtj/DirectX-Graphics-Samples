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

Texture2D<float> g_texInputCacheValue : register(t0);
Texture2D<float> g_texInputCurrentFrameValue : register(t1);
RWTexture2D<float> g_texOutputCacheValue : register(u0);
ConstantBuffer<RTAO_TemporalCache_ReverseReprojectConstantBuffer> cb : register(b0);

[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    float cachedValue = g_texInputCacheValue[DTid];
    float currentFrameValue = g_texInputCurrentFrameValue[DTid];
    float a = max(cb.invCacheFrameAge, cb.minSmoothingFactor);

    g_texOutputCacheValue[DTid] = lerp(cachedValue, currentFrameValue, a);
}