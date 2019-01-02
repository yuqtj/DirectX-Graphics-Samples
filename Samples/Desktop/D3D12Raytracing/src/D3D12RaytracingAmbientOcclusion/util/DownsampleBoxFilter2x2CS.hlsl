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

Texture2D<float4> g_texInput : register(t0);
RWTexture2D<float4> g_texOutput : register(u0);

// Downsample linear 2x2 -> 1x1
[numthreads(DownsampleBoxFilter2x2::ThreadGroup::Width, DownsampleBoxFilter2x2::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	uint2 index = DTid * 2;
	float4 samples[4] = {
		g_texInput[index + uint2(0,0)],
		g_texInput[index + uint2(1,0)],
		g_texInput[index + uint2(0,1)],
		g_texInput[index + uint2(1,1)],
	};
	// todo handle weights for OOB samples
	float4 result = 0.25f * (samples[0] + samples[1] + samples[2] + samples[3]);

	g_texOutput[DTid] = result;
}