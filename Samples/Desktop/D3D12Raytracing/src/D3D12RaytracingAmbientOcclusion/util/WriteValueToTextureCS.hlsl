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

RWTexture2D<UINT> g_texOutput : register(u0);

// Downsample linear 2x2 -> 1x1
[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	g_texOutput[DTid] = DTid.x < 1920 ? 1 : 0;
}