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
SamplerState LinearSampler : register(s0);
ConstantBuffer<DownsampleFilterConstantBuffer> g_CB : register(b0);


[numthreads(DownsampleGaussianFilter::ThreadGroup::Width, DownsampleGaussianFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	int2 index = DTid * 2 + int2(1,1);
	float weights[3] = { 0.27901f, 0.44198f, 0.27901f };
	float4 samples[3][3] = {
		{ g_texInput[index + int2(-1, -1)], g_texInput[index + int2(-1, 0)], g_texInput[index + int2(-1, 1)] },
		{ g_texInput[index + int2( 0, -1)], g_texInput[index + int2( 0, 0)], g_texInput[index + int2( 0, 1)] },
		{ g_texInput[index + int2( 1, -1)], g_texInput[index + int2( 1, 0)], g_texInput[index + int2( 1, 1)] }
	};

	// Horizontal blur
	float4 resultHorizontalBlur[3];
	for (uint i = 0; i < 3; i++)
	{
		resultHorizontalBlur[i] = (float4)0;
		for (uint j = 0; j < 3; j++)
		{
			resultHorizontalBlur[i] += weights[j] * samples[i][j];
		}
	}

	// Vertical Blur
	float4 result = (float4)0;
	for (uint j = 0; j < 3; j++)
	{
		result += weights[j] * resultHorizontalBlur[j];
	}

	g_texOutput[DTid] = result;
}