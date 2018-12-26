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

#define SAMPLE(offsetX, offsetY) g_texInput[index + int2(offsetX, offsetY)]

[numthreads(DownsampleGaussianFilter::ThreadGroup::Width, DownsampleGaussianFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	int2 index = DTid * 2 + int2(1,1);
	float weights[5] = { 0.06136f, 0.24477f, 0.38774f, 0.24477f, 0.06136f };

	float4 samples[5][5] = {
		{ SAMPLE(-2, -2),	SAMPLE(-2, -1),	SAMPLE(-2, 0),	SAMPLE(-2, 1),	SAMPLE(-2, 2) },
		{ SAMPLE(-1, -2),	SAMPLE(-1, -1),	SAMPLE(-1, 0),	SAMPLE(-1, 1),	SAMPLE(-1, 2) },
		{ SAMPLE(0, -2),	SAMPLE(0, -1),  SAMPLE(0, 0),	SAMPLE(0, 1),	SAMPLE(0, 2) },
		{ SAMPLE(1, -2),	SAMPLE(1, -1),	SAMPLE(1, 0),	SAMPLE(1, 1),	SAMPLE(1, 2) },
		{ SAMPLE(2, -2),	SAMPLE(2, -1),	SAMPLE(2, 0),	SAMPLE(2, 1),	SAMPLE(2, 2) }
	};
	// Horizontal blur
	float4 resultHorizontalBlur[5];
	for (uint i = 0; i < 5; i++)
	{
		resultHorizontalBlur[i] = (float4)0;
		for (uint j = 0; j < 5; j++)
		{
			resultHorizontalBlur[i] += weights[j] * samples[i][j];
		}
	}

	// Vertical Blur
	float4 result = (float4)0;
	for (uint j = 0; j < 5; j++)
	{
		result += weights[j] * resultHorizontalBlur[j];
	}

	g_texOutput[DTid] = result;
}