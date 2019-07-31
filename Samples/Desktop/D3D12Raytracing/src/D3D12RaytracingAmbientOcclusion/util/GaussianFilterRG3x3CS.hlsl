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

Texture2D<float2> g_texInput : register(t0);
RWTexture2D<float2> g_texOutput : register(u0);
ConstantBuffer<TextureDimConstantBuffer> cb : register(b0);

SamplerState MirroredLinearSampler : register(s0);

static const float weights[3][3] =
{
    { 0.077847, 0.123317, 0.077847 },
    { 0.123317, 0.195346, 0.123317 },
    { 0.077847, 0.123317, 0.077847 },
};

// ToDo cleanup
#define APPROXIMATE_GAUSSIAN_3X3_VIA_HW_FILTERING 1

#if APPROXIMATE_GAUSSIAN_3X3_VIA_HW_FILTERING
// Approximate 3x3 gaussian filter using HW bilinear filtering.
// Ref: Moller2018, Real-Time Rendering (Fourth Edition), p517
// Performance improvement over 3x3 2D version (4K on 2080 Ti): 0.18ms -> 0.11ms
// Todo Test against w separable version
[numthreads(GaussianFilter::ThreadGroup::Width, GaussianFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    // Set weights based on availability of neighbor samples.
    float4 weights;

    // ToDo test indexing into array instead.
    // Non-border pixels
    if (DTid.x > 0 && DTid.y > 0 && DTid.x < cb.textureDim.x - 1 && DTid.y < cb.textureDim.y - 1) 
    {
        weights = float4(0.077847 + 0.123317 + 0.123317 + 0.195346, 
                         0.077847 + 0.123317,
                         0.077847 + 0.123317,
                         0.077847);
    }
    // Top-left corner
    else if (DTid.x == 0 && DTid.y == 0)     
    {
        weights = float4(0.195346, 0.123317, 0.123317, 0.077847) / 0.519827;
    }
    // Top-right corner
    else if (DTid.x == cb.textureDim.x - 1 && DTid.y == 0) 
    {
        weights = float4(0.123317 + 0.195346, 0, 0.201164, 0) / 0.519827;
    }
    // Bottom-left corner
    else if (DTid.x == 0 && DTid.y == cb.textureDim.y - 1) 
    {
        weights = float4(0.123317 + 0.195346, 0.077847 + 0.123317, 0, 0) / 0.519827;
    }
    // Bottom-right corner
    else if (DTid.x == cb.textureDim.x - 1 && DTid.y == cb.textureDim.y - 1) 
    {
        weights = float4(0.077847 + 0.123317 + 0.123317 + 0.195346, 0, 0, 0) / 0.519827;
    }
    // Left border
    else if (DTid.x == 0) 
    {
        weights = float4(0.123317 + 0.195346, 0.077847 + 0.123317, 0.123317, 0.077847) / 0.720991;
    }
    // Right border
    else if (DTid.x == cb.textureDim.x - 1)
    {
        weights = float4(0.077847 + 0.123317 + 0.123317 + 0.195346, 0, 0.077847 + 0.123317, 0) / 0.720991;
    }
    // Top border
    else if (DTid.y == 0) 
    {
        weights = float4(0.123317 + 0.195346, 0.123317, 0.077847 + 0.123317, 0.077847) / 0.720991;
    }
    // Bottom border
    else
    {
        weights = float4(0.077847 + 0.123317 + 0.123317 + 0.195346, 0.077847 + 0.123317, 0, 0) / 0.720991;
    }

    const float2 offsets[3] = {
        float2(0.5, 0.5) + float2(-0.123317 / (0.123317 + 0.195346), -0.123317 / (0.123317 + 0.195346)),
        float2(0.5, 0.5) + float2(1, -0.077847 / (0.077847 + 0.123317)),
        float2(0.5, 0.5) + float2(-0.077847 / (0.077847 + 0.123317), 1) };

    float2 samples[4];
    samples[0] = g_texInput.SampleLevel(MirroredLinearSampler, (DTid + offsets[0]) * cb.invTextureDim, 0);
    samples[1] = g_texInput.SampleLevel(MirroredLinearSampler, (DTid + offsets[1]) * cb.invTextureDim, 0);
    samples[2] = g_texInput.SampleLevel(MirroredLinearSampler, (DTid + offsets[2]) * cb.invTextureDim, 0);
    samples[3] = g_texInput[DTid + 1];

    float4 samplesR = float4(samples[0].x, samples[1].x, samples[2].x, samples[3].x);
    float4 samplesG = float4(samples[0].y, samples[1].y, samples[2].y, samples[3].y);

    g_texOutput[DTid] = float2(dot(samplesR, weights), dot(samplesG, weights));
}

#else
ToDo or remove
void AddFilterContribution(inout float weightedValueSum, inout float weightSum, in int row, in int col, in int2 DTid)
{
    int2 id = DTid + int2(row - 1, col - 1);
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        weightedValueSum += weights[col][row] * g_texInput[id];
        weightSum += weights[col][row];
    }
}

[numthreads(GaussianFilter::ThreadGroup::Width, GaussianFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    float weightSum = 0;
    float weightedValueSum = 0;

    [unroll]
    for (UINT r = 0; r < 3; r++)
        [unroll]
        for (UINT c = 0; c < 3; c++)
            AddFilterContribution(weightedValueSum, weightSum, r, c, DTid);

        g_texOutput[DTid] = weightedValueSum / weightSum;
}
#endif