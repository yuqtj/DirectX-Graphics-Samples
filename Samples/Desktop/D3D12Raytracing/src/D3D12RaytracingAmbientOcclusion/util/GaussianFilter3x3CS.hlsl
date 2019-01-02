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

Texture2D<float> g_texInput : register(t0);
RWTexture2D<float> g_texOutput : register(u0);
ConstantBuffer<GaussianFilterConstantBuffer> cb : register(b0);

static float kernel1D[3] = { 0.27901, 0.44198, 0.27901 };
static const float weights[3][3] =
{
    { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2] },
    { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2] },
    { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2] },
};


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

    AddFilterContribution(weightedValueSum, weightSum, 0, 0, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 1, 0, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 2, 0, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 0, 1, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 1, 1, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 2, 1, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 0, 2, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 1, 2, DTid);
    AddFilterContribution(weightedValueSum, weightSum, 2, 2, DTid);

	g_texOutput[DTid] = weightedValueSum / weightSum;
}