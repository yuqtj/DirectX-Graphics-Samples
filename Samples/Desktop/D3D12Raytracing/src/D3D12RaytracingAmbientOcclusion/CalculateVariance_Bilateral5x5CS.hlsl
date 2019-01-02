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
#include "RaytracingHlslCompat.h"

Texture2D<float> g_inValues : register(t0);
Texture2D<float4> g_inNormal : register(t1);
Texture2D<float> g_inDepth : register(t2);
Texture2D<uint> g_inNormalOct : register(t3);
RWTexture2D<float> g_outVariance : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);


void AddFilterContribution(inout float weightedValueSum, inout float weightedSquaredValueSum, inout float weightSum, inout UINT numWeights, in float value, in float depth, in float3 normal, float obliqueness, in uint row, in uint col, in uint2 DTid)
{
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    int2 id = int2(DTid) + (int2(row - 2, col - 2) );
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        float iValue = g_inValues[id];
#if COMPRES_NORMALS
        uint normalOct = g_inNormalOct[id];
        float3 iNormal = i_octahedral_32(normalOct, 16u);
#else
        float3 iNormal = g_inNormal[id].xyz;
#endif 
        float  iDepth = g_inDepth[id];

        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
        float w = w_n * w_d;

        float weightedValue = w * iValue;
        weightedValueSum += weightedValue;
        weightedSquaredValueSum += weightedValue * iValue;
        weightSum += w;
        numWeights += w > 0.0001f ? 1 : 0;
    }
}

// Calculates local per-pixel variance ~ Sum(X^2)/N - mean^2;
[numthreads(CalculateVariance_Bilateral::ThreadGroup::Width, CalculateVariance_Bilateral::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    const uint N = 5;
    const float kernel1D[N] = { 1.f / 16, 1.f / 4, 3.f / 8, 1.f / 4, 1.f / 16 };
    const float kernel[N][N] =
    {
        { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2], kernel1D[0] * kernel1D[3], kernel1D[0] * kernel1D[4] },
        { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2], kernel1D[1] * kernel1D[3], kernel1D[1] * kernel1D[4] },
        { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2], kernel1D[2] * kernel1D[3], kernel1D[2] * kernel1D[4] },
        { kernel1D[3] * kernel1D[0], kernel1D[3] * kernel1D[1], kernel1D[3] * kernel1D[2], kernel1D[3] * kernel1D[3], kernel1D[3] * kernel1D[4] },
        { kernel1D[4] * kernel1D[0], kernel1D[4] * kernel1D[1], kernel1D[4] * kernel1D[2], kernel1D[4] * kernel1D[3], kernel1D[4] * kernel1D[4] },
    };
#if COMPRES_NORMALS
    uint normalOct = g_inNormalOct[DTid];
    float3 normal = i_octahedral_32(normalOct, 16u);
    float obliqueness = 1.f;
#else
    float4 normal4 = g_inNormal[DTid];
    float3 normal = normal4.xyz;
    float obliqueness = max(0.0001f, pow(normal4.w, 10));
#endif 

    float  depth = g_inDepth[DTid];
    float  value = g_inValues[DTid];

    UINT numWeights = 1;
    float weightedValueSum = value;
    float weightedSquaredValueSum = value * value;
    float weightSum = 1.f;  // ToDo check for missing value

    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 0, 0, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 0, 1, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 0, 2, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 0, 3, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 0, 4, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 1, 0, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 1, 1, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 1, 2, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 1, 3, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 1, 4, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 2, 0, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 2, 1, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 2, 3, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 2, 4, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 3, 0, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 3, 1, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 3, 2, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 3, 3, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 3, 4, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 4, 0, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 4, 1, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 4, 2, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 4, 3, DTid);
    AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value,depth, normal, obliqueness, 4, 4, DTid);

    float variance;
    if (numWeights > 1)
    {
        float invWeightSum = weightSum > 0.0001f ? 1 / weightSum : 0.f;
        float mean = invWeightSum * weightedValueSum;
        variance = (numWeights / float(numWeights - 1)) * (invWeightSum * weightedSquaredValueSum - mean * mean);
    }
    else
    {
        variance = 0;
    }
    g_outVariance[DTid] = variance;
}