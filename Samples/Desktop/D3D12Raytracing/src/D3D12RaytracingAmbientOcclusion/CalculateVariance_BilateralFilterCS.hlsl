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

// Desc: Calculate Variance via Bilateral kernel.
// Uses normal and depth weights.
// Pitfalls: 
//  - normal weights may limit number of samples for small round objects
//  - depth weights may limit number of samples for thin objects (i.e. grass).

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValues : register(t0);
Texture2D<float> g_inDepth : register(t1);  // ToDo use from normal tex directly
Texture2D<float4> g_inNormalDepth : register(t2);

RWTexture2D<float> g_outVariance : register(u0);
RWTexture2D<float> g_outMean : register(u1);
ConstantBuffer<CalculateVariance_BilateralFilterConstantBuffer> cb: register(b0);

// ToDo use common helper
void LoadNormalAndDepth(Texture2D<float4> normalDepthTexture, in int2 texIndex, out float3 normal, out float depth, out float4 packedValue)
{
    packedValue = normalDepthTexture[texIndex];
    depth = packedValue.z;
    normal = DecodeNormal(packedValue.xy);
}


// ToDo add support for dxdy and perspective correct interpolation?
void AddFilterContribution(inout float weightedValueSum, inout float weightedSquaredValueSum, inout float weightSum, inout UINT numWeights, in float value, in float depth, in float3 normal, float obliqueness, in uint kernelRadius, in uint row, in uint col, in uint2 DTid)
{
    int2 id = int2(DTid) + (int2(row - kernelRadius, col - kernelRadius) );
    if (IsWithinBounds(id, cb.textureDim))
    {
        float iValue = g_inValues[id];

        float4 packedValue;
        float3 iNormal;
        float iDepth;
        LoadNormalAndDepth(g_inNormalDepth, id, iNormal, iDepth, packedValue);

        float w_d = cb.useDepthWeights ? exp(-abs(depth - iDepth) * obliqueness / (cb.depthSigma)) : 1.f;
        float w_n = cb.useNormalWeights ? pow(max(0, dot(normal, iNormal)), cb.normalSigma) : 1.f;
        float w = w_n * w_d;
        
        float SmallValue = 0.001f;
        if (w > SmallValue)
        {
            float weightedValue = w * iValue;
            weightedValueSum += weightedValue;
            weightedSquaredValueSum += weightedValue * iValue;
            weightSum += w;
            numWeights += 1;
        }
    }
}

// Calculates local per-pixel variance ~ Sum(X^2)/N - mean^2;
[numthreads(CalculateVariance_Bilateral::ThreadGroup::Width, CalculateVariance_Bilateral::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    float3 normal;
    float depth;
    float4 packedValue;
    LoadNormalAndDepth(g_inNormalDepth, DTid, normal, depth, packedValue);
    // ToDo use ddxy
    float obliqueness = max(0.0001f, pow(packedValue.w, 10));

    float  value = g_inValues[DTid];

    UINT numWeights = 1;
    float weightedValueSum = value;
    float weightedSquaredValueSum = value * value;
    float weightSum = 1.f;  // ToDo check for missing value

    const uint kernelRadius = cb.kernelWidth >> 1;

   // [unroll]
    for (UINT r = 0; r < cb.kernelWidth; r++)
       // [unroll]
        for (UINT c = 0; c < cb.kernelWidth; c++)
            if (r != kernelRadius || c != kernelRadius)
                 AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, kernelRadius, r, c, DTid);

    float variance = 0;
    float mean = 0;
    if (numWeights > 1)
    {
        float invWeightSum = 1 / weightSum;
        mean = invWeightSum * weightedValueSum;

        // Apply Bessel's correction to the estimated variance, divide by N-1, 
        // since the true population mean is not known. It is only estimated as the sample mean.
        float besselCorrection = numWeights / float(numWeights - 1);
        variance = besselCorrection * (invWeightSum * weightedSquaredValueSum - mean * mean);

        variance = max(0, variance);    // Ensure variance doesn't go negative due to imprecision.
    }
    if (cb.outputMean)
    {
        g_outMean[DTid] = mean;
    }
    g_outVariance[DTid] = variance;
}
