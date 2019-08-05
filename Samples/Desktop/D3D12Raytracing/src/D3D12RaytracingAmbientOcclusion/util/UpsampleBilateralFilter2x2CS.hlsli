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
#include "..\RaytracingShaderHelper.hlsli"

Texture2D<ValueType> g_inValue : register(t0);
Texture2D<NormalDepthTexFormat> g_inLowResNormalDepth : register(t1);
Texture2D<NormalDepthTexFormat> g_inHiResNormalDepth : register(t2);
Texture2D<float2> g_inHiResPartialDistanceDerivative : register(t3);
RWTexture2D<ValueType> g_outValue : register(u0);

// ToDo standardize cb vs g_CB
ConstantBuffer<DownAndUpsampleFilterConstantBuffer> g_CB : register(b0);

SamplerState ClampSampler : register(s0);

// ToDo consider 3x3 tap upsample instead 2x2

// ToDo remove outNormal if not written to.
//RWTexture2D<float4> g_texOutNormal : register(u1);

// ToDo use common bilateral sample fnc across all shaders
// ToDo comment
// ToDo reuse same in all resampling and atrous filter?
// Returns normalized weights for Bilateral Upsample.
float4 BilateralUpsampleWeights(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float4 BilinearWeights, in float2 ddxy)
{
    float4 depthWeights = 1;
    float4 normalWeights = 1;

    if (g_CB.useDepthWeights)
    {
        float depthThreshold = 1;     // ToDo standardize depth vs distance
        float fEpsilon = 1e-6 * ActualDistance;

        if (g_CB.useDynamicDepthThreshold)
        {
            float maxPixelDistance = 3; // Scale to compensate for the fact that the downsampled depth value may come from up to 3 pixels away in the high-res texture scale.

            // ToDo consider ddxy per dimension or have a 1D max(Ddxy) resource?
            // ToDo perspective correction?
            depthThreshold = maxPixelDistance * dot(1, ddxy);
        }
        // ToDo correct weights to weights in the whole project same for treshold and weight
        float fScale = 1.f / depthThreshold;
        depthWeights = min(1.0 / (fScale * abs(SampleDistances - ActualDistance) + fEpsilon), 1);

        depthWeights *= depthWeights >= 0.5;   // ToDo revise - this is same as comparing to depth tolerance

    }

    if (g_CB.useNormalWeights)
    {
        const uint normalExponent = 32;
        normalWeights =  
            float4(
                pow(saturate(dot(ActualNormal, SampleNormals[0])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[1])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[2])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[3])), normalExponent));

        // Ensure a non-zero weight in case none of the normals match.
        normalWeights += 1e-3f;
    }



    BilinearWeights = g_CB.useBilinearWeights ? BilinearWeights : 0.25;
    bool4 isActive = SampleDistances != 0;

    float4 weights = isActive * normalWeights * depthWeights * BilinearWeights;

    float weightSum = dot(weights, 1);
    float4 nWeights = weightSum > 1e-5f ? weights / weightSum : BilinearWeights; // Default to bilinear weights if all weights are too small.

    return nWeights;
}

[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    // Process each 2x2 high res quad at a time, starting from [-1,-1] 
    // so that subsequent internal high-res pixel quads are within low-res quads.
    int2 topLeftHiResIndex = (DTid << 1) + int2(-1, -1);
    int2 topLeftLowResIndex = (topLeftHiResIndex + int2(-1, -1)) >> 1;
    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

    
    float4  vHiResDepths;
    float3 hiResNormals[4];
    float2 hiResTexturePos = (topLeftHiResIndex + 0.5) * g_CB.invHiResTextureDim;
    {
        uint4 packedEncodedNormalDepths = g_inHiResNormalDepth.GatherRed(ClampSampler, hiResTexturePos).wzxy;
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            if (g_CB.useDepthWeights || g_CB.useNormalWeights)
                DecodeNormalDepth(packedEncodedNormalDepths[i], hiResNormals[i], vHiResDepths[i]);
        }
    }

    float4  vLowResDepths;
    float3 lowResNormals[4];
    float2 lowResTexturePos = (topLeftLowResIndex + 0.5) * g_CB.invLowResTextureDim;
    {
        uint4 packedEncodedNormalDepths = g_inLowResNormalDepth.GatherRed(ClampSampler, lowResTexturePos).wzxy;
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            if (g_CB.useDepthWeights || g_CB.useNormalWeights)
                DecodeNormalDepth(packedEncodedNormalDepths[i], lowResNormals[i], vLowResDepths[i]);
        }
    }

    // ToDo use gather
#if VALUE_NUM_COMPONENTS == 1
    float4 vLowResValues = g_inValue.GatherRed(ClampSampler, lowResTexturePos).wzxy;
#elif VALUE_NUM_COMPONENTS == 2
    float2x4 vLowResValues = {
        g_inValue.GatherRed(ClampSampler, lowResTexturePos).wzxy,
        g_inValue.GatherGreen(ClampSampler, lowResTexturePos).wzxy
    };
#endif


    const float4 bilinearWeights[4] = {
        float4(9, 3, 3, 1) / 16,
        float4(3, 9, 1, 3) / 16,
        float4(3, 1, 9, 3) / 16,
        float4(1, 3, 3, 9) / 16
    };
    
    // ToDO standarddize ddxy vs dxdy
    float2x4 ddxy2x4 = 0;
    if (g_CB.useDepthWeights && g_CB.useDynamicDepthThreshold)
    {
        ddxy2x4[0] = g_inHiResPartialDistanceDerivative.GatherRed(ClampSampler, hiResTexturePos).wzxy;
        ddxy2x4[1] = g_inHiResPartialDistanceDerivative.GatherGreen(ClampSampler, hiResTexturePos).wzxy;
    }

    float4x2 ddxy = {
        ddxy2x4._11, ddxy2x4._21,
        ddxy2x4._12, ddxy2x4._22,
        ddxy2x4._13, ddxy2x4._23,
        ddxy2x4._14, ddxy2x4._24,
    };

    {
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            float actualDistance = vHiResDepths[i];
            float3 actualNormal = hiResNormals[i];

            float4 nWeights = BilateralUpsampleWeights(actualDistance, actualNormal, vLowResDepths, lowResNormals, bilinearWeights[i], ddxy[i]);

            // ToDo revise - take an average if none match?
#if VALUE_NUM_COMPONENTS == 1
            float outValue = dot(nWeights, vLowResValues);
            g_outValue[topLeftHiResIndex + srcIndexOffsets[i]] = actualDistance < DISTANCE_ON_MISS ? outValue : vLowResValues[i];
#elif VALUE_NUM_COMPONENTS == 2
            float2 outValue = float2(dot(nWeights, vLowResValues[0]), dot(nWeights, vLowResValues[1]));
            g_outValue[topLeftHiResIndex + srcIndexOffsets[i]] = actualDistance < DISTANCE_ON_MISS ? outValue : vLowResValues._11_21;
#endif
        }
    }
}