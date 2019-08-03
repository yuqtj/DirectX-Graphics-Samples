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
#include "RaytracingShaderHelper.hlsli"
#include "RTAO\Shaders\RTAO.hlsli"

// ToDo scale weights by cachedFrameAge?
// ToDo some pixels here and there on mirror boundaries fail temporal reprojection even for static scene/camera
// ToDo sharp edges fail temporal reprojection due to clamping even for static scene

// ToDO pack value and depth beforehand?
// ToDo standardize in vs input, out vs output
Texture2D<float> g_texInputCurrentFrameValue : register(t0);
Texture2D<float2> g_texInputCurrentFrameLocalMeanVariance : register(t1);
Texture2D<float2> g_texInputCurrentFrameRayHitDistance : register(t2);

// ToDo combine some outputs?
RWTexture2D<float> g_texInputOutputValue : register(u0);
RWTexture2D<uint>  g_texInputOutputFrameAge : register(u1);
RWTexture2D<float> g_texInputOutputSquaredMeanValue : register(u2);
RWTexture2D<float> g_texInputOutputRayHitDistance : register(u3);
RWTexture2D<float> g_texOutputVariance : register(u4);

// ToDo remove
RWTexture2D<float4> g_texOutputDebug1 : register(u10);
RWTexture2D<float4> g_texOutputDebug2 : register(u11);

ConstantBuffer<RTAO_TemporalSupersampling_BlendWithCurrentFrameConstantBuffer> cb : register(b0);

[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint frameAge = g_texInputOutputFrameAge[DTid];

    uint value = g_texInputCurrentFrameValue[DTid];
    uint isValidValue = value != RTAO::InvalidAOValue;

    if (isValidValue)
    {
        frameAge += 1;
    }
    float valueSquaredMean = value * value;
    float rayHitDistance = g_texInputOutputRayHitDistance[DTid];

    float2 localMeanVariance = g_texInputCurrentFrameLocalMeanVariance[DTid];
    float localMean = localMeanVariance.x;
    float localVariance = localMeanVariance.y;
    float variance = localVariance;

    uint maxFrameAge = 1 / cb.minSmoothingFactor;
    frameAge = isValidValue ? min(frameAge + 1, maxFrameAge) : frameAge;
    
    if (frameAge > 1)
    {
        // Clamp value to mean +/- std.dev of local neighborhood to surpress ghosting on value changing due to other occluder movements.
        // Ref: Salvi2016, Temporal Super-Sampling
        float cachedValue = g_texInputOutputValue[DTid];
        if (cb.clampCachedValues)
        {

            float localStdDev = max(cb.stdDevGamma * sqrt(localVariance), cb.minStdDevTolerance);
            float nonClampedCachedValue = cachedValue;
            cachedValue = clamp(cachedValue, localMean - localStdDev, localMean + localStdDev);

            // Scale down the frame age based on how strongly the cached value got clamped to give more weight to new samples
            // ToDo avoid saturate?
            float frameAgeScale = saturate(cb.clampDifferenceToFrameAgeScale * abs(cachedValue - nonClampedCachedValue));
            // ToDo round to nearest integer?
            frameAge = lerp(frameAge, 0, frameAgeScale);
        }

        // ToDo: use moving average (Koskela2019) for the first few samples 
        // to even out the weights for the noisy start instead of weighting first samples much more.
        float invFrameAge = 1.f / (frameAge + 1.f);
        float a = cb.forceUseMinSmoothingFactor ? cb.minSmoothingFactor : max(invFrameAge, cb.minSmoothingFactor);
        a = isValidValue ? a : 0;

        // Value.
        value = lerp(cachedValue, value, a);
        // ToDo use an helper 0/1 resource instead ?
        value = isValidValue ? value : -value;

        // Value Squared Mean.
        float cachedSquaredMeanValue = g_texInputOutputSquaredMeanValue[DTid];
        valueSquaredMean = lerp(cachedSquaredMeanValue, valueSquaredMean, a);

        // Variance.
        float temporalVariance = valueSquaredMean - value * value;
        temporalVariance = max(0, temporalVariance);    // Ensure variance doesn't go negative due to imprecision.
        variance = frameAge >= cb.minFrameAgeToUseTemporalVariance ? temporalVariance : localVariance;
        
        // RayHitDistance.
        float cachedRayHitDistance = g_texInputOutputRayHitDistance[DTid];
        rayHitDistance = lerp(cachedRayHitDistance, rayHitDistance, a);
    }

    g_texInputOutputFrameAge[DTid] = frameAge;
    g_texInputOutputValue[DTid] = value;
    g_texInputOutputSquaredMeanValue[DTid] = valueSquaredMean;
    g_texInputOutputRayHitDistance[DTid] = rayHitDistance;
}