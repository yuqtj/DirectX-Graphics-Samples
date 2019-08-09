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
Texture2D<float> g_texInputCurrentFrameRayHitDistance : register(t2);
Texture2D<uint4> g_texInputReprojected_FrameAge_Value_SquaredMeanValue_RayHitDistance : register(t3);

// ToDo combine some outputs?
RWTexture2D<float> g_texInputOutputValue : register(u0);
RWTexture2D<uint2> g_texInputOutputFrameAge : register(u1);
RWTexture2D<float> g_texInputOutputSquaredMeanValue : register(u2);
RWTexture2D<float> g_texInputOutputRayHitDistance : register(u3);
RWTexture2D<float> g_texOutputVariance : register(u4);
RWTexture2D<float> g_texOutputBlurStrength: register(u5);

// ToDo remove
RWTexture2D<float4> g_texOutputDebug1 : register(u10);
RWTexture2D<float4> g_texOutputDebug2 : register(u11);

ConstantBuffer<RTAO_TemporalSupersampling_BlendWithCurrentFrameConstantBuffer> cb : register(b0);

[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint4 encodedCachedValues = g_texInputReprojected_FrameAge_Value_SquaredMeanValue_RayHitDistance[DTid];
    uint packedFrameAgeRaysToGenerate = encodedCachedValues.x;
    uint frameAge;
    uint numRaysToGenerateOrDenoisePasses;
    Unpack_R16_to_R8G8_UINT(packedFrameAgeRaysToGenerate, frameAge, numRaysToGenerateOrDenoisePasses);


    bool isRayCountValue = !(numRaysToGenerateOrDenoisePasses & 0x80);
    uint numRaysToGenerate = isRayCountValue ? numRaysToGenerateOrDenoisePasses : 0;
    uint numDenoisePasses = 0x7F & numRaysToGenerateOrDenoisePasses;

    float4 cachedValues = float4(frameAge, f16tof32(encodedCachedValues.yzw));
    //g_texOutputDebug1[DTid] = cachedValues;

    float value = g_texInputCurrentFrameValue[DTid];
    bool isValidValue = value != RTAO::InvalidAOValue;

    float valueSquaredMean = value * value;
    float rayHitDistance;
    float variance;
    
    // ToDo can this ever fail? reproject sets frameage to 1 at minimum.
    if (frameAge > 0)
    {     
        uint maxFrameAge = 1 / cb.minSmoothingFactor;
        frameAge = isValidValue ? min(frameAge + 1, maxFrameAge) : frameAge;

        float cachedValue = cachedValues.y;
        // Clamp value to mean +/- std.dev of local neighborhood to surpress ghosting on value changing due to other occluder movements.
        // Ref: Salvi2016, Temporal Super-Sampling

        float2 localMeanVariance = g_texInputCurrentFrameLocalMeanVariance[DTid];
        float localMean = localMeanVariance.x;
        float localVariance = localMeanVariance.y;
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
        float invFrameAge = 1.f / frameAge;
        float a = cb.forceUseMinSmoothingFactor ? cb.minSmoothingFactor : max(invFrameAge, cb.minSmoothingFactor);
        float MaxSmoothingFactor = 1;// 0.2;
        a = min(a, MaxSmoothingFactor);
        a = isValidValue ? a : 0;

        // Value.
        value = lerp(cachedValue, value, a);

        // Value Squared Mean.
        float cachedSquaredMeanValue = cachedValues.z; 
        valueSquaredMean = lerp(cachedSquaredMeanValue, valueSquaredMean, a);

        // Variance.
        float temporalVariance = valueSquaredMean - value * value;
        temporalVariance = max(0, temporalVariance);    // Ensure variance doesn't go negative due to imprecision.
        variance = frameAge >= cb.minFrameAgeToUseTemporalVariance ? temporalVariance : localVariance;
        variance = max(0.1, variance);

        // RayHitDistance.
        rayHitDistance = g_texInputCurrentFrameRayHitDistance[DTid];
        float cachedRayHitDistance = cachedValues.w;
        rayHitDistance = lerp(cachedRayHitDistance, rayHitDistance, a);


        // ToDo use an helper 0/1 resource instead ?
#if RTAO_MARK_CACHED_VALUES_NEGATIVE
        value = isValidValue ? value : -value;
#endif
    }
    else if (isValidValue)
    {
        frameAge = isValidValue ? 1 : 0;
#if RTAO_GAUSSIAN_BLUR_AFTER_TSS
        value = isValidValue ? value : -value;
#endif
        rayHitDistance = isValidValue ? g_texInputCurrentFrameRayHitDistance[DTid] : 22;
        variance = isValidValue ? g_texInputCurrentFrameLocalMeanVariance[DTid].y : 1;
        valueSquaredMean = isValidValue ? valueSquaredMean : RTAO::InvalidAOValue;
    }
#if STOP_TRACING_AND_DENOISING_AFTER_FEW_FRAMES
    if (isValidValue)
    {
        if (numRaysToGenerate > 1)
        {
            numRaysToGenerateOrDenoisePasses = numRaysToGenerate - 1;
        }
        else // Switch to denoise count
        {
            // + 2 ==
            //        + 1 since the subtraction happens before a denoise pass.
            //        + 1 since the value is set in the pass when a ray was cast.
            numRaysToGenerateOrDenoisePasses = (cb.numFramesToDenoiseAfterLastTracedRay + 2) | 0x80;
        }
    }
    else if (!isRayCountValue)
    {
        numRaysToGenerateOrDenoisePasses = (max(numDenoisePasses, 1) - 1) | 0x80;
    }
#else
    numRaysToGenerateOrDenoisePasses = 33;
#endif

    float frameAgeRatio = min(frameAge, cb.blurStrength_MaxFrameAge) / float(cb.blurStrength_MaxFrameAge);
    float blurStrength = pow(1 - frameAgeRatio, cb.blurDecayStrength);

    g_texInputOutputFrameAge[DTid] = uint2(frameAge, numRaysToGenerateOrDenoisePasses);
    g_texInputOutputValue[DTid] = value;
    g_texInputOutputSquaredMeanValue[DTid] = valueSquaredMean;
    g_texInputOutputRayHitDistance[DTid] = rayHitDistance;
    g_texOutputVariance[DTid] = variance; 
    g_texOutputBlurStrength[DTid] = blurStrength;
}