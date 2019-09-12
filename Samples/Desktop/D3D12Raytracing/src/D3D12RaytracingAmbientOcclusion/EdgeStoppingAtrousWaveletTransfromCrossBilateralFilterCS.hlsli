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
#include "Kernels.hlsli"
#include "RTAO/Shaders/RTAO.hlsli"


Texture2D<float> g_inValues : register(t0);

Texture2D<NormalDepthTexFormat> g_inNormalDepth : register(t1);
Texture2D<float> g_inVariance : register(t4);   // ToDo remove
Texture2D<float> g_inSmoothedVariance : register(t5); 
Texture2D<float> g_inHitDistance : register(t6);   // ToDo remove?
Texture2D<float2> g_inPartialDistanceDerivatives : register(t7);   // ToDo remove?
Texture2D<uint2> g_inFrameAge : register(t8);

RWTexture2D<float> g_outFilteredValues : register(u0);
RWTexture2D<float> g_outFilteredVariance : register(u1);
RWTexture2D<float4> g_outDebug1 : register(u3);
RWTexture2D<float4> g_outDebug2 : register(u4);

ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

float DepthThreshold(float distance, float2 ddxy, float2 pixelOffset)
{
    float depthThreshold;

    // Pespective correction for the non-linear interpolation
    if (cb.perspectiveCorrectDepthInterpolation)
    {
        // Calculate depth with perspective correction.
        // Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/visibility-problem-depth-buffer-depth-interpolation
        // Given depth buffer interpolation for finding z at offset q along z0 to z1
        //      z =  1 / (1 / z0 * (1 - q) + 1 / z1 * q)
        // and z1 = z0 + ddxy, where z1 is at a unit pixel offset [1, 1]
        // z can be calculated via ddxy as
        //
        //      z = (z0 + ddxy) / (1 + (1-q) / z0 * ddxy) 

        float z0 = distance;
        float2 zxy = (z0 + ddxy) / (1 + ((1 - pixelOffset) / z0) * ddxy);
        depthThreshold = dot(1, abs(zxy - z0)); // ToDo this should be sqrt(dot(zxy - z0, zxy - z0))
    }
    else
    {
        depthThreshold = dot(1, abs(pixelOffset * ddxy));
    }

    return depthThreshold;
}

void AddFilterContribution(
    inout float weightedValueSum, 
    inout float weightedVarianceSum, 
    inout float weightSum, 
    in float value, 
    in float stdDeviation,
    in float depth, 
    in float3 normal, 
    in float2 ddxy,
    in uint row, 
    in uint col,
    in uint2 kernelStep,
    in uint2 DTid,
    in uint kernelStepShift,
    in float2 varianceSigmaScale)   // ToDo or remove
{

    const float valueSigma = cb.valueSigma;
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;
 
    int2 pixelOffset;
    float kernelWidth;
    float varianceScale = 1;

    pixelOffset = int2(row - FilterKernel::Radius, col - FilterKernel::Radius) * kernelStep;
    int2 id = int2(DTid) + pixelOffset;

    if (IsWithinBounds(id, cb.textureDim))
    {
        float iDepth;
        float3 iNormal;
        DecodeNormalDepth(g_inNormalDepth[id], iNormal, iDepth);
        float iValue = g_inValues[id];

        bool iIsValidValue = iValue != RTAO::InvalidAOValue;
        if (!iIsValidValue || iDepth == 0)
        {
            return;
        }


        // ToDo explain/remove
        float w_s = 1;
#if RTAO_MARK_CACHED_VALUES_NEGATIVE
        if (iValue < 0)
        {
            w_s = cb.staleNeighborWeightScale;
            iValue = -iValue;
        }
#endif

        // 
        // ToDo explain / remove
        float w_fa = 1;
        uint iFrameAge = g_inFrameAge[id].x;

        // Enforce frame age of at least 1 for reprojection for valid values.
        // This is because the denoiser will fill in invalid values with filtered 
        // ones if it can. But it doesn't increase frame age.
        iFrameAge = max(iFrameAge, 1);

        w_fa = cb.weightByFrameAge ? iFrameAge : 1;
        
        // Value based weight.
        float w_x = 1;
        if (valueSigma > 0)
        {
            const float errorOffset = 0.005f;
            float e_x = -abs(value - iValue) / (valueSigma * stdDeviation + errorOffset);
            w_x = exp(e_x);
        }

 
        // Normal based weight.
        float w_n = 1;
        if (normalSigma > 0)
        {
            w_n = pow(max(0, dot(normal, iNormal)), normalSigma);
        }


        // Depth based weight.
        float w_d = 1;
        if (depthSigma > 0)
        {
            float2 pixelOffsetForDepth = pixelOffset;

            // Account for sample offset in bilateral downsampled partial depth derivative buffer.
            if (cb.usingBilateralDownsampledBuffers)
            {
                float2 offsetSign = sign(pixelOffset);
                pixelOffsetForDepth = pixelOffset + offsetSign * float2(0.5, 0.5);
            }

            float depthFloatPrecision = FloatPrecision(max(depth, iDepth), cb.DepthNumMantissaBits);

            // ToDo dedupe with CrossBilateralWeights.hlsli?
            // ToDo test or remove
            if (cb.useProjectedDepthTest)
            {
                float zC = GetDepthAtPixelOffset(depth, ddxy, pixelOffsetForDepth);
                float depthThreshold = abs(zC - depth);
                float depthTolerance = depthSigma * depthThreshold + depthFloatPrecision;
                w_d = min(depthTolerance / (abs(zC - iDepth) + FLT_EPSILON), 1);

            }
            else
            {
                float depthThreshold = DepthThreshold(depth, ddxy, abs(pixelOffsetForDepth));
                float depthTolerance = depthSigma * depthThreshold + depthFloatPrecision;
                w_d = min(depthTolerance / (abs(depth - iDepth) + FLT_EPSILON), 1);
            }

            // ToDo Explain
            w_d *= w_d >= cb.depthWeightCutoff;
        }


        // Filter kernel weight.
        float w_h = FilterKernel::Kernel[row][col];
        

        float w = w_fa * w_s * w_h * w_n * w_x * w_d;
   

        weightedValueSum += w * iValue;
        weightSum += w;

        // ToDo standardize cb naming
        if (cb.outputFilteredVariance)
        {
            float iVariance = g_inVariance[id];
            weightedVarianceSum += w * w * iVariance;   // ToDo rename to sqWeight...
        }
    }
}

// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilterCS::ThreadGroup::Width, AtrousWaveletTransformFilterCS::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint2 Gid : SV_GroupID)
{
    if (!IsWithinBounds(DTid, cb.textureDim))
    {
        return;
    }

    // Initialize values to the current pixel / center filter kernel value.
    float value = g_inValues[DTid];
    float3 normal;
    float depth;
    DecodeNormalDepth(g_inNormalDepth[DTid], normal, depth);
    uint2 frameAgeRaysToGenerate = g_inFrameAge[DTid];
    uint frameAge = frameAgeRaysToGenerate.x;
    uint numRaysToGenerateOrDenoisePasses = frameAgeRaysToGenerate.y;

    
    bool isValidValue = value != RTAO::InvalidAOValue;
    float filteredValue = isValidValue && value < 0 ? -value : value;
    float variance = g_inSmoothedVariance[DTid];
    float filteredVariance = variance;

    if (depth != 0)
    {
        float w_c = 1;
#if RTAO_MARK_CACHED_VALUES_NEGATIVE
        if (isValidValue && value < 0)
        {
            // ToDo clean up, document or remove
            w_c = cb.staleNeighborWeightScale;
            value = -value;
        }
#endif

        float2 ddxy = g_inPartialDistanceDerivatives[DTid];

        float weightSum = 0;
        float weightedValueSum = 0;
        float weightedVarianceSum = 0;
        float stdDeviation = 1;

        if (isValidValue)
        {
            float pixelWeight = cb.weightByFrameAge ? frameAge : 1;
            weightSum = pixelWeight * FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius];
            weightedValueSum = weightSum * value;
            weightedVarianceSum = FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius] * FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius]
                * variance;
            stdDeviation = sqrt(variance);
        }

        // Calculate a kernel step given a ray hit distance.
        uint2 kernelStep = 1 << cb.kernelStepShift;
            // Blur more aggressively on smaller kernels.
            // ToDo remove?

        // Adaptive kernel size
        // Scale the kernel span by AO ray hit distance. 
        // This helps filter out lower frequency noise, a.k.a. boiling artifacts.
        float2 varianceSigmaScale = 1; 
        if (cb.useAdaptiveKernelSize)
        {
            float avgRayHitDistance = isValidValue ? g_inHitDistance[DTid] : 0;

            float perPixelViewAngle = (FOVY / cb.textureDim.y) * PI / 180; 
            float tan_a = tan(perPixelViewAngle);
            float2 projectedSurfaceDim = GetProjectedSurfaceDimensionsPerPixel(depth, ddxy, tan_a);

            // Calculate kernel width as a ratio of hitDistance / projected surface dim per pixel
            float k = 0.5 * cb.minHitDistanceToKernelWidthScale;
            kernelStep = max(1, round(k * avgRayHitDistance / projectedSurfaceDim));

            uint2 targetKernelStep = clamp(kernelStep, (cb.minKernelWidth - 1) / 2, (cb.maxKernelWidth - 1) / 2);
            uint2 adjustedKernelStep = cb.kernelStepShift > 0 ? lerp(1, targetKernelStep, (cb.kernelStepShift-1) / 5.0) : targetKernelStep;

            kernelStep = adjustedKernelStep;

            varianceSigmaScale = log2(kernelStep);
        }

        uint kernelStepShift = cb.kernelStepShift;

        if (variance >= cb.minVarianceToDenoise)
        {
            // Add contributions from the neighborhood.
            [unroll]
            for (UINT r = 0; r < FilterKernel::Width; r++)
            [unroll]
            for (UINT c = 0; c < FilterKernel::Width; c++)
                if (r != FilterKernel::Radius || c != FilterKernel::Radius)
                    AddFilterContribution(
                        weightedValueSum, 
                        weightedVarianceSum, 
                        weightSum, 
                        value, 
                        stdDeviation, 
                        depth, 
                        normal, 
                        ddxy, 
                        r, 
                        c, 
                        kernelStep, 
                        DTid, 
                        kernelStepShift, 
                        varianceSigmaScale);
        }

        float smallValue = 1e-6f;
        if (weightSum > smallValue)
        {
            filteredValue = weightedValueSum / weightSum;
            if (cb.outputFilteredVariance)
            {
                filteredVariance = weightedVarianceSum / (weightSum * weightSum);
            }
        }
        else
        {
            filteredValue = RTAO::InvalidAOValue;
            filteredVariance = 0;
        }
    }

    g_outFilteredValues[DTid] = filteredValue;
    if (cb.outputFilteredVariance)
    {
        g_outFilteredVariance[DTid] = filteredVariance;
    }
}