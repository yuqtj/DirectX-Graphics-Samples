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


// ToDo
// Optimization
// - Frontload/limit to indices of valid neighbors to avoid stalling threads randomly on sparse inputs.

Texture2D<float> g_inValues : register(t0); // ToDo input is 3841x2161 instead of 2160p..

Texture2D<NormalDepthTexFormat> g_inNormalDepth : register(t1);
Texture2D<float> g_inVariance : register(t4);   // ToDo remove
Texture2D<float> g_inSmoothedVariance : register(t5); 
Texture2D<float> g_inHitDistance : register(t6);   // ToDo remove?
Texture2D<float2> g_inPartialDistanceDerivatives : register(t7);   // ToDo remove?
Texture2D<uint2> g_inFrameAge : register(t8);

RWTexture2D<float> g_outFilteredValues : register(u0);
RWTexture2D<float> g_outFilteredVariance : register(u1);
#if !WORKAROUND_ATROUS_VARYING_OUTPUTS 
RWTexture2D<float> g_outFilterWeightSum : register(u2);
#endif
RWTexture2D<float4> g_outDebug1 : register(u3);
RWTexture2D<float4> g_outDebug2 : register(u4);

ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> g_CB: register(b0);

#define MAX_FRAME_AGE 32    // ToDo pass

float DepthThreshold(float distance, float2 ddxy, float2 pixelOffset)
{
    float depthThreshold;

    float fEpsilon = (1.001-distance) * g_CB.depthSigma * 1e-4f;// depth * 1e-6f;// *0.001;// 0.0024;// 12f;     // ToDo finalize the value
    // ToDo rename to perspective correction
#if 0
    if (0 && g_CB.perspectiveCorrectDepthInterpolation)
    {
        float fovAngleY = FOVY;   // ToDO pass from the app
        float2 resolution = g_CB.textureDim;

        // adjust the depth threshold based on slope angle.
        float pixelOffsetLen = length(pixelOffset);
        float unitDistanceDelta = length(pixelOffset * ddxy) / pixelOffsetLen;
        float slopeAngle = asin(obliqueness);// atan(1 / unitDistanceDelta);
        float pixelAngle = pixelOffsetLen * (fovAngleY / resolution.y) * PI / 180;
        depthThreshold = distance * (((sin(slopeAngle) / sin(slopeAngle - pixelAngle)) - 1) + fEpsilon);
    }
    else
#endif
    {
#if 1
        // Todo rename ddxy to dxdy?
        // ToDo use a common helper
        // ToDo rename to: Perspective correct interpolation
        // Pespective correction for the non-linear interpolation
        if (g_CB.perspectiveCorrectDepthInterpolation)
        {
            // Calculate depth via interpolation with perspective correction
            // Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/visibility-problem-depth-buffer-depth-interpolation
            // Given depth buffer interpolation for finding z at offset q along z0 to z1
            //      z =  1 / (1 / z0 * (1 - q) + 1 / z1 * q)
            // and z1 = z0 + ddxy, where z1 is at a unit pixel offset [1, 1]
            // z can be calculated via ddxy as
            //
            //      z = (z0 + ddxy) / (1 + (1-q) / z0 * ddxy) 
         
            float z0 = distance;
            float2 zxy = (z0 + ddxy) / (1 + ((1 - pixelOffset) / z0) * ddxy);
            depthThreshold = dot(1, abs(zxy - z0));
        }
        else
        {
            depthThreshold = dot(1, abs(pixelOffset * ddxy));
        }
#else
        depthThreshold = length(pixelOffset * ddxy);
#endif
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
    in float weightScale,
    in float2 varianceSigmaScale)
{

    const float valueSigma = g_CB.valueSigma;
    const float normalSigma = g_CB.normalSigma;
    const float depthSigma = g_CB.depthSigma;
 
    int2 pixelOffset;
    float kernelWidth;
    float varianceScale = 1;

    // ToDo
    // Denoising improvemnts
    // - 
#if 0
    // ToDo
    // RTX Gems p314 - scale kernel width on avg hit distance and tspp. Higher tspp lower kernel.
    // https://research.nvidia.com/sites/default/files/pubs/2018-05_Combining-Analytic-Direct//I3D2018_combining.pdf
    // - scale kernel based on variance
    if (g_CB.useAdaptiveKernelSize)
    {
        // Calculate kernel width as a ratio of hitDistance / projected surface width per pixel
        float perPixelViewAngle = (FOVY / g_CB.textureDim.y) * PI / 180;
        // ToDo finetune min oblique parameter.
        float projectedSurfaceScale = max(0.4, obliqueness); // Avoid having very low kernel widths at low oblique surfaces. This limits filtering and creates noticeable noise.
        kernelWidth = g_CB.minHitDistanceToKernelWidthScale * minHitDistance * projectedSurfaceScale / (2 * depth * tan(perPixelViewAngle / 2)); // ToDo review math here.
        
        kernelWidth = clamp(kernelWidth, g_CB.minKernelWidth, g_CB.maxKernelWidth);

        float nIterations = 5;


        // Blur more aggressively on smaller kernels.
        // ToDo remove?
        varianceScale = lerp(g_CB.varianceSigmaScaleOnSmallKernels, 1, saturate((kernelWidth - g_CB.minKernelWidth) / 33));

        // Calculate pixel offset per iteration.
        float maxKernelRadius = ceil((kernelWidth - 1) / 2);
#if 1
        // Lower the maxKernelRadius a notch not to overshoot on last iteration with ceil when calculating the pixel offset delta.
        float stepBase = pow(maxKernelRadius - 0.1, 1 / (nIterations - 1));
        float curPixelOffsetDelta = max(g_CB.kernelStepShift + 1, ceil(pow(stepBase, g_CB.kernelStepShift)));
#else
        float pixelOffsetStep = max(1, maxKernelRadius / nIterations);
        float curPixelOffsetDelta = floor((g_CB.kernelStepShift + 1) * pixelOffsetStep);
#endif
        if (curPixelOffsetDelta > maxKernelRadius)
        {
            return;
        }
        pixelOffset = int2(row - FilterKernel::Radius, col - FilterKernel::Radius) * curPixelOffsetDelta;
    }
    else
    {
        pixelOffset = int2(row - FilterKernel::Radius, col - FilterKernel::Radius) << kernelStepShift;
    }
#endif
    pixelOffset = int2(row - FilterKernel::Radius, col - FilterKernel::Radius) * kernelStep;

    //varianceScale = lerp(1, 0.1, dot(1, abs(pixelOffset))/33);
    //varianceScale = 1.0 / (1 + dot(abs(pixelOffset), varianceSigmaScale));
    int2 id = int2(DTid) + pixelOffset;

    if (IsWithinBounds(id, g_CB.textureDim))
    {
        float iDepth;
        float3 iNormal;
        DecodeNormalDepth(g_inNormalDepth[id], iNormal, iDepth);
        float iValue = g_inValues[id];

        if (iValue == RTAO::InvalidAOValue ||
            iDepth == 0)
        {
            return;
        }

        float w_c = 1;

#if RTAO_MARK_CACHED_VALUES_NEGATIVE
        if (iValue < 0)
        {
            w_c = g_CB.staleNeighborWeightScale;// 0.065;
            iValue = -iValue;
        }
#endif
        const float errorOffset = 0.005f;
        float e_x = valueSigma  > 0.001f ? -abs(value - iValue) / (varianceScale * valueSigma * stdDeviation + errorOffset) : 0;
 
        // ToDo loosen up weights for low frameAge? and/or 2nd+ pass
        // ToDo standardize index vs id
        // Ref: SVGF
        // ToDo
        float w_n = pow(max(0, dot(normal, iNormal)), normalSigma);

        // ToDo explain 1 -
        // Make the 0 start at 1 == depthDelta/depthTolerance
        // ToDo finalize obliqueness
        // ToDo obliqueness is incorrect for reflected rays
        //float minObliqueness = depthSigma;//  0.02; // Avoid weighting by depth at very sharp angles. Depend on weighting by normals.
        float2 pixelOffsetForDepth = pixelOffset;
        
        // ToDo use actual pixel offsets from bilateral downsample?
        // Account for sample offset in bilateral downsampled partial depth derivative buffer.
        // 
        if (g_CB.usingBilateralDownsampledBuffers)
        {
            float2 offsetSign = sign(pixelOffset);
            pixelOffsetForDepth = pixelOffset + offsetSign * float2(0.5, 0.5);
        }
        float depthThreshold = DepthThreshold(depth, ddxy, abs(pixelOffsetForDepth));

#if 1
        float depthFloatPrecision = FloatPrecision(max(depth, iDepth), g_CB.DepthNumMantissaBits);

        float depthTolerance = depthSigma * depthThreshold + depthFloatPrecision;
     
        float w_d = 1;
        if (depthSigma > 0.01f)
        {
            if (g_CB.useProjectedDepthTest)
            {
                float zC = GetDepthAtPixelOffset(depth, ddxy, pixelOffsetForDepth);
                float depthThreshold = abs(zC - depth);
                float depthTolerance = depthSigma * depthThreshold + depthFloatPrecision;
                w_d = min(depthTolerance / (abs(zC - iDepth) + FLT_EPSILON), 1);

                if (pixelOffset.x == 0 && pixelOffset.y - 1)
                {
                    //g_outDebug1[DTid] = float4(zC, depthThreshold, depthTolerance, w_d);
                }

            }
            else
            {
                float depthSigma = 1; // TODO remove
                float depthTolerance = depthSigma * depthThreshold + depthFloatPrecision;
                w_d = min(depthTolerance / (abs(depth - iDepth) + FLT_EPSILON), 1);
            }
        }
        //float e_d = depthSigma > 0.01f ? -abs(depth - iDepth) / (depthTolerance + FLT_EPSILON) : 0;
        w_d *= w_d >= g_CB.depthWeightCutoff;
#else   
        float fMinEpsilon = 512 * FLT_EPSILON; // Minimum depth threshold epsilon to avoid acne due to ray/triangle floating precision limitations.
        float fMinDepthScaledEpsilon = 48 * 1e-6  * depth;  // Depth threshold to surpress differences that surface at larger depth from the camera.
        float fEpsilon = fMinEpsilon + fMinDepthScaledEpsilon;
        // ToDo revvise divEpsilon
        float divEpsilon = 1e-6f;
        float depthWeight = min((depthSigma * depthThreshold + fEpsilon) / (abs(depth - iDepth) + divEpsilon), 1);
        float w_d = exp(e_d);
#endif

        float w_h = FilterKernel::Kernel[row][col];

        float w_x =  exp(e_x);
        float w_xd = w_x * w_d;

        float w = w_h * w_n *w_xd;
        w *= w_c;// *weightScale;


        uint iFrameAge = 1;// g_inFrameAge[id].x;
        // Enforce frame age of at least 1 for reprojection for valid values.
        // This is because the denoiser will fill in invalid values with filtered 
        // ones if it can. But it doesn't increase frame age.
        iFrameAge = max(iFrameAge, 1);

        float iPixelWeight = iFrameAge;
        w *= iPixelWeight;
        
        weightedValueSum += w * iValue;
        weightSum += w;


        // ToDo standardize g_CB naming
        if (g_CB.outputFilteredVariance)
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
    // ToDo add early exit if this pixel is processing inactive result.
    // ToDo double check all CS for out of bounds.
    if (!IsWithinBounds(DTid, g_CB.textureDim))
    {
        return;
    }

    float depth;
    float3 normal;
    DecodeNormalDepth(g_inNormalDepth[DTid], normal, depth);
    uint2 frameAgeRaysToGenerate = g_inFrameAge[DTid];
    uint frameAge = frameAgeRaysToGenerate.x;
    uint numRaysToGenerateOrDenoisePasses = frameAgeRaysToGenerate.y;

    bool isRayCountValue = !(numRaysToGenerateOrDenoisePasses & 0x80);
    bool doDenoisingPass = isRayCountValue ? true : 0x7F & numRaysToGenerateOrDenoisePasses;
    
    float value = g_inValues[DTid];
    bool isValidValue = value != RTAO::InvalidAOValue;
    float filteredValue = isValidValue && value < 0 ? -value : value;
    float variance = g_inSmoothedVariance[DTid];
    float filteredVariance = variance;

    if (depth != 0 && (doDenoisingPass || g_CB.forceDenoisePass))
//        frameAge <= g_CB.maxFrameAgeToDenoise)
    {
        // Slow start fading away denoising strenght half way through.
        //float t = (2 * max(frameAge, (g_CB.maxFrameAgeToDenoise + 1) / 2) - g_CB.maxFrameAgeToDenoise) / g_CB.maxFrameAgeToDenoise;
        float neighborWeightScale = 1;// g_CB.weightScale; // g_CB.normalSigma < 64 ? g_CB.weightScale * lerp(1, 0, t) : 1;  // ToDo cleanup

        float w_c = 1;
#if RTAO_MARK_CACHED_VALUES_NEGATIVE
        if (value < 0)
        {
            // ToDo
            //w_c = g_CB.staleNeighborWeightScale;
            value = -value;
        }
#endif

        float2 ddxy = g_inPartialDistanceDerivatives[DTid];

        float2 pixelOffsetForDepth = float2(0, 1);
        if (g_CB.usingBilateralDownsampledBuffers)
        {
            pixelOffsetForDepth += float2(0.5, 0.5);
        }
#if 0
        float depthThreshold = DepthThreshold(depth, ddxy, pixelOffsetForDepth);

        float depthFloatPrecision = FloatPrecision(depth, g_CB.DepthNumMantissaBits);

        float depthTolerance = depthThreshold + depthFloatPrecision;

        g_outDebug1[DTid] = float4(
            depthThreshold,
            depthFloatPrecision,
            depthTolerance,
            0);
#endif

        float weightSum = 0;
        float weightedValueSum = 0;
        float weightedVarianceSum = 0;
        float stdDeviation = 1;

        if (isValidValue)
        {
            float pixelWeight = 1;// frameAge;
            weightSum = pixelWeight * FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius];
            weightedValueSum = weightSum * value;
            weightedVarianceSum = FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius] * FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius]
                * variance;
            stdDeviation = sqrt(variance);
        }

        // Calculate a kernel step given a ray hit distance.
        uint2 kernelStep = 1 << g_CB.kernelStepShift;
            // Blur more aggressively on smaller kernels.
            // ToDo remove?

        float2 varianceSigmaScale = 1; 
        if (frameAge > 16 && g_CB.useAdaptiveKernelSize)
        {
            float avgRayHitDistance = g_inHitDistance[DTid];

            float perPixelViewAngle = (FOVY / g_CB.textureDim.y) * PI / 180; 
            float tan_a = tan(perPixelViewAngle);
            float2 projectedSurfaceDim = GetProjectedSurfaceDimensionsPerPixel(depth, ddxy, tan_a);

            // Calculate kernel width as a ratio of hitDistance / projected surface dim per pixel
            float k = 0.5 * g_CB.minHitDistanceToKernelWidthScale;
            kernelStep = max(1, round(k * avgRayHitDistance / projectedSurfaceDim));


            uint2 targetKernelStep = clamp(kernelStep, g_CB.minKernelWidth / 2, g_CB.maxKernelWidth / 2);
            kernelStep = lerp(1, targetKernelStep, g_CB.kernelStepShift / 10.0);
            g_outDebug1[DTid] = float4(projectedSurfaceDim, kernelStep);
            g_outDebug2[DTid] = float4(projectedSurfaceDim, kernelStep);

            varianceSigmaScale = log2(kernelStep);
        }

        uint kernelStepShift = g_CB.kernelStepShift;// frameAge >= 8 ? g_CB.kernelStepShift : (frameAge + 2) % 3;

        if (variance >= g_CB.minVarianceToDenoise)
        {
            // Add contributions from the neighborhood.
            [unroll]
            for (UINT r = 0; r < FilterKernel::Width; r++)
                [unroll]
            for (UINT c = 0; c < FilterKernel::Width; c++)
                if (r != FilterKernel::Radius || c != FilterKernel::Radius)
                    AddFilterContribution(weightedValueSum, weightedVarianceSum, weightSum, value, stdDeviation, depth, normal, ddxy, r, c, kernelStep, DTid, kernelStepShift, neighborWeightScale, varianceSigmaScale);
        }

        float smallValue = 1e-6f;
        if (weightSum > smallValue)
        {
            //float filteredValue = weightSum > (FilterKernel::Kernel[FilterKernel::Radius][FilterKernel::Radius] + 0.00001) ? weightedValueSum / weightSum : valueSum / numValues;
            filteredValue = weightedValueSum / weightSum;
            filteredVariance = weightedVarianceSum / (weightSum * weightSum);
        }
        else
        {
            filteredValue = RTAO::InvalidAOValue;
            filteredVariance = 0;
        }
    }

    g_outFilteredValues[DTid] = filteredValue;
    if (g_CB.outputFilteredVariance)
    {
        g_outFilteredVariance[DTid] = filteredVariance;
    }

}