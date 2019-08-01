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
#include "RTAO\Shaders\RTAO.hlsli"

// ToDo some pixels here and there on mirror boundaries fail temporal reprojection even for static scene/camera
// ToDo sharp edges fail temporal reprojection due to clamping even for static scene

// ToDO pack value and depth beforehand?
// ToDo standardize in vs input, out vs output
Texture2D<float4> g_texInputCachedNormalDepth : register(t0);        // ToDo standardize cache vs cached
Texture2D<float> g_texInputCachedValue : register(t1);
Texture2D<float> g_texInputCurrentFrameValue : register(t2);
Texture2D<float4> g_texInputCurrentFrameNormalDepth : register(t3);
Texture2D<float4> g_texInputReprojectedNormalDepth : register(t4); // ToDo standardize naming across files
Texture2D<float2> g_texInputTextureSpaceMotionVector : register(t5); // ToDo standardize naming across files
Texture2D<uint> g_texInputCacheFrameAge : register(t6);

Texture2D<float2> g_texInputCurrentFrameLocalMeanVariance : register(t7);
Texture2D<float> g_texInputCachedCoefficientSquaredMean : register(t8);

Texture2D<float2> g_texInputCurrentFrameLinearDepthDerivative : register(t9); // ToDo standardize naming across files

// ToDo combine some outputs?
RWTexture2D<float> g_texOutputCachedValue : register(u0);
RWTexture2D<uint> g_texOutputCacheFrameAge : register(u1);
RWTexture2D<float> g_texOutputCoefficientSquaredMean: register(u2);
RWTexture2D<float> g_texOutputVariance: register(u3);
RWTexture2D<float4> g_texOutputDebug1 : register(u4);
RWTexture2D<float4> g_texOutputDebug2 : register(u5);

ConstantBuffer<RTAO_TemporalCache_ReverseReprojectConstantBuffer> cb : register(b0);

SamplerState LinearSampler : register(s0);
SamplerState ClampSampler : register(s1);

#define DEBUG_OUTPUT 0 // ToDo remove

// ToDo
// - Fix heavy disocclusion on min/magnifaction. Use bilateraly downsampled mip maps?
//   - this happens only when starting to move not on smooth move.


// Calculates a depth threshold for surface at angle beta from camera plane
// based on threshold for a surface at angle alpha
float CalculateAdjustedDepthThreshold(
    float d1,       // ddxy for surface at angle alpha from camera plane
    float alpha,    // angle in radians for surface with threshold d1 
    float beta,     // angle in radians for target surface
    float rho)      // view angle for the pixel
{
    // ToDo add derivation comment
    return d1
           * (sin(beta) / sin(alpha))
           * (cos(rho + alpha) / cos(rho + beta));
   
}

float CalculateAdjustedDepthThreshold(
    float d,          // ddxy for surface with normal
    float z,       // Linear depth for current frame
    float _z,      // Linear depth for prev frame
    float3 normal,    // normal for a surface with threshold d 
    float3 _normal)    // normal of a target surface
{
    float _d = d * _z / z;


    float3 forwardRay = GenerateForwardCameraRayDirection(cb.projectionToWorldWithCameraEyeAtOrigin);
    float3 _forwardRay = GenerateForwardCameraRayDirection(cb.prevProjectionToWorldWithCameraEyeAtOrigin);

    float alpha = acos(dot(normal, forwardRay));
    float beta = acos(dot(_normal, _forwardRay));

    float rho = (FOVY * PI / 180) * cb.invTextureDim.y;

    return CalculateAdjustedDepthThreshold(_d, alpha, beta, rho);
}

// ToDo cleanup with AdjustedDepth...
// ToDO move this to DDXY computation?
// ToDo rename to ddxy dxdy and standardize.
float DepthThreshold(float distance, float2 ddxy, float2 pixelOffset)
{
    float depthThreshold;
    // Todo rename ddxy to dxdy?
    // ToDo use a common helper
    // ToDo rename to: Perspective correct interpolation
    // Pespective correction for the non-linear interpolation
    if (cb.perspectiveCorrectDepthInterpolation)
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
        depthThreshold = dot(1, abs(zxy - z0)); // ToDo this should be sqrt(dot(zxy - z0, zxy - z0))?
    }
    else
    {
        depthThreshold = dot(1, abs(pixelOffset * ddxy));
    }

    return depthThreshold;
}

float4 BilateralResampleWeights(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float2 offset, in uint2 actualIndex, in int2 sampleIndices[4], in float cacheDdxy)
{
    bool4 isWithinBounds = bool4(
        IsWithinBounds(sampleIndices[0], cb.textureDim),
        IsWithinBounds(sampleIndices[1], cb.textureDim),
        IsWithinBounds(sampleIndices[2], cb.textureDim),
        IsWithinBounds(sampleIndices[3], cb.textureDim));

    float4 depthMask = 1;
    if (cb.useDepthWeights)
    {
        float depthThreshold = cacheDdxy;

// Using exact precision values fails the depth test on some views, particularly at smaller resolutions.
        // Scale the tolerance a bit.
        float depthFloatPrecision = FloatPrecision(ActualDistance, cb.DepthNumMantissaBits);
       
        float depthTolerance = cb.depthSigma * depthThreshold + depthFloatPrecision;
        float4 depthWeights = min(depthTolerance / (abs(SampleDistances - ActualDistance) + FLT_EPSILON), 1);
        // ToDo Should there be a distance falloff with a cutoff below 1?
        // ToDo revise the coefficient
        depthMask = depthWeights >= 0.5 ? depthWeights : 0;   // ToDo revise - this is same as comparing to depth tolerance

        // ToDo handle invalid distances, i.e disabled pixels?
        //weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0; // ToDo?

    }
    

    float4 normalWeights = 1;
    if (cb.useNormalWeights)
    {
        const uint normalExponent = 32;
        const float minNormalWeight = 1e-3f; // ToDo pass as parameter
        float normalSigma = 1.1;        // ToDo remove/finetune/document. There's some less than 1 weights even for same/very similar  normals (i.e. house wall tiles)(due to low bit format?)
        
        float4 NdotSampleN = float4(
            dot(ActualNormal, SampleNormals[0]),
            dot(ActualNormal, SampleNormals[1]),
            dot(ActualNormal, SampleNormals[2]),
            dot(ActualNormal, SampleNormals[3]));
        normalWeights = pow(saturate(normalSigma*NdotSampleN), normalExponent) >= minNormalWeight;
    }

    float4 bilinearWeights = 
        float4(
            (1 - offset.x) * (1 - offset.y),
            offset.x * (1 - offset.y),
            (1 - offset.x) * offset.y,
            offset.x * offset.y);

    // ToDo use depth weights instead of mask?
    // ToDo can we prevent diffusion across plane?
    float4 weights = isWithinBounds * bilinearWeights * depthMask * normalWeights;    // ToDo invalidate samples too pixel offcenter? <0.1


#if DEBUG_OUTPUT
    g_texOutputDebug1[actualIndex] = float4(isWithinBounds);
    g_texOutputDebug2[actualIndex] = float4(depthMask);
#endif

    return weights;
}



// ToDo use common helper
void LoadDepthAndNormal(Texture2D<float4> inNormalDepthTexture, in uint2 texIndex, out float depth, out float3 normal)
{
    if (cb.useNormalWeights)
    {
        float4 encodedNormalAndDepth = inNormalDepthTexture[texIndex];
        depth = encodedNormalAndDepth.z;
        normal = DecodeNormal(encodedNormalAndDepth.xy);
    }
}


float GetLinearDepth(float3 viewPosition, float3 forwardCameraRay)
{
    return dot(viewPosition, forwardCameraRay);
}


[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    float2 texturePos = (DTid.xy + 0.5f) * cb.invTextureDim;
    float2 cacheFrameTexturePos = texturePos - g_texInputTextureSpaceMotionVector[DTid];
    
    // Find the nearest integer index smaller than the texture position.
    // The floor() ensures the that value sign is taken into consideration.
    int2 topLeftCacheFrameIndex = floor(cacheFrameTexturePos * cb.textureDim - 0.5);

    // ToDo why this doesn't match cacheFrameTexturePos??
    float2 adjustedCacheFrameTexturePos = (topLeftCacheFrameIndex + 0.5) * cb.invTextureDim;

    float2 cachePixelOffset = cacheFrameTexturePos * cb.textureDim - topLeftCacheFrameIndex - 0.5;

    const int2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

    int2 cacheIndices[4] = {
        topLeftCacheFrameIndex + srcIndexOffsets[0],
        topLeftCacheFrameIndex + srcIndexOffsets[1],
        topLeftCacheFrameIndex + srcIndexOffsets[2],
        topLeftCacheFrameIndex + srcIndexOffsets[3] };
    // ToDo conditional loads if really needed?
    // ToDo use gather
    float3 cacheNormals[4]; 
    float4 vCacheDepths;
    {
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            float dummy;
            LoadDepthAndNormal(g_texInputCachedNormalDepth, cacheIndices[i], vCacheDepths[i], cacheNormals[i]);
        }
    }

#if 0
    // Calculate linear depth in the cache frame.
    // We avoid converting depth between linear and log as log depth loses a lot of precision very quickly as depth increases.
    float3 viewPos = ScreenPosToWorldPos(DTid, linearDepth, cb.textureDim, cb.zNear, float3(0,0,0), cb.projectionToWorldWithCameraEyeAtOrigin);

    // Add the camera translation change 
    viewPos += cb.prevToCurrentFrameCameraTranslation.xyz;

    float3 cameraDirection = GenerateForwardCameraRayDirection(cb.projectionToWorldWithCameraEyeAtOrigin);
    float3 cacheCameraDirection = GenerateForwardCameraRayDirection(cb.prevProjectionToWorldWithCameraEyeAtOrigin);
    float cacheLinearDepth = dot(viewPos, cacheCameraDirection);
#endif

    float2 dxdy = g_texInputCurrentFrameLinearDepthDerivative[DTid];
    // ToDo should this be done separately for both X and Y dimensions?
    float  ddxy = dot(1, dxdy);

    float3 _normal;
    float _depth;
    LoadDepthAndNormal(g_texInputReprojectedNormalDepth, DTid, _depth, _normal);


    // Account for 0.5 sample offset in bilateral downsampled partial depth derivative buffer.
    if (cb.usingBilateralDownsampledBuffers)
    {
        float2 pixelOffset = float2(1.5, 1.5);
        // ToDo review using _depth with current frame dxdy ...
        ddxy = DepthThreshold(_depth, dxdy, pixelOffset);
    }

    // ToDo retest/finetune depth testing. Moving car back and forth fails. Needs world space distance & depth sigma of 2+.
    float cacheDdxy = ddxy;
    if (cb.useWorldSpaceDistance)
    {
        float3 normal;
        float depth;
        LoadDepthAndNormal(g_texInputCurrentFrameNormalDepth, DTid, depth, normal);

        cacheDdxy = CalculateAdjustedDepthThreshold(ddxy, depth, _depth, normal, _normal);
    }

    float value = g_texInputCurrentFrameValue[DTid];
    float mergedValue;

    float4 weights = BilateralResampleWeights(_depth, _normal, vCacheDepths, cacheNormals, cachePixelOffset, DTid, cacheIndices, cacheDdxy);

    float4 vCacheValues = g_texInputCachedValue.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
#if 1
    bool4 isValidCacheValue = vCacheValues != RTAO::InvalidAOValue;

    //weights = isValidValue ? weights : 0;
    weights.x = vCacheValues.x != RTAO::InvalidAOValue ? weights.x : 0;
    weights.y = vCacheValues.y != RTAO::InvalidAOValue ? weights.y : 0;
    weights.z = vCacheValues.z != RTAO::InvalidAOValue ? weights.z : 0;
    weights.w = vCacheValues.w != RTAO::InvalidAOValue ? weights.w : 0;
#endif
    float weightSum = dot(1, weights);



#if 0
    // ToDo dedupe with GetClipSpacePosition()...
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 currentFrameTexturePos = xy * cb.invTextureDim;

    float aspectRatio = cb.textureDim.x / cb.textureDim.y;
    float maxScreenSpaceReprojectionDistance = 0.01;// cb.minSmoothingFactor * 0.1f; // ToDo
     ToDo scale this based on depth?
    float screenSpaceReprojectionDistanceAsWidthPercentage = min(1, length((currentFrameTexturePos - cacheFrameTexturePos) * float2(1, aspectRatio)));
#endif

    //&& screenSpaceReprojectionDistanceAsWidthPercentage <= maxScreenSpaceReprojectionDistance;
    uint frameAge;
    float mergedValueSquaredMean;      // ToDo better prefix than merged?
    float outVariance;

    
    uint maxFrameAge = 1 / cb.minSmoothingFactor - 1;// minSmoothingFactor;

    float2 localMeanVariance = g_texInputCurrentFrameLocalMeanVariance[DTid];
    float localMean = localMeanVariance.x;
    float localVariance = localMeanVariance.y;

    bool isCacheValueValid = weightSum > 1e-3f; // ToDo
    if (isCacheValueValid)
    {
        uint4 vCacheFrameAge = g_texInputCacheFrameAge.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
        float4 vCacheValueSquaredMean = g_texInputCachedCoefficientSquaredMean.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;

        float4 nWeights = weights / weightSum;   // Normalize the weights.
        float cachedValue = dot(nWeights, vCacheValues);
        float cachedValueSquaredMean = dot(nWeights, vCacheValueSquaredMean);


        // ToDo revisit this and potentially make it UI adjustable - weight ^ 2 ?,...
        // Scale the frame age by the total weight. This is to keep the frame age low for 
        // total contributions that have very low reprojection weight. While its preferred to get 
        // a weighted value even for reprojections that have low weights but still
        // satisfy consistency tests, the frame age needs to be kept small so that the actual calculated values
        // are quickly filled in over a few frames. Otherwise, bad estimates from reprojections,
        // such as on disocclussions of surfaces on rotation, are kept around long enough to create 
        // visible streaks that fade away very slow.
        // Example: rotating camera around dragon's nose up close. 
        float frameAgeScale = 1;// saturate(weightSum);

        float cacheFrameAge = frameAgeScale * dot(nWeights, vCacheFrameAge);
        frameAge = round(cacheFrameAge);

        // Clamp value to mean +/- std.dev of local neighborhood to surpress ghosting on value changing due to other occluder movements.
        // Ref: Salvi2016, Temporal Super-Sampling
        float frameAgeClamp = 0;


        if (cb.clampCachedValues)
        {
            float localStdDev = max(cb.stdDevGamma * sqrt(localVariance), cb.minStdDevTolerance);
            float nonClampedCachedValue = cachedValue;
            cachedValue = clamp(cachedValue, localMean - localStdDev, localMean + localStdDev); 
            
            // Scale the frame age based on how strongly the cached value got clamped.
            // ToDo avoid saturate?
            frameAgeClamp = saturate(cb.clampDifferenceToFrameAgeScale * abs(cachedValue - nonClampedCachedValue));
            // ToDo round to nearest integer?
            frameAge = lerp(frameAge, 0, frameAgeClamp);
        }
        //frameAgeClamp = screenSpaceReprojectionDistanceAsWidthPercentage / maxScreenSpaceReprojectionDistance;
        //uint maxFrame
        //frameAge = lerp(frameAge, 0, frameAgeClamp);

        // ToDo: use moving average (Koskela2019) for the first few samples 
        // to even out the weights for the noisy start instead of weighting first samples much more.
        float invFrameAge = 1.f / (frameAge + 1.f);
        float a = cb.forceUseMinSmoothingFactor ? cb.minSmoothingFactor : max(invFrameAge, cb.minSmoothingFactor);

        bool isValidValue = value != RTAO::InvalidAOValue;

        mergedValue = isValidValue ? lerp(cachedValue, value, a) : cachedValue;

        // ToDo remove/fix
        //mergedValue = saturate(mergedValue);
        // ToDo hack - remove
        mergedValue = isValidValue ? mergedValue : -mergedValue;
        mergedValueSquaredMean = isValidValue ? lerp(cachedValueSquaredMean, value * value, a) : cachedValueSquaredMean;


        float temporalVariance = mergedValueSquaredMean - mergedValue * mergedValue;
        temporalVariance = max(0, temporalVariance);    // Ensure variance doesn't go negative due to imprecision.

        outVariance = frameAge >= cb.minFrameAgeToUseTemporalVariance ? temporalVariance : localVariance;
        // ToDo If no valid samples found:
        //  - use largest motion vector from 3x3
        //  - try 3x3 area
        //  - default to average?

        frameAge = isValidValue ? min(frameAge + 1, maxFrameAge) : frameAge;
    }
    else // ToDo initialize values to this instead of branch?
    {
        //ToDo interpolate from neighbors
        bool isValidValue = value != RTAO::InvalidAOValue;
        mergedValue = value;
        mergedValueSquaredMean = isValidValue ? value * value : RTAO::InvalidAOValue;
        frameAge = isValidValue ? 1 : 0;
        outVariance = localVariance;
    }
    g_texOutputCachedValue[DTid] = mergedValue;
    g_texOutputCacheFrameAge[DTid] = frameAge;
    g_texOutputCoefficientSquaredMean[DTid] = mergedValueSquaredMean;
    g_texOutputVariance[DTid] = outVariance;

}