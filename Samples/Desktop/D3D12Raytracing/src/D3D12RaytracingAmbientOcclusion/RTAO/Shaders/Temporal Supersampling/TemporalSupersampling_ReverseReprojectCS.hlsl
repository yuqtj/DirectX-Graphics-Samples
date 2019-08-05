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

// ToDo Desc
// Desc: Sample temporal cache via reverse reprojection.
// If no valid values have been retrieved from the cache, the frameAge is set to 0.

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "RTAO\Shaders\RTAO.hlsli"

// ToDo some pixels here and there on mirror boundaries fail temporal reprojection even for static scene/camera
// ToDo sharp edges fail temporal reprojection due to clamping even for static scene

// ToDO pack value and depth beforehand?
// ToDo standardize in vs input, out vs output
Texture2D<NormalDepthTexFormat> g_texInputCurrentFrameNormalDepth : register(t0);
// ToDo should ddxy be calculated from reprojectedNormalDepth?
Texture2D<float2> g_texInputCurrentFrameLinearDepthDerivative : register(t1); // ToDo standardize naming across files

Texture2D<NormalDepthTexFormat> g_texInputReprojectedNormalDepth : register(t2);  // ToDo add encoded prefix
Texture2D<float2> g_texInputTextureSpaceMotionVector : register(t3);

Texture2D<NormalDepthTexFormat> g_texInputCachedNormalDepth : register(t4);
Texture2D<float> g_texInputCachedValue : register(t5);  // ToDo store 1bit 0/1 in an auxilary reseource instead?
Texture2D<uint> g_texInputCachedFrameAge : register(t6);
Texture2D<float> g_texInputCachedValueSquaredMean : register(t7);
Texture2D<float> g_texInputCachedRayHitDistance : register(t8);

// ToDo combine some outputs?
RWTexture2D<uint> g_texOutputCachedFrameAge : register(u0);
RWTexture2D<float4> g_texOutputReprojectedCachedValues : register(u1);


// ToDo remove
RWTexture2D<float4> g_texOutputDebug1 : register(u10);
RWTexture2D<float4> g_texOutputDebug2 : register(u11);

ConstantBuffer<RTAO_TemporalSupersampling_ReverseReprojectConstantBuffer> cb : register(b0);

SamplerState ClampSampler : register(s0);

// ToDo
// - Fix heavy disocclusion on min/magnifaction. Use bilateraly downsampled mip maps?
//   - this happens only when starting to move not on smooth move.
// standardize naming cache vs cached
// Optimizations:
//  - split into several passes>?
//  - condition to only necessary reads on frameAge 1 and/or invalid value
//  - on 0 motion vector read in only 1 cached value

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
    float d,            // ddxy for surface with normal
    float z,            // Linear depth for current frame
    float _z,           // Linear depth for prev frame
    float3 normal,      // normal for a surface with threshold d 
    float3 _normal)     // normal of a target surface
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

        // ToDo test, tighten the formats
// Using exact precision values fails the depth test on some views, particularly at smaller resolutions.
        // Scale the tolerance a bit.
        float depthFloatPrecision = FloatPrecision(ActualDistance, cb.DepthNumMantissaBits);

        float depthTolerance = cb.depthSigma * depthThreshold + depthFloatPrecision;
        float4 depthWeights = min(depthTolerance / (abs(SampleDistances - ActualDistance) + FLT_EPSILON), 1);
        // ToDo Should there be a distance falloff with a cutoff below 1?
        // ToDo revise the coefficient
        depthMask = depthWeights >= 0.5 ? depthWeights : 0;   // ToDo revise - this is same as comparing to depth tolerance

        //g_texOutputDebug1[actualIndex] = float4(depthTolerance, ActualDistance, SampleDistances.xy);
        //g_texOutputDebug2[actualIndex] = depthWeights;
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
        normalWeights = pow(saturate(normalSigma * NdotSampleN), normalExponent) >= minNormalWeight;
    }

    float4 bilinearWeights =
        float4(
        (1 - offset.x) * (1 - offset.y),
            offset.x * (1 - offset.y),
            (1 - offset.x) * offset.y,
            offset.x * offset.y);

    // Clamp to pixels on tiny offsets.
    bilinearWeights *= bilinearWeights >= 1e-3f;


    // ToDo use depth weights instead of mask?
    // ToDo can we prevent diffusion across plane?
    float4 weights = isWithinBounds * bilinearWeights * depthMask * normalWeights;    // ToDo invalidate samples too pixel offcenter? <0.1

    // Avoid imprecision issues.
    weights = weights > 0.999 ? 1 : weights;
    weights = weights < 0.001 ? 0 : weights;

    return weights;
}

float4 BilateralResampleWeights2(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float2 pixelOffset, in uint2 actualIndex, in int2 sampleIndices[4], in float2 dxdy)
{
    bool4 isWithinBounds = bool4(
        IsWithinBounds(sampleIndices[0], cb.textureDim),
        IsWithinBounds(sampleIndices[1], cb.textureDim),
        IsWithinBounds(sampleIndices[2], cb.textureDim),
        IsWithinBounds(sampleIndices[3], cb.textureDim));

    bool4 isActive = SampleDistances != 0;

    float4 depthWeights = 1;
    if (cb.useDepthWeights)
    {        
        // ToDo revise
        // Account for the pixel offset due to dxdy being bilaterally downsampled.
        // Since the source and target pixel could have been up to 1 pixel away in the higher resolution
        // resource, it could be up to 1.5 pixel away in the lower/quarter resolution.
        float pOffset = 0;
        if (cb.usingBilateralDownsampledBuffers)
        {
            pOffset = 0.5;
        }

        // Get sample pixel offsets from the actualPixel given a pixelOffset from the top-left one.
        float4x2 samplePixelOffsets = {
            float2(-pOffset,-pOffset) - pixelOffset,
            float2(1 + pOffset, -pOffset) - pixelOffset,
            float2(-pOffset,1 + pOffset) - pixelOffset,
            float2(1 + pOffset, 1 + pOffset) - pixelOffset,
        };

        // Calculate expected depths at sample pixels given current depth, dxdy and an offset to the sample pixels.
        float4 vExpectedDepths;
        [unroll]
        for (uint i=0; i < 4; i++)
            vExpectedDepths[i] = GetDepthAtPixelOffset(ActualDistance, dxdy, samplePixelOffsets[i]);
       
        float4 vDepthThresholds = abs(vExpectedDepths - ActualDistance);
        float depthFloatPrecision = FloatPrecision(ActualDistance, cb.DepthNumMantissaBits);
        float4 vDepthTolerances = cb.depthSigma * vDepthThresholds + depthFloatPrecision;

        float fEpsilon = 1e-6 * ActualDistance;
        depthWeights = min(vDepthTolerances / (abs(SampleDistances - vExpectedDepths) + fEpsilon), 1);
        g_texOutputDebug2[actualIndex] = depthWeights;
        // ToDo Should there be a distance falloff with a cutoff below 1?
        // ToDo revise the coefficient
        depthWeights *= depthWeights >= 0.5;   // ToDo revise - this is same as comparing to depth tolerance
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
        normalWeights = pow(saturate(normalSigma * NdotSampleN), normalExponent) >= minNormalWeight;
    }

    float4 bilinearWeights =
        float4(
        (1 - pixelOffset.x) * (1 - pixelOffset.y),
            pixelOffset.x * (1 - pixelOffset.y),
            (1 - pixelOffset.x) * pixelOffset.y,
            pixelOffset.x * pixelOffset.y);

    // ToDo use depth weights instead of mask?
    // ToDo can we prevent diffusion across plane?
    float4 weights = isWithinBounds * isActive * bilinearWeights * depthWeights * normalWeights;    // ToDo invalidate samples too pixel offcenter? <0.1

    return weights;
}


[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    float3 _normal;
    float _depth;
    DecodeNormalDepth(g_texInputReprojectedNormalDepth[DTid], _normal, _depth);
    float2 textureSpaceMotionVector = g_texInputTextureSpaceMotionVector[DTid];

    //g_texOutputDebug1[DTid] = float4(_normal, _depth);
    // ToDo compare against common predefined value
    if (_depth == 0 || textureSpaceMotionVector.x > 1e2f)
    {
        g_texOutputCachedFrameAge[DTid] = 0;
        return;
    }

    float2 texturePos = (DTid.xy + 0.5f) * cb.invTextureDim;
    float2 cacheFrameTexturePos = texturePos - textureSpaceMotionVector;

    // Find the nearest integer index smaller than the texture position.
    // The floor() ensures the that value sign is taken into consideration.
    int2 topLeftCacheFrameIndex = floor(cacheFrameTexturePos * cb.textureDim - 0.5);

    // ToDo why this doesn't match cacheFrameTexturePos??
    float2 adjustedCacheFrameTexturePos = (topLeftCacheFrameIndex + 0.5) * cb.invTextureDim;

    float2 cachePixelOffset = cacheFrameTexturePos * cb.textureDim - 0.5 - topLeftCacheFrameIndex;

    const int2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

    int2 cacheIndices[4] = {
        topLeftCacheFrameIndex + srcIndexOffsets[0],
        topLeftCacheFrameIndex + srcIndexOffsets[1],
        topLeftCacheFrameIndex + srcIndexOffsets[2],
        topLeftCacheFrameIndex + srcIndexOffsets[3] };
    // ToDo conditional loads if really needed?
    float3 cacheNormals[4];
    float4 vCacheDepths;
#if NORMAL_DEPTH_R8G8B16_ENCODING
    {
        uint4 packedEncodedNormalDepths = g_texInputCachedNormalDepth.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            DecodeNormalDepth(packedEncodedNormalDepths[i], cacheNormals[i], vCacheDepths[i]);
        }
    }
#else
    {
        float4 encodedNormalX = g_texInputCachedNormalDepth.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
        float4 encodedNormalY = g_texInputCachedNormalDepth.GatherGreen(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
        [unroll]
        for (int i = 0; i < 4; i++)
        {
            cacheNormals[i] = DecodeNormal(float2(encodedNormalX[i], encodedNormalY[i]));
        }

        vCacheDepths = g_texInputCachedNormalDepth.GatherBlue(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
    }
#endif

    float2 dxdy = abs(g_texInputCurrentFrameLinearDepthDerivative[DTid]);
    // ToDo should this be done separately for both X and Y dimensions?
    // ToDo adjust ddxy for each cache pixel using pixel offsets to that pixel index?
    float  ddxy = dot(1, dxdy);


    // Account for 0.5 sample offset in bilateral downsampled partial depth derivative buffer.
    if (cb.usingBilateralDownsampledBuffers)
    {
        float2 pixelOffset = float2(1.5, 1.5);
        // ToDo review using _depth with current frame dxdy ...
        ddxy = DepthThreshold(_depth, dxdy, pixelOffset);
    }

    // ToDo retest/finetune depth testing. Moving car back and forth fails. Needs world space distance & depth sigma of 2+.
    // If not needed, remove along with the texture
    float cacheDdxy = ddxy;
    if (cb.useWorldSpaceDistance)
    {
        float3 normal;
        float depth;
        DecodeNormalDepth(g_texInputCurrentFrameNormalDepth[DTid], normal, depth);
        cacheDdxy = CalculateAdjustedDepthThreshold(ddxy, depth, _depth, normal, _normal);
    }

    float4 weights = BilateralResampleWeights(_depth, _normal, vCacheDepths, cacheNormals, cachePixelOffset, DTid, cacheIndices, cacheDdxy);
    //float4 weights = BilateralResampleWeights2(_depth, _normal, vCacheDepths, cacheNormals, cachePixelOffset, DTid, cacheIndices, dxdy);

    // Invalidate weights for invalid values in the cache.
    float4 vCacheValues = g_texInputCachedValue.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
    weights = vCacheValues != RTAO::InvalidAOValue ? weights : 0;

    float weightSum = dot(1, weights);
    
    float cachedValue = RTAO::InvalidAOValue;
    float cachedValueSquaredMean = 0;
    float cachedRayHitDistance = 0;

#if 0
    // ToDo dedupe with GetClipSpacePosition()...
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 currentFrameTexturePos = xy * cb.invTextureDim;

    float aspectRatio = cb.textureDim.x / cb.textureDim.y;
    float maxScreenSpaceReprojectionDistance = 0.01;// cb.minSmoothingFactor * 0.1f; // ToDo
    ToDo scale this based on depth ?
        float screenSpaceReprojectionDistanceAsWidthPercentage = min(1, length((currentFrameTexturePos - cacheFrameTexturePos) * float2(1, aspectRatio)));
    //&& screenSpaceReprojectionDistanceAsWidthPercentage <= maxScreenSpaceReprojectionDistance;

   //frameAgeClamp = screenSpaceReprojectionDistanceAsWidthPercentage / maxScreenSpaceReprojectionDistance;
   //uint maxFrame
   //frameAge = lerp(frameAge, 0, frameAgeClamp);
#endif
    uint frameAge;
    bool areCacheValuesValid = weightSum > 1e-3f; // ToDo
    if (areCacheValuesValid)
    {
        uint4 vCachedFrameAge = g_texInputCachedFrameAge.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
        float4 nWeights = weights / weightSum;   // Normalize the weights.

        g_texOutputDebug1[DTid] = float4(weights.xyz, weightSum);
        g_texOutputDebug2[DTid] = nWeights;

        // ToDo revisit this and potentially make it UI adjustable - weight ^ 2 ?,...
        // Scale the frame age by the total weight. This is to keep the frame age low for 
        // total contributions that have very low reprojection weight. While its preferred to get 
        // a weighted value even for reprojections that have low weights but still
        // satisfy consistency tests, the frame age needs to be kept small so that the actual calculated values
        // are quickly filled in over a few frames. Otherwise, bad estimates from reprojections,
        // such as on disocclussions of surfaces on rotation, are kept around long enough to create 
        // visible streaks that fade away very slow.
        // Example: rotating camera around dragon's nose up close. 
        float frameAgeScale = saturate(weightSum);

        float cachedFrameAge = frameAgeScale * dot(nWeights, vCachedFrameAge);
        frameAge = round(cachedFrameAge);

        // Nudge frame age down on non-zero reprojection so that new rays are shot and denoising kicks in.
        if (frameAge == 33 && dot(1, textureSpaceMotionVector * cb.textureDim) > 0.001)
        {
            frameAge = 10;
        }

        // ToDo move this to a separate pass?
        if (frameAge > 0)
        {
            float4 vCacheValues = g_texInputCachedValue.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
            cachedValue = dot(nWeights, vCacheValues);

            float4 vCachedValueSquaredMean = g_texInputCachedValueSquaredMean.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
            cachedValueSquaredMean = dot(nWeights, vCachedValueSquaredMean);

            float4 vCachedRayHitDistances = g_texInputCachedRayHitDistance.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
            cachedRayHitDistance = dot(nWeights, vCachedRayHitDistances);
        }
    }
    else
    {
        // ToDo take an average? and set frameAge low?
        // No valid values can be retrieved from the cache.
        frameAge = 0;
    }
    g_texOutputCachedFrameAge[DTid] = frameAge;
    g_texOutputReprojectedCachedValues[DTid] = float4(frameAge, cachedValue, cachedValueSquaredMean, cachedRayHitDistance);
}