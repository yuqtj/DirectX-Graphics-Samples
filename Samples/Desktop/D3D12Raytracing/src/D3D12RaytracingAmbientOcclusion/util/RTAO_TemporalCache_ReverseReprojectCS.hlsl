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

// ToDo some pixels here and there on mirror boundaries fail temporal reprojection even for static scene/camera

// ToDO pack value and depth beforehand?
Texture2D<float> g_texInputCachedValue : register(t0);
Texture2D<float> g_texInputCurrentFrameValue : register(t1);
Texture2D<float> g_texInputCachedDepth : register(t2);
Texture2D<float> g_texInputCurrentFrameDepth : register(t3);
Texture2D<float4> g_texInputCachedNormal : register(t4);        // ToDo standardize cache vs cached
Texture2D<float4> g_texInputCurrentFrameNormal : register(t5);
Texture2D<uint> g_texInputCacheFrameAge : register(t6);
Texture2D<float> g_texInputCurrentFrameMean : register(t7);
Texture2D<float> g_texInputCurrentFrameVariance : register(t8);
Texture2D<float2> g_texInputCurrentFrameLinearDepthDerivative : register(t9); // ToDo standardize naming across files
Texture2D<float2> g_texInputTextureSpaceMotionVector : register(t10); // ToDo standardize naming across files
Texture2D<float4> g_texInputReprojectedHitPosition : register(t11); // ToDo standardize naming across files
Texture2D<float4> g_texInputCachedHitPosition : register(t12); // ToDo standardize naming across files

RWTexture2D<float> g_texOutputCachedValue : register(u0);
RWTexture2D<uint> g_texOutputCacheFrameAge : register(u1);
RWTexture2D<float4> g_texOutputDebug1 : register(u2);
RWTexture2D<float4> g_texOutputDebug2 : register(u3);

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


float4 BilateralResampleWeights(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float2 offset, in uint2 actualIndex, in uint2 sampleIndices[4], in float cacheDdxy)
{
    uint4 isWithinBounds = uint4(
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
        float floatPrecision = 1.25f * FloatPrecision(ActualDistance, cb.DepthNumMantissaBits);
       
        float depthTolerance = max(cb.depthSigma * depthThreshold, floatPrecision);
        float4 depthWeigths = min(depthTolerance / (abs(SampleDistances - ActualDistance) + FLT_EPSILON), 1);
        depthMask = depthWeigths >= 1 ? depthWeigths : 0;   // ToDo revise
#if DEBUG_OUTPUT
        g_texOutputDebug2[actualIndex] = float4(depthWeigths);
#endif
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
    
    int2 topLeftCacheFrameIndex = int2(cacheFrameTexturePos * cb.textureDim - 0.5);

    // ToDo why this doesn't match cacheFrameTexturePos??
    float2 adjustedCacheFrameTexturePos = (topLeftCacheFrameIndex + 0.5) * cb.invTextureDim;

    float2 cachePixelOffset = cacheFrameTexturePos * cb.textureDim - topLeftCacheFrameIndex - 0.5;

    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

    uint2 cacheIndices[4] = {
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
            LoadDepthAndNormal(g_texInputCachedNormal, cacheIndices[i], vCacheDepths[i], cacheNormals[i]);
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

    float cacheDdxy = ddxy;
    float3 _normal;
    float _depth;
    LoadDepthAndNormal(g_texInputReprojectedHitPosition, DTid, _depth, _normal);


    if (cb.useWorldSpaceDistance)
    {
        float3 normal;
        float dummy;
        float linearDepth = g_texInputCurrentFrameDepth[DTid];
        LoadDepthAndNormal(g_texInputCurrentFrameNormal, DTid, dummy, normal);

        cacheDdxy = CalculateAdjustedDepthThreshold(ddxy, linearDepth, _depth, normal, _normal);
    }

    float value = g_texInputCurrentFrameValue[DTid];
    float mergedValue;

    float4 weights = BilateralResampleWeights(_depth, _normal, vCacheDepths, cacheNormals, cachePixelOffset, DTid, cacheIndices, cacheDdxy);
    float weightSum = dot(1, weights);


#if DEBUG_OUTPUT
    g_texOutputDebug1[actualIndex] = float4(ActualDistance, cacheDdxy, weightSum, 0);
#endif

#if 0
    // ToDo dedupe with GetClipSpacePosition()...
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 currentFrameTexturePos = xy * cb.invTextureDim;

    float aspectRatio = cb.textureDim.x / cb.textureDim.y;
    float maxScreenSpaceReprojectionDistance = 0.01;// cb.minSmoothingFactor * 0.1f; // ToDo
     ToDo scale this based on depth?
    float screenSpaceReprojectionDistanceAsWidthPercentage = min(1, length((currentFrameTexturePos - cacheFrameTexturePos) * float2(1, aspectRatio)));
#endif

    bool isCacheValueValid = weightSum > 1e-3f; // ToDo
    //&& screenSpaceReprojectionDistanceAsWidthPercentage <= maxScreenSpaceReprojectionDistance;
    uint isDisoccluded;
    uint frameAge;

    uint maxFrameAge = 1 / cb.minSmoothingFactor - 1;// minSmoothingFactor;
    if (isCacheValueValid)
    {
        isDisoccluded = false;

        float4 vCacheValues = g_texInputCachedValue.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;
        uint4 vCacheFrameAge = g_texInputCacheFrameAge.GatherRed(ClampSampler, adjustedCacheFrameTexturePos).wzxy;

        weights /= weightSum;   // Normalize the weights.
        float cachedValue = dot(weights, vCacheValues);
        float cacheFrameAge = dot(weights, vCacheFrameAge);
        frameAge = round(cacheFrameAge);

        // Clamp value to mean +/- std.dev of local neighborhood to surpress ghosting on value changing due to other occluder movements.
        // This will prevent reusing values outside the expected range.
        // Ref: Salvi2016, Temporal Super-Sampling
        float frameAgeClamp = 0;

        if (cb.clampCachedValues)
        {
            float localMean = g_texInputCurrentFrameMean[DTid];
            float localVariance = g_texInputCurrentFrameVariance[DTid];
            float localStdDev = max(cb.stdDevGamma * sqrt(localVariance), cb.minStdDevTolerance);
            float prevCachedValue = cachedValue;
            cachedValue = clamp(cachedValue, localMean - localStdDev, localMean + localStdDev); 
            
            float frameAgeAdjustmentDueClamping = 1; // todo
            // Scale the frame age based on how strongly the cached value got clamped.
            frameAgeClamp = frameAgeAdjustmentDueClamping * abs(cachedValue - prevCachedValue);
            frameAge = lerp(frameAge, 0, frameAgeClamp);
        }
        //frameAgeClamp = screenSpaceReprojectionDistanceAsWidthPercentage / maxScreenSpaceReprojectionDistance;
        //uint maxFrame
        //frameAge = lerp(frameAge, 0, frameAgeClamp);

        float invFrameAge = 1.f / (frameAge + 1.f);
        float a = cb.forceUseMinSmoothingFactor ? cb.minSmoothingFactor : max(invFrameAge, cb.minSmoothingFactor);
        mergedValue = lerp(cachedValue, value, a);
        
        // ToDo If no valid samples found:
        //  - use largest motion vector from 3x3
        //  - try 3x3 area
        //  - default to average?
    }
    else // ToDo initialize values to this instead of branch?
    {
        isDisoccluded = true;
        mergedValue = value;
        frameAge = 0;
    }
   
    g_texOutputCachedValue[DTid] = mergedValue;
    g_texOutputCacheFrameAge[DTid] = min(frameAge + 1, maxFrameAge);
}