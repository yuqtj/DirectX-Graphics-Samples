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

// ToDO pack value and depth beforehand?
Texture2D<float> g_texInputCachedValue : register(t0);
Texture2D<float> g_texInputCurrentFrameValue : register(t1);
Texture2D<float> g_texInputCachedDepth : register(t2);
Texture2D<float> g_texInputCurrentFrameDepth : register(t3);
Texture2D<float4> g_texInputCachedNormal : register(t4);
Texture2D<float4> g_texInputCurrentFrameNormal : register(t5);
Texture2D<uint> g_texInputCacheFrameAge : register(t6);
Texture2D<float> g_texInputCurrentFrameMean : register(t7);
Texture2D<float> g_texInputCurrentFrameVariance : register(t8);
Texture2D<float2> g_texInputCurrentFrameLinearDepthDerivative : register(t9); // ToDo standardize naming across files

RWTexture2D<float> g_texOutputCachedValue : register(u0);
RWTexture2D<uint> g_texOutputCacheFrameAge : register(u1);
RWTexture2D<float4> g_texOutputDebug1 : register(u2);
RWTexture2D<float4> g_texOutputDebug2 : register(u3);

ConstantBuffer<RTAO_TemporalCache_ReverseReprojectConstantBuffer> cb : register(b0);

SamplerState LinearSampler : register(s0);


// ToDo
// - Fix heavy disocclusion on min/magnifaction. Use bilateraly downsampled mip maps?
//   - this happens only when starting to move not on smooth move.

#if 0
// Retrieves pixel's position in world space.
// linearDepth - linear depth in [0, 1] range   // ToDo
float3 CalculateWorldPositionFromLinearDepth(in uint2 DTid, in float linearDepth)
{
    // Convert to non-linear depth.
#if USE_NORMALIZED_Z
    float linearDistance = linearDepth * (cb.zFar - cb.zNear) + cb.zNear;
#else
    float linearDistance = linearDepth;
#endif
    float logDepth = cb.zFar / (cb.zFar - cb.zNear) - cb.zFar * cb.zNear

    // Calculate Normalized Device Coordinates xyz = { [-1,1], [-1,1], [0,-1] }
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 screenPos = 2 * xy * cb.invTextureDim - 1;   // Convert to [-1, 1]
    //screenPos.y = -screenPos.y;                       // Invert Y for DirectX-style coordinates.
    float3 ndc = float3(screenPos, logDepth);  

    float4 viewPosition = mul(float4(ndc, 1), cb.invProj);
    //viewPosition /= viewPosition.w; // Perspective division
    float4 worldPosition = mul(viewPosition, cb.invView);
    
    return worldPosition.xyz;
}
#else
// Retrieves pixel's position in clip space.
// linearDepth - linear depth in [0, 1] range   // ToDo
float4 GetClipSpacePosition(in uint2 DTid, in float linearDepth)
{
    // Convert to non-linear depth.
#if USE_NORMALIZED_Z
    ToDo
    float linearDistance = linearDepth * (cb.zFar - cb.zNear) + cb.zNear;
#else
    float linearDistance = linearDepth;
#endif

    // Calculate Normalized Device Coordinates xyz = {[-1,1], [-1,1], [0,-1]}
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 screenPos = 2 * xy * cb.invTextureDim - 1;   // Convert to [-1, 1].
    screenPos.y = -screenPos.y;                         // Invert Y for DirectX-style coordinates.
    float logDepth = ViewToLogDepth(linearDepth, cb.zNear, cb.zFar);
    float3 ndc = float3(screenPos, logDepth);

    float A = cb.zFar / (cb.zFar - cb.zNear);
    float B = -cb.zNear * cb.zFar / (cb.zFar - cb.zNear);
    float w = B / (logDepth - A);

    float4 projPos = float4(ndc, 1) * w;                // Reverse perspective division.
#if 0 
    float4 viewPos = mul(projPos, cb.invProj);
    float4 worldPos = mul(viewPos, cb.invView);

    return worldPos.xyz;
#else
    return projPos;
#endif
}
#endif



float4 BilateralResampleWeights(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float2 offset, in uint2 sampleIndices[4], in float cacheDdxy, in uint2 DTid )
{
    uint4 isWithinBounds = uint4(
        IsWithinBounds(sampleIndices[0], cb.textureDim),
        IsWithinBounds(sampleIndices[1], cb.textureDim),
        IsWithinBounds(sampleIndices[2], cb.textureDim),
        IsWithinBounds(sampleIndices[3], cb.textureDim));

    float4 depthMask = 1;
    if (cb.useDepthWeights)
    {
        // ToDo improve depth test on zoom in. A lot of pixels get invalidated on the ground plane on close zoom in.
#if 0
        // ToDo remove - blurs more/incorrectly
        float4 depthDiffs = abs(SampleDistances - ActualDistance);
        float4 depthWeigthsMask = depthDiffs < cb.depthTolerance * ActualDistance;

        // Weight the depths based of smallest difference.
        float minDepthDiff = min(min(min(depthDiffs.x, depthDiffs.y), depthDiffs.z), depthDiffs.w);
        depthWeigths = min(depthWeigthsMask * (minDepthDiff + 1e-6) / (depthDiffs + 1e-6), 1);
#endif
        float depthThreshold = cb.depthTolerance * ActualDistance;

        depthMask = isWithinBounds &&
            abs(SampleDistances - ActualDistance) < depthThreshold;
    }


    float4 normalWeights = 1;
    if (cb.useNormalWeights)
    {
        const uint normalExponent = 32;
        const float minNormalWeight = 1e-3f; // ToDo pass as parameter
        normalWeights =
            float4(
                pow(saturate(dot(ActualNormal, SampleNormals[0])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[1])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[2])), normalExponent),
                pow(saturate(dot(ActualNormal, SampleNormals[3])), normalExponent) >= minNormalWeight);
    }

    float4 bilinearWeights = float4(
        (1 - offset.x) * (1 - offset.y),
        offset.x * (1 - offset.y),
        (1 - offset.x) * offset.y,
        offset.x * offset.y);

    // ToDo use depth weights instead of mask?
    // ToDo can we prevent diffusion across plane?
    float4 weights = bilinearWeights * depthMask * normalWeights;    // ToDo invalidate samples too pixel offcenter? <0.1

    weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0; // ToDo - the environment map depth is lower...

    float weightSum = dot(weights, 1);

    float minWeightSum = 1e-3f; 
    return weightSum >= minWeightSum ? weights / (dot(weights, 1) + FLT_EPSILON) : 0;
}

float4 BilateralResampleWeights2(in float ActualDistance, in float3 ActualNormal, in float4 SampleDistances, in float3 SampleNormals[4], in float2 offset, in uint2 actualIndex, in uint2 sampleIndices[4], in float cacheDdxy, out float weightSum)
{
    uint4 isWithinBounds = uint4(
        IsWithinBounds(sampleIndices[0], cb.textureDim),
        IsWithinBounds(sampleIndices[1], cb.textureDim),
        IsWithinBounds(sampleIndices[2], cb.textureDim),
        IsWithinBounds(sampleIndices[3], cb.textureDim));

    float4 depthMask = 1;
    if (cb.useDepthWeights)
    {
        float2 ddxy = abs(g_texInputCurrentFrameLinearDepthDerivative[actualIndex]);  // ToDo move to caller

#if 0
        float depthThreshold = 1.f;     // ToDo standardize depth vs distance
        float fEpsilon = 1e-6 * ActualDistance;


        // ToDo consider ddxy per dimension or have a 1D max(Ddxy) resource?
        depthThreshold = max(ddxy.x, ddxy.y);
        
        // ToDo correct weights to weigths in the whole project same for treshold and weigth
        float fScale = 1.f / depthThreshold;
        float4 depthWeigths = min(1.0 / (fScale * abs(SampleDistances - ActualDistance) + fEpsilon), 1);
#else
        float depthThreshold = cacheDdxy;// cb.depthTolerance * ActualDistance; //cacheDdxy;

#if 0
        float4 clipSpacePos = GetClipSpacePosition(actualIndex + uint2(1,1), ActualDistance);
        float4 cacheClipSpacePos = mul(clipSpacePos, cb.reverseProjectionTransform);
        float3 cacheNDCpos = cacheClipSpacePos.xyz;
        cacheNDCpos /= cacheClipSpacePos.w;                             // Perspective division.
        cacheNDCpos.y = -cacheNDCpos.y;                                 // Invert Y for DirectX-style coordinates.
        float cacheLinearDepth = LogToViewDepth(cacheNDCpos.z, cb.zNear, cb.zFar);
#endif
        //depthThreshold = abs(cacheLinearDepth - ActualDistance);
        float fMinEpsilon = cb.floatEpsilonDepthTolerance * 512 * FLT_EPSILON; // Minimum depth threshold epsilon to avoid acne due to ray/triangle floating precision limitations.
        // ToDo should it be exponential?
        float fMinDepthScaledEpsilon = cb.depthDistanceBasedDepthTolerance * 48 * 1e-6  * ActualDistance;  // Depth threshold to surpress differences that surface at larger depth from the camera.
        float fEpsilon = fMinEpsilon + fMinDepthScaledEpsilon;

        // ToDo use exp or mininengine weighting?
        float4 depthWeigths = min((cb.depthSigma * depthThreshold + fEpsilon)/ (abs(SampleDistances - ActualDistance) + FLT_EPSILON), 1);
        //float4 e_d = min(1 - abs(SampleDistances - ActualDistance) / (cb.depthSigma * depthThreshold + fEpsilon), 0);
        //float4 depthWeigths = exp(e_d);
        //float4 e_d = min(1 - abs(depth - iDepth) / (cb.depthSigm * depthThreshold + fEpsilon), 0) : 0;
        //depthWeigths = exp(e_d);
#endif
        depthMask = depthWeigths >= 1 ? depthWeigths : 0;   // ToDo revise
    }
    

    float4 normalWeights = 1;
    if (cb.useNormalWeights)
    {
        const uint normalExponent = 32;
        const float minNormalWeight = 1e-3f; // ToDo pass as parameter
        float normalSigma = 1.1;        // ToDo remove/finetune/document. There's some less than 1 weights even for same/very similar  normals (i.e. house wall tiles)(due to low bit format?)
        normalWeights =
            float4(
                pow(saturate(normalSigma*dot(ActualNormal, SampleNormals[0])), normalExponent),
                pow(saturate(normalSigma*dot(ActualNormal, SampleNormals[1])), normalExponent),
                pow(saturate(normalSigma*dot(ActualNormal, SampleNormals[2])), normalExponent),
                pow(saturate(normalSigma*dot(ActualNormal, SampleNormals[3])), normalExponent) >= minNormalWeight);
    }

    float4 bilinearWeights = float4(
        (1 - offset.x) * (1 - offset.y),
        offset.x * (1 - offset.y),
        (1 - offset.x) * offset.y,
        offset.x * offset.y);

    // ToDo use depth weights instead of mask?
    // ToDo can we prevent diffusion across plane?
    float4 weights = bilinearWeights * depthMask * normalWeights;    // ToDo invalidate samples too pixel offcenter? <0.1

    weights = SampleDistances < DISTANCE_ON_MISS ? weights : 0; // ToDo?

    //float weightSum = dot(weights, 1);
    weightSum = dot(weights, 1);

    float minWeightSum = 1e-3f;
    return weightSum >= minWeightSum ? weights / (dot(weights, 1) + FLT_EPSILON) : 0;
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
    // ToDo remove
#if !COMPRES_NORMALS || !PACK_NORMAL_AND_DEPTH
    Not supported
#endif
}


float GetLinearDepth(float3 viewPosition, float3 forwardCameraRay)
{
    return dot(viewPosition, forwardCameraRay);
}


[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{

    float linearDepth = g_texInputCurrentFrameDepth[DTid];
    float3 normal;
    float dummy;
    LoadDepthAndNormal(g_texInputCurrentFrameNormal, DTid, dummy, normal);
  
    float4 clipSpacePos = GetClipSpacePosition(DTid, linearDepth);
    
    // Reverse project into previous frame
    float4 cacheClipSpacePos = mul(clipSpacePos, cb.reverseProjectionTransform);

    float3 cacheNDCpos = cacheClipSpacePos.xyz;
    cacheNDCpos /= cacheClipSpacePos.w;                             // Perspective division.
    cacheNDCpos.y = -cacheNDCpos.y;                                 // Invert Y for DirectX-style coordinates.
    
    
    // Suffers from loss of precision moving to and back from log representation.
    float cacheLinearDepth2 = LogToViewDepth(cacheNDCpos.z, cb.zNear, cb.zFar);
    float2 cacheFrameTexturePos = (cacheNDCpos.xy + 1) * 0.5f;      // [-1,1] -> [0, 1]


    int2 topLeftCacheFrameIndex = int2(cacheFrameTexturePos * cb.textureDim - 0.5);
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
    {
        for (int i = 0; i < 4; i++)
        {
            float dummy;
            LoadDepthAndNormal(g_texInputCachedNormal, cacheIndices[i], dummy, cacheNormals[i]);
        }
    }

    float4 vCacheDepths = float4(
        g_texInputCachedDepth[cacheIndices[0]],
        g_texInputCachedDepth[cacheIndices[1]],
        g_texInputCachedDepth[cacheIndices[2]],
        g_texInputCachedDepth[cacheIndices[3]]);

    // Calculate linear depth in the cache frame.
    // We avoid converting depth between linear and log as log depth loses a lot of precision very quickly as depth increases.
    float3 viewPos = ScreenPosToWorldPos(DTid, linearDepth, cb.textureDim, cb.zNear, float3(0,0,0), cb.projectionToWorldWithCameraEyeAtOrigin);

    // Add the camera translation change 
    viewPos += cb.prevToCurrentFrameCameraTranslation.xyz;

    float3 cameraDirection = GenerateForwardCameraRayDirection(cb.projectionToWorldWithCameraEyeAtOrigin);
    float3 cacheCameraDirection = GenerateForwardCameraRayDirection(cb.prevProjectionToWorldWithCameraEyeAtOrigin);
    float cacheLinearDepth = dot(viewPos, cacheCameraDirection);


#if 1
    float2 dxdy = g_texInputCurrentFrameLinearDepthDerivative[DTid];  // ToDo move to caller
        // ToDo should this be done separately for both X and Y dimensions?
    float  ddxy = dot(1, dxdy);

    float4 clipSpacePosDdxy = GetClipSpacePosition(DTid + 1, cb.zNear + FLT_EPSILON + ddxy);
    //float4 clipSpacePosDdxy = GetClipSpacePosition(DTid + 1, linearDepth + ddxy);
    float4 cacheClipSpacePosDdxy = mul(clipSpacePosDdxy, cb.reverseProjectionTransform);
    float3 cacheNDCposDxdxy = cacheClipSpacePosDdxy.xyz;
    cacheNDCposDxdxy /= cacheClipSpacePosDdxy.w;                                // Perspective division.
    cacheNDCposDxdxy.y = -cacheNDCposDxdxy.y;                                   // Invert Y for DirectX-style coordinates.
    float cacheLinearDepthDxdy = LogToViewDepth(cacheNDCposDxdxy.z, cb.zNear, cb.zFar); // Log to linear depth.

    float2 cacheFrameTexturePosDxdy = (cacheNDCposDxdxy.xy + 1) * 0.5f;         // [-1,1] -> [0, 1]

    // Calculate depth derivative in cached frame
    float cacheDdxy;
    {
        // TODo use larger values than FLT_EPSILON to avoid division by 0?
        cacheDdxy = ddxy;// ToDo reproject adjust - and test camera moving fast upwards over the ground plane
        g_texOutputDebug1[DTid] = float4(linearDepth, viewPos);
        g_texOutputDebug2[DTid] = float4(vCacheDepths.x, cacheLinearDepth, ddxy, cacheDdxy);
    }

    float2 pixelOffset = abs(cacheFrameTexturePos - cacheFrameTexturePosDxdy) * cb.textureDim;



    
#if 0
    float4 cacheClipPos[4] = {
        GetClipSpacePosition(cacheIndices[0], vCacheDepths.x),
        GetClipSpacePosition(cacheIndices[1], vCacheDepths.y),
        GetClipSpacePosition(cacheIndices[2], vCacheDepths.z),
        GetClipSpacePosition(cacheIndices[3], vCacheDepths.w)};

    float4 cacheWorldPos = float4(
        mul(clipSpacePos, cb.invViewProjAndCameraTranslation),
        mul(clipSpacePos, cb.invViewProjAndCameraTranslation),
        mul(clipSpacePos, cb.invViewProjAndCameraTranslation),
        mul(clipSpacePos, cb.invViewProjAndCameraTranslation));
#endif
#endif

    float4 vCacheValues = float4(
        g_texInputCachedValue[cacheIndices[0]],
        g_texInputCachedValue[cacheIndices[1]],
        g_texInputCachedValue[cacheIndices[2]],
        g_texInputCachedValue[cacheIndices[3]]);

    uint4 vCacheFrameAge = uint4(
        g_texInputCacheFrameAge[cacheIndices[0]],
        g_texInputCacheFrameAge[cacheIndices[1]],
        g_texInputCacheFrameAge[cacheIndices[2]],
        g_texInputCacheFrameAge[cacheIndices[3]]);

    
    

    float value = g_texInputCurrentFrameValue[DTid];
    float mergedValue;
#if 0
    float4 weights = BilateralResampleWeights(cacheLinearDepth, normal, vCacheDepths, cacheNormals, cachePixelOffset, cacheIndices, cacheDdxy, DTid);
#else
    float weightSum2;
    float4 weights = BilateralResampleWeights2(cacheLinearDepth, normal, vCacheDepths, cacheNormals, cachePixelOffset, DTid, cacheIndices, cacheDdxy, weightSum2);
#endif


    float weightSum = dot(1, weights);

    // ToDo dedupe with GetClipSpacePosition()...
    float2 xy = DTid + 0.5f;                            // Center in the middle of the pixel.
    float2 currentFrameTexturePos = xy * cb.invTextureDim;

    float aspectRatio = cb.textureDim.x / cb.textureDim.y;
    float maxScreenSpaceReprojectionDistance = 0.01;// cb.minSmoothingFactor * 0.1f;
    // ToDo scale this based on depth?
    float screenSpaceReprojectionDistanceAsWidthPercentage = min(1, length((currentFrameTexturePos - cacheFrameTexturePos) * float2(1, aspectRatio)));

    bool isCacheValueValid = weightSum > 1e-3f;
    //&& screenSpaceReprojectionDistanceAsWidthPercentage <= maxScreenSpaceReprojectionDistance;
    uint isDisoccluded;
    uint frameAge;

    uint maxFrameAge = 1 / 0.1 - 1;// minSmoothingFactor;
    if (isCacheValueValid)
    {
        isDisoccluded = false;
        float cachedValue = dot(weights, vCacheValues);

        float cacheFrameAge = dot(weights, vCacheFrameAge);
        frameAge = uint(cacheFrameAge + 0.5f);              // HLSL rounds down when converting float to uint, bump up the value by 0.5 before rounding.

        // Clamp value to mean +/- std.dev of local neighborhood to surpress ghosting on value changing due to other occluder movements.
        // This will prevent reusing values outside the expected range.
        // Ref: Salvi2016, Temporal Super-sampling
        float frameAgeClamp = 0;

        if (cb.clampCachedValues)
        {
            float localMean = g_texInputCurrentFrameMean[DTid];
            float localVariance = g_texInputCurrentFrameVariance[DTid];
            float localStdDev = cb.stdDevGamma * sqrt(localVariance);
            float prevCachedValue = cachedValue;
            cachedValue = clamp(cachedValue, localMean - localStdDev, localMean + localStdDev); 
            
            // Scale the frame age based on how strongly the cached value got clamped.
            // TODo frameAgeClamp = abs(cachedValue - prevCachedValue);

        }
        //frameAgeClamp = screenSpaceReprojectionDistanceAsWidthPercentage / maxScreenSpaceReprojectionDistance;
        //uint maxFrame
        //frameAge = lerp(frameAge, 0, frameAgeClamp);

        float invFrameAge = 1.f / (frameAge + 1);
        float minSmoothingFactor = 0.1f;// cb.minSmoothingFactor;
        float a = cb.forceUseMinSmoothingFactor ? cb.minSmoothingFactor : max(invFrameAge, minSmoothingFactor);
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
   
    //float distToPixelCenter = length((cacheFrameTexturePos * cb.textureDim - 0.5f) - topLeftCacheFrameIndex);
    g_texOutputCachedValue[DTid] = mergedValue;
    //g_texOutputCachedValue[DTid] = abs(cacheLinearDepthNear - cacheLinearDepthDxdy);
    //g_texOutputCachedValue[DTid] = cacheClipSpacePosDdxy.z - cacheClipSpacePos.z;

   // g_texOutputCachedValue[DTid] = clipSpacePosDdxy.z - clipSpacePos.z;
    //g_texOutputCachedValue[DTid] = ddxy;

    //g_texOutputCachedValue[DTid] = length(cacheClipSpacePosDdxy - cacheClipSpacePos);
    //g_texOutputCachedValue[DTid] = cacheDdxy;
    g_texOutputCacheFrameAge[DTid] = min(frameAge + 1, maxFrameAge);
    //g_texOutputCachedValue[DTid] = cacheClipSpacePos.x;
}