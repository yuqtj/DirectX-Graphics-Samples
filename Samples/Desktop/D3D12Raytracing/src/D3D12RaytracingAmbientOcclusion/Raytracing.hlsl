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

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

// Remove /Zpr and use column-major? It might be slightly faster

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "RandomNumberGenerator.hlsli"
#include "SSAO/GlobalSharedHlslCompat.h" // ToDo remove
#include "util/AnalyticalTextures.hlsli"

#define HitDistanceOnMiss -1        // ToDo unify with DISTANCE_ON_MISS

// ToDo split to Raytracing for GBUffer and AO?

// ToDo excise non-GBuffer parts out for separate timings? Such as 

// ToDo dedupe code triangle normal calc,..
// ToDo pix doesn't show output for AO pass

//***************************************************************************
//*****------ Shader resources bound via root signatures -------*************
//***************************************************************************

// Scene wide resources.
//  g_* - bound via a global root signature.
//  l_* - bound via a local root signature.
RaytracingAccelerationStructure g_scene : register(t0, space0);
RWTexture2D<float4> g_renderTarget : register(u0);			// ToDo remove

// ToDo prune redundant
// ToDo move this to local ray gen root sig
RWTexture2D<uint> g_rtGBufferCameraRayHits : register(u5);
// {MaterialId, 16b 2D texCoords}
RWTexture2D<uint2> g_rtGBufferMaterialInfo : register(u6);  // 16b {1x Material Id, 3x Diffuse.RGB}. // ToDo compact to 8b?
RWTexture2D<float4> g_rtGBufferPosition : register(u7);
RWTexture2D<float4> g_rtGBufferNormal : register(u8);
RWTexture2D<float> g_rtGBufferDistance : register(u9);

Texture2D<uint> g_texGBufferPositionHits : register(t5); 
Texture2D<uint2> g_texGBufferMaterialInfo : register(t6);     // 16b {1x Material Id, 3x Diffuse.RGB}       // ToDo rename to material like in composerenderpasses
Texture2D<float4> g_texGBufferPositionRT : register(t7);
Texture2D<float4> g_texGBufferNormal : register(t8);
Texture2D<float4> g_texGBufferDistance : register(t9);
TextureCube<float4> g_texEnvironmentMap : register(t12);
Texture2D<float> g_filterWeightSum : register(t13);
Texture2D<uint> g_texInputAOFrameAge : register(t14);

// ToDo remove AOcoefficient and use AO hits instead?
//todo remove rt?
RWTexture2D<float> g_rtAOcoefficient : register(u10);
RWTexture2D<uint> g_rtAORayHits : register(u11);
RWTexture2D<float> g_rtVisibilityCoefficient : register(u12);
RWTexture2D<float> g_rtGBufferDepth : register(u13);
RWTexture2D<float4> g_rtGBufferNormalRGB : register(u14);   // for SSAO. ToDo cleanup
RWTexture2D<float> g_rtAORayHitDistance : register(u15);
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
RWTexture2D<float2> g_rtPartialDepthDerivatives : register(u16);
#endif
// ToDo pack motion vector, depth and normal into float4
RWTexture2D<float2> g_rtTextureSpaceMotionVector : register(u17);
RWTexture2D<float4> g_rtReprojectedHitPosition : register(u18);

ConstantBuffer<SceneConstantBuffer> CB : register(b0);          // ToDo standardize CB var naming
StructuredBuffer<PrimitiveMaterialBuffer> g_materials : register(t3);
StructuredBuffer<AlignedHemisphereSample3D> g_sampleSets : register(t4);

StructuredBuffer<float3x4> g_prevFrameBottomLevelASInstanceTransform : register(t15);


SamplerState LinearWrapSampler : register(s0);


// ToDo: https://developer.nvidia.com/content/understanding-structured-buffer-performance
/*******************************************************************************************************/
// Per-object resources
ConstantBuffer<PrimitiveConstantBuffer> l_materialCB : register(b0, space1);

#if ONLY_SQUID_SCENE_BLAS
StructuredBuffer<Index> l_indices : register(t0, space1);
// Vertex buffers for current and previous frame.
// Current frame array index is ~ (currentFrameVBindex & 1) for animated geometry, 0 otherwise.
StructuredBuffer<VertexPositionNormalTextureTangent> l_vertices[2] : register(t1, space1);  // t1-t2
#else
ByteAddressBuffer l_indices : register(t0, space1);
StructuredBuffer<VertexPositionNormalTexture> l_vertices[2] : register(t1, space1); // t1-t2
#endif
Texture2D<float3> l_texDiffuse : register(t3, space1);
Texture2D<float3> l_texNormalMap : register(t4, space1);
/*******************************************************************************************************/


float GetPlaneConstant(in float3 planeNormal, in float3 pointOnThePlane)
{
    // Given a plane equation N * P + d = 0
    // d = - N * P
    return -dot(planeNormal, pointOnThePlane);
}

bool IsPointOnTheNormalSideOfPlane(in float3 P, in float3 planeNormal, in float3 pointOnThePlane)
{
    float d = GetPlaneConstant(planeNormal, pointOnThePlane);
    return dot(P, planeNormal) + d > 0;
}

float3 ReflectPointThroughAPlane(in float3 P, in float3 planeNormal, in float3 pointOnThePlane)
{
    //           |
    //           |
    //  P ------ C ------ R
    //           |
    //           |
    // Given a point P, plane with normal N and constant d, the projection point C of P onto plane is:
    // C = P + t*N
    //
    // Then the reflected point R of P through the plane can be computed using t as:
    // R = P + 2*t*N

    // Given C = P + t*N, and C lying on the plane,
    // C*N + d = 0
    // then
    // C = - d/N
    // -d/N = P + t*N
    // 0 = d + P*N + t*N*N
    // t = -(d + P*N) / N*N

    float d = GetPlaneConstant(planeNormal, pointOnThePlane);
    float3 N = planeNormal;
    float t = -(d + dot(P, N)) / dot(N, N);

    return P + 2 * t * N;
}

// ToDo standardize pos vs position in shaders
// Calculate screen texture-space motion vector from a previous to the current frame.
void CalculateMotionVectorForReflectedPoint(
    out float2 motionVector,
    out float prevFrameDepth, 
    out float3 prevFrameNormal,
    in uint instanceIndex,
    in float3 hitObjectPosition,
    in float3 hitObjectNormal,
    in uint3 vertexIndices,
    in BuiltInTriangleIntersectionAttributes attr,
    in uint reflectorInstanceIndex,
    in float3 reflectorHitObjectPosition,
    in float3 reflectorObjectNormal)
{
    // ToDo cleanup matrix multiplication order

    // Variables starting with underscore _ denote values in the previous frame.
    float3x4 _BLASTransform = g_prevFrameBottomLevelASInstanceTransform[instanceIndex];

    // Calculate hit object position of the hit in the previous frame.
    float3 _hitObjectPosition = hitObjectPosition;  // For non-animated geometry, object position are same frame to frame.
    if (l_materialCB.isVertexAnimated)
    {
        UINT _vbID = (CB.currentFrameVBindex + 1) & 1;
        float3 _vertices[3] = {
            l_vertices[_vbID][vertexIndices[0]].position,
            l_vertices[_vbID][vertexIndices[1]].position,
            l_vertices[_vbID][vertexIndices[2]].position };
        _hitObjectPosition = HitAttribute(_vertices, attr);
    }

    // Transform the hit object position to world space.
    float3 _hitPosition = mul(_BLASTransform, float4(_hitObjectPosition, 1));
    
    // Calculate normal at the hit in the previous frame.
    // BLAS Transforms in this sample are uniformly scaled so it's OK to directly apply the BLAS transform.
    // ToDo add a note that the transform is expected to have uniform scaling
    prevFrameNormal = normalize(mul((float3x3)_BLASTransform, hitObjectNormal));

    // Reflect the previous frame hitPosition through the reflector in the previous frame.
    float3x4 _reflectorBLASTransform = g_prevFrameBottomLevelASInstanceTransform[reflectorInstanceIndex];
    float3 _reflectorHitPosition = mul(_reflectorBLASTransform, float4(reflectorHitObjectPosition, 1));
    float3 _reflectorNormal = mul((float3x3)_reflectorBLASTransform, reflectorObjectNormal);      // Skipping normalization as it's not required for the uses of the transformed normal here.
    
    // If the hit position is behind the reflector in the previous frame, set the motion vector to be out of bounds
    // ToDo attempt direct lookup instead of reflected point?
    if (!IsPointOnTheNormalSideOfPlane(_hitPosition, _reflectorNormal, _reflectorHitPosition))
    {
        motionVector = float2(1, 1);
        return;
    }

    // Calculate depth of the reflected hit position in the previous frame.
    // ToDo explain
    float3 _mirrorReflectedHitPosition = ReflectPointThroughAPlane(_hitPosition, _reflectorNormal, _reflectorHitPosition);
    float3 _mirrorReflectedHitViewPosition = _mirrorReflectedHitPosition - CB.prevCameraPosition.xyz;
    float3 _cameraDirection = GenerateForwardCameraRayDirection(CB.prevProjToWorldWithCameraEyeAtOrigin);
    prevFrameDepth = dot(_mirrorReflectedHitViewPosition, _cameraDirection);

    float4 _clipSpacePosition = mul(float4(_mirrorReflectedHitPosition, 1), CB.prevViewProj);   // ToDO is 3x3 multiply enough here?
    float2 _texturePosition = ClipSpaceToTexturePosition(_clipSpacePosition);

    // ToDO pass in inverted dimensions?
    float2 xy = DispatchRaysIndex().xy + 0.5f;   // Center in the middle of the pixel.
    float2 texturePosition = xy / (float2) DispatchRaysDimensions().xy;

    motionVector = texturePosition - _texturePosition;
}

// Calculate screen texture-space motion vector from a previous to the current frame.
void CalculateMotionVector(
    out float2 motionVector,
    out float prevFrameDepth,
    out float3 prevFrameNormal,
    in uint instanceIndex,
    in float3 hitObjectPosition,
    in float3 hitObjectNormal,
    in uint3 vertexIndices,
    in BuiltInTriangleIntersectionAttributes attr)
{ 
    // Calculate hit position of the hit in the previous frame.
    float3x4 _BLASTransform = g_prevFrameBottomLevelASInstanceTransform[instanceIndex];

    // Calculate hit object position of the hit in the previous frame.
    float3 _hitObjectPosition = hitObjectPosition; // For non-animated geometry, object position are same frame to frame.
    if (l_materialCB.isVertexAnimated)
    {
        float3 _vertices[3];
        UINT _vbID = (CB.currentFrameVBindex + 1) & 1;
        _vertices[0] = l_vertices[_vbID][vertexIndices[0]].position;
        _vertices[1] = l_vertices[_vbID][vertexIndices[1]].position;
        _vertices[2] = l_vertices[_vbID][vertexIndices[2]].position;
        _hitObjectPosition = HitAttribute(_vertices, attr);
    }

    // Calculate hit position in the previous frame.
    float3 _hitPosition = mul(_BLASTransform, float4(_hitObjectPosition, 1));

    float3 _hitViewPosition = _hitPosition - CB.prevCameraPosition.xyz;
    float3 _cameraDirection = GenerateForwardCameraRayDirection(CB.prevProjToWorldWithCameraEyeAtOrigin);

    prevFrameDepth = dot(_hitViewPosition, _cameraDirection);

    // Calculate normal at the hit in the previous frame.
    // BLAS Transforms in this sample are uniformly scaled so it's OK to directly apply the BLAS transform.
    prevFrameNormal = normalize(mul((float3x3)_BLASTransform, hitObjectNormal));

    // Calculate screen space position of the hit in the previous frame.
    float4 _clipSpacePosition = mul(float4(_hitPosition, 1), CB.prevViewProj);
    float2 _texturePosition = ClipSpaceToTexturePosition(_clipSpacePosition);

    // ToDO pass in inverted dimensions?
    // ToDo should this add 0.5f?
    float2 xy = DispatchRaysIndex().xy + 0.5f;   // Center in the middle of the pixel.
    float2 texturePosition = xy / DispatchRaysDimensions().xy;
    
    motionVector = texturePosition - _texturePosition;
}


//***************************************************************************
//*****------ TraceRay wrappers for radiance and shadow rays. -------********
//***************************************************************************

// Trace a radiance ray into the scene and returns a shaded color.
float4 TraceRadianceRay(in Ray ray, in UINT currentRayRecursionDepth)
{
    if (currentRayRecursionDepth >= MAX_RAY_RECURSION_DEPTH)
    {
        return float4(0, 0, 0, 0);
    }

    // Set the ray's extents.
    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    // Set TMin to a zero value to avoid aliasing artifacts along contact areas.
    // Note: make sure to enable face culling so as to avoid surface face fighting.
	// ToDo Tmin
    rayDesc.TMin = 0.001;
    rayDesc.TMax = 10000;
    RayPayload rayPayload = { float4(0, 0, 0, 0), currentRayRecursionDepth + 1 };
    TraceRay(g_scene,
#if FACE_CULLING
		RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
#else
		0,
#endif
		TraceRayParameters::InstanceMask,
        TraceRayParameters::HitGroup::Offset[RayType::Radiance],
        TraceRayParameters::HitGroup::GeometryStride,
        TraceRayParameters::MissShader::Offset[RayType::Radiance],
        rayDesc, rayPayload);

    return rayPayload.color;
}

// Trace a shadow ray and return true if it hits any geometry.
// ToDo add surface normal and skip tracing a ray for surfaces facing away.
bool TraceShadowRayAndReportIfHit(out float tHit, in Ray ray, in UINT currentRayRecursionDepth, in float TMax = 10000)
{
    if (currentRayRecursionDepth >= MAX_RAY_RECURSION_DEPTH)
    {
        return false;
    }

    // Set the ray's extents.
    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    // Set TMin to a zero value to avoid aliasing artifacts along contact areas. // ToDo update comment re-floating error
    // Note: make sure to enable back-face culling so as to avoid surface face fighting.
    rayDesc.TMin = 0.0;
	rayDesc.TMax = TMax;

    // Initialize shadow ray payload.
    // Set the initial value to a hit at TMax. 
    // Miss shader will set it to HitDistanceOnMiss.
    // This way closest and any hit shaders can be skipped if true tHit is not needed. 
    ShadowRayPayload shadowPayload = { TMax };

    UINT rayFlags =
#if FACE_CULLING            // ToDo remove one path?
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES
#else
        0
#endif
#if REPRO_INVISIBLE_WALL
        | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
#endif
        | RAY_FLAG_FORCE_OPAQUE;             // ~skip any hit shaders
    
    // Skip closest hit shaders of tHit time is not needed.
    if (!CB.useShadowRayHitTime)
    {
        rayFlags |= RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
    }

    TraceRay(g_scene,
        rayFlags,
        TraceRayParameters::InstanceMask,
        TraceRayParameters::HitGroup::Offset[RayType::Shadow],
        TraceRayParameters::HitGroup::GeometryStride,
        TraceRayParameters::MissShader::Offset[RayType::Shadow],
        rayDesc, shadowPayload);
    
    // Report a hit if Miss Shader didn't set the value to HitDistanceOnMiss.
    tHit = shadowPayload.tHit;

    return shadowPayload.tHit > (HitDistanceOnMiss + 0.001f);
}

// Trace a camera ray into the scene.
// rx, ry - auxilary rays offset in screen space by one pixel in x, y directions.
GBufferRayPayload TraceGBufferRay(in Ray ray, in Ray rx, in Ray ry, in UINT currentRayRecursionDepth, float tMin = NEAR_PLANE, float tMax = FAR_PLANE)
{
	// Set the ray's extents.
	RayDesc rayDesc;
	rayDesc.Origin = ray.origin;
	rayDesc.Direction = ray.direction;
	// ToDo update comments about Tmins
	// Set TMin to a zero value to avoid aliasing artifacts along contact areas.
	// Note: make sure to enable face culling so as to avoid surface face fighting.
	// ToDo Tmin - this should be offset along normal.
	rayDesc.TMin = tMin;
	rayDesc.TMax = tMax;
#if ALLOW_MIRRORS
	GBufferRayPayload rayPayload = { currentRayRecursionDepth + 1, false, (uint2)0, (float3)0, (float3)0, 0, (float3)0, (float3)0, rx, ry, 0, 0
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        , 0, 0
#endif
    };
#else
	GBufferRayPayload rayPayload = { false, (uint2)0, (float3)0, (float3)0, rx, ry };
#endif
	TraceRay(g_scene,
#if FACE_CULLING
		RAY_FLAG_CULL_BACK_FACING_TRIANGLES
#else
		0
#endif
		| RAY_FLAG_FORCE_OPAQUE,             // ~skip any hit shaders,
		TraceRayParameters::InstanceMask,
		TraceRayParameters::HitGroup::Offset[RayType::GBuffer],
		TraceRayParameters::HitGroup::GeometryStride,
		TraceRayParameters::MissShader::Offset[RayType::GBuffer],
		rayDesc, rayPayload);

	return rayPayload;
}


// ToDo comment
// MinHitDistance - minimum hit distance of all AO ray hits.
float CalculateAO(out uint numShadowRayHits, out float minHitDistance, in uint2 DTid, in UINT numSamples, in float3 hitPosition, in float3 surfaceNormal, in float3 surfaceAlbedo = float3(1,1,1))
{
    numShadowRayHits = 0;
    minHitDistance = CB.RTAO_maxTheoreticalShadowRayHitTime;
    float occlussionCoefSum = 0;

#if AO_HITPOSITION_BASED_SEED
#if AO_SAMPLES_SPREAD_ACCROSS_PIXELS
    // Neighboring samples NxN share a sample set.
    // Get a sample set ID and seed shared across neighboring pixels.
    uint numSampleSetsInX = (DispatchRaysDimensions().x + CB.numPixelsPerDimPerSet - 1) / CB.numPixelsPerDimPerSet;
    uint2 sampleSetId = DispatchRaysIndex().xy / CB.numPixelsPerDimPerSet;
    uint2 pixelZeroId = sampleSetId * CB.numPixelsPerDimPerSet;
    float3 pixelZeroHitPosition = g_texGBufferPositionRT[pixelZeroId].xyz;      // ToDo remove?

    uint sampleSetSeed = (sampleSetId.y * numSampleSetsInX + sampleSetId.x) * hash(pixelZeroHitPosition) + CB.seed;

    uint RNGState = RNG::SeedThread(sampleSetSeed);
    uint sampleSetJump = RNG::Random(RNGState, 0, CB.numSampleSets - 1) * CB.numSamplesPerSet;

    // Get a pixel ID within the shared set across neighboring pixels.
    uint2 pixeIDPerSet2D = DispatchRaysIndex().xy % CB.numPixelsPerDimPerSet;
    uint pixeIDPerSet = pixeIDPerSet2D.y * CB.numPixelsPerDimPerSet + pixeIDPerSet2D.x;

    // ToDo is RNG being used here any useful?
    uint numPixelsPerSet = CB.numPixelsPerDimPerSet * CB.numPixelsPerDimPerSet;
    uint sampleJump = (pixeIDPerSet + RNG::Random(RNGState, 0, numPixelsPerSet - 1)) % numPixelsPerSet;
    sampleJump *= CB.numSamplesToUse;

#if AO_PROGRESSIVE_SAMPLING
    sampleJump += numSamples * g_texInputAOFrameAge[DTid];
#endif

    for (uint i = 0; i < numSamples; i++)
    {
        // Load a pregenerated random sample from the sample set.
        float3 sample = g_sampleSets[sampleSetJump + sampleJump + i].value;
#else
	// Seed:
	// - DispatchRaysDimensions to break correlation among neighboring pixels.
	// - hash(hitPosition) to break correlation for the same pixel but differet hitPosition when moving camera/objects.
    uint seed = (DispatchRaysDimensions().x * DispatchRaysIndex().y + DispatchRaysIndex().x) * hash(hitPosition) + CB.seed;

	uint RNGState = RNG::SeedThread(seed);
	uint sampleSetJump = RNG::Random(RNGState, 0, CB.numSampleSets - 1) * CB.numSamples;
	uint sampleJump = 0; //RNG::Random(RNGState, 0, CB.numSamples - 1);

    for (uint i = 0; i < CB.numSamplesToUse; i++)
    {
        // Load a pregenerated random sample from the sample set.
        float3 sample = g_sampleSets[sampleSetJump + (sampleJump + i) % CB.numSamples].value;
#endif
        // Calculate coordinate system for the hemisphere
        float3 u, v, w;
        w = surfaceNormal;

        // ToDo is this needed
        // Break hemisphere coordinate correlation
        float x = RNG::Random01(RNGState);
        float y = RNG::Random01(RNGState);
        float z = RNG::Random01(RNGState);
        float3 right = normalize(float3(x, y, z) * 2 - 1);

        //float3 right = normalize(float3(0.0072, 1.0, 0.0034));
        v = normalize(cross(w, right));
        u = cross(v, w);

        float3 rayDirection = sample.x * u + sample.y * v + sample.z * w;

        // ToDo hitPosition adjustment - fix crease artifacts
        // Todo fix noise on flat surface / box
        // ToDo remove unnecessary normalize()
        Ray shadowRay = { hitPosition + CB.RTAO_TraceRayOffsetAlongNormal * surfaceNormal, normalize(rayDirection) };

#else
    uint2 BTid = DispatchRaysIndex().xy & 3; // 4x4 BlockThreadID
    uint RNGState = RNG::SeedThread(CB.seed + (BTid.y << 2 | BTid.x));

    for (uint i = 0; i < CB.numSamplesToUse; i++)
    {
        // Compute random normal using cylindrical coordinates
        float theta = RNG::Random01ex(RNGState) * 6.2831853;
        float height = RNG::Random01(RNGState) * 2.0 - 1.0;
        float radius = sqrt(1.0 - height * height);
        float sinT, cosT; sincos(theta, sinT, cosT);
        float3 rayDirection = float3(cosT*radius, sinT*radius, height);
        if (dot(rayDirection, surfaceNormal) < 0.0)
            rayDirection = -rayDirection;

        Ray shadowRay = { hitPosition + 1e-3 * surfaceNormal, rayDirection };
#endif

#if REPRO_DEVICE_REMOVAL_ON_HARD_CODED_AO_COEF
        const float tMax = CB.RTAO_maxShadowRayHitTime;
        float tHit;
        if (TraceShadowRayAndReportIfHit(tHit, shadowRay, 0, tMax))
        {
            // ToDo Setting coef to 1 and then 0.5 below causes device removal. Wth?
            float occlusionCoef = 1;
            if (CB.RTAO_IsExponentialFalloffEnabled)
            {
                //float t = tHit / tMax;
                //float lambda = CB.RTAO_exponentialFalloffDecayConstant;
                occlusionCoef = 0;    // Causes device removal
                //occlusionCoef = 1 - t;  // Works...
            }
            occlussionCoefSum += occlusionCoef;// (1.f - CB.RTAO_minimumAmbientIllumnination) * occlusionCoef;

            numShadowRayHits++;
        }
    }
    float occlusionCoef = saturate(occlussionCoefSum / numSamples);
    float ambientCoef = 1.f - occlusionCoef;
#elif REPRO_INVISIBLE_WALL
        const float tMax = CB.RTAO_maxShadowRayHitTime;
        float tHit;
        if (TraceShadowRayAndReportIfHit(tHit, shadowRay, 0, tMax))
        {
            float occlusionCoef = 0;
            float t = tHit / tMax;
            if (CB.RTAO_IsExponentialFalloffEnabled && t >= CB.RTAO_minimumAmbientIllumnination)
            {
                float lambda = CB.RTAO_exponentialFalloffDecayConstant;
                occlusionCoef = 1 - t;// exp(-lambda * t*t);
            }
            //occlussionCoefSum += occlusionCoef;// (1.f - CB.RTAO_minimumAmbientIllumnination) * occlusionCoef;
            occlussionCoefSum = max(occlusionCoef, occlussionCoefSum);// (1.f - CB.RTAO_minimumAmbientIllumnination) * occlusionCoef;

            numShadowRayHits++;
        }
    }
    occlussionCoefSum *= numSamples;
    float occlusionCoef = saturate(occlussionCoefSum / numSamples);
    float ambientCoef = 1.f - occlusionCoef;
#else

        float RTAO_TraceRayOffsetAlongNormal;
        float RTAO_TraceRayOffsetAlongRayDirection;
        const float tMax = CB.RTAO_maxShadowRayHitTime;
        float tHit;
        if (TraceShadowRayAndReportIfHit(tHit, shadowRay, CB.RTAO_TraceRayOffsetAlongRayDirection, tMax))
        {
            float occlusionCoef = 1;
            if (CB.RTAO_IsExponentialFalloffEnabled)
            {
                float theoreticalTMax = CB.RTAO_maxTheoreticalShadowRayHitTime;
                float t = tHit / theoreticalTMax;
                float lambda = CB.RTAO_exponentialFalloffDecayConstant;
                // Note: update tMax calculation on falloff expression change.
                occlusionCoef = exp(-lambda * t*t);
            }
            occlussionCoefSum += (1.f - CB.RTAO_minimumAmbientIllumnination) * occlusionCoef;

            minHitDistance = min(tHit, minHitDistance);
            numShadowRayHits++;
        }
    }
#if AO_ANY_HIT_FULL_OCCLUSION
    float ambientCoef = numShadowRayHits > 0 ? 0 : 1;
#else
    float occlusionCoef = saturate(occlussionCoefSum / numSamples);
    float ambientCoef = 1.f - occlusionCoef;
#endif
#endif
	
    // Approximate interreflections of light from blocking surfaces which are generally not completely dark.
    // Ref: Ch 11.3.3 Accounting for Interreflections, Real-Time Rendering (4th edition).
    // The approximation assumes:
    //      o All surfaces incoming and outgoing radiance is the same 
    //      o Current surface color is the same as that of the occluders
    // Since this sample uses scalar ambient coefficient, we use the scalar luminance of the surface color.
    // This will generally brighten the AO making it closer to the result of full Global Illumination, including interreflections.
    if (CB.RTAO_approximateInterreflections)
    {
        float kA = ambientCoef;
        float rho = CB.RTAO_diffuseReflectanceScale * RGBtoLuminance(surfaceAlbedo);

        ambientCoef = kA / (1 - rho * (1 - kA));
    }

	return ambientCoef;
}


float CalculateAO(in float3 hitPosition, in uint2 DTid, UINT numSamples, in float3 surfaceNormal)
{
	uint numShadowRayHits;
    float minHitDistance;
	return CalculateAO(numShadowRayHits, minHitDistance, DTid, numSamples, hitPosition, surfaceNormal);
}

//***************************************************************************
//********************------ Ray gen shader.. -------************************
//***************************************************************************


#if TEST_EARLY_EXIT
[shader("raygeneration")]
void MyRayGenShader_Visibility()
{
    if (g_texInputAOFrameAge[DispatchRaysIndex().xy] > 0)
    {
        g_rtVisibilityCoefficient[DispatchRaysIndex().xy] = 1;
    }
}
#else
[shader("raygeneration")]
void MyRayGenShader_Visibility()
{
	bool hit = g_texGBufferPositionHits[DispatchRaysIndex().xy] > 0;
	uint shadowRayHits = 0;
	bool inShadow = false;
	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DispatchRaysIndex().xy].xyz;
#if COMPRES_NORMALS
        float3 surfaceNormal = DecodeNormal(g_texGBufferNormal[DispatchRaysIndex().xy].xy);
#else
		float3 surfaceNormal = g_texGBufferNormal[DispatchRaysIndex().xy].xyz;
#endif
        Ray visibilityRay = { hitPosition + 0.001f * surfaceNormal, normalize(CB.lightPosition.xyz - hitPosition) };

        float tHit;
		inShadow = TraceShadowRayAndReportIfHit(tHit, visibilityRay, 0);
	}

	// ToDo add option to be true distance and do contact hardening
	g_rtVisibilityCoefficient[DispatchRaysIndex().xy] = inShadow ? 0 : 1;
}
#endif

// ToDo remove
inline Ray GenerateCameraRayViaInterpolation(uint2 index, in float3 cameraPosition, in float4x4 projectionToWorldWithCameraEyeAtOrigin, float2 jitter = float2(0, 0))
{
#if 1
    float2 xy = index + 0.5f; // center in the middle of the pixel.
    xy += jitter;
    float2 screenPos = xy / DispatchRaysDimensions().xy * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates.
    screenPos.y = -screenPos.y;

    // ToDo remove CB ref
    Ray ray;
    ray.origin = cameraPosition;
    ray.direction = normalize(CB.cameraAt.xyz - cameraPosition) + screenPos.x * CB.cameraRight.xyz + screenPos.y * CB.cameraUp.xyz;

#else
    float3 pos00 = ScreenPosToWorldPos(uint2(0, 0), projectionToWorldWithCameraEyeAtOrigin);
    float3 pos10 = ScreenPosToWorldPos(uint2(1, 0), projectionToWorldWithCameraEyeAtOrigin);
    float3 pos01 = ScreenPosToWorldPos(uint2(0, 1), projectionToWorldWithCameraEyeAtOrigin);

    Ray ray;
    ray.origin = cameraPosition;
    ray.direction = normalize(pos00 + index.x * (pos10 - pos00) + index.y * (pos01 - pos00));
#endif
    return ray;
}


[shader("raygeneration")]
void MyRayGenShader_GBuffer()
{
    // ToDo make sure all pixels get written to or clear buffers beforehand. 

	// Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
	Ray ray = GenerateCameraRay(DispatchRaysIndex().xy, CB.cameraPosition.xyz, CB.projectionToWorldWithCameraEyeAtOrigin, CB.cameraJitter);
    //Ray ray = GenerateCameraRayViaInterpolation(DispatchRaysIndex().xy, CB.cameraPosition.xyz, CB.projectionToWorldWithCameraEyeAtOrigin, cameraJitter);

    Ray rx, ry;
    GetAuxilaryCameraRays(CB.cameraPosition.xyz, CB.projectionToWorldWithCameraEyeAtOrigin, rx, ry);

	// Cast a ray into the scene and retrieve GBuffer information.
	UINT currentRayRecursionDepth = 0;
	GBufferRayPayload rayPayload = TraceGBufferRay(ray, rx, ry, currentRayRecursionDepth);

	// Write out GBuffer information to rendertargets.
	// ToDo Test conditional write on all output 
	g_rtGBufferCameraRayHits[DispatchRaysIndex().xy] = (rayPayload.hit ? 1 : 0);
    g_rtGBufferMaterialInfo[DispatchRaysIndex().xy] = rayPayload.materialInfo;

    // ToDo Use calculated hitposition based on distance from GBuffer instead?
	g_rtGBufferPosition[DispatchRaysIndex().xy] = float4(rayPayload.hitPosition, 0);

    float rayLength = DISTANCE_ON_MISS;
    float obliqueness = 0;

    
#if  REPRO_BLOCKY_ARTIFACTS_NONUNIFORM_CB_REFERENCE_SSAO // CB value is incorrect on rayPayload.hit boundaries causing blocky artifacts when within if (hit) block
    if (rayPayload.hit)
    {
        float3 raySegment = rayPayload.hitPosition - CB.cameraPosition.xyz;
#else

    // ToDo dedupe
    //float4 viewSpaceHitPosition = float4(rayPayload.hitPosition - CB.cameraPosition.xyz, 1);
    if (rayPayload.hit)
    {
        // Calculate depth value.
        //float4 homogeneousScreenSpaceHitPosition = mul(viewSpaceHitPosition, CB.viewProjection);
        //float4 screenSpaceHitPosition = homogeneousScreenSpaceHitPosition / homogeneousScreenSpaceHitPosition.w;


#endif
       // float forwardFacing = dot(rayPayload.surfaceNormal, raySegment) / rayLength;
       // obliqueness = -forwardFacing;// min(f16tof32(0x7BFF), rcp(max(forwardFacing, 1e-5)));
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
        //obliqueness = -dot(rayPayload.surfaceNormal, rayPayload.hitPosition);
#endif
        rayLength = rayPayload.tHit;
        obliqueness = rayPayload.obliqueness;
    }


#if COMPRES_NORMALS
    // compress normal
    // ToDo normalize depth to [0,1] as floating point has higher precision around 0.
    // ToDo need to normalize hit distance as well
#if USE_NORMALIZED_Z
     float linearDistance = NormalizeToRange(rayLength, CB.Zmin, CB.Zmax);
#else
    float linearDistance = rayLength;// (rayLength - CB.Zmin) / (CB.Zmax - CB.Zmin);
#endif
    // Calculate z-depth
    float3 cameraDirection = GenerateForwardCameraRayDirection(CB.projectionToWorldWithCameraEyeAtOrigin);
    float linearDepth = linearDistance * dot(ray.direction, cameraDirection);

#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
    float rxLinearDepth = rayPayload.rxTHit * dot(rx.direction, cameraDirection);
    float ryLinearDepth = rayPayload.ryTHit * dot(ry.direction, cameraDirection);
    float2 ddxy = abs(float2(rxLinearDepth, ryLinearDepth) - linearDepth);
    g_rtPartialDepthDerivatives[DispatchRaysIndex().xy] = ddxy;
#endif
#if 1
    float nonLinearDepth = rayPayload.hit ?
        (FAR_PLANE + NEAR_PLANE - 2.0 * NEAR_PLANE * FAR_PLANE / linearDepth) / (FAR_PLANE - NEAR_PLANE)
        : 1;
    nonLinearDepth = (nonLinearDepth + 1.0) / 2.0;
    //linearDepth = rayLength = nonLinearDepth;
#endif

    g_rtGBufferNormal[DispatchRaysIndex().xy] = float4(EncodeNormal(rayPayload.surfaceNormal), linearDepth, obliqueness);
#else
    #if PACK_NORMAL_AND_DEPTH
        obliqueness = rayLength;
    #endif
    g_rtGBufferNormal[DispatchRaysIndex().xy] = float4(rayPayload.surfaceNormal, obliqueness);
#endif
    g_rtGBufferDistance[DispatchRaysIndex().xy] = linearDepth;

    // ToDo revise + check https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
    // Convert distance to nonLinearDepth.
    {

        g_rtGBufferDepth[DispatchRaysIndex().xy] = linearDepth;// nonLinearDepth;
    }
    g_rtGBufferNormalRGB[DispatchRaysIndex().xy] = rayPayload.hit ? float4(rayPayload.surfaceNormal, 0) : float4(0,0,0,0);
}

[shader("raygeneration")]
void MyRayGenShader_AO()
{
    uint2 DTid = DispatchRaysIndex().xy;

	bool hit = g_texGBufferPositionHits[DTid] > 0;
	uint numShadowRayHits = 0;
	float ambientCoef = 1;  // ToDo 1 or 0?
    float minHitDistance = HitDistanceOnMiss;
	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
#if COMPRES_NORMALS
        float3 surfaceNormal = DecodeNormal(g_texGBufferNormal[DispatchRaysIndex().xy].xy);
#else
		float3 surfaceNormal = g_texGBufferNormal[DTid].xyz;
#endif

        float3 surfaceAlbedo = float3(1, 1, 1);
        if (CB.RTAO_approximateInterreflections)
        {
            uint2 materialInfo = g_texGBufferMaterialInfo[DTid];
            UINT materialID;
            DecodeMaterial16b(materialInfo, materialID, surfaceAlbedo);
            surfaceAlbedo = surfaceAlbedo;
        }

        UINT numSamples = CB.numSamplesToUse;

        if (CB.RTAO_UseAdaptiveSampling)
        {
            float filterWeightSum = g_filterWeightSum[DTid].x;
            float clampedFilterWeightSum = min(filterWeightSum, CB.RTAO_AdaptiveSamplingMaxWeightSum);
            float sampleScale = 1 - (clampedFilterWeightSum / CB.RTAO_AdaptiveSamplingMaxWeightSum);
            
            UINT minSamples = CB.RTAO_AdaptiveSamplingMinSamples;
            UINT extraSamples = CB.numSamplesToUse - minSamples;

            if (CB.RTAO_AdaptiveSamplingMinMaxSampling)
            {
                numSamples = minSamples + (sampleScale >= 0.001 ? extraSamples : 0);
            }
            else
            {
                float scaleExponent = CB.RTAO_AdaptiveSamplingScaleExponent;
                numSamples = minSamples + UINT(pow(sampleScale, scaleExponent) * extraSamples);
            }
        }
		ambientCoef = CalculateAO(numShadowRayHits, minHitDistance, DTid, numSamples, hitPosition, surfaceNormal, surfaceAlbedo);
	}

	g_rtAOcoefficient[DispatchRaysIndex().xy] = ambientCoef;
#if GBUFFER_AO_COUNT_AO_HITS
	// ToDo test perf impact of writing this
	g_rtAORayHits[DispatchRaysIndex().xy] = numShadowRayHits;
#endif

    if (CB.useShadowRayHitTime)
    {
#if USE_NORMALIZED_Z
        minHitDistance *= 1 / (CB.Zmax - CB.Zmin); // ToDo pass by CB? 
#endif
      g_rtAORayHitDistance[DispatchRaysIndex().xy] = minHitDistance;
    }
}

[shader("raygeneration")]
void MyRayGenShaderQuarterRes_AO()
{
    uint2 DTid = DispatchRaysIndex().xy * 2;

	bool hit = g_texGBufferPositionHits[DTid] > 0;
	uint numShadowRayHits = 0;
	float ambientCoef = 0;
    float minHitDistance;
	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
#if COMPRES_NORMALS
        float3 surfaceNormal = DecodeNormal(g_texGBufferNormal[DispatchRaysIndex().xy].xy);
#else
		float3 surfaceNormal = g_texGBufferNormal[DTid].xyz;
#endif
        // ToDo Standardize naming AO vs AmbientOcclusion ?
		ambientCoef = CalculateAO(numShadowRayHits, minHitDistance, DTid, CB.numSamplesToUse, hitPosition, surfaceNormal);
	}

	g_rtAOcoefficient[DispatchRaysIndex().xy] = ambientCoef;
#if GBUFFER_AO_COUNT_AO_HITS
	// ToDo test perf impact of writing this
	g_rtAORayHits[DispatchRaysIndex().xy] = numShadowRayHits;
#endif
}

//***************************************************************************
//******************------ Closest hit shaders -------***********************
//***************************************************************************

// ToDo remove
[shader("closesthit")]
void MyClosestHitShader(inout RayPayload rayPayload, in BuiltInTriangleIntersectionAttributes attr)
{
#if ONLY_SQUID_SCENE_BLAS
	uint startIndex = PrimitiveIndex() * 3;
	const uint3 indices = {l_indices[startIndex], l_indices[startIndex + 1], l_indices[startIndex + 2]};
#else
	// Get the base index of the triangle's first 16 bit index.
	uint indexSizeInBytes = 2;
	uint indicesPerTriangle = 3;
	uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;

	uint baseIndex = PrimitiveIndex() * triangleIndexStride;

	// Load up three 16 bit indices for the triangle.
	const uint3 indices = Load3x16BitIndices(baseIndex, l_indices);
#endif
	// Retrieve corresponding vertex normals for the triangle vertices.
    UINT vbID = l_materialCB.isVertexAnimated & (CB.currentFrameVBindex & 1);
    float3 vertexNormals[3] = {
        l_vertices[vbID][indices[0]].normal,
        l_vertices[vbID][indices[1]].normal,
        l_vertices[vbID][indices[2]].normal };

#if FLAT_FACE_NORMALS
	BuiltInTriangleIntersectionAttributes attrCenter;
	attrCenter.barycentrics.x = attrCenter.barycentrics.y = 1.f / 3;
	float3 triangleNormal = normalize(HitAttribute(vertexNormals, attrCenter));
#else
	float3 triangleNormal = HitAttribute(vertexNormals, attr);
#endif

    // PERFORMANCE TIP: it is recommended to avoid values carry over across TraceRay() calls. 
    // Therefore, in cases like retrieving HitWorldPosition(), it is recomputed every time.
#if AO_ONLY
	float ambientCoef = CalculateAO(HitWorldPosition(), 0, CB.numSamplesToUse, triangleNormal);
	float4 color = ambientCoef * float4(0.75, 0.75, 0.75, 0.75);
#else
    // Shadow component.
    // Trace a shadow ray.
    float3 hitPosition = HitWorldPosition();
	
	// ToDo
//    Ray shadowRay = { hitPosition + 0.0001f * triangleNormal, normalize(CB.lightPosition.xyz - hitPosition) };
  //  bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, rayPayload.recursionDepth);
	
    // Calculate final color.
	float ambientCoef = CalculateAO(HitWorldPosition(), 0, CB.numSamplesToUse, triangleNormal);
	float4 color = float4(1, 0, 0, 0); //ToDo
    //float3 phongColor = CalculatePhongLighting(triangleNormal, shadowRayHit, ambient, l_materialCB.diffuse, l_materialCB.specular, l_materialCB.specularPower);
	//float4 color =  float4(phongColor, 1);
#endif     

    rayPayload.color = color;
}

[shader("closesthit")]
void MyClosestHitShader_ShadowRay(inout ShadowRayPayload rayPayload, in BuiltInTriangleIntersectionAttributes attr)
{
    rayPayload.tHit = RayTCurrent();
}

[shader("closesthit")]
void MyClosestHitShader_GBuffer(inout GBufferRayPayload rayPayload, in BuiltInTriangleIntersectionAttributes attr)
{
#if ONLY_SQUID_SCENE_BLAS
	uint startIndex = PrimitiveIndex() * 3;
	const uint3 indices = { l_indices[startIndex], l_indices[startIndex + 1], l_indices[startIndex + 2] };
#else
	// Get the base index of the triangle's first 16 bit index.
	uint indexSizeInBytes = 2;
	uint indicesPerTriangle = 3;
	uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;

	uint baseIndex = PrimitiveIndex() * triangleIndexStride;

	// Load up three 16 bit indices for the triangle.
	const uint3 indices = Load3x16BitIndices(baseIndex, l_indices);
#endif

    // Retrieve vertices for the hit triangle.
    UINT vbID = l_materialCB.isVertexAnimated & (CB.currentFrameVBindex & 1);
    VertexPositionNormalTextureTangent vertices[3] = {
        l_vertices[vbID][indices[0]],
        l_vertices[vbID][indices[1]],
        l_vertices[vbID][indices[2]]};

    float2 vertexTexCoords[3] = { vertices[0].textureCoordinate, vertices[1].textureCoordinate, vertices[2].textureCoordinate };
    float2 texCoord = HitAttribute(vertexTexCoords, attr);

    UINT materialID = l_materialCB.materialID;
    PrimitiveMaterialBuffer material = g_materials[materialID];

    // Load triangle normal.
    float3 normal;
    float3 objectNormal;
    {
        // Retrieve corresponding vertex normals for the triangle vertices.
        float3 vertexNormals[3] = { vertices[0].normal, vertices[1].normal, vertices[2].normal };

#if FLAT_FACE_NORMALS
        BuiltInTriangleIntersectionAttributes attrCenter;
        attrCenter.barycentrics.x = attrCenter.barycentrics.y = 1.f / 3;
        // ToDo input normals should be normalized already
        objectNormal = normalize(HitAttribute(vertexNormals, attrCenter));
#else
        objectNormal = normalize(HitAttribute(vertexNormals, attr));
#endif
#if !FACE_CULLING
        float orientation = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 1 : -1;
        objectNormal *= orientation;
#endif

        // BLAS Transforms in this sample are uniformly scaled so it's OK to directly apply the BLAS transform.
        // ToDo add a note that the transform is expected to have uniform scaling
        normal = normalize(mul((float3x3)ObjectToWorld3x4(), objectNormal));
    }

    rayPayload.obliqueness = dot(normal, -WorldRayDirection());

#if 0
    float3 hitPosition = HitAttribute(vertices, attr);
    
#else
    float3 hitPosition = HitWorldPosition();
#endif
    float2 ddx = 0;
    float2 ddy = 0;

    // Calculate auxilary rays' intersection points with the triangle.
    float3 px, py;
    px = RayPlaneIntersection(hitPosition, normal, rayPayload.rx.origin, rayPayload.rx.direction);
    py = RayPlaneIntersection(hitPosition, normal, rayPayload.ry.origin, rayPayload.ry.direction);

    if (material.hasDiffuseTexture || 
        (CB.RTAO_UseNormalMaps && material.hasNormalTexture) ||
        (material.type == MaterialType::AnalyticalCheckerboardTexture))
    {
        float3 vertexTangents[3] = { vertices[0].tangent, vertices[1].tangent, vertices[2].tangent };
        float3 tangent = HitAttribute(vertexTangents, attr);
        float3 bitangent = normalize(cross(tangent, normal));
        
        CalculateUVDerivatives(normal, tangent, bitangent, hitPosition, px, py, ddx, ddy);
        // ToDo. Lower by 0.5 to sharpen texture filtering.
        //ddx *= 0.5;
        //ddy *= 0.5;
    }

    // Apply NormalMap
    if (CB.RTAO_UseNormalMaps && material.hasNormalTexture)
    {
        float3 tangent;
        if (material.hasPerVertexTangents)
        {
            float3 vertexTangents[3] = { vertices[0].tangent, vertices[1].tangent, vertices[2].tangent };
            tangent = HitAttribute(vertexTangents, attr);
        }
        else
        {
            float3 v0 = vertices[0].position;
            float3 v1 = vertices[1].position;
            float3 v2 = vertices[2].position;
            float2 uv0 = vertices[0].textureCoordinate;
            float2 uv1 = vertices[1].textureCoordinate;
            float2 uv2 = vertices[2].textureCoordinate;
            tangent = CalculateTangent(v0, v1, v2, uv0, uv1, uv2);
        }

        float3 bumpNormal = normalize(l_texNormalMap.SampleGrad(LinearWrapSampler, texCoord, ddx, ddy).xyz) * 2.f - 1.f;
        normal = BumpMapNormalToWorldSpaceNormal(bumpNormal, normal, tangent);
    }

    UINT rayRecursionDepth = rayPayload.rayRecursionDepth;
#if ALLOW_MIRRORS
    if (material.isMirror && rayPayload.rayRecursionDepth < MAX_RAY_RECURSION_DEPTH)
    {
        // ToDo offset hitposition, add comment.
#if TURN_MIRRORS_SEETHROUGH
        // Calculate refracted rx, ry
        // No refraction ,just passing the rays through.
        Ray rx = rayPayload.rx;
        Ray ry = rayPayload.ry;
        Ray ray = { hitPosition + 0.05f * WorldRayDirection(), WorldRayDirection() };
#else
        // Calculate reflected rx, ry
        Ray rx = { px, reflect(rayPayload.rx.direction, normal) };
        Ray ry = { py, reflect(rayPayload.ry.direction, normal) };
        Ray ray = { hitPosition, reflect(WorldRayDirection(), normal) };
#endif

        rayPayload = TraceGBufferRay(ray, rx, ry, rayPayload.rayRecursionDepth, NEAR_PLANE, FAR_PLANE - RayTCurrent());
        rayPayload.tHit += RayTCurrent();
	}
    else
#endif
    {
        float3 diffuse;
        if (material.type == MaterialType::Default)
        {
            if (material.hasDiffuseTexture)
            {
                diffuse = l_texDiffuse.SampleGrad(LinearWrapSampler, texCoord, ddx, ddy).xyz;
            }
            else
            {
                diffuse = material.diffuse;
            }
        } 
        else if (material.type == MaterialType::AnalyticalCheckerboardTexture)
        {
#if 0 
            float2 ddx_uv;
            float2 ddy_uv;
            float2 uv = TexCoords(hitPosition);

            CalculateRayDifferentials(ddx_uv, ddy_uv, uv, hitPosition, normal, CB.cameraPosition.xyz, CB.projectionToWorldWithCameraEyeAtOrigin);
            diffuse = CheckersTextureBoxFilter(uv, ddx_uv, ddy_uv, 25);
#else // ToDo fix derivatives
            float2 uv = hitPosition.xz / 2;
            float checkers = CheckersTextureBoxFilter(uv, ddx, ddy);
            diffuse = checkers > 0.5 ? float3(50, 68, 90) / 255 : float3(170, 170, 170) / 255;
#endif
        }
        rayPayload.hit = true;
        rayPayload.materialInfo = EncodeMaterial16b(materialID, diffuse);
        rayPayload.hitPosition = hitPosition;
        rayPayload.surfaceNormal = normal;
        rayPayload.tHit = RayTCurrent();
        rayPayload.instanceIndex = InstanceIndex();
        rayPayload.hitObjectPosition = HitObjectPosition();
        rayPayload.objectNormal = objectNormal;

#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        float3 rxRaySegment = px - rayPayload.rx.origin;
        float3 ryRaySegment = py - rayPayload.ry.origin;
        rayPayload.rxTHit += length(rxRaySegment);
        rayPayload.ryTHit += length(ryRaySegment);
        rayPayload.BLASinstanceIndex = InstanceIndex();
#endif
    }


    if (rayRecursionDepth == 1)
    {
        float2 motionVector;
        float prevFrameDepth;
        float3 prevFrameNormal;

        // Primary ray hits.
        if (rayPayload.rayRecursionDepth == 1)
        {
            CalculateMotionVector(
                motionVector,
                prevFrameDepth,
                prevFrameNormal,
                InstanceIndex(),
                HitObjectPosition(),
                objectNormal,
                indices,
                attr);
        }
        // Single reflected/refracted ray hits.
        else if (rayPayload.rayRecursionDepth == 2)
        {
            CalculateMotionVectorForReflectedPoint(
                motionVector,
                prevFrameDepth,
                prevFrameNormal,
                rayPayload.instanceIndex,
                rayPayload.hitObjectPosition,
                rayPayload.objectNormal,
                indices,
                attr,
                InstanceIndex(),
                HitObjectPosition(),
                objectNormal);
        }
        // Multiple reflected/refracted ray hits.
        else
        {
            // Not supported
            // Set motion vetor to point out of bounds.
            motionVector = float2(1, 1);
        }
        // Store the motion vector.
        // ToDo store it here or pass via rayPayload?
        g_rtTextureSpaceMotionVector[DispatchRaysIndex().xy] = motionVector;
        g_rtReprojectedHitPosition[DispatchRaysIndex().xy] = float4(prevFrameNormal, prevFrameDepth);
    }
}


//***************************************************************************
//**********************------ Miss shaders -------**************************
//***************************************************************************

[shader("miss")]
void MyMissShader(inout RayPayload rayPayload)
{
	
    rayPayload.color = BackgroundColor;
}

[shader("miss")]
void MyMissShader_ShadowRay(inout ShadowRayPayload rayPayload)
{
    rayPayload.tHit = HitDistanceOnMiss;
}

// ToDo - remove miss shader for GBuffer
[shader("miss")]
void MyMissShader_GBuffer(inout GBufferRayPayload rayPayload)
{
	rayPayload.hit = false;
#if USE_ENVIRONMENT_MAP
    float3 color = g_texEnvironmentMap.SampleLevel(LinearWrapSampler, WorldRayDirection(), 0).xyz;
    rayPayload.materialInfo = EncodeMaterial16b(0, color);
#endif
}


#endif // RAYTRACING_HLSL