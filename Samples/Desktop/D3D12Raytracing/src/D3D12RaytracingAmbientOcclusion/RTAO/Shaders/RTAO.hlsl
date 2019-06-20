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
#include "..\..\RaytracingHlslCompat.h"
#include "..\..\RaytracingShaderHelper.hlsli"
#include "..\..\RandomNumberGenerator.hlsli"
#include "RaySorting.hlsli"

#define HitDistanceOnMiss -1        // ToDo unify with DISTANCE_ON_MISS - should be 0 as we're using non-negative low precision formats

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
Texture2D<float> g_texShadowMap : register(t21);
Texture2D<float4> g_texAORaysDirectionOriginDepthHit : register(t22);
Texture2D<uint2> g_texAOSourceToSortedRayIndex : register(t23);


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
RWTexture2D<float4> g_rtColor : register(u19);
RWTexture2D<float4> g_rtAODiffuse : register(u20);
RWTexture2D<float> g_rtShadowMap : register(u21);
RWTexture2D<float4> g_rtAORaysDirectionOriginDepth : register(u22);
RWTexture2D<float4> g_rtGBufferNormalDepthLowPrecision : register(u23);

ConstantBuffer<SceneConstantBuffer> CB : register(b0);          // ToDo standardize CB var naming
StructuredBuffer<PrimitiveMaterialBuffer> g_materials : register(t3);
StructuredBuffer<AlignedHemisphereSample3D> g_sampleSets : register(t4);

StructuredBuffer<float3x4> g_prevFrameBottomLevelASInstanceTransform : register(t15);


SamplerState LinearWrapSampler : register(s0);
SamplerComparisonState ShadowMapSamplerComp : register(s1);
SamplerState ShadowMapSampler : register(s2);

// ToDo: https://developer.nvidia.com/content/understanding-structured-buffer-performance
/*******************************************************************************************************/
// Per-object resources
ConstantBuffer<PrimitiveConstantBuffer> l_materialCB : register(b0, space1);

#if ONLY_SQUID_SCENE_BLAS
StructuredBuffer<Index> l_indices : register(t0, space1);
StructuredBuffer<VertexPositionNormalTextureTangent> l_vertices : register(t1, space1); // Current frame vertex buffer.
StructuredBuffer<VertexPositionNormalTextureTangent> l_verticesPrevFrame : register(t2, space1); 
#else
ByteAddressBuffer l_indices : register(t0, space1);
StructuredBuffer<VertexPositionNormalTexture> l_vertices : register(t1, space1); // Current frame vertex buffer.
StructuredBuffer<VertexPositionNormalTexture> l_verticesPrevFrame : register(t2, space1);
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

float3 ReflectPointThroughPlane(in float3 P, in float3 planeNormal, in float3 pointOnThePlane)
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


// Reflects a point across a planar mirror. 
// Returns FLT_MAX if the point is behind the mirror.
float3 ReflectFrontPointThroughPlane(
    in float3 p,
    in float3 mirrorSurfacePoint,
    in float3 mirrorNormal)
{
    if (!IsPointOnTheNormalSideOfPlane(p, mirrorNormal, mirrorSurfacePoint))
    {
        // ToDo attempt direct lookup using hit position in raygen instead of reflected point?
        return FLT_MAX; // ToDo is this safe?
    }

    return ReflectPointThroughPlane(p, mirrorNormal, mirrorSurfacePoint);
}

float3 GetWorldHitPositionInPreviousFrame(
    in float3 hitObjectPosition,
    in uint BLASInstanceIndex,
    in uint3 vertexIndices,
    in BuiltInTriangleIntersectionAttributes attr,
    out float3x4 _BLASTransform)
{
    // Variables prefixed with underscore _ denote values in the previous frame.

    // Calculate hit object position of the hit in the previous frame.
    float3 _hitObjectPosition;
    if (l_materialCB.isVertexAnimated)
    {
        float3 _vertices[3] = {
            l_verticesPrevFrame[vertexIndices[0]].position,
            l_verticesPrevFrame[vertexIndices[1]].position,
            l_verticesPrevFrame[vertexIndices[2]].position };
        _hitObjectPosition = HitAttribute(_vertices, attr);
    }
    else // non-vertex animated geometry        // ToDo apply this at declaration instead and avoid else?
    {
        _hitObjectPosition = hitObjectPosition;
    }

    // Transform the hit object position to world space.
    _BLASTransform = g_prevFrameBottomLevelASInstanceTransform[BLASInstanceIndex];
    return mul(_BLASTransform, float4(_hitObjectPosition, 1));
}


//***************************************************************************
//*********************------ TraceRay wrappers. -------*********************
//***************************************************************************

// Trace a shadow ray and return true if it hits any geometry.
bool TraceAORayAndReportIfHit(out float tHit, in Ray ray, in UINT currentRayRecursionDepth, in bool retrieveTHit = true, in float TMax = 10000, in bool acceptFirstHit = false)
{
    if (currentRayRecursionDepth >= CB.maxShadowRayRecursionDepth)
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
        | RAY_FLAG_CULL_NON_OPAQUE;             // ~skip transparent objects
    
    if (acceptFirstHit || !retrieveTHit)
    {
        // Performance TIP: Accept first hit if true hit is not neeeded,
        // or has minimal to no impact (in AO). The peformance gain can
        // be substantial.
        rayFlags |= RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    }

    // Skip closest hit shaders of tHit time is not needed.
    if (!retrieveTHit) 
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

    return shadowPayload.tHit > 0;
}

bool TraceAORayAndReportIfHit(out float tHit, in Ray ray, in float3 N, in UINT currentRayRecursionDepth, in bool retrieveTHit = true, in float TMax = 10000)
{
    // Only trace if the surface is facing the target.
    if (dot(ray.direction, N) > 0)
    {
        return TraceAORayAndReportIfHit(tHit, ray, currentRayRecursionDepth, retrieveTHit, TMax);
    }
    return false;
}

bool TraceAORayAndReportIfHit(in float3 hitPosition, in float3 direction, in float3 N, in GBufferRayPayload rayPayload, in float TMax = 10000)
{
    float tOffset = 0.001f;
    Ray visibilityRay = { hitPosition + tOffset * N, direction };
    float dummyTHit;
    return TraceAORayAndReportIfHit(dummyTHit, visibilityRay, N, rayPayload.rayRecursionDepth, false, TMax);
}


// ToDo comment
// MinHitDistance - minimum hit distance of all AO ray hits.
// ToDo ensure AO doesn't cast beyond max recursion depth
float CalculateAO(out uint numShadowRayHits, out float minHitDistance, in uint2 DTid, in UINT numSamples, in float3 hitPosition, in float3 surfaceNormal, in float3 surfaceAlbedo = float3(1,1,1), in float linearDepth = 0)
{
    numShadowRayHits = 0;
    minHitDistance = CB.RTAO_maxTheoreticalShadowRayHitTime;
    float occlussionCoefSum = 0;

    // Calculate coordinate system for the hemisphere.
    // ToDo AO has square alias due to same hemisphere
    float3 u, v, w;
    w = surfaceNormal;
#if 0
    w = float3(0, 1, 0);
#endif
    float3 right = float3(0.0072, 0.999994132f, 0.0034);
    v = normalize(cross(w, right));
    u = cross(v, w);
    

    // Calculate offsets to the pregenerated sample set.
    uint sampleSetJump;     // Offset to the start of the sample set
    uint sampleJump;        // Offset to the first sample for this pixel within a sample set.
    {
        // Neighboring samples NxN share a sample set, but use different samples within a set.
        // Sharing a sample set lets the pixels in the group get a better coverage of the hemisphere 
        // than if each pixel used a separate sample set with less samples per set.
        
        // Get a common sample set ID and seed shared across neighboring pixels.
        uint numSampleSetsInX = (DispatchRaysDimensions().x + CB.numPixelsPerDimPerSet - 1) / CB.numPixelsPerDimPerSet;
        uint2 sampleSetId = DTid / CB.numPixelsPerDimPerSet;

        // Get a common hitPosition to adjust the sampleSeed by. 
        // This breaks noise correlation on camera movement which otherwise results 
        // in noise pattern swimming across the screen on camera movement.
        uint2 pixelZeroId = sampleSetId * CB.numPixelsPerDimPerSet;
        float3 pixelZeroHitPosition = g_texGBufferPositionRT[pixelZeroId].xyz;      // ToDo remove?
        uint sampleSetSeed = (sampleSetId.y * numSampleSetsInX + sampleSetId.x) * hash(pixelZeroHitPosition) + CB.seed;
        uint RNGState = RNG::SeedThread(sampleSetSeed);
        
        sampleSetJump = RNG::Random(RNGState, 0, CB.numSampleSets - 1) * CB.numSamplesPerSet;

        // Get a pixel ID within the shared set across neighboring pixels.
        uint2 pixeIDPerSet2D = DTid % CB.numPixelsPerDimPerSet;
        uint pixeIDPerSet = pixeIDPerSet2D.y * CB.numPixelsPerDimPerSet + pixeIDPerSet2D.x;

        // Randomize starting sample position within a sample set per neighbor group 
        // to break group to group correlation resulting in square alias.
        uint numPixelsPerSet = CB.numPixelsPerDimPerSet * CB.numPixelsPerDimPerSet;
        sampleJump = pixeIDPerSet + RNG::Random(RNGState, 0, numPixelsPerSet - 1);
        sampleJump *= CB.numSamplesToUse;
    }
#if AO_PROGRESSIVE_SAMPLING
    sampleJump += numSamples * g_texInputAOFrameAge[DTid];
#endif

#if AO_SPP_N_MAX == 1
    uint i = 0;
#else
    for (uint i = 0; i < numSamples; i++)
#endif
    {
        // Load a pregenerated random sample from the sample set.
        float3 sample = g_sampleSets[sampleSetJump + ((sampleJump + i) % CB.numSamplesPerSet)].value;

        // ToDo remove unnecessary normalize()
        float3 rayDirection = normalize(sample.x * u + sample.y * v + sample.z * w);
        //rayDirection = normalize(float3(1, 1, 1));
        if (CB.RTAO_UseSortedRays)
        {
            uint2 DTid = DispatchRaysIndex().xy;
            g_rtAORaysDirectionOriginDepth[DTid] = float4(EncodeNormal(rayDirection), linearDepth, 0);
        }

        // ToDo hitPosition adjustment - fix crease artifacts
        // Todo fix noise on flat surface / box
        Ray shadowRay = { hitPosition + CB.RTAO_TraceRayOffsetAlongNormal * surfaceNormal, rayDirection };

        const float tMax = CB.RTAO_maxShadowRayHitTime;
        float tHit;
        if (TraceAORayAndReportIfHit(tHit, shadowRay, 0, CB.useShadowRayHitTime, tMax, true))
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
    float occlusionCoef = saturate(occlussionCoefSum / (float)numSamples);
    float ambientCoef = 1.f - occlusionCoef;
#endif
	
    // Approximate interreflections of light from blocking surfaces which are generally not completely dark and tend to have similar radiance.
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


[shader("raygeneration")]
void RayGenShader_AO()
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
        float4 normalDepth = g_texGBufferNormal[DTid];
        float3 surfaceNormal = DecodeNormal(normalDepth.xy);
        float linearDepth = normalDepth.z;
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
		ambientCoef = CalculateAO(numShadowRayHits, minHitDistance, DTid, numSamples, hitPosition, surfaceNormal, surfaceAlbedo, linearDepth);
	}
    else
    {
        if (CB.RTAO_UseSortedRays)
        {
            g_rtAORaysDirectionOriginDepth[DTid] = float4(0, 0, 0, 0);
        }
    }

    if (!CB.RTAO_UseSortedRays)
    {
        g_rtAOcoefficient[DTid] = ambientCoef;
    }
#if GBUFFER_AO_COUNT_AO_HITS
	// ToDo test perf impact of writing this
	g_rtAORayHits[DTid] = numShadowRayHits;
#endif

    if (CB.useShadowRayHitTime)
    {
#if USE_NORMALIZED_Z
        minHitDistance *= 1 / (CB.Zmax - CB.Zmin); // ToDo pass by CB? 
#endif
      g_rtAORayHitDistance[DTid] = minHitDistance;
    }
}

[shader("raygeneration")]
void RayGenShader_AO_sortedRays()
{
#if RTAO_RAY_SORT_1DRAYTRACE
    uint index1D = DispatchRaysIndex().x; 
    uint2 rayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);

    // Calculate 2D DTid from a 1D index where
    // - 1D index maps to a valid index with ray tracing buffer dimensions
    // - pixels are row major within a ray group.
    // - ray groups are row major within the raytracing buffer dimensions.
    // Adjust for the fact ray groups may go beyond raytracing dimension.

    // Find the ray group row index.
    uint numValidPixelsInRow = CB.raytracingDim.x;
    uint rowOfRayGroupSize = rayGroupDim.y * numValidPixelsInRow;
    uint rayGroupRowIndex = index1D / rowOfRayGroupSize;
    
    // ToDo replace module with subtraction
    // Find the ray group column index.
    uint numValidPixelsInColumn = CB.raytracingDim.y;
    uint numRowsInCurrentRayGroup = min((rayGroupRowIndex + 1) * rayGroupDim.y, numValidPixelsInColumn) - rayGroupRowIndex * rayGroupDim.y;
    uint currentRow_RayGroupSize = numRowsInCurrentRayGroup * rayGroupDim.x;
    uint index1DWithinRayGroupRow = index1D - rayGroupRowIndex * rowOfRayGroupSize;
    uint rayGroupColumnIndex = index1DWithinRayGroupRow / currentRow_RayGroupSize;
    uint2 rayGroupIndex = uint2(rayGroupColumnIndex, rayGroupRowIndex);

    // Find the thread offset index within the ray group.
    uint currentRayGroup_index1D = index1DWithinRayGroupRow - rayGroupIndex.x * currentRow_RayGroupSize;
    uint currentRayGroupWidth = min((rayGroupIndex.x + 1) * rayGroupDim.x, numValidPixelsInRow) - rayGroupIndex.x * rayGroupDim.x;
    uint rayThreadRowIndex = currentRayGroup_index1D / currentRayGroupWidth;
    uint rayThreadColumnIndex = currentRayGroup_index1D - rayThreadRowIndex * currentRayGroupWidth;
    uint2 rayThreadIndex = uint2(rayThreadColumnIndex, rayThreadRowIndex);
    
    uint2 DTid = rayGroupIndex * rayGroupDim + rayThreadIndex;
    uint2 rayGroupBase = rayGroupIndex * rayGroupDim;
    uint2 rayGroupRayIndex = g_texAOSourceToSortedRayIndex[DTid];
    uint2 rayIndex = rayGroupBase + rayGroupRayIndex;
    

    uint numShadowRayHits = 0;
    float ambientCoef = 1;  // ToDo 1 or 0?
    float occlussionCoefSum = 0;
    float minHitDistance = CB.RTAO_maxTheoreticalShadowRayHitTime;
    uint numSamples = 1;

    if (IsActiveRay(rayGroupRayIndex))
    {
#else
    uint2 DTid = DispatchRaysIndex().xy;
    uint2 rayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);
    uint2 rayGroupBase = (DTid / rayGroupDim) * rayGroupDim;
    uint2 rayGroupRayIndex = g_texAOSourceToSortedRayIndex[DTid];
    uint2 rayIndex = rayGroupBase + rayGroupThreadIndex;
    ToDo
#endif

        //    bool hit = g_texGBufferPositionHits[DTid] > 0;
        float3 rayDirectionOrigin = g_texAORaysDirectionOriginDepthHit[rayIndex].xyz;
        float rayOriginDepth = rayDirectionOrigin.z;

        float3 rayDirection = DecodeNormal(rayDirectionOrigin.xy);
        float3 hitPosition = g_texGBufferPositionRT[rayIndex].xyz;

        // ToDo remove?
        float3 surfaceNormal = DecodeNormal(g_texGBufferNormal[rayIndex].xy);


        // ToDo hitPosition adjustment - fix crease artifacts
        // Todo fix noise on flat surface / box
        // ToDo use normal
        Ray shadowRay = { hitPosition + CB.RTAO_TraceRayOffsetAlongNormal * surfaceNormal, rayDirection };
        //Ray shadowRay = { hitPosition + CB.RTAO_TraceRayOffsetAlongNormal * rayDirection, rayDirection };

        const float tMax = CB.RTAO_maxShadowRayHitTime; // ToDo make sure its FLT_10BIT_MAX or less since we use 10bit origin depth in RaySort
        float tHit;
        if (TraceAORayAndReportIfHit(tHit, shadowRay, 0, CB.useShadowRayHitTime, tMax, true))
        {
            float occlusionCoef = 1;
            if (CB.RTAO_IsExponentialFalloffEnabled)
            {
                float theoreticalTMax = CB.RTAO_maxTheoreticalShadowRayHitTime;
                float t = tHit / theoreticalTMax;
                float lambda = CB.RTAO_exponentialFalloffDecayConstant;
                // Note: update tMax calculation on falloff expression change.
                occlusionCoef = exp(-lambda * t * t);
            }
            occlussionCoefSum += (1.f - CB.RTAO_minimumAmbientIllumnination) * occlusionCoef;

            minHitDistance = min(tHit, minHitDistance);
            numShadowRayHits++;
        }
    
#if AO_ANY_HIT_FULL_OCCLUSION
        ambientCoef = numShadowRayHits > 0 ? 0 : 1;
#else
        float occlusionCoef = saturate(occlussionCoefSum / (float)numSamples);
        ambientCoef = 1.f - occlusionCoef;
#endif

        // Approximate interreflections of light from blocking surfaces which are generally not completely dark and tend to have similar radiance.
        // Ref: Ch 11.3.3 Accounting for Interreflections, Real-Time Rendering (4th edition).
        // The approximation assumes:
        //      o All surfaces incoming and outgoing radiance is the same 
        //      o Current surface color is the same as that of the occluders
        // Since this sample uses scalar ambient coefficient, we use the scalar luminance of the surface color.
        // This will generally brighten the AO making it closer to the result of full Global Illumination, including interreflections.
        if (CB.RTAO_approximateInterreflections)
        {
            uint2 materialInfo = g_texGBufferMaterialInfo[rayIndex];
            UINT materialID;
            float3 surfaceAlbedo;
            DecodeMaterial16b(materialInfo, materialID, surfaceAlbedo);

            float kA = ambientCoef;
            float rho = CB.RTAO_diffuseReflectanceScale * RGBtoLuminance(surfaceAlbedo);

            ambientCoef = kA / (1 - rho * (1 - kA));
        }
    }

#if AVOID_SCATTER_WRITES_FOR_SORTED_RAY_RESULTS
    g_rtAOcoefficient[DTid] = ambientCoef;
#if GBUFFER_AO_COUNT_AO_HITS
    // ToDo test perf impact of writing this
    g_rtAORayHits[DTid] = numShadowRayHits;
#endif

    if (CB.useShadowRayHitTime)
    {
#if USE_NORMALIZED_Z
        minHitDistance *= 1 / (CB.Zmax - CB.Zmin); // ToDo pass by CB? 
#endif
        //g_rtAORayHitDistance[DTid] = minHitDistance;
        g_rtAORayHitDistance[DTid] = minHitDistance;
    }
#else
    g_rtAOcoefficient[rayIndex] = ambientCoef;
#if GBUFFER_AO_COUNT_AO_HITS
    // ToDo test perf impact of writing this
    g_rtAORayHits[rayIndex] = numShadowRayHits;
#endif

    if (CB.useShadowRayHitTime)
    {
#if USE_NORMALIZED_Z
        minHitDistance *= 1 / (CB.Zmax - CB.Zmin); // ToDo pass by CB? 
#endif
        //g_rtAORayHitDistance[DTid] = minHitDistance;
        g_rtAORayHitDistance[rayIndex] = minHitDistance;
    }
#endif
}


[shader("raygeneration")]
void RayGenShaderQuarterRes_AO()
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
        float3 surfaceNormal = DecodeNormal(g_texGBufferNormal[DTid].xy);
#else
		float3 surfaceNormal = g_texGBufferNormal[DTid].xyz;
#endif
        // ToDo Standardize naming AO vs AmbientOcclusion ?
		ambientCoef = CalculateAO(numShadowRayHits, minHitDistance, DTid, CB.numSamplesToUse, hitPosition, surfaceNormal);
	}

	g_rtAOcoefficient[DTid] = ambientCoef;
#if GBUFFER_AO_COUNT_AO_HITS
	// ToDo test perf impact of writing this
	g_rtAORayHits[DTid] = numShadowRayHits;
#endif
}

//***************************************************************************
//******************------ Closest hit shaders -------***********************
//***************************************************************************

[shader("closesthit")]
void ClosestHitShader_AORay(inout ShadowRayPayload rayPayload, in BuiltInTriangleIntersectionAttributes attr)
{
    rayPayload.tHit = RayTCurrent();
}

//***************************************************************************
//**********************------ Miss shaders -------**************************
//***************************************************************************

[shader("miss")]
void MissShader_AORay(inout ShadowRayPayload rayPayload)
{
    rayPayload.tHit = HitDistanceOnMiss;
}



#endif // RAYTRACING_HLSL