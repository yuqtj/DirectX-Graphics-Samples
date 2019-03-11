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

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "RandomNumberGenerator.hlsli"
#include "SSAO/GlobalSharedHlslCompat.h" // ToDo remove

// ToDo split to Raytracing for GBUffer and AO?

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

// ToDo remove AOcoefficient and use AO hits instead?
RWTexture2D<float> g_rtAOcoefficient : register(u10);
RWTexture2D<uint> g_rtAORayHits : register(u11);
RWTexture2D<float> g_rtVisibilityCoefficient : register(u12);
RWTexture2D<float> g_rtGBufferDepth : register(u13);
RWTexture2D<float4> g_rtGBufferNormalRGB : register(u14);   // for SSAO. ToDo cleanup


ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
StructuredBuffer<PrimitiveMaterialBuffer> g_materials : register(t3);
StructuredBuffer<AlignedHemisphereSample3D> g_sampleSets : register(t4);

SamplerState LinearWrapSampler : register(s0);

// Per-object resources
ConstantBuffer<PrimitiveConstantBuffer> l_materialCB : register(b1);


// ToDo remove space0
#if ONLY_SQUID_SCENE_BLAS
StructuredBuffer<Index> l_indices : register(t1, space0);
StructuredBuffer<VertexPositionNormalTextureTangent> l_vertices : register(t2, space0);
#else
ByteAddressBuffer l_indices : register(t1, space0);
StructuredBuffer<VertexPositionNormalTexture> l_vertices : register(t2, space0);
#endif
Texture2D<float3> l_texDiffuse : register(t10);
Texture2D<float3> l_texNormalMap : register(t11);




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
    // Miss shader will set it to -1.
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
    if (!g_sceneCB.useShadowRayHitTime)
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
    
    // Report a hit if Miss Shader didn't set the value to -1.
    tHit = shadowPayload.tHit;

    return shadowPayload.tHit > (-1 + 0.001f);
}

// Trace a camera ray into the scene.
// rx, ry - auxilary rays offset in screen space by one pixel in x, y directions.
GBufferRayPayload TraceGBufferRay(in Ray ray, in Ray rx, in Ray ry, in UINT currentRayRecursionDepth)
{
	// Set the ray's extents.
	RayDesc rayDesc;
	rayDesc.Origin = ray.origin;
	rayDesc.Direction = ray.direction;
	// ToDo update comments about Tmins
	// Set TMin to a zero value to avoid aliasing artifacts along contact areas.
	// Note: make sure to enable face culling so as to avoid surface face fighting.
	// ToDo Tmin - this should be offset along normal.
	rayDesc.TMin = 0.001;
	rayDesc.TMax = 10000;
#if ALLOW_MIRRORS
	GBufferRayPayload rayPayload = { currentRayRecursionDepth + 1, false, (uint2)0, (float3)0, (float3)0, rx, ry };
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
float CalculateAO(out uint numShadowRayHits, in UINT numSamples, in float3 hitPosition, in float3 surfaceNormal, in float3 surfaceAlbedo = float3(1,1,1))
{
    numShadowRayHits = 0;
    float occlussionCoefSum = 0;

#if AO_HITPOSITION_BASED_SEED
#if AO_SAMPLES_SPREAD_ACCROSS_PIXELS
    // Neighboring samples NxN share a sample set.
    // Get a sample set ID and seed shared across neighboring pixels.
    uint numSampleSetsInX = (DispatchRaysDimensions().x + g_sceneCB.numPixelsPerDimPerSet - 1) / g_sceneCB.numPixelsPerDimPerSet;
    uint2 sampleSetId = DispatchRaysIndex().xy / g_sceneCB.numPixelsPerDimPerSet;
    uint2 pixelZeroId = sampleSetId * g_sceneCB.numPixelsPerDimPerSet;
    float3 pixelZeroHitPosition = g_texGBufferPositionRT[pixelZeroId].xyz;

    uint sampleSetSeed = (sampleSetId.y * numSampleSetsInX + sampleSetId.x) * hash(pixelZeroHitPosition) + g_sceneCB.seed;

    uint RNGState = RNG::SeedThread(sampleSetSeed);
    uint sampleSetJump = RNG::Random(RNGState, 0, g_sceneCB.numSampleSets - 1) * g_sceneCB.numSamplesPerSet;

    // Get a pixel ID within the shared set across neighboring pixels.
    uint2 pixeIDPerSet2D = DispatchRaysIndex().xy % g_sceneCB.numPixelsPerDimPerSet;
    uint pixeIDPerSet = pixeIDPerSet2D.y * g_sceneCB.numPixelsPerDimPerSet + pixeIDPerSet2D.x;

    // ToDo is RNG being used here any useful?
    uint numPixelsPerSet = g_sceneCB.numPixelsPerDimPerSet * g_sceneCB.numPixelsPerDimPerSet;
    uint sampleJump = (pixeIDPerSet + RNG::Random(RNGState, 0, numPixelsPerSet - 1)) % numPixelsPerSet;
    sampleJump *= g_sceneCB.numSamplesToUse;

    for (uint i = 0; i < numSamples; i++)
    {
        // Load a pregenerated random sample from the sample set.
        float3 sample = g_sampleSets[sampleSetJump + sampleJump + i].value;
#else
	// Seed:
	// - DispatchRaysDimensions to break correlation among neighboring pixels.
	// - hash(hitPosition) to break correlation for the same pixel but differet hitPosition when moving camera/objects.
    uint seed = (DispatchRaysDimensions().x * DispatchRaysIndex().y + DispatchRaysIndex().x) * hash(hitPosition) + g_sceneCB.seed;

	uint RNGState = RNG::SeedThread(seed);
	uint sampleSetJump = RNG::Random(RNGState, 0, g_sceneCB.numSampleSets - 1) * g_sceneCB.numSamples;
	uint sampleJump = 0; RNG::Random(RNGState, 0, g_sceneCB.numSamples - 1);

    for (uint i = 0; i < g_sceneCB.numSamplesToUse; i++)
    {
        // Load a pregenerated random sample from the sample set.
        float3 sample = g_sampleSets[sampleSetJump + (sampleJump + i) % g_sceneCB.numSamples].value;
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
        Ray shadowRay = { hitPosition + 0.001f * surfaceNormal, normalize(rayDirection) };

#else
    uint2 BTid = DispatchRaysIndex().xy & 3; // 4x4 BlockThreadID
    uint RNGState = RNG::SeedThread(g_sceneCB.seed + (BTid.y << 2 | BTid.x));

    for (uint i = 0; i < g_sceneCB.numSamplesToUse; i++)
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
        const float tMax = g_sceneCB.maxShadowRayHitTime;
        float tHit;
        if (TraceShadowRayAndReportIfHit(tHit, shadowRay, 0, tMax))
        {
            // ToDo Setting coef to 1 and then 0.5 below causes device removal. Wth?
            float occlusionCoef = 1;
            if (g_sceneCB.RTAO_IsExponentialFalloffEnabled)
            {
                //float t = tHit / tMax;
                //float lambda = g_sceneCB.RTAO_exponentialFalloffDecayConstant;
                occlusionCoef = 0;    // Causes device removal
                //occlusionCoef = 1 - t;  // Works...
            }
            occlussionCoefSum += occlusionCoef;// (1.f - g_sceneCB.RTAO_minimumAmbientIllumnination) * occlusionCoef;

            numShadowRayHits++;
        }
    }
    float occlusionCoef = saturate(occlussionCoefSum / numSamples);
    float ambientCoef = 1.f - occlusionCoef;
#elif REPRO_INVISIBLE_WALL
        const float tMax = g_sceneCB.maxShadowRayHitTime;
        float tHit;
        if (TraceShadowRayAndReportIfHit(tHit, shadowRay, 0, tMax))
        {
            float occlusionCoef = 0;
            float t = tHit / tMax;
            if (g_sceneCB.RTAO_IsExponentialFalloffEnabled && t >= g_sceneCB.RTAO_minimumAmbientIllumnination)
            {
                float lambda = g_sceneCB.RTAO_exponentialFalloffDecayConstant;
                occlusionCoef = 1 - t;// exp(-lambda * t*t);
            }
            //occlussionCoefSum += occlusionCoef;// (1.f - g_sceneCB.RTAO_minimumAmbientIllumnination) * occlusionCoef;
            occlussionCoefSum = max(occlusionCoef, occlussionCoefSum);// (1.f - g_sceneCB.RTAO_minimumAmbientIllumnination) * occlusionCoef;

            numShadowRayHits++;
        }
    }
    occlussionCoefSum *= numSamples;
    float occlusionCoef = saturate(occlussionCoefSum / numSamples);
    float ambientCoef = 1.f - occlusionCoef;
#else
        const float tMax = g_sceneCB.maxShadowRayHitTime;
        float tHit;
        if (TraceShadowRayAndReportIfHit(tHit, shadowRay, 0, tMax))
        {
            float occlusionCoef = 1;
            if (g_sceneCB.RTAO_IsExponentialFalloffEnabled)
            {
                float t = tHit / tMax;
                float lambda = g_sceneCB.RTAO_exponentialFalloffDecayConstant;
                occlusionCoef = exp(-lambda * t*t);
            }
            occlussionCoefSum += (1.f - g_sceneCB.RTAO_minimumAmbientIllumnination) * occlusionCoef;

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
    if (g_sceneCB.RTAO_approximateInterreflections)
    {
        float kA = ambientCoef;
        float rho = g_sceneCB.RTAO_diffuseReflectanceScale * RGBtoLuminance(surfaceAlbedo);

        ambientCoef = kA / (1 - rho * (1 - kA));
    }

	return ambientCoef;
}


float CalculateAO(in float3 hitPosition, UINT numSamples, in float3 surfaceNormal)
{
	uint numShadowRayHits;
	return CalculateAO(numShadowRayHits, numSamples, hitPosition, surfaceNormal);
}

//***************************************************************************
//********************------ Ray gen shader.. -------************************
//***************************************************************************

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
        float3 surfaceNormal = Decode(g_texGBufferNormal[DispatchRaysIndex().xy].xy);
#else
		float3 surfaceNormal = g_texGBufferNormal[DispatchRaysIndex().xy].xyz;
#endif
        Ray visibilityRay = { hitPosition + 0.001f * surfaceNormal, normalize(g_sceneCB.lightPosition.xyz - hitPosition) };

        float tHit;
		inShadow = TraceShadowRayAndReportIfHit(tHit, visibilityRay, 0);
	}

	// ToDo add option to be true distance and do contact hardening
	g_rtVisibilityCoefficient[DispatchRaysIndex().xy] = inShadow ? 0 : 1;
}

[shader("raygeneration")]
void MyRayGenShader_GBuffer()
{
	// ToDo remove
#if CAMERA_JITTER
	uint seed = DispatchRaysDimensions().x * DispatchRaysIndex().y + DispatchRaysIndex().x;// +g_sceneCB.seed;
	uint RNGState = RNG::SeedThread(seed);
	float2 cameraJitter = 2 * (float2(RNG::Random01(RNGState), RNG::Random01(RNGState)) - 0.5f);
	cameraJitter *= 0.5f;
#else
	float2 cameraJitter = float2(0, 0);
#endif

	// Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
	Ray ray = GenerateCameraRay(DispatchRaysIndex().xy, g_sceneCB.cameraPosition.xyz, g_sceneCB.projectionToWorldWithCameraEyeAtOrigin, cameraJitter);

    Ray rx, ry;
    GetAuxilaryCameraRays(g_sceneCB.cameraPosition.xyz, g_sceneCB.projectionToWorldWithCameraEyeAtOrigin, rx, ry);

	// Cast a ray into the scene and retrieve GBuffer information.
	UINT currentRayRecursionDepth = 0;
	GBufferRayPayload rayPayload = TraceGBufferRay(ray, rx, ry, currentRayRecursionDepth);

	// Write out GBuffer information to rendertargets.
	// ToDo Test conditional write on all output 
	g_rtGBufferCameraRayHits[DispatchRaysIndex().xy] = (rayPayload.hit ? 1 : 0);
    g_rtGBufferMaterialInfo[DispatchRaysIndex().xy] = rayPayload.materialInfo;

    // ToDo Use calculated hitposition based on distance from GBuffer instead?
	g_rtGBufferPosition[DispatchRaysIndex().xy] = float4(rayPayload.hitPosition, 0);

    float rayLength = FLT_MAX;
    float obliqueness = 0;

    
#if  REPRO_BLOCKY_ARTIFACTS_NONUNIFORM_CB_REFERENCE_SSAO // CB value is incorrect on rayPayload.hit boundaries causing blocky artifacts when within if (hit) block
    if (rayPayload.hit)
    {
        float3 raySegment = g_sceneCB.cameraPosition.xyz - rayPayload.hitPosition;
#else

    float3 raySegment = g_sceneCB.cameraPosition.xyz - rayPayload.hitPosition;
    if (rayPayload.hit)
    {
#endif
        rayLength = length(raySegment);
        float forwardFacing = dot(rayPayload.surfaceNormal, raySegment) / rayLength;
        obliqueness = forwardFacing;// min(f16tof32(0x7BFF), rcp(max(forwardFacing, 1e-5)));
    }


#if COMPRES_NORMALS
    // compress normal
    // ToDo review precision of 16bit format - particularly rayLength (renormalize ray length?)
    g_rtGBufferNormal[DispatchRaysIndex().xy] = float4(Encode(rayPayload.surfaceNormal), rayLength, obliqueness);
#else
    #if PACK_NORMAL_AND_DEPTH
        obliqueness = rayLength;
    #endif
    g_rtGBufferNormal[DispatchRaysIndex().xy] = float4(rayPayload.surfaceNormal, obliqueness);
#endif
    g_rtGBufferDistance[DispatchRaysIndex().xy] = rayLength;

    // ToDo revise
    // Convert distance to nonLinearDepth.
    {
#if 1
        float nonLinearDepth =  rayPayload.hit ? 
                                    (FAR_PLANE + NEAR_PLANE - 2.0 * NEAR_PLANE * FAR_PLANE / rayLength) / (FAR_PLANE - NEAR_PLANE)
                                :   1;
        nonLinearDepth = (nonLinearDepth + 1.0) / 2.0;
#elif 0

        float nonLinearDepth =  rayPayload.hit ?
                                    (1 - (1 - rayLength) / (rayLength * ZSCALE))
                                :   0;
#else
        float nonLinearDepth = rayPayload.hit ? rayLength : 1;

#endif
        g_rtGBufferDepth[DispatchRaysIndex().xy] = nonLinearDepth;
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
	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
#if COMPRES_NORMALS
        float3 surfaceNormal = Decode(g_texGBufferNormal[DispatchRaysIndex().xy].xy);
#else
		float3 surfaceNormal = g_texGBufferNormal[DTid].xyz;
#endif

        float3 surfaceAlbedo = float3(1, 1, 1);
        if (g_sceneCB.RTAO_approximateInterreflections)
        {
            uint2 materialInfo = g_texGBufferMaterialInfo[DTid];
            UINT materialID;
            DecodeMaterial16b(materialInfo, materialID, surfaceAlbedo);
            surfaceAlbedo = surfaceAlbedo;
        }

        UINT numSamples = g_sceneCB.numSamplesToUse;

        if (g_sceneCB.RTAO_UseAdaptiveSampling)
        {
            float filterWeightSum = g_filterWeightSum[DTid].x;
            numSamples = filterWeightSum <= g_sceneCB.RTAO_AdaptiveSamplingMaxWeightSum ? 4* numSamples : numSamples;
        }
		ambientCoef = CalculateAO(numShadowRayHits, numSamples, hitPosition, surfaceNormal, surfaceAlbedo);
	}

	g_rtAOcoefficient[DispatchRaysIndex().xy] = ambientCoef;
#if GBUFFER_AO_COUNT_AO_HITS
	// ToDo test perf impact of writing this
	g_rtAORayHits[DispatchRaysIndex().xy] = numShadowRayHits;
#endif
}

[shader("raygeneration")]
void MyRayGenShaderQuarterRes_AO()
{
    uint2 DTid = DispatchRaysIndex().xy * 2;

	bool hit = g_texGBufferPositionHits[DTid] > 0;
	uint numShadowRayHits = 0;
	float ambientCoef = 0;
	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
#if COMPRES_NORMALS
        float3 surfaceNormal = Decode(g_texGBufferNormal[DispatchRaysIndex().xy].xy);
#else
		float3 surfaceNormal = g_texGBufferNormal[DTid].xyz;
#endif
        // ToDo Standardize naming AO vs AmbientOcclusion ?
		ambientCoef = CalculateAO(numShadowRayHits, g_sceneCB.numSamplesToUse, hitPosition, surfaceNormal);
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
	float3 vertexNormals[3] = { l_vertices[indices[0]].normal, l_vertices[indices[1]].normal, l_vertices[indices[2]].normal};

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
	float ambientCoef = CalculateAO(HitWorldPosition(), g_sceneCB.numSamplesToUse, triangleNormal);
	float4 color = ambientCoef * float4(0.75, 0.75, 0.75, 0.75);
#else
    // Shadow component.
    // Trace a shadow ray.
    float3 hitPosition = HitWorldPosition();
	
	// ToDo
//    Ray shadowRay = { hitPosition + 0.0001f * triangleNormal, normalize(g_sceneCB.lightPosition.xyz - hitPosition) };
  //  bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, rayPayload.recursionDepth);
	
    // Calculate final color.
	float ambientCoef = CalculateAO(HitWorldPosition(), g_sceneCB.numSamplesToUse, triangleNormal);
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

    // Retrieve texture coordinates for the hit.
    VertexPositionNormalTextureTangent vertices[3] = { l_vertices[indices[0]], l_vertices[indices[1]], l_vertices[indices[2]] };
    float2 vertexTexCoords[3] = { vertices[0].textureCoordinate, vertices[1].textureCoordinate, vertices[2].textureCoordinate };
    float2 texCoord = HitAttribute(vertexTexCoords, attr);


    UINT materialID = l_materialCB.materialID;
    PrimitiveMaterialBuffer material = g_materials[materialID];

    // Load triangle normal.
    float3 normal;
    {
        // Retrieve corresponding vertex normals for the triangle vertices.
        float3 vertexNormals[3] = { vertices[0].normal, vertices[1].normal, vertices[2].normal };

#if FLAT_FACE_NORMALS
        BuiltInTriangleIntersectionAttributes attrCenter;
        attrCenter.barycentrics.x = attrCenter.barycentrics.y = 1.f / 3;
        // ToDo input normals should be normalized already
        normal = normalize(HitAttribute(vertexNormals, attrCenter));
#else
        normal = normalize(HitAttribute(vertexNormals, attr));
#endif
#if !FACE_CULLING
        float orientation = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 1 : -1;
        normal *= orientation;
#endif
    }

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
    
    if (material.hasDiffuseTexture || material.hasNormalTexture)
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
    if (material.hasNormalTexture)
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

        rayPayload = TraceGBufferRay(ray, rx, ry, rayPayload.rayRecursionDepth);
	}
    else
#endif
    {
        float3 diffuse;
        if (material.hasDiffuseTexture)
        {
            diffuse = l_texDiffuse.SampleGrad(LinearWrapSampler, texCoord, ddx, ddy).xyz;
        }
        else
        {
            diffuse = material.diffuse;
        }
        rayPayload.hit = true;
        rayPayload.materialInfo = EncodeMaterial16b(materialID, diffuse);
        rayPayload.hitPosition = hitPosition;
        rayPayload.surfaceNormal = normal;
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
    rayPayload.tHit = -1;
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