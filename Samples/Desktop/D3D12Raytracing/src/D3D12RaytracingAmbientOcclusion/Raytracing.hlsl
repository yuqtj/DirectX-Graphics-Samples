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

// ToDo dedupe code triangle normal calc,..
// ToDo pix doesn't show output for AO pass

//***************************************************************************
//*****------ Shader resources bound via root signatures -------*************
//***************************************************************************

// Scene wide resources.
//  g_* - bound via a global root signature.
//  l_* - bound via a local root signature.
RaytracingAccelerationStructure g_scene : register(t0, space0);
RWTexture2D<float4> g_renderTarget : register(u0);

// ToDo move this to local ray gen root sig
RWTexture2D<uint> g_rtGBufferCameraRayHits : register(u5);
RWTexture2D<uint> g_rtGBufferMaterialID : register(u6);
RWTexture2D<float4> g_rtGBufferPosition : register(u7);
RWTexture2D<float4> g_rtGBufferNormal : register(u8);
Texture2D<uint> g_texGBufferPositionHits : register(t5);
Texture2D<uint> g_texGBufferMaterialID : register(t6);
Texture2D<float4> g_texGBufferPositionRT : register(t7);
Texture2D<float4> g_texGBufferNormal : register(t8);
RWTexture2D<float> g_rtAOcoefficient : register(u9);
RWTexture2D<uint> g_rtAORayHits : register(u10);

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
StructuredBuffer<PrimitiveMaterialBuffer> g_materials : register(t3);
StructuredBuffer<AlignedHemisphereSample3D> g_sampleSets : register(t4);



// Per-object resources
ConstantBuffer<PrimitiveConstantBuffer> l_materialCB : register(b1);


#if ONLY_SQUID_SCENE_BLAS
StructuredBuffer<Index> l_indices : register(t1, space0);
StructuredBuffer<VertexPositionNormalTextureTangent> l_vertices : register(t2, space0);
#else
ByteAddressBuffer l_indices : register(t1, space0);
StructuredBuffer<VertexPositionNormalTexture> l_vertices : register(t2, space0);
#endif



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
bool TraceShadowRayAndReportIfHit(in Ray ray, in UINT currentRayRecursionDepth)
{
    if (currentRayRecursionDepth >= MAX_RAY_RECURSION_DEPTH)
    {
        return false;
    }

    // Set the ray's extents.
    RayDesc rayDesc;
    rayDesc.Origin = ray.origin;
    rayDesc.Direction = ray.direction;
    // Set TMin to a zero value to avoid aliasing artifcats along contact areas.
    // Note: make sure to enable back-face culling so as to avoid surface face fighting.
    rayDesc.TMin = 0.0;
	rayDesc.TMax = AO_RAY_T_MAX;// 0000;	// ToDo set this to dist to light

    // Initialize shadow ray payload.
    // Set the initial value to true since closest and any hit shaders are skipped. 
    // Shadow miss shader, if called, will set it to false.
    ShadowRayPayload shadowPayload = { true };
    TraceRay(g_scene,
#if FACE_CULLING
		RAY_FLAG_CULL_BACK_FACING_TRIANGLES
#else
		0
#endif
		| RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
        | RAY_FLAG_FORCE_OPAQUE             // ~skip any hit shaders
        | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // ~skip closest hit shaders,
        TraceRayParameters::InstanceMask,
        TraceRayParameters::HitGroup::Offset[RayType::Shadow],
        TraceRayParameters::HitGroup::GeometryStride,
        TraceRayParameters::MissShader::Offset[RayType::Shadow],
        rayDesc, shadowPayload);

    return shadowPayload.hit;
}

// Trace a camera ray into the scene.
GBufferRayPayload TraceGBufferRay(in Ray ray, in UINT currentRayRecursionDepth)
{
	// Set the ray's extents.
	RayDesc rayDesc;
	rayDesc.Origin = ray.origin;
	rayDesc.Direction = ray.direction;
	// ToDo update comments about Tmins
	// Set TMin to a zero value to avoid aliasing artifacts along contact areas.
	// Note: make sure to enable face culling so as to avoid surface face fighting.
	// ToDo Tmin
	rayDesc.TMin = 0.001;
	rayDesc.TMax = 10000;
#if ALLOW_MIRRORS
	GBufferRayPayload rayPayload = {false, 0, (float3)0, (float3)0, currentRayRecursionDepth + 1 };
#else
	GBufferRayPayload rayPayload = { false, 0, (float3)0, (float3)0 };
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
float CalculateAO(in float3 hitPosition, in float3 surfaceNormal, out uint numShadowRayHits)
{
	uint seed = DispatchRaysDimensions().x * DispatchRaysIndex().y + DispatchRaysIndex().x + g_sceneCB.seed;

	uint RNGState = RNG::SeedThread(seed);
	uint sampleSetJump = RNG::Random(RNGState, 0, g_sceneCB.numSampleSets - 1) * g_sceneCB.numSamples;

	uint sampleJump = RNG::Random(RNGState, 0, g_sceneCB.numSamples - 1);
	numShadowRayHits = 0;
	for (uint i = 0; i < g_sceneCB.numSamplesToUse; i++)
	{
		float3 sample = g_sampleSets[sampleSetJump + (sampleJump + i) % g_sceneCB.numSamples].value;

		// Calculate coordinate system for the hemisphere
		float3 u, v, w;
		w = surfaceNormal;

		// Break hemisphere coordinate correlation - ToDo is this needed with >#N sample sets? Compare perf
		float x = RNG::Random01(RNGState);
		float y = RNG::Random01(RNGState);
		float z = RNG::Random01(RNGState);
		float3 right = normalize(float3(x, y, z) + 0.001f*w.yzx);

		//        float3 right = normalize(float3(0.0072, 1.0, 0.0034));
		v = normalize(cross(w, right));
		u = cross(v, w);

		float3 rayDirection = sample.x * u + sample.y * v + sample.z * w;

		// ToDo hitPosition adjustment - fix crease artifacts
		// Todo fix noise on flat surface / box
		Ray shadowRay = { hitPosition + 0.001f * surfaceNormal, normalize(rayDirection) };

		if (TraceShadowRayAndReportIfHit(shadowRay, 0))
		{
			numShadowRayHits++;
		}
	}
#if AO_ANY_HIT_FULL_OCCLUSION
	float ambientCoef = numShadowRayHits > 0 ? 0 : 1;
#else
	float ambientCoef = 1.f - ((float)numShadowRayHits / g_sceneCB.numSamplesToUse);
#endif
	
	return ambientCoef;
}


float CalculateAO(in float3 hitPosition, in float3 surfaceNormal)
{
	uint numShadowRayHits;
	return CalculateAO(hitPosition, surfaceNormal, numShadowRayHits);
}

//***************************************************************************
//********************------ Ray gen shader.. -------************************
//***************************************************************************

[shader("raygeneration")]
void MyRayGenShader_PrimaryAndAO()
{
#if RAYGEN_SINGLE_COLOR_SHADING
	g_renderTarget[DispatchRaysIndex().xy] = float4(1,0,0,1);
	return;
#endif
    // Generate a ray for a camera pixel corresponding to an index from the dispatched 2D grid.
    Ray ray = GenerateCameraRay(DispatchRaysIndex().xy, g_sceneCB.cameraPosition.xyz, g_sceneCB.projectionToWorldWithCameraEyeAtOrigin );
 
    // Cast a ray into the scene and retrieve a shaded color.
    UINT currentRecursionDepth = 0;
	float4 color = TraceRadianceRay(ray, currentRecursionDepth);

    // Write the raytraced color to the output texture.
    g_renderTarget[DispatchRaysIndex().xy] = color;
}

[shader("raygeneration")]
void MyRayGenShader_GBuffer()
{
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

	// Cast a ray into the scene and retrieve GBuffer information.
	UINT currentRayRecursionDepth = 0;
	GBufferRayPayload rayPayload = TraceGBufferRay(ray, currentRayRecursionDepth);

	// Write out GBuffer information to rendertargets.
	// ToDo Test conditional write
	g_rtGBufferCameraRayHits[DispatchRaysIndex().xy] = (rayPayload.hit ? 1 : 0);
	g_rtGBufferMaterialID[DispatchRaysIndex().xy] = rayPayload.materialID;
	g_rtGBufferPosition[DispatchRaysIndex().xy] = float4(rayPayload.hitPosition, 0);
	g_rtGBufferNormal[DispatchRaysIndex().xy] = float4(rayPayload.surfaceNormal, 0);
}

[shader("raygeneration")]
void MyRayGenShader_AO()
{
	bool hit = g_texGBufferPositionHits[DispatchRaysIndex().xy] > 0;
	uint shadowRayHits = 0;
	float ambientCoef = 0;
	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DispatchRaysIndex().xy].xyz;
		float3 surfaceNormal = g_texGBufferNormal[DispatchRaysIndex().xy].xyz;
		ambientCoef = CalculateAO(hitPosition, surfaceNormal, shadowRayHits);
	}

	g_rtAOcoefficient[DispatchRaysIndex().xy] = ambientCoef;
#if GBUFFER_AO_COUNT_AO_HITS
	// ToDo test perf impact of writing this
	g_rtAORayHits[DispatchRaysIndex().xy] = shadowRayHits;
#endif
}

// ToDo PIX feedbackpix
// - Add resource names to Resources tab
// - Keep Show/Hide DXR resource state across events
// - There's no shader item in DispatchRays like it is in Dispatch
// - For shader records - rename root sig to local root sig.
// - Pipeline tab - show DXR resources can take a lot of time (10s for 2k records) - show first 20 right away? Resizing the tab is slow as well.
//  Sart analysis on the right side of the window

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

#if NORMAL_SHADING
	rayPayload.color = float4(triangleNormal, 1.0f);
	return;
#endif

    // PERFORMANCE TIP: it is recommended to avoid values carry over across TraceRay() calls. 
    // Therefore, in cases like retrieving HitWorldPosition(), it is recomputed every time.
#if AO_ONLY
	float ambientCoef = CalculateAO(HitWorldPosition(), triangleNormal);
	float4 color = ambientCoef * float4(0.75, 0.75, 0.75, 0.75);
#else
    // Shadow component.
    // Trace a shadow ray.
    float3 hitPosition = HitWorldPosition();
	
	// ToDo
//    Ray shadowRay = { hitPosition + 0.0001f * triangleNormal, normalize(g_sceneCB.lightPosition.xyz - hitPosition) };
  //  bool shadowRayHit = TraceShadowRayAndReportIfHit(shadowRay, rayPayload.recursionDepth);
	
    // Calculate final color.
	float ambientCoef = CalculateAO(HitWorldPosition(), triangleNormal);
	float4 color = float4(1, 0, 0, 0); //ToDo
    //float3 phongColor = CalculatePhongLighting(triangleNormal, shadowRayHit, ambient, l_materialCB.diffuse, l_materialCB.specular, l_materialCB.specularPower);
	//float4 color =  float4(phongColor, 1);
#endif     

    rayPayload.color = color;
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

	// Retrieve corresponding vertex normals for the triangle vertices.
	float3 vertexNormals[3] = { l_vertices[indices[0]].normal, l_vertices[indices[1]].normal, l_vertices[indices[2]].normal };
	
#if FLAT_FACE_NORMALS
	BuiltInTriangleIntersectionAttributes attrCenter;
	attrCenter.barycentrics.x = attrCenter.barycentrics.y = 1.f / 3;
	// ToDo input normals should be normalized already
	float3 triangleNormal = normalize(HitAttribute(vertexNormals, attrCenter));
#else
	float3 triangleNormal = HitAttribute(vertexNormals, attr);
#endif

#if !FACE_CULLING
	float orientation = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE ? 1 : -1;
	triangleNormal *= orientation;
#endif
	rayPayload.hit = true;
	rayPayload.materialID = l_materialCB.materialID;
	rayPayload.hitPosition = HitWorldPosition();
	rayPayload.surfaceNormal = triangleNormal;

#if ALLOW_MIRRORS
	PrimitiveMaterialBuffer material = g_materials[rayPayload.materialID];
	if (material.isMirror && rayPayload.rayRecursionDepth < MAX_RAY_RECURSION_DEPTH)
	{
#if TURN_MIRRORS_SEETHROUGH
		Ray passThroughRay = { rayPayload.hitPosition + 0.05f*WorldRayDirection(), WorldRayDirection() };
		rayPayload = TraceGBufferRay(passThroughRay, rayPayload.rayRecursionDepth);

#else
		Ray reflectionRay = { rayPayload.hitPosition, reflect(WorldRayDirection(), rayPayload.surfaceNormal) };
		rayPayload = TraceGBufferRay(reflectionRay, rayPayload.rayRecursionDepth);
#endif
	}
#endif
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
    rayPayload.hit = false;
}

// ToDo - remove miss shader for GBuffer
[shader("miss")]
void MyMissShader_GBuffer(inout GBufferRayPayload rayPayload)
{
	rayPayload.hit = false;
}


#endif // RAYTRACING_HLSL