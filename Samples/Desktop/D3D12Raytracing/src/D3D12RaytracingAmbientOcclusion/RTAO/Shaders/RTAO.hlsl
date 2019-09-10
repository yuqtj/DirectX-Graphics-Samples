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
#include "Ray Sorting/RaySorting.hlsli"
#include "RTAO.hlsli"

// ToDo pix doesn't show output for AO pass

RaytracingAccelerationStructure g_scene : register(t0);

// ToDo remove unneccessary, move ray computation to CS
// ToDo switch to depth == 0 for hit/no hit?
Texture2D<float4> g_texRayOriginPosition : register(t7);
Texture2D<NormalDepthTexFormat> g_texRayOriginSurfaceNormalDepth : register(t8);
Texture2D<NormalDepthTexFormat> g_texAORaysDirectionOriginDepth : register(t22);
Texture2D<uint2> g_texAOSortedToSourceRayIndexOffset : register(t23);
Texture2D<float4> g_texAOSurfaceAlbedo : register(t24);

// ToDo remove ? 
Texture2D<uint> g_texInputAOFrameAge : register(t14);

// ToDo remove AOcoefficient and use AO hits instead?
//todo remove rt?
RWTexture2D<float> g_rtAOcoefficient : register(u10);
RWTexture2D<uint> g_rtAORayHits : register(u11);
RWTexture2D<float> g_rtAORayHitDistance : register(u15);
RWTexture2D<NormalDepthTexFormat> g_rtAORaysDirectionOriginDepth : register(u22);

ConstantBuffer<RTAOConstantBuffer> cb : register(b0);          // ToDo standardize cb var naming
StructuredBuffer<AlignedHemisphereSample3D> g_sampleSets : register(t4);

// Delay the include so that resource references resolve.
#include "RayGen.hlsli"


//***************************************************************************
//*********************------ TraceRay wrappers. -------*********************
//***************************************************************************

// Trace an AO ray and return true if it hits any geometry.
bool TraceAORayAndReportIfHit(out float tHit, in Ray ray, in float TMax, in float3 surfaceNormal)
{
    RayDesc rayDesc;

    // Nudge the origin along the surface normal a bit to avoid 
    // starting from behind the surface
    // due to float calculations imprecision.
    rayDesc.Origin = ray.origin + 0.001 * surfaceNormal;
    rayDesc.Direction = ray.direction;

    // Set the ray's extents.
    rayDesc.TMin = 0.0;
	rayDesc.TMax = TMax;

    // Initialize shadow ray payload.
    // Set the initial value to a hit at TMax. 
    // This way closest and any hit shaders can be skipped if true tHit is not needed. 
    ShadowRayPayload shadowPayload = { TMax };

    UINT rayFlags =
#if FACE_CULLING            // ToDo remove one path?
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES
#else
        0
#endif
        // Skip transparent objects.
        | RAY_FLAG_CULL_NON_OPAQUE;        

    // ToDo remove?
    // ToDo test visual impact
    // ToDo test perf impact 1.7 -> 1.55 ms
    bool acceptFirstHit = true;
    if (acceptFirstHit)
    {
        // ToDo test perf impact
        // Performance TIP: Accept first hit if true hit is not neeeded,
        // or has minimal to no impact (in AO). The peformance gain can
        // be substantial.
        rayFlags |= RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    }

    // ToDo test perf impact
    // Skip closest hit shaders of tHit time is not needed.
    // rayFlags |= RAY_FLAG_SKIP_CLOSEST_HIT_SHADER; 

    TraceRay(g_scene,
        rayFlags,
        RTAOTraceRayParameters::InstanceMask,
        RTAOTraceRayParameters::HitGroup::Offset[RTAORayType::AO],
        RTAOTraceRayParameters::HitGroup::GeometryStride,
        RTAOTraceRayParameters::MissShader::Offset[RTAORayType::AO],
        rayDesc, shadowPayload);
    
    tHit = shadowPayload.tHit;

    // Report a hit if Miss Shader didn't set the value to HitDistanceOnMiss.
    return RTAO::HasAORayHitAnyGeometry(tHit);
}

// Traces a given AO ray. 
// Returns its tHit and a calculated ambient coefficient.
float CalculateAO(out float tHit, in uint2 srcPixelIndex, in Ray AOray, in float3 surfaceNormal)
{
    float ambientCoef = 1;
    const float tMax = cb.RTAO_maxShadowRayHitTime; // ToDo make sure its FLT_10BIT_MAX or less since we use 10bit origin depth in RaySort
    if (TraceAORayAndReportIfHit(tHit, AOray, tMax, surfaceNormal))
    {
        float occlusionCoef = 1;
        if (cb.RTAO_IsExponentialFalloffEnabled)
        {
            float theoreticalTMax = cb.RTAO_maxTheoreticalShadowRayHitTime;
            float t = tHit / theoreticalTMax;
            float lambda = cb.RTAO_exponentialFalloffDecayConstant;
            occlusionCoef = exp(-lambda * t * t);
        }
        ambientCoef = 1 - (1 - cb.RTAO_MinimumAmbientIllumination) * occlusionCoef;

        // Approximate interreflections of light from blocking surfaces which are generally not completely dark and tend to have similar radiance.
        // Ref: Ch 11.3.3 Accounting for Interreflections, Real-Time Rendering (4th edition).
        // The approximation assumes:
        //      o All surfaces incoming and outgoing radiance is the same 
        //      o Current surface color is the same as that of the occluders
        // Since this sample uses scalar ambient coefficient, we use the scalar luminance of the surface color.
        // This will generally brighten the AO making it closer to the result of full Global Illumination, including interreflections.
        if (cb.RTAO_approximateInterreflections)
        {
            // ToDo test perf impact of reading the texture and move this to compose pass
            float3 surfaceAlbedo = g_texAOSurfaceAlbedo[srcPixelIndex].xyz;

            float kA = ambientCoef;
            float rho = cb.RTAO_diffuseReflectanceScale * RGBtoLuminance(surfaceAlbedo);

            ambientCoef = kA / (1 - rho * (1 - kA));
        }
    }

    return ambientCoef;
}

//***************************************************************************
//********************------ Ray gen shader.. -------************************
//***************************************************************************

[shader("raygeneration")]
void RayGenShader()
{
#if 0
    uint2 srcRayIndex = DispatchRaysIndex().xy;
    float3 hitPosition = g_texRayOriginPosition[srcRayIndex].xyz;
    Ray AORay = { hitPosition, float3(0.2, 0.4, 0.2) };
    const float tMax = cb.RTAO_maxShadowRayHitTime; // ToDo make sure its FLT_10BIT_MAX or less since we use 10bit origin depth in RaySort
    float3 surfaceNormal = float3(0, 1, 0);
    float tHit;
    TraceAORayAndReportIfHit(tHit, AORay, tMax, surfaceNormal);
#else
    // ToDo move to a CS if always using a raysort.
    uint2 srcRayIndex = DispatchRaysIndex().xy;
    
    // ToDo
    float3 surfaceNormal;
    float depth;
    // ToDO use full precision here?
    DecodeNormalDepth(g_texRayOriginSurfaceNormalDepth[srcRayIndex], surfaceNormal, depth);
	bool hit = depth != 0;   // ToDo use a common func to determine
    float tHit = RTAO::RayHitDistanceOnMiss;
    float ambientCoef = RTAO::InvalidAOValue;
	if (hit)
	{
		float3 hitPosition = g_texRayOriginPosition[srcRayIndex].xyz;     
        float3 rayDirection = GetRandomRayDirection(srcRayIndex, surfaceNormal, cb.raytracingDim);
        Ray AORay = { hitPosition, rayDirection };
        ambientCoef = CalculateAO(tHit, srcRayIndex, AORay, surfaceNormal);
    }
#endif

    g_rtAOcoefficient[srcRayIndex] = ambientCoef;
    g_rtAORayHitDistance[srcRayIndex] = RTAO::HasAORayHitAnyGeometry(tHit) ? tHit : cb.RTAO_maxTheoreticalShadowRayHitTime;
 
#if GBUFFER_AO_COUNT_AO_HITS
    // ToDo test perf impact of writing this
    g_rtAORayHits[srcRayIndex] = RTAO::HasAORayHitAnyGeometry(tHit);
#endif
}

// Retrieves 2D source and sorted ray indices from a 1D ray index where
// - every valid (i.e. is within ray tracing buffer dimensions) 1D index maps to a valid 2D index.
// - pixels are row major within a ray group.
// - ray groups are row major within the raytracing buffer dimensions.
// - rays are sorted per ray group.
// Overflowing ray group dimensions on the borders are clipped to valid raytracing dimnesions.
// Returns whether the retrieved ray is active.
bool Get2DRayIndices(out uint2 sortedRayIndex2D, out uint2 srcRayIndex2D, in uint index1D)
{
    uint2 rayGroupDim = uint2(SortRays::RayGroup::Width, SortRays::RayGroup::Height);

    // Find the ray group row index.
    uint numValidPixelsInRow = cb.raytracingDim.x;
    uint rowOfRayGroupSize = rayGroupDim.y * numValidPixelsInRow;
    uint rayGroupRowIndex = index1D / rowOfRayGroupSize;

    // Find the ray group column index.
    uint numValidPixelsInColumn = cb.raytracingDim.y;
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

    // Get the corresponding source index
    sortedRayIndex2D = rayGroupIndex * rayGroupDim + rayThreadIndex;
    uint2 rayGroupBase = rayGroupIndex * rayGroupDim;
    uint2 rayGroupRayIndexOffset = g_texAOSortedToSourceRayIndexOffset[sortedRayIndex2D];   // ToDo rename to encoded
    srcRayIndex2D = rayGroupBase + GetRawRayIndexOffset(rayGroupRayIndexOffset);

    return IsActiveRay(rayGroupRayIndexOffset);
}

[shader("raygeneration")]
void RayGenShader_sortedRays()
{
    uint DTid_1D = DispatchRaysIndex().x; 
    uint2 srcRayIndex;
    uint2 sortedRayIndex;
    bool isActiveRay = Get2DRayIndices(sortedRayIndex, srcRayIndex, DTid_1D);

    uint2 srcRayIndexFullRes = srcRayIndex;
    if (cb.doCheckerboardSampling)
    {
        UINT pixelStepX = 2;
        bool isEvenPixelY = (srcRayIndex.y & 1) == 0;
        UINT pixelOffsetX = isEvenPixelY != cb.areEvenPixelsActive;
        srcRayIndexFullRes.x = srcRayIndex.x * pixelStepX + pixelOffsetX;
    }

    float tHit = RTAO::RayHitDistanceOnMiss;
    float ambientCoef = RTAO::InvalidAOValue;
    if (isActiveRay)
    {
        float dummy;
        float3 rayDirection;
        DecodeNormalDepth(g_texAORaysDirectionOriginDepth[srcRayIndex], rayDirection, dummy);
        float3 hitPosition = g_texRayOriginPosition[srcRayIndexFullRes].xyz;

        // ToDo test trading for using ray direction insteads
        float3 surfaceNormal;
        float depth;

        // ToDo use ray direction instead?
        DecodeNormalDepth(g_texRayOriginSurfaceNormalDepth[srcRayIndexFullRes], surfaceNormal, depth);

        Ray AORay = { hitPosition, rayDirection };
        ambientCoef = CalculateAO(tHit, srcRayIndexFullRes, AORay, surfaceNormal);
    }

    uint2 outPixel = srcRayIndexFullRes;

    g_rtAOcoefficient[outPixel] = ambientCoef;
    g_rtAORayHitDistance[outPixel] = RTAO::HasAORayHitAnyGeometry(tHit) ? tHit : cb.RTAO_maxTheoreticalShadowRayHitTime;

#if GBUFFER_AO_COUNT_AO_HITS
    // ToDo test perf impact of writing this
    g_rtAORayHits[outPixel] = HasAORayHitAnyGeometry(tHit);
#endif
}

//***************************************************************************
//******************------ Closest hit shaders -------***********************
//***************************************************************************

[shader("closesthit")]
void ClosestHitShader(inout ShadowRayPayload rayPayload, in BuiltInTriangleIntersectionAttributes attr)
{
    rayPayload.tHit = RayTCurrent();
}

//***************************************************************************
//**********************------ Miss shaders -------**************************
//***************************************************************************

[shader("miss")]
void MissShader(inout ShadowRayPayload rayPayload)
{
    rayPayload.tHit = RTAO::RayHitDistanceOnMiss;
}



#endif // RAYTRACING_HLSL