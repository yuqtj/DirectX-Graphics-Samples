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
#include "Ray sorting/RayGen.hlsli"
#include "RTAO.hlsli"

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
RaytracingAccelerationStructure g_scene : register(t0);

// ToDo remove unneccessary, move ray computation to CS
// ToDo switch to depth == 0 for hit/no hit?
Texture2D<float4> g_texRayOriginPosition : register(t7);
Texture2D<NormalDepthTexFormat> g_texRayOriginSurfaceNormalDepth : register(t8);
Texture2D<NormalDepthTexFormat> g_texAORaysDirectionOriginDepthHit : register(t22);
Texture2D<uint2> g_texAOSortedToSourceRayIndexOffset : register(t23);
Texture2D<float4> g_texAOSurfaceAlbedo : register(t24);


// ToDo remove ? 
Texture2D<float> g_filterWeightSum : register(t13);
Texture2D<uint> g_texInputAOFrameAge : register(t14);

// ToDo remove AOcoefficient and use AO hits instead?
//todo remove rt?
RWTexture2D<float> g_rtAOcoefficient : register(u10);
RWTexture2D<uint> g_rtAORayHits : register(u11);
RWTexture2D<float> g_rtAORayHitDistance : register(u15);
RWTexture2D<NormalDepthTexFormat> g_rtAORaysDirectionOriginDepth : register(u22);

ConstantBuffer<RTAOConstantBuffer> CB : register(b0);          // ToDo standardize CB var naming
StructuredBuffer<AlignedHemisphereSample3D> g_sampleSets : register(t4);





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
    rayDesc.Origin = ray.origin + CB.RTAO_TraceRayOffsetAlongNormal * surfaceNormal;
    rayDesc.Direction = ray.direction;

    // Set the ray's extents.
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
        // Skip transparent objects.
        | RAY_FLAG_CULL_NON_OPAQUE;        

    // ToDo remove?
    // ToDo test visual impact
    // ToDo test perf impact 1.7 -> 1.55 ms
    bool acceptFirstHit = true;
    if (acceptFirstHit || !CB.useShadowRayHitTime)
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

Ray GenerateRandomAORay(in uint2 srcRayIndex, in float3 hitPosition, in float3 surfaceNormal)
{
    // Calculate coordinate system for the hemisphere.
    // ToDo AO has square alias due to same hemisphere
    float3 u, v, w;
    w = surfaceNormal;

    // ToDo revisit this
    // Get a vector that's not parallel to w;
#if 0
    float3 right = float3(0.0072f, 0.999994132f, 0.0034f);
#else
    float3 right = 0.3f * w + float3(-0.72f, 0.56f, -0.34f);
#endif
    v = normalize(cross(w, right));
    u = cross(v, w);


    // Calculate offsets to the pregenerated sample set.
    uint sampleSetJump;     // Offset to the start of the sample set
    uint sampleJump;        // Offset to the first sample for this pixel within a sample set.
    {
        // Neighboring samples NxN share a sample set, but use different samples within a set.
        // Sharing a sample set lets the pixels in the group get a better coverage of the hemisphere 
        // than if each pixel used a separate sample set with less samples pregenerated per set.

        // Get a common sample set ID and seed shared across neighboring pixels.
        uint numSampleSetsInX = (DispatchRaysDimensions().x + CB.numPixelsPerDimPerSet - 1) / CB.numPixelsPerDimPerSet;
        uint2 sampleSetId = srcRayIndex / CB.numPixelsPerDimPerSet;

        // Get a common hitPosition to adjust the sampleSeed by. 
        // This breaks noise correlation on camera movement which otherwise results 
        // in noise pattern swimming across the screen on camera movement.
        uint2 pixelZeroId = sampleSetId * CB.numPixelsPerDimPerSet;
        float3 pixelZeroHitPosition = g_texRayOriginPosition[pixelZeroId].xyz;      // ToDo remove?
        uint sampleSetSeed = (sampleSetId.y * numSampleSetsInX + sampleSetId.x) * hash(pixelZeroHitPosition) + CB.seed;
        uint RNGState = RNG::SeedThread(sampleSetSeed);

        sampleSetJump = RNG::Random(RNGState, 0, CB.numSampleSets - 1) * CB.numSamplesPerSet;

        // Get a pixel ID within the shared set across neighboring pixels.
        uint2 pixeIDPerSet2D = srcRayIndex % CB.numPixelsPerDimPerSet;
        uint pixeIDPerSet = pixeIDPerSet2D.y * CB.numPixelsPerDimPerSet + pixeIDPerSet2D.x;

        // Randomize starting sample position within a sample set per neighbor group 
        // to break group to group correlation resulting in square alias.
        uint numPixelsPerSet = CB.numPixelsPerDimPerSet * CB.numPixelsPerDimPerSet;
        sampleJump = pixeIDPerSet + RNG::Random(RNGState, 0, numPixelsPerSet - 1);
    }

    // Load a pregenerated random sample from the sample set.
    float3 sample = g_sampleSets[sampleSetJump + (sampleJump % CB.numSamplesPerSet)].value;

    // ToDo remove unnecessary normalize()
    float3 rayDirection = normalize(sample.x * u + sample.y * v + sample.z * w);

    Ray AORay = { hitPosition, rayDirection };

    return AORay;
}


// Traces a given AO ray. 
// Returns its tHit and a calculated ambient coefficient.
float CalculateAO(out float tHit, in uint2 srcPixelIndex, in Ray AOray, in float3 surfaceNormal)
{
    float ambientCoef = 1;
    const float tMax = CB.RTAO_maxShadowRayHitTime; // ToDo make sure its FLT_10BIT_MAX or less since we use 10bit origin depth in RaySort
    if (TraceAORayAndReportIfHit(tHit, AOray, tMax, surfaceNormal))
    {
        float occlusionCoef = 1;
        if (CB.RTAO_IsExponentialFalloffEnabled)
        {
            float theoreticalTMax = CB.RTAO_maxTheoreticalShadowRayHitTime;
            float t = tHit / theoreticalTMax;
            float lambda = CB.RTAO_exponentialFalloffDecayConstant;
            occlusionCoef = exp(-lambda * t * t);
        }
        ambientCoef = 1 - (1 - CB.RTAO_MinimumAmbientIllumination) * occlusionCoef;

        // Approximate interreflections of light from blocking surfaces which are generally not completely dark and tend to have similar radiance.
        // Ref: Ch 11.3.3 Accounting for Interreflections, Real-Time Rendering (4th edition).
        // The approximation assumes:
        //      o All surfaces incoming and outgoing radiance is the same 
        //      o Current surface color is the same as that of the occluders
        // Since this sample uses scalar ambient coefficient, we use the scalar luminance of the surface color.
        // This will generally brighten the AO making it closer to the result of full Global Illumination, including interreflections.
        if (CB.RTAO_approximateInterreflections)
        {
            // ToDo test perf impact of reading the texture and move this to compose pass
            float3 surfaceAlbedo = g_texAOSurfaceAlbedo[srcPixelIndex].xyz;

            float kA = ambientCoef;
            float rho = CB.RTAO_diffuseReflectanceScale * RGBtoLuminance(surfaceAlbedo);

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
    const float tMax = CB.RTAO_maxShadowRayHitTime; // ToDo make sure its FLT_10BIT_MAX or less since we use 10bit origin depth in RaySort
    float3 surfaceNormal = float3(0, 1, 0);
    float tHit;
    TraceAORayAndReportIfHit(tHit, AORay, tMax, surfaceNormal);
#else
    // ToDo move to a CS if always using a raysort.
    uint2 srcRayIndex = DispatchRaysIndex().xy;
    
#if 0
    uint2 windowSize = uint2(1920, 1080) / 8;
    uint2 topLeft = uint2(6,1) * windowSize;
    uint2 botRight = topLeft + windowSize;
    if (!(srcRayIndex.x >= topLeft.x && srcRayIndex.y >= topLeft.y &&
        srcRayIndex.x < botRight.x && srcRayIndex.y < botRight.y))
    {
        return;
    }
#endif
    // ToDo
    float3 surfaceNormal;
    float depth;
    // ToDO use full precision here?
    DecodeNormalDepth(g_texRayOriginSurfaceNormalDepth[srcRayIndex], surfaceNormal, depth);
	bool hit = depth != 0;   // ToDo use a common func to determine
	if (hit)
	{
		float3 hitPosition = g_texRayOriginPosition[srcRayIndex].xyz;
            
        
        //if (CB.RTAO_UseAdaptiveSampling)
        //{
        //    float filterWeightSum = g_filterWeightSum[srcRayIndex].x;
        //    float clampedFilterWeightSum = min(filterWeightSum, CB.RTAO_AdaptiveSamplingMaxWeightSum);
        //    float sampleScale = 1 - (clampedFilterWeightSum / CB.RTAO_AdaptiveSamplingMaxWeightSum);
        //    
        //    UINT minSamples = CB.RTAO_AdaptiveSamplingMinSamples;
        //    UINT extraSamples = CB.numSamplesToUse - minSamples;

        //    if (CB.RTAO_AdaptiveSamplingMinMaxSampling)
        //    {
        //        numSamples = minSamples + (sampleScale >= 0.001 ? extraSamples : 0);
        //    }
        //    else
        //    {
        //        float scaleExponent = CB.RTAO_AdaptiveSamplingScaleExponent;
        //        numSamples = minSamples + UINT(pow(sampleScale, scaleExponent) * extraSamples);
        //    }
        //}
        
        float tHit;
#if 1
        Ray AORay = GenerateRandomAORay(srcRayIndex, hitPosition, surfaceNormal);
#else
        Ray AORay = { hitPosition, normalize(float3(0.2, 0.4, 0.2)) };
#endif
        float ambientCoef = CalculateAO(tHit, srcRayIndex, AORay, surfaceNormal);

        if (CB.RTAO_UseSortedRays)
        {
            //g_rtAORaysDirectionOriginDepth[srcRayIndex] = float4(EncodeNormal(AORay.direction), depth, 0);
        }
        else
        {
            g_rtAOcoefficient[srcRayIndex] = ambientCoef;
            g_rtAORayHitDistance[srcRayIndex] = RTAO::HasAORayHitAnyGeometry(tHit) ? tHit : CB.RTAO_maxTheoreticalShadowRayHitTime;
        }

#if GBUFFER_AO_COUNT_AO_HITS
        // ToDo test perf impact of writing this
        g_rtAORayHits[srcRayIndex] = RTAO::HasAORayHitAnyGeometry(tHit);
#endif
	}
    else
    {
        if (CB.RTAO_UseSortedRays)
        {
            g_rtAORaysDirectionOriginDepth[srcRayIndex] = 0;
        }

#if GBUFFER_AO_COUNT_AO_HITS
        // ToDo test perf impact of writing this
        g_rtAORayHits[srcRayIndex] = 0;
#endif
    }
#endif
}

#define SKIP_INVALID_RAYS 0

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

#if SKIP_INVALID_RAYS
    uint rayGroupSize = rayGroupDim.y * rayGroupDim.x;
    uint indexMultiplier = 2;
    uint _index1D = index1D * indexMultiplier;

    uint2 rayGroupIndex;
    uint2 rayThreadIndex;
    if (_index1D < CB.raytracingDim.y * CB.raytracingDim.x)
    {
        uint rayGroupIndex1D = _index1D / rayGroupSize;
        uint2 rayGroups = CB.raytracingDim / rayGroupDim;
        rayGroupIndex = uint2(rayGroupIndex1D % rayGroups.x, rayGroupIndex1D / rayGroups.x);
        uint rayThreadIndex1D = (_index1D % rayGroupSize) / indexMultiplier;
        rayThreadIndex = uint2(rayThreadIndex1D % rayGroupDim.x, rayThreadIndex1D / rayGroupDim.x);
    }
    else
    {
        return false;
    }
#else
    // Find the ray group row index.
    uint numValidPixelsInRow = CB.raytracingDim.x;
    uint rowOfRayGroupSize = rayGroupDim.y * numValidPixelsInRow;
    uint rayGroupRowIndex = index1D / rowOfRayGroupSize;

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
#endif

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

    float tHit = RTAO::RayHitDistanceOnMiss;
    float ambientCoef = RTAO::InvalidAOValue;
    if (isActiveRay)
    {
        // ToDo use higher res normal, ray direction?
        // ToDo split raydirection and origin into two resources?
        float dummy;
        float3 rayDirection;
        DecodeNormalDepth(g_texAORaysDirectionOriginDepthHit[srcRayIndex], rayDirection, dummy);
        float3 hitPosition = g_texRayOriginPosition[srcRayIndex].xyz;

        // ToDo test trading for using ray direction insteads
        float3 surfaceNormal;
        float depth;
        DecodeNormalDepth(g_texRayOriginSurfaceNormalDepth[srcRayIndex], surfaceNormal, depth);

        Ray AORay = { hitPosition, rayDirection };
        ambientCoef = CalculateAO(tHit, srcRayIndex, AORay, surfaceNormal);
    }

#if AVOID_SCATTER_WRITES_FOR_SORTED_RAY_RESULTS
    uint2 outPixel = sortedRayIndex;
#else
    uint2 outPixel = srcRayIndex;
#endif

    g_rtAOcoefficient[outPixel] = ambientCoef;
    g_rtAORayHitDistance[outPixel] = RTAO::HasAORayHitAnyGeometry(tHit) ? tHit : CB.RTAO_maxTheoreticalShadowRayHitTime;

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