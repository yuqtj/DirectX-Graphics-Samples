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

// Desc: Generates adaptive count of rays using heuristics for RTAO
// Generates rays based on the per-pixel frame age and generates
// 1-4 or 1-16 rays per 2x2 or 4x4 pixel quad based on the per-quad minimum frame age.
// Inactive rays have ray origin depth set to 0.
//
#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "RandomNumberGenerator.hlsli"
#include "RaySorting.hlsli"

Texture2D<NormalDepthTexFormat> g_texRayOriginSurfaceNormalDepth : register(t0);
Texture2D<float4> g_texRayOriginPosition : register(t1);
Texture2D<uint2> g_texFrameAge : register(t2);

// ToDo use higher bit format?
RWTexture2D<NormalDepthTexFormat> g_rtRaysDirectionOriginDepth : register(u0);

ConstantBuffer<AdaptiveRayGenConstantBuffer> CB: register(b0);
StructuredBuffer<AlignedHemisphereSample3D> g_sampleSets : register(t3);


groupshared uint FrameAgeCache[DefaultComputeShaderParams::ThreadGroup::Height][DefaultComputeShaderParams::ThreadGroup::Width];

float3 GetRandomRayDirection(in uint2 srcRayIndex, in float3 surfaceNormal)
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
        uint numSampleSetsInX = (CB.textureDim.x + CB.numPixelsPerDimPerSet - 1) / CB.numPixelsPerDimPerSet;
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

    return rayDirection;
}


// ToDo
// Limitations:
// -    TextureDim and CsDim must be a multiple of QuadDim
// Address:
// - pixel neighborhood sampling. Carry sample set id with each pixel?
// - support 2+ spp per ray
// Comment
// - rename frameAge to numTemporalSamples?
[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint2 GTid : SV_GroupThreadID)
{
    float3 surfaceNormal;
    float rayOriginDepth;
    DecodeNormalDepth(g_texRayOriginSurfaceNormalDepth[DTid], surfaceNormal, rayOriginDepth);

#if 1
    float3 rayDirection = 0;
    if (rayOriginDepth != INVALID_RAY_ORIGIN_DEPTH)
    {
        // Generate rays for two pixels in a checkerboard pattern.
        if (CB.doCheckerboardRayGeneration)
        {
            bool isEvenPixel = ((DTid.x + DTid.y) & 1) == 0;
            if (isEvenPixel == CB.checkerboardGenerateRaysForEvenPixels)
            {
                rayDirection = GetRandomRayDirection(DTid, surfaceNormal);
            }
            else
            {
                // Invalidate this ray entry.
                rayOriginDepth = INVALID_RAY_ORIGIN_DEPTH;
            }
        }
        else // CB.MaxRaysPerQuad == 4
        {
            rayDirection = GetRandomRayDirection(DTid, surfaceNormal);
        }
    }

#else
    // Load the frame age for the whole quad into shared memory.
    uint2 frameAgeAndNumRaysToGenerate = g_texFrameAge[DTid & 0xFFFE];
    uint frameAge = frameAgeAndNumRaysToGenerate.x;
    uint numRaysToGenerateOrDenoisePasses = frameAgeAndNumRaysToGenerate.y;

    bool isRayCountValue = !(numRaysToGenerateOrDenoisePasses & 0x80);
    uint numRaysToGenerate = isRayCountValue ? numRaysToGenerateOrDenoisePasses : 0;

    FrameAgeCache[GTid.y][GTid.x] = rayOriginDepth != INVALID_RAY_ORIGIN_DEPTH ? frameAge : CB.MaxFrameAge;
    GroupMemoryBarrierWithGroupSync();

    uint2 quadIndex = GTid / CB.QuadDim;
    uint2 quadThreadIndex = GTid % CB.QuadDim;
    uint quadThreadIndex1D = quadThreadIndex.y * CB.QuadDim.x + quadThreadIndex.x;
    uint2 quadStartGTid = quadIndex * CB.QuadDim;
    
    // Find the minimum frameAge per quad.
    uint minQuadFrameAge = CB.MaxFrameAge;
    for (uint r = 0; r < CB.QuadDim.y; r++)
        for (uint c = 0; c < CB.QuadDim.x; c++)
        {
            minQuadFrameAge = min(minQuadFrameAge, FrameAgeCache[quadStartGTid.y + r][quadStartGTid.x + c]);
        }

    // Calculate number of rays to generate per quad.
    uint MaxNumRaysPerQuad = CB.QuadDim.x * CB.QuadDim.y;
    uint numRaysToGeneratePerQuad = min(MaxNumRaysPerQuad, CB.MaxRaysPerQuad);
#if 0
    if (minQuadFrameAge >= CB.MinFrameAgeForAdaptiveSampling)
    {
        float t = (minQuadFrameAge - CB.MinFrameAgeForAdaptiveSampling) / float(CB.MaxFrameAge - CB.MinFrameAgeForAdaptiveSampling);
        numRaysToGeneratePerQuad = lerp(MaxNumRaysPerQuad, 1, t);
    }
    numRaysToGeneratePerQuad = min(numRaysToGeneratePerQuad, CB.MaxRaysPerQuad);
#endif
    uint StartID = (CB.FrameID * numRaysToGeneratePerQuad) % MaxNumRaysPerQuad;

    // Generate the rays.
    float3 rayDirection = 0;
    if (rayOriginDepth != INVALID_RAY_ORIGIN_DEPTH)
    {
        // Check whether this pixel is due to generate a ray.
#if 0
        // ToDo make sure to generate no more than one ray per pixel.

#else
        if (numRaysToGenerate == 0)
        {

            rayOriginDepth = INVALID_RAY_ORIGIN_DEPTH;
        }
#if 0
        else if (quadThreadIndex1D == CB.FrameID)
        {
            rayDirection = GetRandomRayDirection(DTid, surfaceNormal);
        }
#else
        else if ((quadThreadIndex1D >= StartID && quadThreadIndex1D < StartID + numRaysToGeneratePerQuad) ||
            // Check for when a valid quad thread index range wraps around.
            (StartID + numRaysToGeneratePerQuad >= MaxNumRaysPerQuad && quadThreadIndex1D < ((StartID + numRaysToGeneratePerQuad) % MaxNumRaysPerQuad)))
        {
            rayDirection = GetRandomRayDirection(DTid, surfaceNormal);
        }
#endif
#endif
        else
        {
            // Invalidate this ray entry.
            rayOriginDepth = INVALID_RAY_ORIGIN_DEPTH;
        }
    }
#endif
    g_rtRaysDirectionOriginDepth[DTid] = EncodeNormalDepth(rayDirection, rayOriginDepth);
}
