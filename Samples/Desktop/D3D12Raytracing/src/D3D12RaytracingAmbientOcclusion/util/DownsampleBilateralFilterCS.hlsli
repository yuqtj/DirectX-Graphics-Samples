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

// ToDo fix black jaggies on the bottom row in downsampled normal 
Texture2D<float> g_inValue : register(t0);
Texture2D<float4> g_inNormalAndDepth : register(t1);
Texture2D<float2> g_inPartialDistanceDerivatives : register(t2);  // update file name to include ddxy
RWTexture2D<float> g_outValue : register(u0);
RWTexture2D<float4> g_outNormalAndDepth : register(u1);
RWTexture2D<float2> g_outPartialDistanceDerivatives : register(u2);   // ToDo rename hits to Geometryits everywhere

// Ref: https://developer.amd.com/wordpress/media/2012/10/ShopfMixedResolutionRendering.pdf
// ToDo comment
float BilateralInterpolation_DepthNormalBilinearAware(
    float ActualDistance,
    float3 ActualNormal,
    float4 SampleDistances,
    float3 SampleNormals[4],
    float4 BilinearWeights,
    float4 SampleValues)
{
    float4 depthWeights = 1.0 / (abs(SampleDistances - ActualDistance) + 1e-6 * ActualDistance);
    float4 normalWeights = float4(
        pow(dot(ActualNormal, SampleNormals[0]), 32),
        pow(dot(ActualNormal, SampleNormals[1]), 32),
        pow(dot(ActualNormal, SampleNormals[2]), 32),
        pow(dot(ActualNormal, SampleNormals[3]), 32)
        );
    float4 weights = normalWeights * depthWeights * BilinearWeights;

    return dot(weights, SampleValues) / dot(weights, 1);
}

float BilateralInterpolation_DepthNormalAware(
    float ActualDistance,
    float3 ActualNormal,
    float4 SampleDistances,
    float3 SampleNormals[4],
    float4 SampleValues)
{
    float4 depthWeights = 1.0 / (abs(SampleDistances - ActualDistance) + 1e-6 * ActualDistance);
    float4 normalWeights = float4(
        pow(saturate(dot(ActualNormal, SampleNormals[0])), 32),
        pow(saturate(dot(ActualNormal, SampleNormals[1])), 32),
        pow(saturate(dot(ActualNormal, SampleNormals[2])), 32),
        pow(saturate(dot(ActualNormal, SampleNormals[3])), 32)
        );
    float4 weights = normalWeights * depthWeights;

    return dot(weights, SampleValues) / dot(weights, 1);
}

// ToDo remove - not applicable for 2x2 downsample
float BilateralInterpolation_DepthBilinearAware(
    float ActualDistance,
    float4 SampleDistances,
    float4 BilinearWeights,
    float4 SampleValues)
{
    float4 depthWeights = 1.0 / (abs(SampleDistances - ActualDistance) + 1e-6 * ActualDistance);
    float4 weights = depthWeights * BilinearWeights;

    return dot(weights, SampleValues) / dot(weights, 1);
}


float BilateralInterpolation_DepthAware(
    float ActualDistance,
    float4 SampleDistances,
    float4 SampleValues)
{
    // ToDo use depth weights from atrous filter?
    float4 depthWeights = 1.0 / (abs(SampleDistances - ActualDistance) + 1e-6 * ActualDistance);
    float4 weights = depthWeights;

    return dot(weights, SampleValues) / dot(weights, 1);
}

// ToDo strip Bilateral from the name?
// Returns a selected depth index when bilateral downsapling.
void GetDepthIndexFromDownsampleDepthBilateral2x2(out UINT outDepthIndex, in float depths[4], in uint2 DTid)
{
    // Choose a alternate min max depth sample in a checkerboard 2x2 pattern to improve depth correlations for bilateral 2x2 upsampling.
    // Ref: http://c0de517e.blogspot.com/2016/02/downsampled-effects-with-depth-aware.html
    bool checkerboardTakeMin = ((DTid.x + DTid.y) & 1) == 0;

    // Invalidate out-of-bounds 0 depths when taking the min depth.
    float4 vDepths = float4(depths[0], depths[1], depths[2], depths[3]);
    vDepths = checkerboardTakeMin
        ? vDepths > 0.001f ? vDepths : DISTANCE_ON_MISS
        : vDepths;

    float lowResDepth = checkerboardTakeMin
                        ?   min(min(min(vDepths.x, vDepths.y), vDepths.z), vDepths.w)
                        :   max(max(max(vDepths.x, vDepths.y), vDepths.z), vDepths.w);

    // Find the corresponding sample index to the the selected sample depth.
    float4 depthDelta = abs(lowResDepth - vDepths);

    outDepthIndex = depthDelta[0] < depthDelta[1] ? 0 : 1;
    outDepthIndex = depthDelta[2] < depthDelta[outDepthIndex] ? 2 : outDepthIndex;
    outDepthIndex = depthDelta[3] < depthDelta[outDepthIndex] ? 3 : outDepthIndex;
}

void LoadDepthAndNormal(in uint2 texIndex, out float4 encodedNormalAndDepth, out float depth, out float3 normal)
{
    encodedNormalAndDepth = g_inNormalAndDepth[texIndex];
    depth = encodedNormalAndDepth.z;
    normal = DecodeNormal(encodedNormalAndDepth.xy);
}

// ToDo remove _DepthAware from the name?

[numthreads(DownsampleValueNormalDepthBilateralFilter::ThreadGroup::Width, DownsampleValueNormalDepthBilateralFilter::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
    uint2 topLeftSrcIndex = DTid << 1;
    const uint2 srcIndexOffsets[4] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };

    float4 encodedNormalsAndDepths[4];
    float  depths[4];
    float3 normals[4];
    for (int i = 0; i < 4; i++)
    {
        LoadDepthAndNormal(topLeftSrcIndex + srcIndexOffsets[i], encodedNormalsAndDepths[i], depths[i], normals[i]);
    }

    // ToDo rename depth to distance?
    float4 vDepths = float4(depths[0], depths[1], depths[2], depths[3]);
    float4 values = float4(
        g_inValue[topLeftSrcIndex],
        g_inValue[topLeftSrcIndex + srcIndexOffsets[1]],
        g_inValue[topLeftSrcIndex + srcIndexOffsets[2]],
        g_inValue[topLeftSrcIndex + srcIndexOffsets[3]]);

    // ToDo min max depth
    UINT outDepthIndex;
    GetDepthIndexFromDownsampleDepthBilateral2x2(outDepthIndex, depths, DTid);

    // ToDo comment on not interpolating actualNormal
    g_outNormalAndDepth[DTid] = encodedNormalsAndDepths[outDepthIndex];

    // ToDo
    // Since we're reducing the resolution by 2, multiple the partial derivatives by 2. Either that or the multiplier should be applied when calculating weights.
    // ToDo it would be cleaner to apply that multiplier at weights calculation. Or recompute the partial derivatives on downsample?
#if 1 // Pespective correction for the non-linear interpolation
    float2 ddxy = g_inPartialDistanceDerivatives[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
    float z0 = depths[outDepthIndex];
    float pixelOffset = 2;
    float2 zxy = (z0 + ddxy) / (1 + ((1 - pixelOffset) / z0) * ddxy);
    g_outPartialDistanceDerivatives[DTid] = abs(zxy - z0);
#else
    g_outPartialDistanceDerivatives[DTid] = 2 * g_inPartialDistanceDerivatives[topLeftSrcIndex + srcIndexOffsets[outDepthIndex]];
#endif

#ifdef BILATERAL_DOWNSAMPLE_VALUE_POINT_SAMPLING
    g_outValue[DTid] = values[outDepthIndex];

#elif defined(BILATERAL_DOWNSAMPLE_DEPTH_WEIGHTED_VALUE_INTERPOLATION)
    float actualDepth = depths[outDepthIndex];
    g_outValue[DTid] = BilateralInterpolation_DepthAware(actualDepth, vDepths, values);

#elif defined(BILATERAL_DOWNSAMPLE_DEPTH_NORMAL_WEIGHTED_VALUE_INTERPOLATION)
    float actualDepth = depths[outDepthIndex];
    float3 actualNormal = normals[outDepthIndex];
    g_outValue[DTid] = BilateralInterpolation_DepthNormalAware(actualDepth, actualNormal, vDepths, normals, values);
#endif
}