
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
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"
#include "RTAO.hlsli"

// Output.
RWTexture2D<float4> g_renderTarget : register(u0);

// Input.
ConstantBuffer<ComposeRenderPassesConstantBuffer> g_CB : register(b0);
Texture2D<uint> g_texGBufferPositionHits : register(t0);
Texture2D<uint2> g_texGBufferMaterial : register(t1);    // 16b {1x Material Id, 3x Diffuse.RGB}
Texture2D<float4> g_texGBufferPositionRT : register(t2);
Texture2D<NormalDepthTexFormat> g_texGBufferNormalDepth : register(t3);	// ToDo merge some GBuffers resources ?
Texture2D<float> g_texAO : register(t5);
StructuredBuffer<PrimitiveMaterialBuffer> g_materials : register(t7);
Texture2D<float> g_texRayHitDistance : register(t9);
Texture2D<uint> g_texTemporalSupersamplingDisocclusionMap : register(t10);  // ToDo is this frameAge? rename
Texture2D<float4> g_texColor : register(t11);
Texture2D<float4> g_texAOSurfaceAlbedo : register(t12);
Texture2D<float4> g_texVariance : register(t13);
Texture2D<float4> g_texLocalMeanVariance : register(t14);


float4 RenderPBRResult(in uint2 DTid)
{
    float4 color;
    bool hit = g_texGBufferPositionHits[DTid] > 0;
    if (hit)
    {
        float3 surfaceNormal;
        DecodeNormal(g_texGBufferNormalDepth[DTid], surfaceNormal);

        uint2 materialInfo = g_texGBufferMaterial[DTid];
        UINT materialID;
        float3 albedo;
        DecodeMaterial16b(materialInfo, materialID, albedo);
        PrimitiveMaterialBuffer material = g_materials[materialID];
        float3 specular = RemoveSRGB(material.Ks);      // ToDo review SRGB calls
        float3 phongColor = g_texColor[DTid].xyz;

        // Subtract the default ambient illumination that has already been added to the color in raytrace pass.
        float ambientCoef = g_CB.isAOEnabled ? g_texAO[DTid] : g_CB.defaultAmbientIntensity;
        ambientCoef -= g_CB.defaultAmbientIntensity;

        float3 ambientColor = ambientCoef * g_texAOSurfaceAlbedo[DTid].xyz;
        color = float4(phongColor + ambientColor, 1);

        // Apply visibility falloff.
        float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
        float t = length(hitPosition);
        color = lerp(color, BackgroundColor, 1.0 - exp(-DISTANCE_FALLOFF * t * t * t * t));
    }
    else
    {
#if USE_ENVIRONMENT_MAP
        uint2 materialInfo = g_texGBufferMaterial[DTid];
        UINT materialID;
        float3 albedo;
        DecodeMaterial16b(materialInfo, materialID, albedo);
        albedo = RemoveSRGB(albedo);
        float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
        float t = (clamp(hitPosition.y, 0.015, 0.025) - 0.015) * 100;       // ToDo
        color = lerp(BackgroundColor, float4(albedo, 1), t);
#else
        color = BackgroundColor;
#endif
    }
    return color;
}

float4 RenderAOResult(in uint2 DTid)
{
    float4 color = float4(1, 1, 1, 1);
    bool hit = g_texGBufferPositionHits[DTid] > 0;
    if (hit)
    {
        float ambientCoef = g_texAO[DTid];
        color = ambientCoef != RTAO::InvalidAOValue ? ambientCoef : 1;
        float4 albedo = float4(1, 1, 1, 1);
        color *= albedo;

        if (g_CB.compositionType == AmbientOcclusionAndDisocclusionMap)
        {
            uint frameAge = g_texTemporalSupersamplingDisocclusionMap[DTid].x;
            color = frameAge == 1 ? float4(1, 0, 0, 1) : color;


            float normalizedFrameAge = min(1.f, frameAge / 32.f);
            float3 minFrameAgeColor = float3(153, 18, 15) / 255;
            float3 maxFrameAgeColor = float3(170, 220, 200) / 255;
            color = float4(lerp(minFrameAgeColor, maxFrameAgeColor, normalizedFrameAge), 1);
        }
    }

    return color;
}

float4 RenderVariance(in uint2 DTid)
{
    float4 color = float4(1, 1, 1, 1);
    bool hit = g_texGBufferPositionHits[DTid] > 0;
    if (hit)
    {
        float variance;
        if (g_CB.compositionType == CompositionType::AmbientOcclusionVariance)
            variance = g_texVariance[DTid].x;
        else
            variance = g_texLocalMeanVariance[DTid].y;
        // ToDo why is minHitDistance 0 or very dark on outer edges?
        float3 minSampleColor = float3(20, 20, 20) / 255;
        float3 maxSampleColor = float3(255, 255, 255) / 255;
        if (g_CB.variance_visualizeStdDeviation)
            variance = sqrt(variance);
        variance *= g_CB.variance_scale;
        color = float4(lerp(minSampleColor, maxSampleColor, variance), 1);
    }

    return color;
}

float4 RenderRayHitDistance(in uint2 DTid)
{
    float4 color = float4(1, 1, 1, 1);
    bool hit = g_texGBufferPositionHits[DTid] > 0;
    if (hit)
    {
        // ToDo why is minHitDistance 0 or very dark on outer edges?
        float3 minDistanceColor = float3(15, 18, 153) / 255;
        float3 maxDistanceColor = float3(170, 220, 200) / 255;
        float hitDistance = g_texRayHitDistance[DTid].x;
        float hitCoef = hitDistance / g_CB.RTAO_MaxRayHitDistance;
        color = hitCoef >= 0.0f ? float4(lerp(minDistanceColor, maxDistanceColor, hitCoef), 1) : float4(1, 1, 1, 1);
    }

    return color;
}

float4 RenderNormalOrDepth(in uint2 DTid)
{
    float4 color = float4(1, 1, 1, 1);
    bool hit = g_texGBufferPositionHits[DTid] > 0;
    if (hit)
    {
        float depth;
        float3 surfaceNormal;
        DecodeNormalDepth(g_texGBufferNormalDepth[DTid], surfaceNormal, depth);

        if (g_CB.compositionType == CompositionType::NormalsOnly)
            color = float4(surfaceNormal, 1);
        else 
            color = depth / 80; // ToDo
    }

    return color;
}

float4 RenderAlbedo(in uint2 DTid)
{
    float4 color = float4(1, 1, 1, 1);
    bool hit = g_texGBufferPositionHits[DTid] > 0;
    if (hit)
    {
        uint2 materialInfo = g_texGBufferMaterial[DTid];
        UINT materialID;
        float3 albedo;
        DecodeMaterial16b(materialInfo, materialID, albedo);
        color = float4(albedo, 1);
    }

    return color;
}

float4 RenderDisocclusionMap(in uint2 DTid)
{
    float4 color = float4(1, 1, 1, 1);
    bool hit = g_texGBufferPositionHits[DTid] > 0;
    if (hit)
    {
        color = g_texTemporalSupersamplingDisocclusionMap[DTid].x == 1 ? float4(1, 0, 0, 0) : float4(1, 1, 1, 1);
    }

    return color;
}

// ToDo Cleanup SRGB here and elsewhere dfealing with in/out colors
[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID )
{
    float4 color;
    switch(g_CB.compositionType)
    {
    case CompositionType::PBRShading: 
        color = RenderPBRResult(DTid);
        break;

    case CompositionType::AmbientOcclusionOnly_Denoised:
    case CompositionType::AmbientOcclusionOnly_TemporallySupersampled:
    case CompositionType::AmbientOcclusionOnly_RawOneFrame:
    case CompositionType::AmbientOcclusionAndDisocclusionMap:
        color = RenderAOResult(DTid);
        break;

    case CompositionType::AmbientOcclusionVariance:
    case CompositionType::AmbientOcclusionLocalVariance:
        color = RenderVariance(DTid);
        break;

    case CompositionType::RTAOHitDistance:
        color = RenderRayHitDistance(DTid);
        break;

    case CompositionType::NormalsOnly:
    case CompositionType::DepthOnly:
        color = RenderNormalOrDepth(DTid);
        break;

    case CompositionType::Diffuse:
        color = RenderAlbedo(DTid);
        break;

    case CompositionType::DisocclusionMap:
        color = RenderDisocclusionMap(DTid);
        break;  
    default:
        color = float4(1, 0, 0, 0);
        break;
    }

	// Write the composited color to the output texture.
    g_renderTarget[DTid] = float4(ApplySRGB(color.rgb), color.a);
}

