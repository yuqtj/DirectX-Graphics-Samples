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
Texture2D<float> g_texFilterWeightSum : register(t8);
Texture2D<float> g_texRayHitDistance : register(t9);
Texture2D<uint> g_texTemporalSupersamplingDisocclusionMap : register(t10);  // ToDo is this frameAge? rename
Texture2D<float4> g_texColor : register(t11);
Texture2D<float4> g_texAOSurfaceAlbedo : register(t12);
Texture2D<float4> g_texVariance : register(t13);
Texture2D<float4> g_texLocalMeanVariance : register(t14);

// ToDo Cleanup SRGB here and elsewhere dfealing with in/out colors
[numthreads(DefaultComputeShaderParams::ThreadGroup::Width, DefaultComputeShaderParams::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID )
{
	if (DTid.x >= g_CB.rtDimensions.x || DTid.y >= g_CB.rtDimensions.y)
	{
		return;
	}

	bool hit = g_texGBufferPositionHits[DTid] > 0;
	float distance = 1e6;
	float4 color;
#if 0
    float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
    if (hit && length(hitPosition.xz + float2(10,-10)) < 60)
    {
#else
    float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
	//if (hit)
    if (true)
	{
#endif
        float depthDummy;
        float3 surfaceNormal;
        DecodeNormalDepth(g_texGBufferNormalDepth[DTid], surfaceNormal, depthDummy);

        // ToDo rename to enable dynamic AO?
        float ambientCoef = g_CB.defaultAmbientIntensity;
        
        if (hit)
        {
            ambientCoef  = g_CB.isAOEnabled ? g_texAO[DTid] : g_CB.defaultAmbientIntensity;
        }

        // ToDo use switch?
        // ToDo rename phong
        // Calculate final color.
        if (g_CB.compositionType == CompositionType::PhongLighting)
        {
            uint2 materialInfo = g_texGBufferMaterial[DTid];
            UINT materialID;
            float3 diffuse;
            DecodeMaterial16b(materialInfo, materialID, diffuse);

            PrimitiveMaterialBuffer material = g_materials[materialID];
            float3 specular = RemoveSRGB(material.Ks);      // ToDo review SRGB calls
            float3 phongColor = g_texColor[DTid].xyz;
         
            // Subtract the default ambient illuminatation that has already been added to the color in raytrace pass.
            ambientCoef -= g_CB.defaultAmbientIntensity;

            float3 ambientColor = ambientCoef * g_texAOSurfaceAlbedo[DTid].xyz;
            color = float4(phongColor + ambientColor, 1);


            // Apply visibility falloff.
            // ToDo incorrect when subtracting camera
            
            distance = length(hitPosition);
            float t = distance;
            
            // ToDo
            color = lerp(color, BackgroundColor, 1.0 - exp(-DISTANCE_FALLOFF * t*t*t*t));
        }
        else if (g_CB.compositionType == CompositionType::AmbientOcclusionOnly_Denoised ||
            g_CB.compositionType == CompositionType::AmbientOcclusionOnly_TemporallySupersampled ||
                 g_CB.compositionType == CompositionType::AmbientOcclusionOnly_RawOneFrame ||
                 g_CB.compositionType == AmbientOcclusionAndDisocclusionMap)
        {
            color = ambientCoef != RTAO::InvalidAOValue ? ambientCoef : 1;// g_CB.defaultAmbientIntensity;
            float4 albedo = float4(1, 1, 1, 1);// float4(0.75f, 0.75f, 0.75f, 1.0f);
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
        else if (g_CB.compositionType == CompositionType::AmbientOcclusionVariance ||
            g_CB.compositionType == CompositionType::AmbientOcclusionLocalVariance)
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
        else if (g_CB.compositionType == CompositionType::RTAOHitDistance)
        {
            // ToDo why is minHitDistance 0 or very dark on outer edges?
            float3 minDistanceColor = float3(15, 18, 153) / 255;
            float3 maxDistanceColor = float3(170, 220, 200) / 255;
            float hitDistance = g_texRayHitDistance[DTid].x;
            float hitCoef = hitDistance / g_CB.RTAO_MaxRayHitDistance;
            color = hitCoef >= 0.0f ? float4(lerp(minDistanceColor, maxDistanceColor, hitCoef), 1) : float4(1, 1, 1, 1);
        }
        else if (g_CB.compositionType == CompositionType::NormalsOnly)
        {
            color = float4(surfaceNormal, 1);
        }
        else if (g_CB.compositionType == CompositionType::DepthOnly)
        {
            color = depthDummy / 80; // ToDo
        }
        else if (g_CB.compositionType == CompositionType::Diffuse)
        {
            uint2 materialInfo = g_texGBufferMaterial[DTid];
            UINT materialID;
            float3 diffuse;
            DecodeMaterial16b(materialInfo, materialID, diffuse);

            color = float4(diffuse, 1); // ToDo
        }
        else if (g_CB.compositionType == CompositionType::DisocclusionMap)
        {
            color = g_texTemporalSupersamplingDisocclusionMap[DTid].x == 1 ? float4(1, 0, 0, 0) : float4(1, 1, 1, 1);
        }
	}
	else
	{
        if (g_CB.compositionType == CompositionType::PhongLighting)
        {
#if USE_ENVIRONMENT_MAP
            uint2 materialInfo = g_texGBufferMaterial[DTid];
            UINT materialID;
            float3 diffuse;
            DecodeMaterial16b(materialInfo, materialID, diffuse);
            diffuse = RemoveSRGB(diffuse);
            float t = (clamp(hitPosition.y, 0.015, 0.025) - 0.015) * 100;
            color = lerp(BackgroundColor, float4(diffuse, 1), t);
#else
            color = BackgroundColor;
#endif
        }
        else
        {
            color = float4(1, 1, 1, 1);
        }
    }

	// Write the composited color to the output texture.
    g_renderTarget[DTid] = float4(ApplySRGB(color.rgb), color.a);
}

