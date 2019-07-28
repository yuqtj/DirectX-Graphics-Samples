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
#include "util/BxDF.hlsli"
#include "RTAO/Shaders/RTAO.hlsli"

// Output.
RWTexture2D<float4> g_renderTarget : register(u0);

// Input.
ConstantBuffer<ComposeRenderPassesConstantBuffer> g_CB : register(b0);
Texture2D<uint> g_texGBufferPositionHits : register(t0);
Texture2D<uint2> g_texGBufferMaterial : register(t1);    // 16b {1x Material Id, 3x Diffuse.RGB}
Texture2D<float4> g_texGBufferPositionRT : register(t2);
Texture2D<float4> g_texGBufferNormal : register(t3);	// ToDo merge some GBuffers resources ?
Texture2D<float> g_texAO : register(t5);
Texture2D<float> g_texVisibility : register(t6);
StructuredBuffer<PrimitiveMaterialBuffer> g_materials : register(t7);
Texture2D<float> g_texFilterWeightSum : register(t8);
Texture2D<float> g_texRayHitDistance : register(t9);
Texture2D<uint> g_texTemporalCacheDisocclusionMap : register(t10);
Texture2D<float4> g_texColor : register(t11);
Texture2D<float4> g_texAOSurfaceAlbedo : register(t12);
Texture2D<float4> g_texVariance : register(t13);

SamplerState LinearWrapSampler : register(s0);

float CalculateDiffuseCoefficient(in float3 hitPosition, in float3 toLightRay, in float3 normal);
float3 CalculateSpecularCoefficient(in float3 hitPosition, in float3 toEyeRay, in float3 toLightRay, in float3 normal, in float specularPower);
float3 CalculatePhongLighting(in float3 normal, in float3 hitPosition, in float3 toEyeRay, in float visibilityCoefficient, in float ambientCoef, in float3 diffuse, in float3 specular, in float specularPower = 50);


// ToDo Cleanup SRGB here and elsewhere dfealing with in/out colors
[numthreads(ComposeRenderPassesCS::ThreadGroup::Width, ComposeRenderPassesCS::ThreadGroup::Height, 1)]
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
#if COMPRES_NORMALS
        float3 surfaceNormal = DecodeNormal(g_texGBufferNormal[DTid].xy);
#else
        float3 surfaceNormal = g_texGBufferNormal[DTid].xyz;
#endif
		float visibilityCoefficient = g_texVisibility[DTid];

        // ToDo rename to enable dynamic AO?
        float ambientCoef = g_CB.defaultAmbientIntensity;
        
        if (hit)
        {
            ambientCoef  = g_CB.enableAO ? g_texAO[DTid] : g_CB.defaultAmbientIntensity;
        }

        // ToDo use switch?
        // Calculate final color.
        if (g_CB.compositionType == CompositionType::PhongLighting)
        {
            uint2 materialInfo = g_texGBufferMaterial[DTid];
            UINT materialID;
            float3 diffuse;
            DecodeMaterial16b(materialInfo, materialID, diffuse);

            PrimitiveMaterialBuffer material = g_materials[materialID];
            float3 toEyeRay = normalize(g_CB.cameraPosition.xyz - hitPosition);
            diffuse = diffuse; 
            float3 specular = RemoveSRGB(material.Ks);
#if 1
            float3 phongColor = g_texColor[DTid].xyz;
         
            // Subtract the default ambient illuminatation that has already been added to the color in raytrace pass.
            ambientCoef -= g_CB.defaultAmbientIntensity;

            float3 ambientColor = ambientCoef * g_texAOSurfaceAlbedo[DTid].xyz;
            color = float4(phongColor + ambientColor, 1);
#elif 0
            float3 phongColor = CalculatePhongLighting(surfaceNormal, hitPosition, toEyeRay, visibilityCoefficient, ambientCoef, diffuse, specular, material.specularPower);
#else
            float3 toLightRay = normalize(g_CB.lightPosition - hitPosition);
            float3 phongColor = Shade(material.type, diffuse, specular, g_CB.lightDiffuseColor.xyz, visibilityCoefficient, ambientCoef, material.roughness, surfaceNormal, toEyeRay, toLightRay);
#endif

            // Apply visibility falloff.
            // ToDo incorrect when subtracting camera
            distance = length(hitPosition);// -g_CB.cameraPosition.xyz);
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
                uint frameAge = g_texTemporalCacheDisocclusionMap[DTid].x;
                color = frameAge == 1 ? float4(1, 0, 0, 1) : color;


                float normalizedFrameAge = min(1.f, frameAge / 32.f);
                float3 minFrameAgeColor = float3(153, 18, 15) / 255;
                float3 maxFrameAgeColor = float3(170, 220, 200) / 255;
                color = float4(lerp(minFrameAgeColor, maxFrameAgeColor, normalizedFrameAge), 1);
            }
        }
        else if (g_CB.compositionType == CompositionType::AmbientOcclusionHighResSamplingPixels)
        {
            // ToDo
            color = float4(1,1,0, 1);
#if 0
            UINT numSamples = g_CB.RTAO_MaxSPP;
            if (g_CB.RTAO_UseAdaptiveSampling)
            {
                float filterWeightSum = g_texFilterWeightSum[DTid].x;
                float clampedFilterWeightSum = min(filterWeightSum, g_CB.RTAO_AdaptiveSamplingMaxWeightSum);
                float sampleScale = 1 - (clampedFilterWeightSum / g_CB.RTAO_AdaptiveSamplingMaxWeightSum);

                UINT minSamples = g_CB.RTAO_AdaptiveSamplingMinSamples;
                UINT extraSamples = g_CB.RTAO_MaxSPP - minSamples;

                if (g_CB.RTAO_AdaptiveSamplingMinMaxSampling)
                {
                    numSamples = minSamples + (sampleScale >= 0.001 ? extraSamples : 0);
                }
                else
                {
                    float scaleExponent = g_CB.RTAO_AdaptiveSamplingScaleExponent;
                    numSamples = minSamples + UINT(pow(sampleScale, scaleExponent) * extraSamples);
                }
            }
            float3 minSampleColor = float3(170, 220, 200) / 255;
            float3 maxSampleColor = float3(153, 18, 15) / 255;
            float sppScale = float(numSamples) / g_CB.RTAO_MaxSPP;
            color = float4(lerp(minSampleColor, maxSampleColor, sppScale), 1);
#endif
        }
        else if (g_CB.compositionType == CompositionType::Variance)
        {
            // ToDo why is minHitDistance 0 or very dark on outer edges?
            float3 minSampleColor = float3(20, 20, 20) / 255;
            float3 maxSampleColor = float3(255, 255, 255) / 255;
            float variance = g_texVariance[DTid].x;
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
            color = float4(0, 0, 0, 1); // ToDo
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
            color = g_texTemporalCacheDisocclusionMap[DTid].x == 1 ? float4(1, 0, 0, 0) : float4(1, 1, 1, 1);
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



//***************************************************************************
//****************------ Utility functions -------***************************
//***************************************************************************


// Diffuse lighting calculation.
float CalculateDiffuseCoefficient(in float3 hitPosition, in float3 toLightRay, in float3 normal)
{
	float fNDotL = saturate(dot(toLightRay, normal));
	return fNDotL;
}

// Phong lighting specular component
float3 CalculateSpecularCoefficient(in float3 hitPosition, in float3 toEyeRay, in float3 toLightRay, in float3 normal, in float specularPower)
{
	float3 reflectedToLightRay = reflect(toLightRay, normal);
    return pow(saturate(dot(reflectedToLightRay, toEyeRay)), specularPower);
}

// Phong lighting model = ambient + diffuse + specular components.
float3 CalculatePhongLighting(in float3 normal, in float3 hitPosition, in float3 toEyeRay,  in float visibilityCoefficient, in float ambientCoef, in float3 diffuse, in float3 specular, in float specularPower)
{
	float3 toLightRay = normalize(g_CB.lightPosition - hitPosition);

	// Diffuse component.
	float Kd = CalculateDiffuseCoefficient(hitPosition, toLightRay, normal);
	float3 diffuseColor = visibilityCoefficient * diffuse * Kd * g_CB.lightDiffuseColor;

	// Specular component.
	float3 specularColor = float3(0, 0, 0);
	if (visibilityCoefficient > 0.99f)
	{
		float3 lightSpecularColor = float3(1, 1, 1);
		float3 Ks = CalculateSpecularCoefficient(hitPosition, toEyeRay, toLightRay, normal, specularPower);
		specularColor = specular * Ks * lightSpecularColor;
	}

	// Ambient component.
	float3 ambientColor = ambientCoef * diffuse;

	// ToDo
	return ambientColor + diffuseColor;//  +specularColor;
}