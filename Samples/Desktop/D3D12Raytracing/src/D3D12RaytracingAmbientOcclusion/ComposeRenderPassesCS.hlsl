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
	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DTid].xyz;
#endif
#if COMPRES_NORMALS
        float3 surfaceNormal = Decode(g_texGBufferNormal[DTid].xy);
#else
        float3 surfaceNormal = g_texGBufferNormal[DTid].xyz;
#endif
		float visibilityCoefficient = g_texVisibility[DTid];
        float ambientCoef = g_CB.enableAO ? g_texAO[DTid] : 1;

        // Calculate final color.
        if (g_CB.compositionType == CompositionType::PhongLighting)
        {
            uint2 materialInfo = g_texGBufferMaterial[DTid];
            UINT materialID;
            float3 diffuse;
            DecodeMaterial16b(materialInfo, materialID, diffuse);

            PrimitiveMaterialBuffer material = g_materials[materialID];
            float3 toEyeRay = normalize(g_CB.cameraPosition.xyz - hitPosition);
            diffuse = RemoveSRGB(diffuse);
            float3 specular = RemoveSRGB(material.specular);
            float3 phongColor = CalculatePhongLighting(surfaceNormal, hitPosition, toEyeRay, visibilityCoefficient, ambientCoef, diffuse, specular, material.specularPower);
            color = float4(phongColor, 1);

            // Apply visibility falloff.
            // ToDo incorrect when subtracting camera
            distance = length(hitPosition);// -g_CB.cameraPosition.xyz);
            float t = distance;
            color = lerp(color, BackgroundColor, 1.0 - exp(-DISTANCE_FALLOFF * t*t*t));
        }
        else if (g_CB.compositionType == CompositionType::AmbientOcclusionOnly)
        {
            color = ambientCoef;
            float4 albedo = float4(1, 1, 1, 1);// float4(0.75f, 0.75f, 0.75f, 1.0f);
            color *= albedo;
        }
        else if (g_CB.compositionType == CompositionType::AmbientOcclusionHighResSamplingPixels)
        {
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
        }
        else if (g_CB.compositionType == CompositionType::NormalsOnly)
        {
            color = float4(surfaceNormal, 1);
        }
        else if (g_CB.compositionType == CompositionType::DepthOnly)
        {
            color = float4(0, 0, 0, 1); // ToDo
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
            color = float4(diffuse, 1);
#else
            color = BackgroundColor;
#endif
        }
        else if (g_CB.compositionType == CompositionType::AmbientOcclusionHighResSamplingPixels)
        {
            color = float4(1, 1, 1, 1);
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
	float3 reflectedToLightRay = normalize(reflect(toLightRay, normal));
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