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

// Output.
RWTexture2D<float4> g_renderTarget : register(u0);

// Input.
ConstantBuffer<ComposeRenderPassesConstantBuffer> g_CB : register(b0);
Texture2D<uint> g_texGBufferPositionHits : register(t0);
Texture2D<uint> g_texGBufferMaterialID : register(t1);
Texture2D<float3> g_texGBufferPositionRT : register(t2);
Texture2D<float3> g_texGBufferNormal : register(t3);	// ToDo merge some GBuffers resources ?
Texture2D<float> g_texAO : register(t5);
Texture2D<float> g_texVisibility : register(t6);
StructuredBuffer<PrimitiveMaterialBuffer> g_materials : register(t7);


float CalculateDiffuseCoefficient(in float3 hitPosition, in float3 toLightRay, in float3 normal);
float3 CalculateSpecularCoefficient(in float3 hitPosition, in float3 toEyeRay, in float3 toLightRay, in float3 normal, in float specularPower);
float3 CalculatePhongLighting(in float3 normal, in float3 hitPosition, in float3 toEyeRay, in float visibilityCoefficient, in float ambientCoef, in float3 diffuse, in float3 specular, in float specularPower = 50);

float3 ApplySRGB(float3 x)
{
    return x < 0.0031308 ? 12.92 * x : 1.055 * pow(abs(x), 1.0 / 2.4) - 0.055;
}

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

	if (hit)
	{
		float3 hitPosition = g_texGBufferPositionRT[DTid];
		float3 surfaceNormal = g_texGBufferNormal[DTid];
		float visibilityCoefficient = g_texVisibility[DTid];
		float ambientCoef = g_texAO[DTid];

		// ToDo incorrect when subtracting camera
		distance = length(hitPosition);// -g_CB.cameraPosition.xyz);

#if AO_ONLY
		// ToDo remove albedo
		color = ambientCoef;
		float4 albedo = float4(0.75f, 0.75f, 0.75f, 1.0f);
		color *= albedo;
#else
		// Calculate final color.
		UINT materialID = g_texGBufferMaterialID[DTid];
		PrimitiveMaterialBuffer material = g_materials[materialID];
		float3 toEyeRay = normalize(g_CB.cameraPosition.xyz - hitPosition);
		float3 phongColor = CalculatePhongLighting(surfaceNormal, hitPosition, toEyeRay, visibilityCoefficient, ambientCoef, material.diffuse, material.specular, material.specularPower);
		color = float4(phongColor, 1);
#endif

		// Apply visibility falloff.
		float t = distance;
		color = lerp(color, BackgroundColor, 1.0 - exp(-DISTANCE_FALLOFF*t*t*t));
	}
	else
	{
		color = BackgroundColor;
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