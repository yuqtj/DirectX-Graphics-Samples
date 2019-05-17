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

// Bidirectional transmission distribution function (BTDF).
// Ref: Ray Tracing from the Ground Up, Suffern

// Legend:
// wi - incident ray
// iorIn - ior of media ray is coming from
// iorOut - ior of media ray is going to
// eta - relative index of refraction, namely iorIn / iorOut

#ifndef BRDF_HLSL
#define BRDF_HLSL

// BTDF - bidirectional transmission distribution function.
// This namespace implements BTDF for a perfect transmitter that uses a single index of refraction (ior)
// and iorOut represent air, i.e. 1.
namespace SimpleBTDF {

    // Sample the value of BTDF.
    // Kt - transmission coefficient
    // normal - surface normal at hit point
    // wo - outgoing/viewing direction
    // wt - transmitted ray
    // ior - index of refraction
    float3 Sample_f(in float Kt, in float3 normal, in float3 wo, in float ior, out float3 wt)
    {
        float cos_thetai = dot(normal, wo);
        float eta = iorIn / iorOut;

        // Cosine of wt's angle.
        float cos_thetat = sqrt(1 - (1 - cos_thetai * cos_thetai) / (eta * eta));

        // Transmitted ray.
        wt = -wo / eta - (cos_thetat - cos_thetai / eta) * normal;

        float3 white = 1;

        return 
        
        
    }
}

#endif // BRDF_HLSL