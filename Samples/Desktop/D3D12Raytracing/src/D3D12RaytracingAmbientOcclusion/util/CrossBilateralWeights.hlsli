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

#ifndef CROSSBILATERALWEIGHTS_HLSLI
#define CROSSBILATERALWEIGHTS_HLSLI

namespace CrossBilateral
{
    namespace Normal
    {
        struct Parameters
        {
            float Sigma;
            float SigmaExponent;
        };

        // Get cross bilateral normal based weights.
        float4 GetWeights(
            in float3 TargetNormal,
            in float3 SampleNormals[4],
            in Parameters Params)
        {
            float4 NdotSampleN = float4(
                dot(TargetNormal, SampleNormals[0]),
                dot(TargetNormal, SampleNormals[1]),
                dot(TargetNormal, SampleNormals[2]),
                dot(TargetNormal, SampleNormals[3]));

            // Apply adjustment scale to the dot product. 
            // Values greater than 1 increase tolerance scale 
            // for unwanted inflated normal differences,
            // such as due to low-precision normal quantization.
            NdotSampleN *= Params.Sigma;

            float4 normalWeights = pow(saturate(NdotSampleN), Params.SigmaExponent);

            return normalWeights;
        }
    }

    // Linear depth.
    namespace Depth
    {
        struct Parameters
        {
            float Sigma;
            float WeightCutoff;
            uint NumMantissaBits;
        };

        // Remap Ddxy that was calculated for a unit pixel offset at a given depth to a new offset.
        // ToDo rename to ddxy dxdy and standardize.
        float2 RemapDdxy(in float depth, in float2 ddxy, in float2 newOffset)
        {
            // Calculate depth via interpolation with perspective correction
            // Ref: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/visibility-problem-depth-buffer-depth-interpolation
            // Given depth buffer interpolation for finding z at offset q along z0 to z1
            //      z =  1 / (1 / z0 * (1 - q) + 1 / z1 * q)
            // and z1 = z0 + ddxy, where z1 is at a unit pixel offset [1, 1]
            // z can be calculated via ddxy as
            //
            //      z = (z0 + ddxy) / (1 + (1-q) / z0 * ddxy) 

            float z0 = depth;
            float2 zxy = (z0 + ddxy) / (1 + ((1 - newOffset) / z0) * ddxy);

            return zxy - z0;;
        }


        float4 GetWeights(
            in float TargetDepth,
            in float2 Ddxy,
            in float4 SampleDepths, // offset in-between the samples to scale ddxy by.
            in Parameters Params)
        {
            float depthThreshold = dot(1, abs(Ddxy));
            // ToDo pass from a cb?
            float depthFloatPrecision = FloatPrecision(TargetDepth, Params.NumMantissaBits);

            float depthTolerance = Params.Sigma * depthThreshold + depthFloatPrecision;
            float4 depthWeights = min(depthTolerance / (abs(SampleDepths - TargetDepth) + FLT_EPSILON), 1);
            depthWeights *= depthWeights >= Params.WeightCutoff;

            return depthWeights;
        }

        float4 GetWeights(
            in float TargetDepth,
            in float2 Ddxy,
            in float4 SampleDepths,
            in float2 SampleOffset, // offset in-between the samples to remap ddxy for.
            in Parameters Params)
        {
            float2 remappedDdxy = RemapDdxy(TargetDepth, Ddxy, SampleOffset);
            return GetWeights(TargetDepth, remappedDdxy, SampleDepths, Params);
        }

    /*
        ToDo 
        // Calculate expected depths at sample pixels given current depth, dxdy and an offset to the sample pixels.
        float4 vExpectedDepths;
        [unroll]
        for (uint i = 0; i < 4; i++)
            vExpectedDepths[i] = GetDepthAtPixelOffset(TargetDepth, dxdy, samplePixelOffsets[i]);

        float4 vDepthThresholds = abs(vExpectedDepths - TargetDepth);
        float depthFloatPrecision = FloatPrecision(TargetDepth, cb.DepthNumMantissaBits);
        float4 vDepthTolerances = cb.depthSigma * vDepthThresholds + depthFloatPrecision;

        float fEpsilon = 1e-6 * TargetDepth;
        depthWeights = min(vDepthTolerances / (abs(SampleDepths - vExpectedDepths) + fEpsilon), 1);
        //g_texOutputDebug2[TargetIndex] = depthWeights;
        // ToDo Should there be a Depth falloff with a cutoff below 1?
        // ToDo revise the coefficient
        depthWeights *= depthWeights >= 0.5;   // ToDo revise - this is same as comparing to depth tolerance
    */
    }

    namespace Bilinear
    {
        // TargetOffset - offset from the top left ([0,0]) sample of the quad samples.
        float4 GetWeights(in float2 TargetOffset)
        {
            float4 bilinearWeights =
                float4(
                    (1 - TargetOffset.x) * (1 - TargetOffset.y),
                    TargetOffset.x * (1 - TargetOffset.y),
                    (1 - TargetOffset.x) * TargetOffset.y,
                    TargetOffset.x * TargetOffset.y);

            return bilinearWeights;
        }
    }

    namespace BilinearDepthNormal
    {
        struct Parameters
        {
            Normal::Parameters Normal;
            Depth::Parameters Depth;
        };

        float4 GetWeights(
            in float TargetDepth,
            in float3 TargetNormal,
            in float2 TargetOffset,
            in float2 Ddxy,
            in float4 SampleDepths,
            in float3 SampleNormals[4],
            in float2 SamplesOffset,
            Parameters Params)
        {
            float4 bilinearWeights = Bilinear::GetWeights(TargetOffset);

            //ToDo add/subtract the targetOffset from each samplesoffset

            float4 depthWeights = Depth::GetWeights(TargetDepth, Ddxy, SampleDepths, SamplesOffset, Params.Depth);
            float4 normalWeights = Normal::GetWeights(TargetNormal, SampleNormals, Params.Normal);

            return bilinearWeights * depthWeights * normalWeights;
        }

        float4 GetWeights(
            in float TargetDepth,
            in float3 TargetNormal,
            in float2 TargetOffset,
            in float2 Ddxy,
            in float4 SampleDepths,
            in float3 SampleNormals[4],
            Parameters Params)
        {
            float4 bilinearWeights = Bilinear::GetWeights(TargetOffset);
            float4 depthWeights = Depth::GetWeights(TargetDepth, Ddxy, SampleDepths, Params.Depth);
            float4 normalWeights = Normal::GetWeights(TargetNormal, SampleNormals, Params.Normal);

            return bilinearWeights * depthWeights * normalWeights;
        }
    }
}

#endif // CROSSBILATERALWEIGHTS_HLSLI