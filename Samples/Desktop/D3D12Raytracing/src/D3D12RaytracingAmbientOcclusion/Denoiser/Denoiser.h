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
#pragma once

// ToDo move to cpp
#include "RaytracingSceneDefines.h"
#include "DirectXRaytracingHelper.h"
#include "RaytracingAccelerationStructure.h"
#include "CameraController.h"
#include "PerformanceTimers.h"
#include "Sampler.h"
#include "GpuKernels.h"
#include "EngineTuning.h"


namespace Denoiser
{
    namespace Args
    {
    }

    class Denoiser
    {
    public:
        // Ctors.
        Denoiser();

        // Public methods.
        void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT maxInstanceContributionToHitGroupIndex);
        void OnUpdate();
        void Execute(GameCore::Camera camera);
        void ReleaseDeviceDependentResources();
        void ReleaseWindowSizeDependentResources() {}; // ToDo
        
        // Getters/Setters.
        void SetResolution(UINT width, UINT height);

    private:
        void TemporalSupersamplingReverseProjection(GameCore::Camera camera);
        void RenderPass_TemporalSupersamplingBlendWithCurrentFrame();
        void MultiPassBlur();

        void CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex);
        void CreateConstantBuffers();
        void CreateAuxilaryDeviceResources();
        void CreateTextureResources();
        void ApplyAtrousWaveletTransformFilter(bool isFirstPass);
        void ApplyAtrousWaveletTransformFilter(const  GpuResource& inValueResource, const  GpuResource& inNormalDepthResource, const  GpuResource& inDepthResource, const  GpuResource& inRayHitDistanceResource, const  GpuResource& inPartialDistanceDerivativesResource, GpuResource* outSmoothedValueResource, GpuResource* varianceResource, GpuResource* smoothedVarianceResource, UINT calculateVarianceTimerId, UINT smoothVarianceTimerId, UINT atrousFilterTimerId);

        GpuResource(&GBufferResources())[GBufferResource::Count];

        void CreateResolutionDependentResources();

        static UINT s_numInstances;
        std::shared_ptr<DX::DeviceResources> m_deviceResources;
        std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;

        UINT m_width;
        UINT m_height;

        // ToDo dedupe resources. Does dpeth need to have 2 instances?   
        GpuResource m_temporalCache[2][TemporalSupersampling::Count]; // ~array[Read/Write ping pong resource][Resources].

        GpuResource m_TSSAOCoefficient[2];    // ToDo why is this not part of m_temporalCache?
        GpuResource m_temporalSupersampling_blendedAOCoefficient[2];
        GpuResource m_cachedFrameAgeValueSquaredValueRayHitDistance;

        // ToDo use a common ping-pong index? 
        // ToDo cleanup readId should be for input to TAO, confusing.
        UINT          m_temporalCacheCurrentFrameResourceIndex = 0;
        UINT          m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex = 0;
        UINT          m_normalDepthCurrentFrameResourceIndex = 0;

        GpuResource m_varianceResources[AOVarianceResource::Count];
        GpuResource m_localMeanVarianceResources[AOVarianceResource::Count];
        GpuResource m_multiPassDenoisingBlurStrength;
        GpuResource m_prevFrameGBufferNormalDepth;

        // GpuKernels
        GpuKernels::FillInCheckerboard      m_fillInCheckerboardKernel;
        GpuKernels::GaussianFilter          m_gaussianSmoothingKernel;
        GpuKernels::TemporalSupersampling_ReverseReproject m_temporalCacheReverseReprojectKernel;
        GpuKernels::TemporalSupersampling_BlendWithCurrentFrame m_temporalCacheBlendWithCurrentFrameKernel;
        GpuKernels::AtrousWaveletTransformCrossBilateralFilter m_atrousWaveletTransformFilter;
        GpuKernels::CalculateVariance       m_calculateVarianceKernel;
        GpuKernels::CalculateMeanVariance   m_calculateMeanVarianceKernel;
        const UINT                          MaxCalculateVarianceKernelInvocationsPerFrame =
            1
            + 1; // Temporal Super-Sampling.


    };
}