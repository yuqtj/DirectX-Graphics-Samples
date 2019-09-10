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
#include "Scene.h"
#include "RTAO/RTAO.h"

class RTAO;

// ToDo rename to AO denoiser?
namespace Denoiser_Args
{
    extern BoolVar Denoising_UseSmoothedVariance;
}

class Denoiser
{
public:
    enum class ResourceType {
        Variance = 0,
        LocalMeanVariance
    };

    enum DenoiseStage {
        Denoise_Stage1_TemporalReverseReproject = 0x1 << 0,
        Denoise_Stage2_Denoise = 0x1 << 1,
        Denoise_StageAll = Denoise_Stage1_TemporalReverseReproject | Denoise_Stage2_Denoise
    };
    // Ctors.
    Denoiser() {}
    ~Denoiser() {} // ToDo

    // Public methods.
    void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap);
    void Run(Scene& scene, Pathtracer& pathtracer, RTAO& rtao, DenoiseStage stage = Denoise_StageAll);
    void SetResolution(UINT width, UINT height);
    void Release();
        
    // Getters/Setters.
    static DXGI_FORMAT ResourceFormat(ResourceType resourceType);
    UINT DenoisingWidth() { return m_denoisingWidth; }
    UINT DenoisingHeight() { return m_denoisingHeight; }

private:
    void TemporalReverseReproject(Scene& scene, Pathtracer& pathtracer);
    void TemporalSupersamplingBlendWithCurrentFrame(RTAO& rtao);
    void BlurDisocclusions(Pathtracer& pathtracer);

    void CreateDeviceDependentResources();
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateTextureResources();
    void ApplyAtrousWaveletTransformFilter(Pathtracer& pathtracer, RTAO& rtao, bool isFirstPass);
    void CreateResolutionDependentResources();

    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;

    UINT m_denoisingWidth = 0;
    UINT m_denoisingHeight = 0;

    // ToDo dedupe resources. Does dpeth need to have 2 instances?   
    GpuResource m_temporalCache[2][TemporalSupersampling::Count]; // ~array[Read/Write ping pong resource][Resources].

    GpuResource m_temporalAOCoefficient[2];    // ToDo why is this not part of m_temporalCache?
    GpuResource m_temporalSupersampling_blendedAOCoefficient[2];
    GpuResource m_cachedFrameAgeValueSquaredValueRayHitDistance;

    // ToDo use a common ping-pong index? 
    // ToDo cleanup readId should be for input to TAO, confusing.
    UINT          m_temporalCacheCurrentFrameResourceIndex = 0;
    UINT          m_temporalCacheCurrentFrameTemporalAOCoefficientResourceIndex = 0;

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
    GpuKernels::FillInMissingValuesFilter m_fillInMissingValuesFilterKernel;
    GpuKernels::BilateralFilter m_bilateralFilterKernel;

    friend class Composition;
};