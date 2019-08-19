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
        void OnRender(D3D12_GPU_VIRTUAL_ADDRESS accelerationStructure, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceHitPositionResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceNormalDepthResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceAlbedoResource, D3D12_GPU_DESCRIPTOR_HANDLE frameAgeResource);
        void ReleaseDeviceDependentResources();
        void ReleaseWindowSizeDependentResources() {}; // ToDo

    private:
        void RenderPass_TemporalSupersamplingReverseProjection();
        void RenderPass_TemporalSupersamplingBlendWithCurrentFrame();
        void MultiPassBlur();

        void CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex);
        void CreateConstantBuffers();
        void CreateAuxilaryDeviceResources();
        void CreateTextureResources();
        void ApplyAtrousWaveletTransformFilter(bool isFirstPass);
        void ApplyAtrousWaveletTransformFilter(const  GpuResource& inValueResource, const  GpuResource& inNormalDepthResource, const  GpuResource& inDepthResource, const  GpuResource& inRayHitDistanceResource, const  GpuResource& inPartialDistanceDerivativesResource, GpuResource* outSmoothedValueResource, GpuResource* varianceResource, GpuResource* smoothedVarianceResource, UINT calculateVarianceTimerId, UINT smoothVarianceTimerId, UINT atrousFilterTimerId);

        void CreateResolutionDependentResources();

        UINT m_width;
        UINT m_height;

        std::shared_ptr<DX::DeviceResources> m_deviceResources;
        std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;

        static UINT s_numInstances;
    };
}