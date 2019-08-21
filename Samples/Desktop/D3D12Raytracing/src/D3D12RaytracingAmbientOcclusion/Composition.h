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


namespace Composition
{
    namespace Args
    {
        extern EnumVar AntialiasingMode;
        extern EnumVar CompositionMode;
    }

    extern std::map<std::wstring, BottomLevelAccelerationStructureGeometry>	m_bottomLevelASGeometries;
    extern std::unique_ptr<RaytracingAccelerationStructureManager> m_accelerationStructure;
    extern GpuResource m_grassPatchVB[UIParameters::NumGrassGeometryLODs][2];      // Two VBs: current and previous frame.
    extern D3D12_GPU_DESCRIPTOR_HANDLE g_nullVertexBufferGPUhandle;

    class Composition
    {
    public:
        // Ctors.
        Composition();

        // Public methods.
        void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT maxInstanceContributionToHitGroupIndex);
        void OnUpdate();
        void Render();
        void ReleaseDeviceDependentResources();
        void ReleaseWindowSizeDependentResources() {}; // ToDo
        void SetResolution(UINT width, UINT height);

    private:
        void CreateComposeRenderPassesCSResources();
        void CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex);
        void CreateConstantBuffers();
        void CreateAuxilaryDeviceResources();
        void RenderRNGVisualizations();
        void CreateSamplesRNGVisualization();
        void CreateTextureResources();
        void CreateResolutionDependentResources();
        void BilateralUpsample(
            UINT hiResWidth,
            UINT hiResHeight,
            GpuKernels::UpsampleBilateralFilter::FilterType filterType,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDepthDerivativesResourceHandle,
            GpuResource* outputHiResValueResource,
            LPCWCHAR passName);

        void UpsampleResourcesForRenderComposePass();


        static UINT s_numInstances;
        std::shared_ptr<DX::DeviceResources> m_deviceResources;
        std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;

        // ToDo cleanup - ReduceSum objects are in m_reduceSumKernel.
        ComPtr<ID3D12PipelineState>         m_computePSOs[ComputeShader::Type::Count];
        ComPtr<ID3D12RootSignature>         m_computeRootSigs[ComputeShader::Type::Count];

        ConstantBuffer<ComposeRenderPassesConstantBuffer>   m_csComposeRenderPassesCB;
        ConstantBuffer<RNGConstantBuffer>   m_csHemisphereVisualizationCB;

        GpuKernels::DownsampleValueNormalDepthBilateralFilter m_downsampleValueNormalDepthBilateralFilterKernel;
    };
}