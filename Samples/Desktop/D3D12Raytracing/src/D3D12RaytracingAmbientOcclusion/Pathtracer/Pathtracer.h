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



namespace Pathtracer_Args
{
    extern NumVar DefaultAmbientIntensity;
}

class Pathtracer
{
public:
    // Ctors.
    Pathtracer();

    // Public methods.
    void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap, Scene& scene);
    void Run(Scene& scene);
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources() {}; // ToDo
    void SetCamera(const GameCore::Camera& camera);
    void SetResolution(UINT GBufferWidth, UINT GBufferHeight, UINT RTAOWidth, UINT RTAOHeight);

    // Getters & Setters.
    GpuResource(&GBufferResources(bool getQuarterResResources = false))[GBufferResource::Count];

    void RequestRecreateRaytracingResources() { m_isRecreateRaytracingResourcesRequested = true; }
private:
    void UpdateConstantBuffer(Scene& scene);
    void CreateDeviceDependentResources(Scene& scene);
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateRootSignatures();
    void CreateRaytracingPipelineStateObject();
    void CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateTextureResources();

    void CreateResolutionDependentResources();
    void BuildShaderTables(Scene& scene);
    void DispatchRays(ID3D12Resource* rayGenShaderTable, UINT width = 0, UINT height = 0);
    void CalculateRayHitCount();
    void DownsampleGBuffer();

    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;

    UINT m_width;        // ToDo rename
    UINT m_height;
    UINT m_quarterResWidth;
    UINT m_quarterResHeight;

    // Raytracing shaders.
    static const wchar_t* c_rayGenShaderNames[RayGenShaderType::Count];
    static const wchar_t* c_closestHitShaderNames[RayType::Count];
    static const wchar_t* c_missShaderNames[RayType::Count];
    static const wchar_t* c_hitGroupNames[RayType::Count];

    // DirectX Raytracing (DXR) attributes
    ComPtr<ID3D12StateObject>   m_dxrStateObject;

    // Shader tables
    ComPtr<ID3D12Resource> m_rayGenShaderTables[RayGenShaderType::Count];
    UINT m_rayGenShaderTableRecordSizeInBytes[RayGenShaderType::Count];
    ComPtr<ID3D12Resource> m_hitGroupShaderTable;
    UINT m_hitGroupShaderTableStrideInBytes = UINT_MAX;
    ComPtr<ID3D12Resource> m_missShaderTable;
    UINT m_missShaderTableStrideInBytes = UINT_MAX;

    // Root signatures
    ComPtr<ID3D12RootSignature> m_raytracingGlobalRootSignature;
    ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature;

    // Raytracing resources.
    ConstantBuffer<PathtracerConstantBuffer> m_CB;
    GpuResource m_GBufferResources[GBufferResource::Count];
    GpuResource m_GBufferQuarterResResources[GBufferResource::Count]; // ToDo remove unused

    D3D12_GPU_DESCRIPTOR_HANDLE m_nullVertexBufferGPUhandle;

    GpuKernels::CalculatePartialDerivatives  m_calculatePartialDerivativesKernel;
    GpuKernels::ReduceSum				m_reduceSumKernel;
    GpuKernels::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter m_downsampleGBufferBilateralFilterKernel; //ToDo rename?

    bool m_isRecreateRaytracingResourcesRequested = false;
};