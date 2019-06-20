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



namespace RTAORayGenShaderType {
    enum Enum {
        AOFullRes = 0,
        AOSortedRays,
        AOQuarterRes,
        Count
    };
}

class RTAO
{
public:
    RTAO();

    void Setup(std::shared_ptr<DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap);
    void OnRender(D3D12_GPU_VIRTUAL_ADDRESS& accelerationStructure);
    void OnUpdate();
    void SetResolution(UINT width, UINT height);
    ID3D12Resource* GetRTAOOutputResource() { return m_ssaoResources.Get(); }

private:
    void CreateDeviceDependentResources();
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateRootSignatures();    
    void CreateRaytracingPipelineStateObject();
    void CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateResources();
    void CalculateAdaptiveSamplingCounts();
    
    void CreateSamplesRNG();
    void CreateDeviceDependentResources();
    void CreateResolutionDependentResources();
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources();
    void RenderRNGVisualizations();
    void BuildShaderTables();
    void DispatchRays(ID3D12Resource* rayGenShaderTable);
    void CalculateRayHitCount();

    UINT m_raytracingWidth;
    UINT m_raytracingHeight;

    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;
    std::mt19937 m_generatorURNG;

    // Raytracing shaders.
    static const wchar_t* c_hitGroupName;
    static const wchar_t* c_rayGenShaderNames[RTAORayGenShaderType::Count];
    static const wchar_t* c_closestHitShaderName;
    static const wchar_t* c_missShaderName;


    // DirectX Raytracing (DXR) attributes
    ComPtr<ID3D12StateObject> m_dxrStateObject;

    // Shader tables
    ComPtr<ID3D12Resource> m_rayGenShaderTables[RTAORayGenShaderType::Count];
    UINT m_rayGenShaderTableRecordSizeInBytes[RTAORayGenShaderType::Count];
    ComPtr<ID3D12Resource> m_hitGroupShaderTable;
    UINT m_hitGroupShaderTableStrideInBytes;
    ComPtr<ID3D12Resource> m_missShaderTable;
    UINT m_missShaderTableStrideInBytes;

    // Root signatures
    ComPtr<ID3D12RootSignature> m_raytracingGlobalRootSignature;
    ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature;

    // Parameters
    bool m_calculateRayHitCounts = false;


};