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


class Pathtracer
{
public:
    // Ctors.
    Pathtracer();

    // Public methods.
    void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT maxInstanceContributionToHitGroupIndex);
    void OnUpdate();
    void OnRender(D3D12_GPU_VIRTUAL_ADDRESS accelerationStructure, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceHitPositionResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceNormalDepthResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceAlbedoResource, D3D12_GPU_DESCRIPTOR_HANDLE frameAgeResource);
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources() {}; // ToDo
    void SetCamera(const GameCore::Camera& camera);
    void SetLight(const XMVECTOR& position, const XMFLOAT3& color);

    // Getters & Setters.
    GpuResource(&GBufferResources(bool retrieveLowResResources))[GBufferResource::Count];
    GpuResource* GetGBufferResources(bool retrieveLowResResources);

    void RequestRecreateRaytracingResources() { m_isRecreateRaytracingResourcesRequested = true; }
private:
    void CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex);
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateRootSignatures();    
    void CreateRaytracingPipelineStateObject();
    void CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateTextureResources();
    
    void CreateResolutionDependentResources();
    void BuildShaderTables(UINT maxInstanceContributionToHitGroupIndex);
    void DispatchRays(ID3D12Resource* rayGenShaderTable, UINT width = 0, UINT height = 0);
    void CalculateRayHitCount();
    void DownsampleGBuffer();

    UINT m_GBufferWidth;        // ToDo rename
    UINT m_GBufferHeight;
    UINT m_raytracingWidth;
    UINT m_raytracingHeight;

    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;

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
    ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature[LocalRootSignature::Type::Count];

    // Raytracing resources.
    GpuResource m_GBufferResources[GBufferResource::Count];
    GpuResource m_GBufferLowResResources[GBufferResource::Count]; // ToDo remove unused
    GpuResource m_prevFrameGBufferNormalDepth;

    ConstantBuffer<PathtracerConstantBuffer> m_sceneCB;

    bool m_isRecreateRaytracingResourcesRequested = false;

    static UINT             s_numInstances;
};