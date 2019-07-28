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



// ToDo move to cpp
namespace RTAORayGenShaderType {
    enum Enum {
        AOFullRes = 0,
        AOSortedRays,
        Count
    };
}

class RTAO
{
public:
    // Ctors.
    RTAO();

    // Public methods.
    void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT maxInstanceContributionToHitGroupIndex);
    void OnUpdate();
    void OnRender(D3D12_GPU_VIRTUAL_ADDRESS accelerationStructure, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceHitPositionResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceNormalDepthResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceAlbedoResource, D3D12_GPU_DESCRIPTOR_HANDLE frameAgeResource);
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources() {}; // ToDo

    // Getters & Setters.
    void SetResolution(UINT width, UINT height);
    DXGI_FORMAT GetAOCoefficientFormat();
    float GetSpp();

    // ToDo return only a subset
    RWGpuResource (&AOResources())[AOResource::Count]{ return m_AOResources; }
    RWGpuResource* GetAOResources(){ return m_AOResources; }

    void RequestRecreateAOSamples() { m_isRecreateAOSamplesRequested = true; }
    void RequestRecreateRaytracingResources() { m_isRecreateRaytracingResourcesRequested = true; }

private:
    void CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex);
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateRootSignatures();    
    void CreateRaytracingPipelineStateObject();
    void CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateTextureResources();
    void CalculateAdaptiveSamplingCounts();
    
    void CreateSamplesRNG();
    void CreateResolutionDependentResources();
    void BuildShaderTables(UINT maxInstanceContributionToHitGroupIndex);
    void DispatchRays(ID3D12Resource* rayGenShaderTable, UINT width = 0, UINT height = 0);
    void CalculateRayHitCount();

    UINT m_raytracingWidth;
    UINT m_raytracingHeight;

    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;
    std::mt19937 m_generatorURNG;

    // ToDo fix artifacts at 4 spp. Looks like selfshadowing on some AOrays in SquidScene

    // Raytracing shaders.
    static const wchar_t* c_hitGroupName;
    static const wchar_t* c_rayGenShaderNames[RTAORayGenShaderType::Count];
    static const wchar_t* c_closestHitShaderName;
    static const wchar_t* c_missShaderName;

    // Raytracing shader resources.
    RWGpuResource   m_AOResources[AOResource::Count];
    RWGpuResource   m_AORayDirectionOriginDepth;
    RWGpuResource   m_sortedToSourceRayIndexOffset;   // Index of a ray in the source array given a sorted index.
    RWGpuResource   m_sourceToSortedRayIndexOffset;   // Index of a ray in the sorted array given a source index.
    RWGpuResource   m_sortedRayGroupDebug;            // ToDo remove
    ConstantBuffer<RTAOConstantBuffer> m_CB;
    Samplers::MultiJittered m_randomSampler;
    StructuredBuffer<AlignedUnitSquareSample2D> m_samplesGPUBuffer;
    StructuredBuffer<AlignedHemisphereSample3D> m_hemisphereSamplesGPUBuffer;

    UINT		    m_numAORayGeometryHits;


#if DEBUG_PRINT_OUT_RTAO_DISPATCH_TIME
    DX::GPUTimer dispatchRayTime;
#endif

    // DirectX Raytracing (DXR) attributes
    ComPtr<ID3D12StateObject>   m_dxrStateObject;

    // Shader tables
    ComPtr<ID3D12Resource> m_rayGenShaderTables[RTAORayGenShaderType::Count];
    UINT m_rayGenShaderTableRecordSizeInBytes[RTAORayGenShaderType::Count];
    ComPtr<ID3D12Resource> m_hitGroupShaderTable;
    UINT m_hitGroupShaderTableStrideInBytes = UINT_MAX;
    ComPtr<ID3D12Resource> m_missShaderTable;
    UINT m_missShaderTableStrideInBytes = UINT_MAX;

    // Root signatures
    ComPtr<ID3D12RootSignature> m_raytracingGlobalRootSignature;
    ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature;

    // Compute shader & resources.
    GpuKernels::ReduceSum		m_reduceSumKernel;
    GpuKernels::AdaptiveRayGenerator m_rayGen;
    GpuKernels::SortRays        m_raySorter;


    bool m_isRecreateAOSamplesRequested = false;
    bool m_isRecreateRaytracingResourcesRequested = false;

    // Parameters
    bool m_calculateRayHitCounts = false;


    static UINT             s_numInstances;
};