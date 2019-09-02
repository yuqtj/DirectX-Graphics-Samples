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
#include "Composition/Composition.h"
#include "Scene.h"

// ToDo move to cpp
namespace RTAORayGenShaderType {
    enum Enum {
        AOFullRes = 0,
        AOSortedRays,
        Count
    };
}

namespace RTAO_Args
{
    extern BoolVar QuarterResAO;
}


class RTAO
{
public:
    enum class ResourceType {
        AOCoefficient = 0,
        RayHitDistance
    };

    // Ctors.
    RTAO();
    ~RTAO();

    // Public methods.
    void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap, Scene& scene);
    void Run(D3D12_GPU_VIRTUAL_ADDRESS accelerationStructure, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceHitPositionResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceNormalDepthResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceAlbedoResource);
    void SetResolution(UINT width, UINT height); 
    void Release();

    // Getters & Setters.
    GpuResource(&AOResources())[AOResource::Count]{ return m_AOResources; }
    static DXGI_FORMAT ResourceFormat(ResourceType resourceType);
    float MaxRayHitTime();
    void SetMaxRayHitTime(float maxRayHitTime); 
    float GetSpp();
    void GetRayGenParameters(bool* isCheckerboardSamplingEnabled, bool* checkerboardLoadEvenPixels);

    UINT RaytracingWidth() { return m_raytracingWidth; }
    UINT RaytracingHeight() { return m_raytracingHeight; }

private:
    void UpdateConstantBuffer(UINT frameIndex);
    void CreateDeviceDependentResources(Scene& scene);
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateRootSignatures();
    void CreateRaytracingPipelineStateObject();
    void CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateTextureResources();

    void CreateSamplesRNG();
    void CreateResolutionDependentResources();
    void BuildShaderTables(Scene& scene);
    void DispatchRays(ID3D12Resource* rayGenShaderTable, UINT width = 0, UINT height = 0);
    void CalculateRayHitCount();

    UINT m_raytracingWidth = 0;
    UINT m_raytracingHeight = 0;

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
    GpuResource   m_AOResources[AOResource::Count];
    GpuResource   m_AORayDirectionOriginDepth;
    GpuResource   m_sortedToSourceRayIndexOffset;   // Index of a ray in the source array given a sorted index.
    
    // ToDo remove
    GpuResource   m_sourceToSortedRayIndexOffset;   // Index of a ray in the sorted array given a source index.
    ConstantBuffer<RTAOConstantBuffer> m_CB;
    Samplers::MultiJittered m_randomSampler;
    StructuredBuffer<AlignedUnitSquareSample2D> m_samplesGPUBuffer;
    StructuredBuffer<AlignedHemisphereSample3D> m_hemisphereSamplesGPUBuffer;

    UINT		    m_numAORayGeometryHits;
    bool            m_checkerboardGenerateRaysForEvenPixels = false;

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


    // Parameters
    bool m_calculateRayHitCounts = false;

    friend class Composition;
};
