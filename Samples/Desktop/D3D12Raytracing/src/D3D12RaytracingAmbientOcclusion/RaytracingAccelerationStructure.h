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

#include "RayTracingHlslCompat.h"
#include "RaytracingSceneDefines.h"

#define SizeOfInUint32(obj) ((sizeof(obj) - 1) / sizeof(UINT32) + 1)

struct AccelerationStructureBuffers
{
    ComPtr<ID3D12Resource> scratch;
    ComPtr<ID3D12Resource> accelerationStructure;
    ComPtr<ID3D12Resource> instanceDesc;    // Used only for top-level AS
    UINT64                 ResultDataMaxSizeInBytes;
};

// ToDO remove?
struct alignas(16) AlignedGeometryTransform3x4
{
	float transform3x4[12];
};

// AccelerationStructure
// A base class for bottom-level and top-level AS.
class AccelerationStructure
{
public:
	AccelerationStructure();
	virtual ~AccelerationStructure() {}
	void ReleaseD3DResources();
	UINT64 RequiredScratchSize() { return std::max(m_prebuildInfo.ScratchDataSizeInBytes, m_prebuildInfo.UpdateScratchDataSizeInBytes); }
	UINT64 RequiredResultDataSizeInBytes() { return m_prebuildInfo.ResultDataMaxSizeInBytes; }
	ID3D12Resource* GetResource() { return m_accelerationStructure.Get(); }
	const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& PrebuildInfo() { return m_prebuildInfo; }
    const std::wstring& GetName() { return m_name; }

protected:
    ComPtr<ID3D12Resource> m_accelerationStructure;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS m_buildFlags;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO m_prebuildInfo;
    std::wstring m_name;

    bool m_isBuilt = false; // whether the AS has been built at least once.
    bool m_isDirty;		    // whether the AS requires to be rebuilt.

	void AllocateResource(ID3D12Device5* device, const wchar_t* resourceName = nullptr);
};

class BottomLevelAccelerationStructureGeometry
{
public:
    std::wstring                    m_name;
    std::vector<GeometryInstance>	m_geometryInstances;
    std::vector<D3DGeometry>        m_geometries;
    std::vector<D3DTexture>         m_textures;
    UINT                            m_numTriangles = 0;
    DXGI_FORMAT                     m_indexFormat;
    UINT                            m_ibStrideInBytes;
    DXGI_FORMAT                     m_vertexFormat;
    UINT                            m_vbStrideInBytes;
    UINT                            m_instanceContributionToHitGroupIndex = 0;

    BottomLevelAccelerationStructureGeometry(const wchar_t* name) : m_name(name) {}
    BottomLevelAccelerationStructureGeometry(const std::wstring& name) : m_name(name) {}
};

class BottomLevelAccelerationStructure : public AccelerationStructure
{
public:
	BottomLevelAccelerationStructure();
	~BottomLevelAccelerationStructure() {}
	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>* GetGeometryDescs() { return &m_geometryDescs; }
	
	// ToDo:
	// UpdateGeometry()

    void Initialize(ID3D12Device5* device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, BottomLevelAccelerationStructureGeometry& bottomLevelASGeometry);
    void Build(ID3D12GraphicsCommandList5* commandList, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress = 0, bool bUpdate = false);
	void UpdateGeometryDescsTransform(D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress);

    UINT GetInstanceContributionToHitGroupIndex() { return m_instanceContributionToHitGroupIndex; }
    void SetInstanceContributionToHitGroupIndex(UINT index) { m_instanceContributionToHitGroupIndex = index; }
	void SetDirty(bool isDirty) { m_isDirty = isDirty; }
	bool IsDirty() { return m_isDirty; }

	const XMMATRIX& GetTransform() { return m_transform; }

private:
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> m_geometryDescs;
    DirectX::XMMATRIX m_transform;

    // Runtime state
    UINT m_instanceContributionToHitGroupIndex;


	void BuildGeometryDescs(BottomLevelAccelerationStructureGeometry& bottomLevelASGeometry);
	void ComputePrebuildInfo(ID3D12Device5* device);
};

class TopLevelAccelerationStructure : public AccelerationStructure
{
public:
	TopLevelAccelerationStructure() {}
    ~TopLevelAccelerationStructure() {};

	void Initialize(ID3D12Device5* device, UINT numBottomLevelASInstanceDescs, std::vector<BottomLevelAccelerationStructure>& vBottomLevelAS, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, const wchar_t* resourceName = nullptr);
	void Build(ID3D12GraphicsCommandList5* commandList, UINT numInstanceDescs, D3D12_GPU_VIRTUAL_ADDRESS InstanceDescs, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, bool bUpdate = false);

private:

	void ComputePrebuildInfo(ID3D12Device5* device, UINT numBottomLevelASInstanceDescs);
};


struct BottomLevelAccelerationStructureInstanceDesc : public D3D12_RAYTRACING_INSTANCE_DESC
{
    void SetTransform(const DirectX::XMMATRIX& transform);
    void GetTransform(DirectX::XMMATRIX* transform);
};
static_assert(sizeof(BottomLevelAccelerationStructureInstanceDesc) == sizeof(D3D12_RAYTRACING_INSTANCE_DESC), L"This is a wrapper used in place of the desc. It has to have the same size");


class RaytracingAccelerationStructureManager
{
public:
    RaytracingAccelerationStructureManager(ID3D12Device5* device, UINT numBottomLevelInstances, UINT frameCount);
    ~RaytracingAccelerationStructureManager() {}

    UINT AddBottomLevelAS(ID3D12Device5* device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, BottomLevelAccelerationStructureGeometry& bottomLevelASGeometry, bool performUpdateOnBuild = false);
    UINT AddBottomLevelASInstance(UINT bottomLevelASindex, UINT instanceContributionToHitGroupIndex, XMMATRIX transform = XMMatrixIdentity(), BYTE InstanceMask = 1);
    void InitializeTopLevelAS(ID3D12Device5* device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, const wchar_t* resourceName = nullptr);
    void Build(ID3D12GraphicsCommandList5* commandList, ID3D12DescriptorHeap* descriptorHeap, UINT frameIndex, bool bForceBuild = false);
    BottomLevelAccelerationStructureInstanceDesc& GetBottomLevelASInstance(UINT bottomLevelASinstanceIndex) { return m_bottomLevelASInstanceDescs[bottomLevelASinstanceIndex]; }
    const StructuredBuffer<BottomLevelAccelerationStructureInstanceDesc>& GetBottomLevelASInstancesBuffer() { return m_bottomLevelASInstanceDescs; }

    ID3D12Resource* GetTopLevelASResource() { return m_topLevelAS.GetResource(); }
    UINT64 GetASMemoryFootprint() { return m_ASmemoryFootprint; }
    UINT GetNumberOfBottomLevelASInstances() { return static_cast<UINT>(m_bottomLevelASInstanceDescs.NumElements()); }

private:
    TopLevelAccelerationStructure	m_topLevelAS;
    std::vector<BottomLevelAccelerationStructure> m_vBottomLevelAS;
    StructuredBuffer<BottomLevelAccelerationStructureInstanceDesc> m_bottomLevelASInstanceDescs;
    UINT numBottomLevelASInstances = 0;

    std::vector<bool> m_vBottomLevelASPerformUpdateOnBuild;

    ComPtr<ID3D12Resource>	m_accelerationStructureScratch;
    UINT64 m_scratchResourceSize = 0;

    UINT64 m_ASmemoryFootprint = 0;
};