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
    bool m_isBuilt = false;

	void AllocateResource(ID3D12Device5* device, const wchar_t* resourceName = nullptr);
};

class BottomLevelAccelerationStructure : public AccelerationStructure
{
public:
	BottomLevelAccelerationStructure();
	~BottomLevelAccelerationStructure() {}
	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>* GetGeometryDescs() { return &m_geometryDescs; }
	
	// ToDo:
	// UpdateGeometry()

	void Initialize(ID3D12Device5* device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, DXGI_FORMAT indexFormat, UINT ibStrideInBytes, UINT vbStrideInBytes, std::vector<GeometryInstance>& geometries, const wchar_t* name = nullptr);
	void Build(ID3D12GraphicsCommandList5* commandList, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress = 0, bool bUpdate = false);
	void BuildInstanceDesc(void* destInstanceDesc);
	void UpdateGeometryDescsTransform(D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress);

    void SetInstanceContributionToHitGroupIndex(UINT index) { m_instanceContributionToHitGroupIndex = index; }
	void SetDirty(bool isDirty) { m_isDirty = isDirty; }
	bool IsDirty() { return m_isDirty; }

	const XMMATRIX& GetTransform() { return m_transform; }

private:
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> m_geometryDescs;
    DirectX::XMMATRIX m_transform;

    // Runtime state
    bool m_isDirty;		// if true, AS requires an update/build.
    UINT m_instanceContributionToHitGroupIndex;


	void BuildGeometryDescs(DXGI_FORMAT indexFormat, UINT ibStrideInBytes, UINT vbStrideInBytes, std::vector<GeometryInstance>& geometries);
	void ComputePrebuildInfo(ID3D12Device5* device);
};

class TopLevelAccelerationStructure : public AccelerationStructure
{
public:
	TopLevelAccelerationStructure() {}
    ~TopLevelAccelerationStructure() {};

	void Initialize(ID3D12Device5* device, UINT numBottomLevelASInstanceDescs, std::vector<BottomLevelAccelerationStructure>& vBottomLevelAS, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, const wchar_t* resourceName = nullptr);
	void Build(ID3D12GraphicsCommandList5* commandList, UINT numInstanceDescs, D3D12_GPU_VIRTUAL_ADDRESS InstanceDescs, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, bool bUpdate = false);
	void UpdateInstanceDescTransforms(std::vector<BottomLevelAccelerationStructure>& vBottomLevelAS);

private:

	void ComputePrebuildInfo(ID3D12Device5* device, UINT numBottomLevelASInstanceDescs);
};


struct BottomLevelAccelerationStructureInstanceDesc : public D3D12_RAYTRACING_INSTANCE_DESC
{
    void SetTransform(const DirectX::XMMATRIX& transform);
    void GetTransform(DirectX::XMMATRIX* transform);
};
static_assert(sizeof(BottomLevelAccelerationStructureInstanceDesc) == sizeof(D3D12_RAYTRACING_INSTANCE_DESC) % 16 == 0, L"This is a wrapper used in place of the desc. It has to have the same size");


class RaytracingAccelerationStructureManager
{
public:
    RaytracingAccelerationStructureManager(ID3D12Device5* device, UINT numBottomLevelInstances, UINT frameCount);
    ~RaytracingAccelerationStructureManager() {}

    UINT AddBottomLevelAS(ID3D12Device5* device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, DXGI_FORMAT indexFormat, UINT ibStrideInBytes, UINT vbStrideInBytes, std::vector<GeometryInstance>& geometries, bool performUpdateOnBuild = false, const wchar_t* name = nullptr);
    UINT AddBottomLevelASInstance(UINT bottomLevelASindex, XMMATRIX transform = XMMatrixIdentity(), UINT instanceContributionToHitGroupIndex = 0, BYTE InstanceMask = 1);
    void InitializeTopLevelAS(ID3D12Device5* device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, const wchar_t* resourceName = nullptr);
    void Build(ID3D12GraphicsCommandList5* commandList, ID3D12DescriptorHeap* descriptorHeap, UINT frameIndex, bool bForceBuild = false);
    BottomLevelAccelerationStructureInstanceDesc& GetBottomLevelASInstance(UINT bottomLevelASinstanceIndex) { return m_bottomLevelASInstanceDescs[bottomLevelASinstanceIndex]; }


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