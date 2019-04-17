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
protected:
	ComPtr<ID3D12Resource> m_accelerationStructure;
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS m_buildFlags;
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO m_prebuildInfo;
	
public:
	AccelerationStructure();
	virtual ~AccelerationStructure() {}
	void ReleaseD3DResources();
	UINT64 RequiredScratchSize() { return std::max(m_prebuildInfo.ScratchDataSizeInBytes, m_prebuildInfo.UpdateScratchDataSizeInBytes); }
	UINT64 RequiredResultDataSizeInBytes() { return m_prebuildInfo.ResultDataMaxSizeInBytes; }
	ID3D12Resource* GetResource() { return m_accelerationStructure.Get(); }
	const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO& PrebuildInfo() { return m_prebuildInfo; }

protected:
	void AllocateResource(ID3D12Device5* device, const wchar_t* resourceName = nullptr);
};

class BottomLevelAccelerationStructure : public AccelerationStructure
{
	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> m_geometryDescs;
	DirectX::XMMATRIX m_transform;

	// Runtime state
	bool m_isDirty;		// if true, AS requires an update/build.
    UINT m_instanceContributionToHitGroupIndex;

public:
	BottomLevelAccelerationStructure();
	~BottomLevelAccelerationStructure() {}
	std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>* GetGeometryDescs() { return &m_geometryDescs; }
	
	// ToDo:
	// UpdateGeometry()

	void Initialize(ID3D12Device5* device, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, DXGI_FORMAT indexFormat, UINT ibStrideInBytes, UINT vbStrideInBytes, std::vector<GeometryInstance>& geometries);
	void Build(ID3D12GraphicsCommandList5* commandList, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress, bool bUpdate = false);
	void BuildInstanceDesc(void* destInstanceDesc, UINT* descriptorHeapIndex);
	void UpdateGeometryDescsTransform(D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress);
	void SetTransform(const DirectX::XMMATRIX& transform)
	{
		m_transform = transform;
	}
    void ApplyTransform(const DirectX::XMMATRIX& transform)
    {
        m_transform = transform * m_transform;
    }

    void SetInstanceContributionToHitGroupIndex(UINT index) { m_instanceContributionToHitGroupIndex = index; }
	void SetDirty(bool isDirty) { m_isDirty = isDirty; }
	bool IsDirty() { return m_isDirty; }

	const XMMATRIX& GetTransform() { return m_transform; }

private:
	void BuildGeometryDescs(DXGI_FORMAT indexFormat, UINT ibStrideInBytes, UINT vbStrideInBytes, std::vector<GeometryInstance>& geometries);
	void ComputePrebuildInfo(ID3D12Device5* device);
};

class TopLevelAccelerationStructure : public AccelerationStructure
{
	StructuredBuffer<D3D12_RAYTRACING_INSTANCE_DESC> m_dxrInstanceDescs;

public:
	TopLevelAccelerationStructure() {}
	~TopLevelAccelerationStructure();

	UINT NumberOfBLAS();

	void Initialize(ID3D12Device5* device, std::vector<BottomLevelAccelerationStructure>& vBottomLevelAS, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, std::vector<UINT>* bottomLevelASinstanceDescsDescritorHeapIndices);
	void Build(ID3D12GraphicsCommandList5* commandList, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, bool bUpdate = false);
	void UpdateInstanceDescTransforms(std::vector<BottomLevelAccelerationStructure>& vBottomLevelAS);

private:
	void ComputePrebuildInfo(ID3D12Device5* device);
	void BuildInstanceDescs(ID3D12Device5* device, std::vector<BottomLevelAccelerationStructure>& vBottomLevelAS, std::vector<UINT>* bottomLevelASinstanceDescsDescritorHeapIndices);
};


class RaytracingAccelerationStructureManager
{

};