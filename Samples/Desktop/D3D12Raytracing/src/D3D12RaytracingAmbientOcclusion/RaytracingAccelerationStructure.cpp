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

#include "stdafx.h"
#include "RaytracingAccelerationStructure.h"
#include "D3D12RaytracingAmbientOcclusion.h"
#include "EngineProfiling.h"

using namespace std;

AccelerationStructure::AccelerationStructure() :
	m_buildFlags(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE),
    m_isBuilt(false),
	m_prebuildInfo{}
{
}

void AccelerationStructure::ReleaseD3DResources()
{
	m_accelerationStructure.Reset();
}

void AccelerationStructure::AllocateResource(ID3D12Device5* device, const wchar_t* resourceName)
{
	// Allocate resource for acceleration structures.
	// Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
	// Default heap is OK since the application doesn�t need CPU read/write access to them. 
	// The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
	// and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
	//  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
	//  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.
	D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
	AllocateUAVBuffer(device, m_prebuildInfo.ResultDataMaxSizeInBytes, &m_accelerationStructure, initialResourceState, resourceName);
}


BottomLevelAccelerationStructure::BottomLevelAccelerationStructure() :
	m_isDirty(true),
    m_instanceContributionToHitGroupIndex(0)
{
}

void BottomLevelAccelerationStructure::UpdateGeometryDescsTransform(D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress)
{
	for (UINT i = 0; i < m_geometryDescs.size(); i++)
	{
		auto& geometryDesc = m_geometryDescs[i];
		geometryDesc.Triangles.Transform3x4 = baseGeometryTransformGPUAddress + i * sizeof(AlignedGeometryTransform3x4);
	}
}

// Build geometry descs for bottom-level AS.
void BottomLevelAccelerationStructure::BuildGeometryDescs(DXGI_FORMAT indexFormat, UINT ibStrideInBytes, UINT vbStrideInBytes, vector<GeometryInstance>& geometries)
{
	// ToDo pass geometry flag from the sample cpp
	// Mark the geometry as opaque. 
	// PERFORMANCE TIP: mark geometry as opaque whenever applicable as it can enable important ray processing optimizations.
	// Note: When rays encounter opaque geometry an any hit shader will not be executed whether it is present or not.
	D3D12_RAYTRACING_GEOMETRY_FLAGS geometryFlags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

	D3D12_RAYTRACING_GEOMETRY_DESC geometryDescTemplate = {};
	geometryDescTemplate.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	geometryDescTemplate.Triangles.IndexFormat = indexFormat;
	geometryDescTemplate.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
	geometryDescTemplate.Flags = geometryFlags;

	m_geometryDescs.reserve(geometries.size());

	for (auto& geometry: geometries)
	{
		auto& geometryDesc = geometryDescTemplate;
		geometryDesc.Triangles.IndexBuffer = geometry.ib.indexBuffer;
		geometryDesc.Triangles.IndexCount = geometry.ib.count;
		geometryDesc.Triangles.VertexBuffer = geometry.vb.vertexBuffer;
		geometryDesc.Triangles.VertexCount = geometry.vb.count;
		geometryDesc.Triangles.Transform3x4 = geometry.transform;

		m_geometryDescs.push_back(geometryDesc);
	}
}

void BottomLevelAccelerationStructure::ComputePrebuildInfo(ID3D12Device5* device)
{
	// Get the size requirements for the scratch and AS buffers.
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &bottomLevelInputs = bottomLevelBuildDesc.Inputs;
    bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    bottomLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    bottomLevelInputs.Flags = m_buildFlags;
    bottomLevelInputs.NumDescs = static_cast<UINT>(m_geometryDescs.size());
    bottomLevelInputs.pGeometryDescs = m_geometryDescs.data();
	
	device->GetRaytracingAccelerationStructurePrebuildInfo(&bottomLevelInputs, &m_prebuildInfo);
	ThrowIfFalse(m_prebuildInfo.ResultDataMaxSizeInBytes > 0);
}

void BottomLevelAccelerationStructure::Initialize(
	ID3D12Device5* device, 
	D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, 
	DXGI_FORMAT indexFormat, 
	UINT ibStrideInBytes, 
	UINT vbStrideInBytes, 
	vector<GeometryInstance>& geometries, 
    const wchar_t* resourceName)
{
	m_buildFlags = buildFlags;
	BuildGeometryDescs(indexFormat, ibStrideInBytes, vbStrideInBytes, geometries);
	ComputePrebuildInfo(device);
	AllocateResource(device, resourceName);
	m_isDirty = true;
}

void BottomLevelAccelerationStructure::Build(
    ID3D12GraphicsCommandList5* commandList, 
    ID3D12Resource* scratch, 
    ID3D12DescriptorHeap* descriptorHeap, 
    D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress, 
    bool bUpdate)
{
	ThrowIfFalse(m_prebuildInfo.ScratchDataSizeInBytes <= scratch->GetDesc().Width, L"Insufficient scratch buffer size provided!");
	
    if (baseGeometryTransformGPUAddress > 0)
    {
        UpdateGeometryDescsTransform(baseGeometryTransformGPUAddress);
    }

    // ToDo cleanup
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &bottomLevelInputs = bottomLevelBuildDesc.Inputs;
	{
        // ToDo remove repeating BOTTOM_LEVEL flag specification
        bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        bottomLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        bottomLevelInputs.Flags = m_buildFlags;
		if (m_isBuilt && bUpdate)
		{
            bottomLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
		}
        bottomLevelInputs.NumDescs = static_cast<UINT>(m_geometryDescs.size());
        bottomLevelInputs.pGeometryDescs = m_geometryDescs.data();

		bottomLevelBuildDesc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
		bottomLevelBuildDesc.DestAccelerationStructureData = m_accelerationStructure->GetGPUVirtualAddress();
	}

	commandList->SetDescriptorHeaps(1, &descriptorHeap);
    commandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);

	m_isDirty = false;
    m_isBuilt = true;
}

void TopLevelAccelerationStructure::ComputePrebuildInfo(ID3D12Device5* device, UINT numBottomLevelASInstanceDescs)
{
	// Get the size requirements for the scratch and AS buffers.
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
    topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    topLevelInputs.Flags = m_buildFlags;
    topLevelInputs.NumDescs = numBottomLevelASInstanceDescs;

	device->GetRaytracingAccelerationStructurePrebuildInfo(&topLevelInputs, &m_prebuildInfo);
	ThrowIfFalse(m_prebuildInfo.ResultDataMaxSizeInBytes > 0);
}

void TopLevelAccelerationStructure::Initialize(ID3D12Device5* device, UINT numBottomLevelASInstanceDescs, vector<BottomLevelAccelerationStructure>& vBottomLevelAS, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, const wchar_t* resourceName)
{
	m_buildFlags = buildFlags;
	ComputePrebuildInfo(device, numBottomLevelASInstanceDescs);
	AllocateResource(device, resourceName);
}

void TopLevelAccelerationStructure::Build(ID3D12GraphicsCommandList5* commandList, UINT numBottomLevelASInstanceDescs, D3D12_GPU_VIRTUAL_ADDRESS bottomLevelASnstanceDescs, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, bool bUpdate)
{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
    {
        topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        topLevelInputs.Flags = m_buildFlags;
        if (m_isBuilt && bUpdate)
        {
            topLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        }
        topLevelInputs.NumDescs = numBottomLevelASInstanceDescs;

        topLevelBuildDesc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        topLevelBuildDesc.DestAccelerationStructureData = m_accelerationStructure->GetGPUVirtualAddress();
    }
    topLevelInputs.InstanceDescs = bottomLevelASnstanceDescs;

    commandList->SetDescriptorHeaps(1, &descriptorHeap);
    commandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);
    m_isBuilt = true;
}

RaytracingAccelerationStructureManager::RaytracingAccelerationStructureManager(ID3D12Device5* device, UINT numBottomLevelInstances, UINT frameCount)
{
    m_bottomLevelASInstanceDescs.Create(device, numBottomLevelInstances, frameCount, L"Bottom-Level Acceleration Structure Instance descs.");
}

// Adds a bottom-level Acceleration Structure.
// Requires a corresponding 1 or more AddBottomLevelASInstance() calls to be added to top-level AS.
// Returns an index to the created bottom-level AS object.
UINT RaytracingAccelerationStructureManager::AddBottomLevelAS(
    ID3D12Device5* device,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags,
    DXGI_FORMAT indexFormat,
    UINT ibStrideInBytes,
    UINT vbStrideInBytes,
    vector<GeometryInstance>& geometries,
    bool performUpdateOnBuild,
    const wchar_t* name)
{
    BottomLevelAccelerationStructure bottomLevelAS;

    if (performUpdateOnBuild)
    {
        buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }

    bottomLevelAS.Initialize(device, buildFlags, indexFormat, ibStrideInBytes, vbStrideInBytes, geometries, name);

    m_ASmemoryFootprint += bottomLevelAS.RequiredResultDataSizeInBytes();
    m_scratchResourceSize = max(bottomLevelAS.RequiredScratchSize(), m_scratchResourceSize);

    UINT bottomLevelASindex = static_cast<UINT>(m_vBottomLevelAS.size());
    m_vBottomLevelAS.push_back(bottomLevelAS);
    m_vBottomLevelASPerformUpdateOnBuild.push_back(performUpdateOnBuild);

    return bottomLevelASindex;
}

// Adds an instance of a bottom-level Acceleration Structure.
// Requires a call InitializeTopLevelAS() call to be added to top-level AS.
UINT RaytracingAccelerationStructureManager::AddBottomLevelASInstance(
    UINT bottomLevelASindex,
    XMMATRIX transform,
    UINT instanceContributionToHitGroupIndex,
    BYTE instanceMask)
{
    ThrowIfFalse(numBottomLevelASInstances < m_bottomLevelASInstanceDescs.NumElements(), L"Not enough instance desc buffer size.");

    UINT instanceIndex = numBottomLevelASInstances++;
    auto& bottomLevelAS = m_vBottomLevelAS[bottomLevelASindex];

    auto& instanceDesc = m_bottomLevelASInstanceDescs[instanceIndex];
    instanceDesc.InstanceMask = instanceMask;
    instanceDesc.InstanceContributionToHitGroupIndex = instanceContributionToHitGroupIndex;
    instanceDesc.AccelerationStructure = bottomLevelAS.GetResource()->GetGPUVirtualAddress();
    XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instanceDesc.Transform), transform);    

    return instanceIndex;
};

// Initializes the top-level Acceleration Structure.
void RaytracingAccelerationStructureManager::InitializeTopLevelAS(
    ID3D12Device5* device,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, 
    const wchar_t* resourceName)
{
    m_topLevelAS.Initialize(device, GetNumberOfBottomLevelASInstances(), m_vBottomLevelAS, buildFlags, resourceName);

    m_ASmemoryFootprint += m_topLevelAS.RequiredResultDataSizeInBytes();
    m_scratchResourceSize = max(m_topLevelAS.RequiredScratchSize(), m_scratchResourceSize);

    AllocateUAVBuffer(device, m_scratchResourceSize, &m_accelerationStructureScratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"Acceleration structure scratch resource");
}

// Builds all bottom-level and top-level Acceleration Structures.
void RaytracingAccelerationStructureManager::Build(
    ID3D12GraphicsCommandList5* commandList, 
    ID3D12DescriptorHeap* descriptorHeap,
    UINT frameIndex,
    bool bForceBuild)
{
    ScopedTimer _prof(L"Acceleration Structure build", commandList);

    m_bottomLevelASInstanceDescs.CopyStagingToGpu(frameIndex);

    // Build all bottom-level AS.
    {
        ScopedTimer _prof(L"Bottom Level AS", commandList);
        for (UINT i = 0; i < m_vBottomLevelAS.size(); i++)
        {
            auto& bottomLevelAS = m_vBottomLevelAS[i];
            if (bForceBuild || bottomLevelAS.IsDirty())
            {
                ScopedTimer _prof(bottomLevelAS.GetName(), commandList);

                bool performUpdate = m_vBottomLevelASPerformUpdateOnBuild[i];
                D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGpuAddress = 0;  // ToDo
                bottomLevelAS.Build(commandList, m_accelerationStructureScratch.Get(), descriptorHeap, baseGeometryTransformGpuAddress, performUpdate);

                // Since a single scratch resource is reused, put a barrier in-between each call.
                commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(bottomLevelAS.GetResource()));
            }
        }
    }

    // Build the top-level AS.
    {
        ScopedTimer _prof(L"Top Level AS", commandList);

        bool performUpdate = false; // Always rebuild top-level Acceleration Structure.
        D3D12_GPU_VIRTUAL_ADDRESS instanceDescs = m_bottomLevelASInstanceDescs.GpuVirtualAddress(frameIndex);
        m_topLevelAS.Build(commandList, GetNumberOfBottomLevelASInstances(), instanceDescs, m_accelerationStructureScratch.Get(), descriptorHeap, performUpdate);

        // ToDo move this barrier righ before when the resource is being accessed for reading.
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_topLevelAS.GetResource()));
    }
}

void BottomLevelAccelerationStructureInstanceDesc::SetTransform(const XMMATRIX& transform)
{
    XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(Transform), transform);
}

void BottomLevelAccelerationStructureInstanceDesc::GetTransform(XMMATRIX* transform)
{
    *transform = XMLoadFloat3x4(reinterpret_cast<XMFLOAT3X4*>(Transform));
}