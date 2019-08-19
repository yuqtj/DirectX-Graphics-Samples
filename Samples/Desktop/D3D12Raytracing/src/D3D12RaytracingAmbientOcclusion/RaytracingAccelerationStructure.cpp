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
    m_isDirty(true),
    m_isBuilt(false),
	m_prebuildInfo{}
{
}

void AccelerationStructure::ReleaseD3DResources()
{
    // ToDo
	g_accelerationStructure.Reset();
}

ID3D12Resource* AccelerationStructure::GetResource()
{
    if (m_compact)
    {
        return m_compactedAccelerationStructure.Get();
    }
    else
    {
        return g_accelerationStructure.Get();
    }
}

void AccelerationStructure::AllocateResource(ID3D12Device5* device)
{
	// Allocate resource for acceleration structures.
	// Acceleration structures can only be placed in resources that are created in the default heap (or custom heap equivalent). 
	// Default heap is OK since the application doesn’t need CPU read/write access to them. 
	// The resources that will contain acceleration structures must be created in the state D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, 
	// and must have resource flag D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS. The ALLOW_UNORDERED_ACCESS requirement simply acknowledges both: 
	//  - the system will be doing this type of access in its implementation of acceleration structure builds behind the scenes.
	//  - from the app point of view, synchronization of writes/reads to acceleration structures is accomplished using UAV barriers.
    // Buffer resources must have 64KB alignment which satisfies the AS resource requirement to have alignment of 256 (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT).
	D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
	AllocateUAVBuffer(device, m_prebuildInfo.ResultDataMaxSizeInBytes, &g_accelerationStructure, initialResourceState, m_name.c_str());

    // ToDo create the compacted AS once we know the compacted size. This will require to update the TLAS's instance descs with compacted resource GPU virtual address.
    // Create the resource for compacted AS.
    wstring resourceName = m_name + wstring(L" (Compacted)");
    AllocateUAVBuffer(device, m_prebuildInfo.ResultDataMaxSizeInBytes, &m_compactedAccelerationStructure, initialResourceState, resourceName.c_str());
}


BottomLevelAccelerationStructure::BottomLevelAccelerationStructure() :
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
void BottomLevelAccelerationStructure::BuildGeometryDescs(BottomLevelAccelerationStructureGeometry& bottomLevelASGeometry)
{
	D3D12_RAYTRACING_GEOMETRY_DESC geometryDescTemplate = {};
	geometryDescTemplate.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
	geometryDescTemplate.Triangles.IndexFormat = bottomLevelASGeometry.m_indexFormat;
	geometryDescTemplate.Triangles.VertexFormat = bottomLevelASGeometry.m_vertexFormat;

	m_geometryDescs.reserve(bottomLevelASGeometry.m_geometryInstances.size());

    // ToDo check input VBs are passed as D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE - as per spec it's required
	for (auto& geometry: bottomLevelASGeometry.m_geometryInstances)
	{
		auto& geometryDesc = geometryDescTemplate;
        geometryDescTemplate.Flags = geometry.geometryFlags;
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
    BottomLevelAccelerationStructureGeometry& bottomLevelASGeometry, 
    bool allowUpdate,
    bool bUpdateOnBuild, 
    bool performCompaction)
{
    m_allowUpdate = allowUpdate;
    m_updateOnBuild = bUpdateOnBuild;
    m_compact = performCompaction;

    m_buildFlags = buildFlags;
    m_name = bottomLevelASGeometry.GetName();
    
    if (allowUpdate)
    {
        m_buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }

    if (m_compact)
    {
        m_buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    }

	BuildGeometryDescs(bottomLevelASGeometry);
	ComputePrebuildInfo(device);
	AllocateResource(device);
    if (m_compact)
    {
        AllocateUAVBuffer(device, sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC), &m_compactionQueryDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"AS Compaction Query Desc");
        AllocateReadBackBuffer(device, sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC), &m_compactionQueryReadBack, D3D12_RESOURCE_STATE_COPY_DEST, L"AS Compaction Query Readback");     
    }

	m_isDirty = true;
    m_isBuilt = false;
}

// The caller must add a UAV barrier before using the resource.
void BottomLevelAccelerationStructure::Build(
    ID3D12GraphicsCommandList4* commandList, 
    ID3D12Resource* scratch, 
    ID3D12DescriptorHeap* descriptorHeap, 
    D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGPUAddress)
{
	ThrowIfFalse(m_prebuildInfo.ScratchDataSizeInBytes <= scratch->GetDesc().Width, L"Insufficient scratch buffer size provided!");
	
    if (baseGeometryTransformGPUAddress > 0)
    {
        UpdateGeometryDescsTransform(baseGeometryTransformGPUAddress);
    }

    currentID = (currentID + 1) % 3;    // ToDo remove or fix up naming, constants
    m_cacheGeometryDescs[currentID].clear();
    m_cacheGeometryDescs[currentID].resize(m_geometryDescs.size());
    copy(m_geometryDescs.begin(), m_geometryDescs.end(), m_cacheGeometryDescs[currentID].begin());

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &bottomLevelInputs = bottomLevelBuildDesc.Inputs;
	{
        // ToDo remove repeating BOTTOM_LEVEL flag specification
        bottomLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        bottomLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        bottomLevelInputs.Flags = m_buildFlags;
		if (m_isBuilt && m_allowUpdate && m_updateOnBuild)
		{
            bottomLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            bottomLevelBuildDesc.SourceAccelerationStructureData = g_accelerationStructure->GetGPUVirtualAddress();
		}
        bottomLevelInputs.NumDescs = static_cast<UINT>(m_cacheGeometryDescs[currentID].size());
        bottomLevelInputs.pGeometryDescs = m_cacheGeometryDescs[currentID].data();

		bottomLevelBuildDesc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
		bottomLevelBuildDesc.DestAccelerationStructureData = g_accelerationStructure->GetGPUVirtualAddress();
	}

	commandList->SetDescriptorHeaps(1, &descriptorHeap);

    if (m_compact)
    {
        // Retrieve the compacted size as part of the build operation.
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postBuildDesc;
        postBuildDesc.DestBuffer = m_compactionQueryDesc->GetGPUVirtualAddress();
        postBuildDesc.InfoType = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;

        commandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 1, &postBuildDesc);

        // Copy the compaction desc result to the readback buffer.      
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::UAV(g_accelerationStructure.Get()),
            CD3DX12_RESOURCE_BARRIER::Transition(m_compactionQueryDesc.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
        commandList->CopyResource(m_compactionQueryReadBack.Get(), m_compactionQueryDesc.Get());
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_compactionQueryDesc.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    }
    else
    {
        commandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);
    }

	m_isDirty = false;
    m_isBuilt = true;
}

// ToDo remove
void BottomLevelAccelerationStructure::ReadbackCompactedSize()
{
    if (m_compact)
    {
        // Readback the compacted size desc.
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC compactedSizeDesc;
        UINT* mappedData = nullptr;
        CD3DX12_RANGE readRange(0, sizeof(compactedSizeDesc));
        ThrowIfFailed(m_compactionQueryReadBack->Map(0, &readRange, reinterpret_cast<void**>(&mappedData)));
        memcpy(&compactedSizeDesc, mappedData, sizeof(compactedSizeDesc));
        m_compactionQueryReadBack->Unmap(0, &CD3DX12_RANGE(0, 0));
    }
}


// Performs compaction, if enabled, of the acceleration structure.
void BottomLevelAccelerationStructure::PostBuild(
    ID3D12GraphicsCommandList4* commandList)
{
    if (m_compact)
    {
        // Make sure the readback is done being written to.
        //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_compactionQueryReadBack.Get()));

#if 0
        // ToDo this has to be read back once GPU CPU synced
     
        // Readback the compacted size desc.
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC compactedSizeDesc;
        UINT* mappedData = nullptr;
        CD3DX12_RANGE readRange(0, sizeof(compactedSizeDesc));
        ThrowIfFailed(m_compactionQueryReadBack->Map(0, &readRange, reinterpret_cast<void**>(&mappedData)));
        memcpy(&compactedSizeDesc, mappedData, sizeof(compactedSizeDesc));
        m_compactionQueryReadBack->Unmap(0, &CD3DX12_RANGE(0, 0));

        // ToDo. For now the compacted AS is created at the same size as non-compacted resource up front to avoid the need to updata TLAS instance descs.
        // Create the resource for compacted AS.
        //wstring resourceName = m_name + wstring(L" (Compacted)");
        //AllocateUAVBuffer(device, compactedSizeDesc.CompactedSizeInBytes, &m_compactedAccelerationStructure, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, resourceName.c_str());
#endif
        // Compact the AS.
        commandList->CopyRaytracingAccelerationStructure(m_compactedAccelerationStructure->GetGPUVirtualAddress(), g_accelerationStructure->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

        // ToDo remove the UAV barrier.
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(m_compactedAccelerationStructure.Get()));

        // ToDo release the source AS after the copy is done.
    }
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

void TopLevelAccelerationStructure::Initialize(
    ID3D12Device5* device, 
    UINT numBottomLevelASInstanceDescs, 
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, 
    bool allowUpdate,
    bool bUpdateOnBuild,
    const wchar_t* resourceName)
{
    m_allowUpdate = allowUpdate;
    m_updateOnBuild = bUpdateOnBuild; 
	m_buildFlags = buildFlags;

    m_name = resourceName;

    if (allowUpdate)
    {
        m_buildFlags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    }

	ComputePrebuildInfo(device, numBottomLevelASInstanceDescs);
	AllocateResource(device);

    m_isDirty = true;
    m_isBuilt = false;
}

void TopLevelAccelerationStructure::Build(ID3D12GraphicsCommandList4* commandList, UINT numBottomLevelASInstanceDescs, D3D12_GPU_VIRTUAL_ADDRESS bottomLevelASnstanceDescs, ID3D12Resource* scratch, ID3D12DescriptorHeap* descriptorHeap, bool bUpdate)
{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS &topLevelInputs = topLevelBuildDesc.Inputs;
    {
        topLevelInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        topLevelInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        topLevelInputs.Flags = m_buildFlags;
        if (m_isBuilt && m_allowUpdate && m_updateOnBuild)
        {
            topLevelInputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        }
        topLevelInputs.NumDescs = numBottomLevelASInstanceDescs;

        topLevelBuildDesc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        topLevelBuildDesc.DestAccelerationStructureData = g_accelerationStructure->GetGPUVirtualAddress();
    }
    topLevelInputs.InstanceDescs = bottomLevelASnstanceDescs;

    commandList->SetDescriptorHeaps(1, &descriptorHeap);
    commandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);
    m_isDirty = false;
    m_isBuilt = true;
}

RaytracingAccelerationStructureManager::RaytracingAccelerationStructureManager(ID3D12Device5* device, UINT numBottomLevelInstances, UINT frameCount)
{
    m_bottomLevelASInstanceDescs.Create(device, numBottomLevelInstances, frameCount, L"Bottom-Level Acceleration Structure Instance descs.");
}

// Adds a bottom-level Acceleration Structure.
// The passed in bottom-level AS geometry must have a unique name.
// Requires a corresponding 1 or more AddBottomLevelASInstance() calls to be added to the top-level AS for the bottom-level AS to be included.
void RaytracingAccelerationStructureManager::AddBottomLevelAS(
    ID3D12Device5* device,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags,
    BottomLevelAccelerationStructureGeometry& bottomLevelASGeometry,
    bool allowUpdate,
    bool performUpdateOnBuild,
    bool performCompaction)
{
    ThrowIfFalse(m_vBottomLevelAS.find(bottomLevelASGeometry.GetName()) == m_vBottomLevelAS.end(),
        L"A bottom level acceleration structure with that name already exists.");

    auto& bottomLevelAS = m_vBottomLevelAS[bottomLevelASGeometry.GetName()];

    bottomLevelAS.Initialize(device, buildFlags, bottomLevelASGeometry, allowUpdate, performUpdateOnBuild, performCompaction);

    m_ASmemoryFootprint += bottomLevelAS.RequiredResultDataSizeInBytes();
    m_scratchResourceSize = max(bottomLevelAS.RequiredScratchSize(), m_scratchResourceSize);

    m_vBottomLevelAS[bottomLevelAS.GetName()] = bottomLevelAS;
}

// Adds an instance of a bottom-level Acceleration Structure.
// Requires a call InitializeTopLevelAS() call to be added to top-level AS.
UINT RaytracingAccelerationStructureManager::AddBottomLevelASInstance(
    const wstring& bottomLevelASname,
    UINT instanceContributionToHitGroupIndex,
    XMMATRIX transform,
    BYTE instanceMask)
{
    ThrowIfFalse(numBottomLevelASInstances < m_bottomLevelASInstanceDescs.NumElements(), L"Not enough instance desc buffer size.");

    UINT instanceIndex = numBottomLevelASInstances++;
    auto& bottomLevelAS = m_vBottomLevelAS[bottomLevelASname];
    
    auto& instanceDesc = m_bottomLevelASInstanceDescs[instanceIndex];
    instanceDesc.InstanceMask = instanceMask;
    instanceDesc.InstanceContributionToHitGroupIndex = instanceContributionToHitGroupIndex != UINT_MAX ? instanceContributionToHitGroupIndex : bottomLevelAS.GetInstanceContributionToHitGroupIndex();
    instanceDesc.AccelerationStructure = bottomLevelAS.GetResource()->GetGPUVirtualAddress();
    XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(instanceDesc.Transform), transform);    

    return instanceIndex;
};

// Initializes the top-level Acceleration Structure.
void RaytracingAccelerationStructureManager::InitializeTopLevelAS(
    ID3D12Device5* device,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags, 
    bool allowUpdate, 
    bool performUpdateOnBuild,
    const wchar_t* resourceName)
{
    m_topLevelAS.Initialize(device, GetNumberOfBottomLevelASInstances(), buildFlags, allowUpdate, performUpdateOnBuild, resourceName);

    m_ASmemoryFootprint += m_topLevelAS.RequiredResultDataSizeInBytes();
    m_scratchResourceSize = max(m_topLevelAS.RequiredScratchSize(), m_scratchResourceSize);

    AllocateUAVBuffer(device, m_scratchResourceSize, &m_accelerationStructureScratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"Acceleration structure scratch resource");
}

// Builds all bottom-level and top-level Acceleration Structures.
void RaytracingAccelerationStructureManager::Build(
    ID3D12GraphicsCommandList4* commandList, 
    ID3D12DescriptorHeap* descriptorHeap,
    UINT frameIndex,
    bool bForceBuild)
{
    ScopedTimer _prof(L"Acceleration Structure build", commandList);

    m_bottomLevelASInstanceDescs.CopyStagingToGpu(frameIndex);

    // Build all bottom-level AS.
    {
        ScopedTimer _prof(L"Bottom Level AS", commandList);
        for (auto& bottomLevelASpair : m_vBottomLevelAS)
        {
            auto& bottomLevelAS = bottomLevelASpair.second;
            if (bForceBuild || bottomLevelAS.IsDirty())
            {
                ScopedTimer _prof(bottomLevelAS.GetName(), commandList);

                D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGpuAddress = 0;  // ToDo
                bottomLevelAS.Build(commandList, m_accelerationStructureScratch.Get(), descriptorHeap, baseGeometryTransformGpuAddress);

                // Since a single scratch resource is reused, put a barrier in-between each call.
                // ToDo add option to use per BLAS scratch with one UAV barrier
                // ToDo fix this for compaction. The get resource will return the compacted resource not source AS. But there's an UAV in Build in that case.
                commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(bottomLevelAS.GetResource()));

                bottomLevelAS.PostBuild(commandList);
            }
            // ToDo call just once when the CPU GPU synced
            bottomLevelAS.ReadbackCompactedSize();
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