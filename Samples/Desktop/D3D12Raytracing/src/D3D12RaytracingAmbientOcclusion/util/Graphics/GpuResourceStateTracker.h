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

#include "GpuResource.h"

// ToDo turn into class and check rwFlags being properly set on access.
class GpuResourceStateTracker
{
public:
    void TransitionResource(ID3D12GraphicsCommandList* commandList, GpuResource& Resource, D3D12_RESOURCE_STATES NewState, bool FlushImmediate = false);
    void BeginResourceTransition(ID3D12GraphicsCommandList* commandList, GpuResource& Resource, D3D12_RESOURCE_STATES NewState, bool FlushImmediate = false);
    void InsertUAVBarrier(ID3D12GraphicsCommandList* commandList, GpuResource& Resource, bool FlushImmediate = false);
    void InsertAliasBarrier(ID3D12GraphicsCommandList* commandList, GpuResource& Before, GpuResource& After, bool FlushImmediate = false);
    void FlushResourceBarriers(ID3D12GraphicsCommandList* commandList);

protected:

    const UINT c_MaxNumBarriers = 16;
    D3D12_RESOURCE_BARRIER m_ResourceBarrierBuffer[c_MaxNumBarriers];
    UINT m_NumBarriersToFlush;
};
