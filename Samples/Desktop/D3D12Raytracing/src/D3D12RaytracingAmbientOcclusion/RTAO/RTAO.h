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

#include "GlobalSharedHlslCompat.h"
#include "RTAOHlslCompat.h"
#include "DeviceResources.h"
#include "GeneralHelper.h"
#include "Lighting.h"
#include "PrimitiveBatch.h"
#include "VertexTypes.h"
#include "DXSampleHelper.h"
#include "DescriptorHeap.h"     // ToDo remove sample's impl of DescriptorHeap?

namespace RTAOPerObjectCSUDesc
{
    enum PerObjectCSUDesc
    {
        SRVDiffuse,
        SRVSpecular,
        SRVNormal,
        CSUCount
    };
}

namespace RTAORootSig
{
    enum RootSig
    {
        RootSceneConstSlot,
        RootRTAOConstSlot,
        RootMaterialConstSlot,
        RootSRVBeginSlot,
        RootSRVEndSlot = RootSRVBeginSlot + 5 - 1,
        RootUAVBeginSlot,
        RootUAVEndSlot = RootUAVBeginSlot + 9 - 1,
        RootSamplerSlot,
        RootCount
    };
}

namespace Parts
{
    constexpr unsigned int MaxParts = 10;
}

namespace RTAOCSUDesc
{
    enum CSUDesc
    {
        CBVScene,
        CBVSampleThickness,
        SRVDepthDownsizeStart,
        SRVDepthDownsizeEnd = SRVDepthDownsizeStart + NUM_BUFFERS - 1,
        SRVDepthTiledStart,
        SRVDepthTiledEnd = SRVDepthTiledStart + NUM_BUFFERS - 1,
        SRVNormalDownsizeStart,
        SRVNormalDownsizeEnd = SRVNormalDownsizeStart + NUM_BUFFERS - 1,
        SRVNormalTiledStart,
        SRVNormalTiledEnd = SRVNormalTiledStart + NUM_BUFFERS - 1,
        SRVMergedStart,
        SRVMergedEnd = SRVMergedStart + NUM_BUFFERS - 1,
        SRVSmoothStart,
        SRVSmoothEnd = SRVSmoothStart + NUM_BUFFERS - 1 - 1,
        SRVHighQualityStart,
        SRVHighQualityEnd = SRVHighQualityStart + NUM_BUFFERS - 1,
        SRVLinearDepth,
        SRVDepth,
        SRVGBuffer,
        SRVRTAO,
        SRVOutFrame,
        SRVPerObjectStart,
        SRVPerObjectEnd = SRVPerObjectStart + Parts::MaxParts * RTAOPerObjectCSUDesc::CSUCount - 1,
        UAVDepthDownsizeStart,
        UAVDepthDownsizeEnd = UAVDepthDownsizeStart + NUM_BUFFERS - 1,
        UAVDepthTiledStart,
        UAVDepthTiledEnd = UAVDepthTiledStart + NUM_BUFFERS - 1,
        UAVNormalDownsizeStart,
        UAVNormalDownsizeEnd = UAVNormalDownsizeStart + NUM_BUFFERS - 1,
        UAVNormalTiledStart,
        UAVNormalTiledEnd = UAVNormalTiledStart + NUM_BUFFERS - 1,
        UAVMergedStart,
        UAVMergedEnd = UAVMergedStart + NUM_BUFFERS - 1,
        UAVSmoothStart,
        UAVSmoothEnd = UAVSmoothStart + NUM_BUFFERS - 1 - 1,
        UAVHighQualityStart,
        UAVHighQualityEnd = UAVHighQualityStart + NUM_BUFFERS - 1,
        UAVLinearDepth,
        UAVRTAO,
        UAVOutFrame,
        CSUCount
    };
}

namespace RTAOSamplerDesc
{
    enum SamplerDesc
    {
        SamplerLinearWrap,
        SamplerLinearBorder,
        SamplerLinearClamp,
        SamplerPointClamp,
        SamplerCount
    };
}

namespace RTAORTVDesc
{
    enum RTVDesc
    {
        RTVGBuffer,
        RTVCount
    };
}

namespace RTAODSVDesc
{
    enum DSVDesc
    {
        DSVGBuffer,
        DSVCount
    };
}

class RTAO
{
public:
    RTAO();

    void Setup(std::shared_ptr<DX::DeviceResources> pDeviceResources);
    void OnRender();
    void OnUpdate();
    void OnSizeChanged(UINT width, UINT height);
    ID3D12Resource* GetRTAOOutputResource() { return m_ssaoResources.Get(); }

private:
    void CreateDeviceDependentResources();
    void CreateDescriptorHeaps();
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateRootSignatures();    
    void CreateRaytracingPipelineStateObject();
    void CreateDxilLibrarySubobject();
    void CreateHitGroupSubobjects();
    void CreateLocalRootSignatureSubobjects();
    void CreateResources();
    void CalculateAdaptiveSamplingCounts();
    
    void CreateSamplesRNG();
    void CreateDeviceDependentResources();
    void CreateWindowSizeDependentResources();
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources();
    void RenderRNGVisualizations();
    void BuildShaderTables();
    void DispatchRays(ID3D12Resource* rayGenShaderTable, uint32_t width, uint32_t height);
    


    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    std::mt19937 m_generatorURNG;

    // Raytracing shaders.
    static const wchar_t* c_hitGroupNames_TriangleGeometry[RayType::Count];
    static const wchar_t* c_rayGenShaderNames[RayGenShaderType::Count];
    static const wchar_t* c_closestHitShaderNames[RayType::Count];
    static const wchar_t* c_missShaderNames[RayType::Count];

    // Shader tables
    ComPtr<ID3D12Resource> m_rayGenShaderTables[RaytracingType::Count][RayGenShaderType::Count];
    UINT m_rayGenShaderTableRecordSizeInBytes[RaytracingType::Count];
    ComPtr<ID3D12Resource> m_hitGroupShaderTable;
    UINT m_hitGroupShaderTableStrideInBytes;
    ComPtr<ID3D12Resource> m_missShaderTable;
    UINT m_missShaderTableStrideInBytes;



};