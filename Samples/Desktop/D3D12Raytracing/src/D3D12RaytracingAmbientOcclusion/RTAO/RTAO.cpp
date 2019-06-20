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
#include "RTAO.h"
#include "GameInput.h"
#include "EngineTuning.h"
#include "EngineProfiling.h"
#include "GpuTimeManager.h"
#include "D3D12RaytracingAmbientOcclusion.h"
#include "CompiledShaders\RTAO.hlsl.h"


// ToDo prune unused
using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;

// Shader entry points.
const wchar_t* RTAO::c_rayGenShaderNames[] =
{
    L"RayGenShader", L"RayGenShader_sortedRays"
};
const wchar_t* RTAO::c_closestHitShaderName = L"ClosestHitShader";
const wchar_t* RTAO::c_missShaderName = L"MissShader";
// Hit groups.
const wchar_t* RTAO::c_hitGroupName = L"HitGroup_Triangle";


// ToDo use singleton interface and prevent multiple objects.
RTAO* g_pRTAO = nullptr;

namespace SceneArgs
{
    void OnRecreateRaytracingResources(void*)
    {
        g_pRTAO->RequestRecreateRaytracingResources();
    }

    void OnRecreateSamples(void*)
    {
        g_pRTAO->RequestRecreateAOSamples();
    }

    IntVar AOTileX(L"Render/AO/Tile X", 1, 1, 128, 1);
    IntVar AOTileY(L"Render/AO/Tile Y", 1, 1, 128, 1);

    BoolVar RTAOUseRaySorting(L"Render/AO/RTAO/Ray Sorting/Enabled", true);
    NumVar RTAORayBinDepthSizeMultiplier(L"Render/AO/RTAO/Ray Sorting/Ray bin depth size (multiplier of MaxRayHitTime)", 0.1f, 0.01f, 10.f, 0.01f);
    BoolVar RTAORaySortingUseOctahedralRayDirectionQuantization(L"Render/AO/RTAO/Ray Sorting/Octahedral ray direction quantization", true);

    // RTAO
    // Adaptive Sampling.
    BoolVar QuarterResAO(L"Render/AO/RTAO/Quarter res", false, OnRecreateRaytracingResources, nullptr);
    BoolVar RTAOAdaptiveSampling(L"Render/AO/RTAO/Adaptive Sampling/Enabled", false);
    BoolVar RTAOUseNormalMaps(L"Render/AO/RTAO/Normal maps", false);
    NumVar RTAOAdaptiveSamplingMaxFilterWeight(L"Render/AO/RTAO/Adaptive Sampling/Filter weight cutoff for max sampling", 0.995f, 0.0f, 1.f, 0.005f);
    BoolVar RTAOAdaptiveSamplingMinMaxSampling(L"Render/AO/RTAO/Adaptive Sampling/Only min/max sampling", false);
    NumVar RTAOAdaptiveSamplingScaleExponent(L"Render/AO/RTAO/Adaptive Sampling/Sampling scale exponent", 0.3f, 0.0f, 10, 0.1f);
    BoolVar RTAORandomFrameSeed(L"Render/AO/RTAO/Random per-frame seed", false);

    // ToDo remove
    NumVar RTAOTraceRayOffsetAlongNormal(L"Render/AO/RTAO/TraceRay/Ray origin offset along surface normal", 0.001f, 0, 0.1f, 0.0001f);
    NumVar RTAOTraceRayOffsetAlongRayDirection(L"Render/AO/RTAO/TraceRay/Ray origin offset fudge along ray direction", 0, 0, 0.1f, 0.0001f);



    const WCHAR* FloatingPointFormatsR[TextureResourceFormatR::Count] = { L"R32_FLOAT", L"R16_FLOAT", L"R8_UNORM" };
    EnumVar RTAO_AmbientCoefficientResourceFormat(L"Render/Texture Formats/AO/RTAO/Ambient Coefficient", TextureResourceFormatR::R8_UNORM, TextureResourceFormatR::Count, FloatingPointFormatsR, OnRecreateRaytracingResources);

  
    // ToDo cleanup RTAO... vs RTAO_..
    IntVar RTAOAdaptiveSamplingMinSamples(L"Render/AO/RTAO/Adaptive Sampling/Min samples", 1, 1, AO_SPP_N* AO_SPP_N, 1);
    IntVar RTAO_KernelStepShift0(L"Render/AO/RTAO/Kernel Step Shifts/0", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift1(L"Render/AO/RTAO/Kernel Step Shifts/1", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift2(L"Render/AO/RTAO/Kernel Step Shifts/2", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift3(L"Render/AO/RTAO/Kernel Step Shifts/3", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift4(L"Render/AO/RTAO/Kernel Step Shifts/4", 0, 0, 10, 1);

    IntVar AOSampleCountPerDimension(L"Render/AO/RTAO/Samples per pixel NxN", AO_SPP_N, 1, AO_SPP_N_MAX, 1, OnRecreateSamples, nullptr);
    IntVar AOSampleSetDistributedAcrossPixels(L"Render/AO/RTAO/Sample set distribution across NxN pixels ", 8, 1, 8, 1, OnRecreateSamples, nullptr);
#if LOAD_PBRT_SCENE
    NumVar RTAOMaxRayHitTime(L"Render/AO/RTAO/Max ray hit time", AO_RAY_T_MAX, 0.0f, 50.0f, 0.2f);
#else
    NumVar RTAOMaxRayHitTime(L"Render/AO/RTAO/Max ray hit time", AO_RAY_T_MAX, 0.0f, 1000.0f, 4);
#endif
    BoolVar RTAOApproximateInterreflections(L"Render/AO/RTAO/Approximate Interreflections/Enabled", true);
    NumVar RTAODiffuseReflectanceScale(L"Render/AO/RTAO/Approximate Interreflections/Diffuse Reflectance Scale", 0.5f, 0.0f, 1.0f, 0.1f);
    NumVar  minimumAmbientIllumination(L"Render/AO/RTAO/Minimum Ambient Illumination", 0.07f, 0.0f, 1.0f, 0.01f);
    BoolVar RTAOIsExponentialFalloffEnabled(L"Render/AO/RTAO/Exponential Falloff", true);
    NumVar RTAO_ExponentialFalloffDecayConstant(L"Render/AO/RTAO/Exponential Falloff Decay Constant", 2.f, 0.0f, 20.f, 0.25f);
    NumVar RTAO_ExponentialFalloffMinOcclusionCutoff(L"Render/AO/RTAO/Exponential Falloff Min Occlusion Cutoff", 0.4f, 0.0f, 1.f, 0.05f);       // ToDo Finetune document perf.
};



RTAO::RTAO()
{
    for (auto& rayGenShaderTableRecordSizeInBytes : m_rayGenShaderTableRecordSizeInBytes)
    {
        rayGenShaderTableRecordSizeInBytes = UINT_MAX;
    }
    m_generatorURNG.seed(1729);
}


void RTAO::Setup(shared_ptr<DeviceResources> deviceResources, shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT numHitGroupShaderRecodsToCreate, UINT FrameCount)
{
    m_deviceResources = deviceResources;
    m_cbvSrvUavHeap = descriptorHeap;

    CreateDeviceDependentResources(FrameCount);
}

// Create resources that depend on the device.
void RTAO::CreateDeviceDependentResources(UINT numHitGroupShaderRecodsToCreate, UINT FrameCount)
{
    auto device = m_deviceResources->GetD3DDevice();
    
    CreateAuxilaryDeviceResources(FrameCount);

    // Initialize raytracing pipeline.

    // Create root signatures for the shaders.
    CreateRootSignatures();

    // Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
    CreateRaytracingPipelineStateObject();

    // Create constant buffers for the geometry and the scene.
    CreateConstantBuffers();

    // Build shader tables, which define shaders and their local root arguments.
    BuildShaderTables(numHitGroupShaderRecodsToCreate);

    // ToDo rename
    CreateSamplesRNG();
}


// ToDo rename
void RTAO::CreateAuxilaryDeviceResources(UINT FrameCount)
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    auto commandList = m_deviceResources->GetCommandList();

    EngineProfiling::RestoreDevice(device, commandQueue, FrameCount);
    ResourceUploadBatch resourceUpload(device);
    resourceUpload.Begin();

    // ToDo move?
    m_reduceSumKernel.Initialize(device, GpuKernels::ReduceSum::Uint);
    m_raySorter.Initialize(device, FrameCount);

    // Upload the resources to the GPU.
    auto finish = resourceUpload.End(commandQueue);

    // Wait for the upload thread to terminate
    finish.wait();
}

// Create constant buffers.
void RTAO::CreateConstantBuffers()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    m_CB.Create(device, FrameCount, L"RTAO Constant Buffer");
}


void RTAO::CreateRootSignatures()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Global Root Signature
    // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
    {
        using namespace GlobalRootSignature;

        // ToDo reorder
        // ToDo use slot index in ranges everywhere
        CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count]; // Perfomance TIP: Order from most frequent to least frequent.
        ranges[Slot::Output].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output textures
        ranges[Slot::GBufferResources].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 5, 5);  // 5 output GBuffer textures
        ranges[Slot::AOResourcesOut].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 10);  // 2 output AO textures
        ranges[Slot::VisibilityResource].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 12);  // 1 output visibility texture
        ranges[Slot::GBufferDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 13);  // 1 output depth texture
        ranges[Slot::GbufferNormalRGB].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 14);  // 1 output normal texture
        ranges[Slot::NormalDepthLowPrecision].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 23);  // 1 output normal depth texture
        ranges[Slot::AORayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 15);  // 1 output ray hit distance texture
        ranges[Slot::MotionVector].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 17);  // 1 output texture space motion vector.
        ranges[Slot::ReprojectedHitPosition].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 18);  // 1 output texture reprojected hit position
        ranges[Slot::Color].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 19);  // 1 output texture shaded color
        ranges[Slot::AODiffuse].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 20);  // 1 output texture AO diffuse
        ranges[Slot::ShadowMapUAV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 21);  // 1 output ShadowMap texture
        ranges[Slot::AORayDirectionOriginDepthHitUAV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 22);  // 1 output AO ray direction and origin depth texture


#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        ranges[Slot::PartialDepthDerivatives].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 16);  // 1 output partial depth derivative texture
#endif
        ranges[Slot::GBufferResourcesIn].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 5);  // 4 input GBuffer textures
        ranges[Slot::EnvironmentMap].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12);  // 1 input environment map texture
        ranges[Slot::FilterWeightSum].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 13);  // 1 input filter weight sum texture
        ranges[Slot::AOFrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 14);  // 1 input AO frame age

        ranges[Slot::ShadowMapSRV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 21);  // 1 ShadowMap texture
        ranges[Slot::AORayDirectionOriginDepthHitSRV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 22);  // 1 AO ray direction and origin depth texture
        ranges[Slot::AOSourceToSortedRayIndex].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 23);  // 1 input AO ray group thread offsets

        CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
        rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[Slot::Output]);
        rootParameters[Slot::GBufferResources].InitAsDescriptorTable(1, &ranges[Slot::GBufferResources]);
        rootParameters[Slot::GBufferResourcesIn].InitAsDescriptorTable(1, &ranges[Slot::GBufferResourcesIn]);
        rootParameters[Slot::AOResourcesOut].InitAsDescriptorTable(1, &ranges[Slot::AOResourcesOut]);
        rootParameters[Slot::VisibilityResource].InitAsDescriptorTable(1, &ranges[Slot::VisibilityResource]);
        rootParameters[Slot::EnvironmentMap].InitAsDescriptorTable(1, &ranges[Slot::EnvironmentMap]);
        rootParameters[Slot::GBufferDepth].InitAsDescriptorTable(1, &ranges[Slot::GBufferDepth]);
        rootParameters[Slot::GbufferNormalRGB].InitAsDescriptorTable(1, &ranges[Slot::GbufferNormalRGB]);
        rootParameters[Slot::NormalDepthLowPrecision].InitAsDescriptorTable(1, &ranges[Slot::NormalDepthLowPrecision]);
        rootParameters[Slot::FilterWeightSum].InitAsDescriptorTable(1, &ranges[Slot::FilterWeightSum]);
        rootParameters[Slot::AORayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::AORayHitDistance]);
        rootParameters[Slot::AOFrameAge].InitAsDescriptorTable(1, &ranges[Slot::AOFrameAge]);
        rootParameters[Slot::AORayDirectionOriginDepthHitSRV].InitAsDescriptorTable(1, &ranges[Slot::AORayDirectionOriginDepthHitSRV]);
        rootParameters[Slot::AOSourceToSortedRayIndex].InitAsDescriptorTable(1, &ranges[Slot::AOSourceToSortedRayIndex]);
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        rootParameters[Slot::PartialDepthDerivatives].InitAsDescriptorTable(1, &ranges[Slot::PartialDepthDerivatives]);
#endif
        rootParameters[Slot::MotionVector].InitAsDescriptorTable(1, &ranges[Slot::MotionVector]);
        rootParameters[Slot::ReprojectedHitPosition].InitAsDescriptorTable(1, &ranges[Slot::ReprojectedHitPosition]);
        rootParameters[Slot::Color].InitAsDescriptorTable(1, &ranges[Slot::Color]);
        rootParameters[Slot::AODiffuse].InitAsDescriptorTable(1, &ranges[Slot::AODiffuse]);
        rootParameters[Slot::ShadowMapSRV].InitAsDescriptorTable(1, &ranges[Slot::ShadowMapSRV]);
        rootParameters[Slot::ShadowMapUAV].InitAsDescriptorTable(1, &ranges[Slot::ShadowMapUAV]);
        rootParameters[Slot::AORayDirectionOriginDepthHitUAV].InitAsDescriptorTable(1, &ranges[Slot::AORayDirectionOriginDepthHitUAV]);

        rootParameters[Slot::AccelerationStructure].InitAsShaderResourceView(0);
        rootParameters[Slot::SceneConstant].InitAsConstantBufferView(0);		// ToDo rename to ConstantBuffer
        rootParameters[Slot::MaterialBuffer].InitAsShaderResourceView(3);
        rootParameters[Slot::SampleBuffers].InitAsShaderResourceView(4);
        rootParameters[Slot::PrevFrameBottomLevelASIstanceTransforms].InitAsShaderResourceView(15);

        CD3DX12_STATIC_SAMPLER_DESC staticSamplers[] =
        {
            // LinearWrapSampler
            CD3DX12_STATIC_SAMPLER_DESC(0, SAMPLER_FILTER),
            // ShadowMapSamplerComp
            CD3DX12_STATIC_SAMPLER_DESC(1, D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT),
            // ShadowMapSampler
            CD3DX12_STATIC_SAMPLER_DESC(2, D3D12_FILTER_MIN_MAG_MIP_POINT)
        };

        CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, ARRAYSIZE(staticSamplers), staticSamplers);
        SerializeAndCreateRootSignature(device, globalRootSignatureDesc, &m_raytracingGlobalRootSignature, L"Global root signature");
    }

    // Local Root Signature
    // This is a root signature that enables a shader to have unique arguments that come from shader tables.
    {
        // Triangle geometry
        {
            using namespace LocalRootSignature::Triangle;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count]; // Perfomance TIP: Order from most frequent to least frequent.
            ranges[Slot::IndexBuffer].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 1);  // 1 buffer - index buffer.
            ranges[Slot::VertexBuffer].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1, 1);  // 1 buffer - current frame vertex buffer.
            ranges[Slot::PreviousFrameVertexBuffer].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2, 1);  // 1 buffer - previous frame vertex buffer.
            ranges[Slot::DiffuseTexture].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3, 1);  // 1 diffuse texture
            ranges[Slot::NormalTexture].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4, 1);  // 1 normal texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::ConstantBuffer].InitAsConstants(SizeOfInUint32(PrimitiveConstantBuffer), 0, 1);
            rootParameters[Slot::IndexBuffer].InitAsDescriptorTable(1, &ranges[Slot::IndexBuffer]);
            rootParameters[Slot::VertexBuffer].InitAsDescriptorTable(1, &ranges[Slot::VertexBuffer]);
            rootParameters[Slot::PreviousFrameVertexBuffer].InitAsDescriptorTable(1, &ranges[Slot::PreviousFrameVertexBuffer]);
            rootParameters[Slot::DiffuseTexture].InitAsDescriptorTable(1, &ranges[Slot::DiffuseTexture]);
            rootParameters[Slot::NormalTexture].InitAsDescriptorTable(1, &ranges[Slot::NormalTexture]);

            CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
            SerializeAndCreateRootSignature(device, localRootSignatureDesc, &m_raytracingLocalRootSignature[LocalRootSignature::Type::Triangle], L"Local root signature: triangle geometry");
        }
    }
}


// DXIL library
// This contains the shaders and their entrypoints for the state object.
// Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
void RTAO::CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    auto lib = raytracingPipeline->CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void*)g_pRTAO, ARRAYSIZE(g_pRTAO));
    lib->SetDXILLibrary(&libdxil);
    // Use default shader exports for a DXIL library/collection subobject ~ surface all shaders.
}

// Hit groups
// A hit group specifies closest hit, any hit and intersection shaders 
// to be executed when a ray intersects the geometry.
void RTAO::CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    // Triangle geometry hit groups
    {
        auto hitGroup = raytracingPipeline->CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();

        hitGroup->SetClosestHitShaderImport(c_closestHitShaderName);
        hitGroup->SetHitGroupExport(c_hitGroupName);
        hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
    }
}


// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void RTAO::CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    // Ray gen and miss shaders in this sample are not using a local root signature and thus one is not associated with them.

    // Hit groups
    // Triangle geometry
    {
        auto localRootSignature = raytracingPipeline->CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localRootSignature->SetRootSignature(m_raytracingLocalRootSignature.Get());
        // Shader association
        auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
        rootSignatureAssociation->AddExport(c_hitGroupName);
    }
}

// Create a raytracing pipeline state object (RTPSO).
// An RTPSO represents a full set of shaders reachable by a DispatchRays() call,
// with all configuration options resolved, such as local signatures and other state.
void RTAO::CreateRaytracingPipelineStateObject()
{
    auto device = m_deviceResources->GetD3DDevice();
    // Ambient Occlusion state object.
    {
        // ToDo review
        // Create 18 subobjects that combine into a RTPSO:
        // Subobjects need to be associated with DXIL exports (i.e. shaders) either by way of default or explicit associations.
        // Default association applies to every exported shader entrypoint that doesn't have any of the same type of subobject associated with it.
        // This simple sample utilizes default shader association except for local root signature subobject
        // which has an explicit association specified purely for demonstration purposes.
        // 1 - DXIL library
        // 8 - Hit group types - 4 geometries (1 triangle, 3 aabb) x 2 ray types (ray, shadowRay)
        // 1 - Shader config
        // 6 - 3 x Local root signature and association
        // 1 - Global root signature
        // 1 - Pipeline config
        CD3DX12_STATE_OBJECT_DESC raytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };

        bool loadShadowShadersOnly = true;
        // DXIL library
        CreateDxilLibrarySubobject(&raytracingPipeline);

        // Hit groups
        CreateHitGroupSubobjects(&raytracingPipeline);

        // ToDo try 2B float payload
#define AO_4B_RAYPAYLOAD 0

        // Shader config
        // Defines the maximum sizes in bytes for the ray rayPayload and attribute structure.
        auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
#if AO_4B_RAYPAYLOAD
        UINT payloadSize = static_cast<UINT>(sizeof(ShadowRayPayload));		// ToDo revise
#else
        UINT payloadSize = static_cast<UINT>(max(max(sizeof(RayPayload), sizeof(ShadowRayPayload)), sizeof(GBufferRayPayload)));		// ToDo revise
#endif
        UINT attributeSize = sizeof(XMFLOAT2);  // float2 barycentrics
        shaderConfig->Config(payloadSize, attributeSize);

        // Local root signature and shader association
        // This is a root signature that enables a shader to have unique arguments that come from shader tables.
        CreateLocalRootSignatureSubobjects(&raytracingPipeline);

        // Global root signature
        // This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
        auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3DX12_GLOBAL_ROOT_SIGNATURE_SUBOBJECT>();
        globalRootSignature->SetRootSignature(m_raytracingGlobalRootSignature.Get());

        // Pipeline config
        // Defines the maximum TraceRay() recursion depth.
        auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
        // PERFOMANCE TIP: Set max recursion depth as low as needed
        // as drivers may apply optimization strategies for low recursion depths.
        UINT maxRecursionDepth = 1;
        pipelineConfig->Config(maxRecursionDepth);

        PrintStateObjectDesc(raytracingPipeline);

        // Create the state object.
        ThrowIfFailed(device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_dxrStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
    }
}


void RTAO::CreateResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

    // ToDo change this to non-PS resouce since we use CS?
    D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    // ToDo remove obsolete resources, QuarterResAO event triggers this so we may not need all low/gbuffer width AO resources.

    // Full-res AO resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_AOResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
        m_AOResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
        for (UINT i = 0; i < AOResource::Count; i++)
        {
            m_AOResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_AOResources[i].uavDescriptorHeapIndex = m_AOResources[0].uavDescriptorHeapIndex + i;
            m_AOResources[i].srvDescriptorHeapIndex = m_AOResources[0].srvDescriptorHeapIndex + i;
        }

        // ToDo pack some resources.

        // ToDo cleanup raytracing resolution - twice for coefficient.
        CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Coefficient], initialResourceState, L"Render/AO Coefficient");

#if ATROUS_DENOISER
        CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
#else
        CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
#endif
        // ToDo 8 bit hit count?
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_UINT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::HitCount], initialResourceState, L"Render/AO Hit Count");

        // ToDo use lower bit float?
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::FilterWeightSum], initialResourceState, L"Render/AO Filter Weight Sum");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::RayHitDistance], initialResourceState, L"Render/AO Hit Distance");
    }

    // ToDo merge low/full-res or only create one at a time?
    // Low-res AO resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_AOLowResResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
        m_AOLowResResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
        for (UINT i = 0; i < AOResource::Count; i++)
        {
            m_AOLowResResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_AOLowResResources[i].uavDescriptorHeapIndex = m_AOLowResResources[0].uavDescriptorHeapIndex + i;
            m_AOLowResResources[i].srvDescriptorHeapIndex = m_AOLowResResources[0].srvDescriptorHeapIndex + i;
        }

        CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::Coefficient], initialResourceState, L"Render/AO LowRes Coefficient");

#if ATROUS_DENOISER
        CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::Smoothed], initialResourceState, L"Render/AO LowRes Smoothed");
#else
        CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::Smoothed], initialResourceState, L"Render/AO LowRes Smoothed");
#endif
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::HitCount], initialResourceState, L"Render/AO LowRes Hit Count");

        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::FilterWeightSum], initialResourceState, L"Render/AO LowRes Filter Weight Sum");

        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::RayHitDistance], initialResourceState, L"Render/AO LowRes Hit Distance");
    }
    
    
    m_sourceToSortedRayIndex.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R8G8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sourceToSortedRayIndex, initialResourceState, L"Source To Sorted Ray Index");

    m_sortedToSourceRayIndex.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R8G8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sortedToSourceRayIndex, initialResourceState, L"Sorted To Source Ray Index");

    m_sortedRayGroupDebug.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R32G32B32A32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sortedRayGroupDebug, initialResourceState, L"Sorted Ray Group Debug");

    m_AORayDirectionOriginDepth.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AORayDirectionOriginDepth, initialResourceState, L"AO Rays Direction, Origin Depth and Hit");
}


// Build shader tables.
// This encapsulates all shader records - shaders and the arguments for their local root signatures.
// For AO, the shaders are simple with only one shader type per shader table.
// numHitGroupShaderRecordsToCreate - since BLAS instances in this sample specify non-zero InstanceContributionToHitGroupIndex, 
//  the sample needs to add as many shader records to all hit group shader tables so that DXR shader addressing lands on a valid shader record.
void RTAO::BuildShaderTables(UINT numHitGroupShaderRecordsToCreate)
{
    auto device = m_deviceResources->GetD3DDevice();

    void* rayGenShaderIDs[RTAORayGenShaderType::Count];
    void* missShaderID;
    void* hitGroupShaderID;

    // A shader name look-up table for shader table debug print out.
    unordered_map<void*, wstring> shaderIdToStringMap;

    auto GetShaderIDs = [&](auto* stateObjectProperties)
    {
        for (UINT i = 0; i < RTAORayGenShaderType::Count; i++)
        {
            rayGenShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(c_rayGenShaderNames[i]);
            shaderIdToStringMap[rayGenShaderIDs[i]] = c_rayGenShaderNames[i];
        }

        missShaderID = stateObjectProperties->GetShaderIdentifier(c_missShaderName);
        shaderIdToStringMap[missShaderID] = c_missShaderName;

        hitGroupShaderID = stateObjectProperties->GetShaderIdentifier(c_hitGroupName);
        shaderIdToStringMap[hitGroupShaderID] = c_hitGroupName;
    };

    // Get shader identifiers.
    UINT shaderIDSize;
    ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
    ThrowIfFailed(m_dxrStateObject.As(&stateObjectProperties));
    GetShaderIDs(stateObjectProperties.Get());
    shaderIDSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

    // ToDo review
    /*************--------- Shader table layout -------*******************
    | -------------------------------------------------------------------
    | -------------------------------------------------------------------
    |Shader table - RayGenShaderTable: 32 | 32 bytes
    | [0]: MyRaygenShader, 32 + 0 bytes
    | -------------------------------------------------------------------

    | -------------------------------------------------------------------
    |Shader table - MissShaderTable: 32 | 64 bytes
    | [0]: MyMissShader, 32 + 0 bytes
    | [1]: MyMissShader_ShadowRay, 32 + 0 bytes
    | -------------------------------------------------------------------

    | -------------------------------------------------------------------
    |Shader table - HitGroupShaderTable: 96 | 196800 bytes
    | [0]: MyHitGroup_Triangle, 32 + 56 bytes
    | [1]: MyHitGroup_Triangle_ShadowRay, 32 + 56 bytes
    | [2]: MyHitGroup_Triangle, 32 + 56 bytes
    | [3]: MyHitGroup_Triangle_ShadowRay, 32 + 56 bytes
    | ...
    | --------------------------------------------------------------------
    **********************************************************************/

    // RayGen shader tables.
    {
        UINT numShaderRecords = 1;
        UINT shaderRecordSize = shaderIDSize; // No root arguments

        for (UINT i = 0; i < RTAORayGenShaderType::Count; i++)
        {
            ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RTAO RayGenShaderTable");
            rayGenShaderTable.push_back(ShaderRecord(rayGenShaderIDs[i], shaderIDSize, nullptr, 0));
            rayGenShaderTable.DebugPrint(shaderIdToStringMap);
            m_rayGenShaderTables[i] = rayGenShaderTable.GetResource();
        }
    }

    // Miss shader table.
    {
        UINT numShaderRecords = 1;
        UINT shaderRecordSize = shaderIDSize; // No root arguments

        ShaderTable missShaderTable(device, numShaderRecords, shaderRecordSize, L"RTAO MissShaderTable");
        missShaderTable.push_back(ShaderRecord(missShaderID, shaderIDSize, nullptr, 0));

        missShaderTable.DebugPrint(shaderIdToStringMap);
        m_missShaderTableStrideInBytes = missShaderTable.GetShaderRecordSize();
        m_missShaderTable = missShaderTable.GetResource();
    }
    
    // Hit group shader table.
    {
        UINT numShaderRecords = numHitGroupShaderRecordsToCreate;
        UINT shaderRecordSize = shaderIDSize; // No root arguments

        ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"RTAO HitGroupShaderTable");

        for (UINT i = 0; i < numShaderRecords; i++)
        {
            hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderID, shaderIDSize, nullptr, 0));
        }
        hitGroupShaderTable.DebugPrint(shaderIdToStringMap);
        m_hitGroupShaderTableStrideInBytes = hitGroupShaderTable.GetShaderRecordSize();
        m_hitGroupShaderTable = hitGroupShaderTable.GetResource();
    }
}


// ToDo rename
void RTAO::CreateSamplesRNG()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    UINT spp = SceneArgs::AOSampleCountPerDimension * SceneArgs::AOSampleCountPerDimension;
    UINT samplesPerSet = spp * SceneArgs::AOSampleSetDistributedAcrossPixels * SceneArgs::AOSampleSetDistributedAcrossPixels;
    UINT NumSampleSets = 83;
    m_randomSampler.Reset(samplesPerSet, NumSampleSets, Samplers::HemisphereDistribution::Cosine);



    // Create shader resources
    {
        // ToDo rename GPU from resource names?
        m_samplesGPUBuffer.Create(device, m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(), FrameCount, L"GPU buffer: Random unit square samples");
        m_hemisphereSamplesGPUBuffer.Create(device, m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(), FrameCount, L"GPU buffer: Random hemisphere samples");

        for (UINT i = 0; i < m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(); i++)
        {
            XMFLOAT3 p = m_randomSampler.GetHemisphereSample3D();
            // Convert [-1,1] to [0,1].
            m_samplesGPUBuffer[i].value = XMFLOAT2(p.x * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
            m_hemisphereSamplesGPUBuffer[i].value = p;
        }
    }
}


// Calculate adaptive per-pixel sampling counts.
// Ref: Bauszat et al. 2011, Adaptive Sampling for Geometry-Aware Reconstruction Filters
// The per-pixel sampling counts are calculated in two steps:
//  - Computing a per-pixel filter weight. This value represents how much of neighborhood contributes
//    to a pixel when filtering. Pixels with small values benefit from filtering only a little and thus will 
//    benefit from increased sampling the most.
//  - Calculating per-pixel sampling count based on the filter weight;
void RTAO::CalculateAdaptiveSamplingCounts()
{
    ThrowIfFalse(false, L"ToDo. Should this be part of AO or result passed in from outside?");
#if 0
    auto commandList = m_deviceResources->GetCommandList();

    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    RWGpuResource& NormalDeptLowPrecisionResource = SceneArgs::QuarterResAO ?
        m_normalDepthLowResLowPrecision[m_normalDepthCurrentFrameResourceIndex]
        : m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex];

    // Transition the output resource to UAV state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::FilterWeightSum].resource.Get(), before, after));
    }
    UINT offsets[5] = { 0, 1, 2, 3, 4 };    // ToDo
    // Calculate filter weight sum for each pixel. 
    {
        ScopedTimer _prof(L"CalculateFilterWeights", commandList);
        m_atrousWaveletTransformFilter.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            GpuKernels::AtrousWaveletTransformCrossBilateralFilter::EdgeStoppingGaussian5x5,
            AOResources[AOResource::Coefficient].gpuDescriptorReadAccess,
            NormalDeptLowPrecisionResource.gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess,
#if PACK_MEAN_VARIANCE
            m_smoothedMeanVarianceResource.gpuDescriptorReadAccess,
#else
            m_smoothedVarianceResource.gpuDescriptorReadAccess,
#endif
            AOResources[AOResource::RayHitDistance].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            &AOResources[AOResource::FilterWeightSum],
            SceneArgs::AODenoiseValueSigma,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            static_cast<TextureResourceFormatRGB::Type>(static_cast<UINT>(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat)),
            offsets,
            1,
            GpuKernels::AtrousWaveletTransformCrossBilateralFilter::Mode::OutputPerPixelFilterWeightSum,
            SceneArgs::ReverseFilterOrder,
            SceneArgs::UseSpatialVariance,
            SceneArgs::RTAODenoisingPerspectiveCorrectDepthInterpolation,
            false,
            SceneArgs::RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor,
            SceneArgs::RTAODenoisingFilterMinKernelWidth,
            static_cast<UINT>((SceneArgs::RTAODenoisingFilterMaxKernelWidthPercentage / 100) * m_raytracingWidth),
            SceneArgs::RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels,
            SceneArgs::QuarterResAO);
    }

    // Transition the output to SRV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::FilterWeightSum].resource.Get(), before, after));
    }
#endif
}


void RTAO::DispatchRays(ID3D12Resource* rayGenShaderTable)
{
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"DispatchRays", commandList);

    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    dispatchDesc.HitGroupTable.StartAddress = m_hitGroupShaderTable->GetGPUVirtualAddress();
    dispatchDesc.HitGroupTable.SizeInBytes = m_hitGroupShaderTable->GetDesc().Width;
    dispatchDesc.HitGroupTable.StrideInBytes = m_hitGroupShaderTableStrideInBytes;
    dispatchDesc.MissShaderTable.StartAddress = m_missShaderTable->GetGPUVirtualAddress();
    dispatchDesc.MissShaderTable.SizeInBytes = m_missShaderTable->GetDesc().Width;
    dispatchDesc.MissShaderTable.StrideInBytes = m_missShaderTableStrideInBytes;
    dispatchDesc.RayGenerationShaderRecord.StartAddress = rayGenShaderTable->GetGPUVirtualAddress();
    dispatchDesc.RayGenerationShaderRecord.SizeInBytes = rayGenShaderTable->GetDesc().Width;
    dispatchDesc.Width = m_raytracingWidth;
    dispatchDesc.Height = m_raytracingHeight;
    dispatchDesc.Depth = 1;
    commandList->SetPipelineState1(m_dxrStateObject.Get());

    commandList->DispatchRays(&dispatchDesc);
};


void RTAO::OnUpdate()
{

    if (m_isRecreateAOSamplesRequested)
    {
        m_isRecreateAOSamplesRequested = false;

        m_deviceResources->WaitForGpu();
        CreateSamplesRNG();
    }

    if (m_isRecreateRaytracingResourcesRequested)
    {
        // ToDo what if scenargs change during rendering? race condition??
        // Buffer them - create an intermediate
        m_isRecreateRaytracingResourcesRequested = false;
        m_deviceResources->WaitForGpu();

        // ToDo split to recreate only whats needed?
        CreateResolutionDependentResources();
        CreateAuxilaryDeviceResources();
    }
}

void RTAO::OnRender(D3D12_GPU_VIRTUAL_ADDRESS& accelerationStructure)
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    if (SceneArgs::RTAOAdaptiveSampling)
    {
        // ToDo move to within AO
        CalculateAdaptiveSamplingCounts();
    }

    ScopedTimer _prof(L"CalculateAmbientOcclusion", commandList);


    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    RWGpuResource& NormalDeptLowPrecisionResource = SceneArgs::QuarterResAO ?
        m_normalDepthLowResLowPrecision[m_normalDepthCurrentFrameResourceIndex]
        : m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex];

    // Transition AO resources to UAV state.        // ToDo check all comments
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::HitCount].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Coefficient].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::RayHitDistance].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_sourceToSortedRayIndex.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_sortedToSourceRayIndex.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_sortedRayGroupDebug.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_AORayDirectionOriginDepth.resource.Get(), before, after),
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());


    // Bind inputs.
    // ToDo use [enum] instead of [0]
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResourcesIn, GBufferResources[0].gpuDescriptorReadAccess);
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::SampleBuffers, m_hemisphereSamplesGPUBuffer.GpuVirtualAddress(frameIndex));
    commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_CB.GpuVirtualAddress(frameIndex));   // ToDo let AO have its own CB.
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::FilterWeightSum, AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);


    //ToDo remove
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOFrameAge, m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalCache::FrameAge].gpuDescriptorReadAccess);

    // Bind output RT.
    // ToDo remove output and rename AOout
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOResourcesOut, AOResources[0].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AORayHitDistance, AOResources[AOResource::RayHitDistance].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AORayDirectionOriginDepthHitUAV, m_AORayDirectionOriginDepth.gpuDescriptorWriteAccess);

    // Bind the heaps, acceleration structure and dispatch rays. 
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, accelerationStructure);

    DispatchRays(m_rayGenShaderTables[RTAORayGenShaderType::AOFullRes].Get(), m_raytracingWidth, m_raytracingHeight);

    // ToDo Remove
    //DispatchRays(m_rayGenShaderTables[SceneArgs::QuarterResAO ? RTAORayGenShaderType::AOQuarterRes : RTAORayGenShaderType::AOFullRes].Get(),
    //    &m_gpuTimers[GpuTimers::Raytracing_AO], m_raytracingWidth, m_raytracingHeight);

    // Transition AO resources to shader resource state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_AORayDirectionOriginDepth.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::UAV(m_AORayDirectionOriginDepth.resource.Get()),  // ToDo
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    if (SceneArgs::RTAOUseRaySorting)
    {
        float rayBinDepthSize = SceneArgs::RTAORayBinDepthSizeMultiplier * SceneArgs::RTAOMaxRayHitTime;
        m_raySorter.Execute(
            commandList,
            rayBinDepthSize,
            m_raytracingWidth,
            m_raytracingHeight,
            GpuKernels::SortRays::FilterType::CountingSort,
            //GpuKernels::SortRays::FilterType::BitonicSort,
            SceneArgs::RTAORaySortingUseOctahedralRayDirectionQuantization,
            m_cbvSrvUavHeap->GetHeap(),
            m_AORayDirectionOriginDepth.gpuDescriptorReadAccess,
            m_sourceToSortedRayIndex.gpuDescriptorWriteAccess,
            m_sortedToSourceRayIndex.gpuDescriptorWriteAccess,
            m_sortedRayGroupDebug.gpuDescriptorWriteAccess);

        // Transition the output to SRV state. 
        {
            D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            D3D12_RESOURCE_BARRIER barriers[] = {
                CD3DX12_RESOURCE_BARRIER::Transition(m_sourceToSortedRayIndex.resource.Get(), before, after),
                CD3DX12_RESOURCE_BARRIER::Transition(m_sortedToSourceRayIndex.resource.Get(), before, after),
                CD3DX12_RESOURCE_BARRIER::Transition(m_sortedRayGroupDebug.resource.Get(), before, after),
                CD3DX12_RESOURCE_BARRIER::UAV(m_sourceToSortedRayIndex.resource.Get()),
                CD3DX12_RESOURCE_BARRIER::UAV(m_sortedToSourceRayIndex.resource.Get())
            };
            commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
        }

        {
            ScopedTimer _prof(L"[Sorted]CalculateAmbientOcclusion", commandList);

            commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());


            // Bind inputs.
            // ToDo use [enum] instead of [0]
            commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResourcesIn, GBufferResources[0].gpuDescriptorReadAccess);
            commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::SampleBuffers, m_hemisphereSamplesGPUBuffer.GpuVirtualAddress(frameIndex));
            commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_CB.GpuVirtualAddress(frameIndex));   // ToDo let AO have its own CB.
            commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::FilterWeightSum, AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);

            // ToDo remove
            commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOFrameAge, m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalCache::FrameAge].gpuDescriptorReadAccess);
            commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AORayDirectionOriginDepthHitSRV, m_AORayDirectionOriginDepth.gpuDescriptorReadAccess);
            commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOSourceToSortedRayIndex, m_sourceToSortedRayIndex.gpuDescriptorReadAccess);

            // Bind output RT.
            // ToDo remove output and rename AOout
            commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOResourcesOut, AOResources[0].gpuDescriptorWriteAccess);
            commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AORayHitDistance, AOResources[AOResource::RayHitDistance].gpuDescriptorWriteAccess);

            // Bind the heaps, acceleration structure and dispatch rays. 
            commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress());

#if RTAO_RAY_SORT_1DRAYTRACE
            DispatchRays(m_rayGenShaderTables[RTAORayGenShaderType::AOSortedRays].Get(), m_raytracingWidth * m_raytracingHeight, 1);
#else
            DispatchRays(m_rayGenShaderTables[RTAORayGenShaderType::AOSortedRays].Get(), m_raytracingWidth, m_raytracingHeight);
#endif
        }
    }

    // Transition AO resources to shader resource state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::HitCount].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Coefficient].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::RayHitDistance].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::UAV(AOResources[AOResource::Coefficient].resource.Get()),  // ToDo
            CD3DX12_RESOURCE_BARRIER::UAV(AOResources[AOResource::RayHitDistance].resource.Get()),  // ToDo
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    // Calculate AO ray hit count.
    if (m_calculateRayHitCounts)
    {
        ScopedTimer _prof(L"CalculateAORayHitCount", commandList);
        CalculateRayHitCount();
    }

    PIXEndEvent(commandList);
}

void RTAO::CalculateRayHitCount()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
    auto commandList = m_deviceResources->GetCommandList();

    RWGpuResource* inputResource = &m_AOResources[AOResource::HitCount]; break;

    m_reduceSumKernel.Execute(
        commandList,
        m_cbvSrvUavHeap->GetHeap(),
        frameIndex,
        type,
        inputResource->gpuDescriptorReadAccess,
        &m_numRayGeometryHits[type]);
};

void RTAO::CreateResolutionDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();

    CreateResources();
    m_reduceSumKernel.CreateInputResourceSizeDependentResources(
        device,
        m_cbvSrvUavHeap.get(),
        FrameCount,
        m_GBufferWidth,
        m_GBufferHeight,
        ReduceSumCalculations::Count);
    m_atrousWaveletTransformFilter.CreateInputResourceSizeDependentResources(device, m_cbvSrvUavHeap.get(), m_raytracingWidth, m_raytracingHeight);
}


void RTAO::SetResolution(UINT width, UINT height)
{
    m_raytracingWidth = width;
    m_raytracingHeight = height;

    CreateResolutionDependentResources();
}
