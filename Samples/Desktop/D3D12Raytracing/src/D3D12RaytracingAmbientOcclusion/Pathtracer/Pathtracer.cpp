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
#include "Pathtracer.h"
#include "GameInput.h"
#include "EngineTuning.h"
#include "EngineProfiling.h"
#include "GpuTimeManager.h"
#include "D3D12RaytracingAmbientOcclusion.h"
#include "CompiledShaders\Pathtracer.hlsl.h"

// ToDo prune unused
using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;


namespace GlobalRootSignature {
    namespace Slot {
        enum Enum {
            Output = 0,
            GBufferResources,
            GBufferResourcesIn,
            AOResourcesOut,	// ToDo cleanup, move to local root sigs 
            VisibilityResource,	// ToDo cleanup, move to local root sigs 
            AccelerationStructure,
            SceneConstant,
            MaterialBuffer,
            SampleBuffers,
            EnvironmentMap,
            FilterWeightSum,
            GBufferDepth,   // ToDo move to the above slot for GBufferResources ?
            GbufferNormalRGB,
            AORayHitDistance,
            AOFrameAge,
            AORayDirectionOriginDepthHitSRV,
            AOSourceToSortedRayIndex,
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
            PartialDepthDerivatives,
#endif
            PrevFrameBottomLevelASIstanceTransforms,
            MotionVector,
            ReprojectedNormalDepth,
            Color,
            AOSurfaceAlbedo,
            Count
        };
    }
}


namespace LocalRootSignature {
    namespace Slot {
        enum Enum {
            ConstantBuffer = 0,
            IndexBuffer,
            VertexBuffer,
            PreviousFrameVertexBuffer,
            DiffuseTexture,
            NormalTexture,
            Count
        };
    }
    struct RootArguments {
        PrimitiveConstantBuffer cb;
        // ToDo add align specifier
        // Bind each resource via a descriptor.
        // This design was picked for simplicity, but one could optimize for shader record size by:
        //    1) Binding multiple descriptors via a range descriptor instead.
        //    2) Storing 4 Byte indices (instead of 8 Byte descriptors) to a global pool resources.
        D3D12_GPU_DESCRIPTOR_HANDLE indexBufferGPUHandle;
        D3D12_GPU_DESCRIPTOR_HANDLE vertexBufferGPUHandle;
        D3D12_GPU_DESCRIPTOR_HANDLE previousFrameVertexBufferGPUHandle;
        D3D12_GPU_DESCRIPTOR_HANDLE diffuseTextureGPUHandle;
        D3D12_GPU_DESCRIPTOR_HANDLE normalTextureGPUHandle;
    };
}

// Shader entry points.
const wchar_t* Pathtracer::c_rayGenShaderNames[] = { L"MyRayGenShader_GBuffer" };
const wchar_t* Pathtracer::c_closestHitShaderNames[] = { L"MyClosestHitShader_GBuffer", L"MyClosestHitShader_ShadowRay" };
const wchar_t* Pathtracer::c_missShaderNames[] = { L"MyMissShader_GBuffer", L"MyMissShader_ShadowRay" };

// Hit groups.
const wchar_t* Pathtracer::c_hitGroupNames[] = { L"MyHitGroup_Triangle_GBuffer", L"MyHitGroup_Triangle_ShadowRay" };

// Singleton instance.
Pathtracer* g_pPathracer = nullptr;

Pathtracer* instance()
{
    return g_pPathracer;
}

void OnRecreateRTAORaytracingResources(void*)
{
    g_pPathracer->RequestRecreateRaytracingResources();
}

void OnRecreateSampleRaytracingResources(void*)
{
    Sample::instance().RequestRecreateRaytracingResources();
}

namespace Pathtracer_Args
{
    // ToDo Reorganize UI, cleanup obsolete.

    // Default ambient intensity for hitPositions that don't have a calculated Ambient coefficient.
    // Calculating AO just for a single hitPosition per pixel can cause visible visual differences
    // in bounces off surfaces that have non-zero Albedo, such as reflection on car paint at sharp angles. 
    // With default Ambient coefficient added to every hit along the ray, the visual difference is decreased.
    NumVar DefaultAmbientIntensity(L"Render/PathTracing/Default ambient intensity", 0.4f, 0, 1, 0.01f);

    IntVar MaxRadianceRayRecursionDepth(L"Render/PathTracing/Max Radiance Ray recursion depth", 3, 1, MAX_RAY_RECURSION_DEPTH, 1);   // ToDo Replace with 3/4 depth as it adds visible differences on spaceship/car
    IntVar MaxShadowRayRecursionDepth(L"Render/PathTracing/Max Shadow Ray recursion depth", 4, 1, MAX_RAY_RECURSION_DEPTH, 1);

    // Avoid tracing rays where they have close to zero visual impact.
    // todo test perf gain or remove.
    // ToDo remove RTAO from name
    NumVar RTAO_minimumFrBounceCoefficient(L"Render/PathTracing/Minimum BRDF bounce contribution coefficient", 0.03f, 0, 1.01f, 0.01f);        // Minimum BRDF coefficient to cast a ray for.
    NumVar RTAO_minimumFtBounceCoefficient(L"Render/PathTracing/Minimum BTDF bounce contribution coefficient", 0.00f, 0, 1.01f, 0.01f);        // Minimum BTDF coefficient to cast a ray for.
   
    BoolVar RTAOUseNormalMaps(L"Render/PathTracing/Normal maps", false);
    const WCHAR* FloatingPointFormatsRG[TextureResourceFormatRG::Count] = { L"R32G32_FLOAT", L"R16G16_FLOAT", L"R8G8_SNORM" };
    // ToDo  ddx needs to be in normalized to use UNORM.
    EnumVar RTAO_PartialDepthDerivativesResourceFormat(L"Render/Texture Formats/PartialDepthDerivatives", TextureResourceFormatRG::R16G16_FLOAT, TextureResourceFormatRG::Count, FloatingPointFormatsRG, Sample::OnRecreateRaytracingResources);
    EnumVar RTAO_MotionVectorResourceFormat(L"Render/Texture Formats/AO/RTAO/Temporal Supersampling/Motion Vector", TextureResourceFormatRG::R16G16_FLOAT, TextureResourceFormatRG::Count, FloatingPointFormatsRG, Sample::OnRecreateRaytracingResources);
}

GpuResource (&GBufferResources(bool getQuarterResResources))[GBufferResource::Count]
{
    return g_pPathracer->GBufferResources(getQuarterResResources);
}


GpuResource(&Pathtracer::GBufferResources(bool getQuarterResResources))[GBufferResource::Count]
{
    if (getQuarterResResources)
        return m_GBufferQuarterResResources;
    else
        return m_GBufferResources;
}
        

    
Pathtracer::Pathtracer()
{
    ThrowIfFalse(g_pPathracer == nullptr, L"There can be only one Pathtracer instance.");
    g_pPathracer = this;

    for (auto& rayGenShaderTableRecordSizeInBytes : m_rayGenShaderTableRecordSizeInBytes)
    {
        rayGenShaderTableRecordSizeInBytes = UINT_MAX;
    }
}

void Pathtracer::Setup(shared_ptr<DeviceResources> deviceResources, shared_ptr<DX::DescriptorHeap> descriptorHeap, Scene& scene)
{
    m_deviceResources = deviceResources;
    m_cbvSrvUavHeap = descriptorHeap;

    CreateDeviceDependentResources(scene);
}

void Pathtracer::Release()
{ 
    // ToDo 

    m_dxrStateObject.Reset();

    m_raytracingGlobalRootSignature.Reset();
    m_raytracingLocalRootSignature.Reset();

    ResetComPtrArray(&m_rayGenShaderTables);
    m_missShaderTable.Reset();
    m_hitGroupShaderTable.Reset();
}

// Create resources that depend on the device.
void Pathtracer::CreateDeviceDependentResources(Scene& scene)
{
    CreateAuxilaryDeviceResources();

    // Initialize raytracing pipeline.

    // Create root signatures for the shaders.
    CreateRootSignatures();

    // Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
    CreateRaytracingPipelineStateObject();

    // Create constant buffers for the geometry and the scene.
    CreateConstantBuffers();

    // Build shader tables, which define shaders and their local root arguments.
    BuildShaderTables(scene);
}


// ToDo rename
void Pathtracer::CreateAuxilaryDeviceResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    auto commandList = m_deviceResources->GetCommandList();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    m_calculatePartialDerivativesKernel.Initialize(device, FrameCount);
    m_downsampleGBufferBilateralFilterKernel.Initialize(device, GpuKernels::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::FilterDepthAware2x2, FrameCount);

    // Create null resource descriptor for the unused second VB in non-animated geometry.
    D3D12_CPU_DESCRIPTOR_HANDLE nullCPUhandle;
    UINT nullHeapIndex = UINT_MAX;
    CreateBufferSRV(nullptr, device, 0, sizeof(VertexPositionNormalTextureTangent), m_cbvSrvUavHeap.get(), &nullCPUhandle, &m_nullVertexBufferGPUhandle, &nullHeapIndex);

}

// Create constant buffers.
void Pathtracer::CreateConstantBuffers()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    m_CB.Create(device, FrameCount, L"Pathtracer Constant Buffer");
}


void Pathtracer::CreateRootSignatures()
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
        ranges[Slot::AORayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 15);  // 1 output ray hit distance texture
        ranges[Slot::MotionVector].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 17);  // 1 output texture space motion vector.
        ranges[Slot::ReprojectedNormalDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 18);  // 1 output texture reprojected hit position
        ranges[Slot::Color].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 19);  // 1 output texture shaded color
        ranges[Slot::AOSurfaceAlbedo].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 20);  // 1 output texture AO diffuse


#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        ranges[Slot::PartialDepthDerivatives].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 16);  // 1 output partial depth derivative texture
#endif
        ranges[Slot::GBufferResourcesIn].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 5);  // 4 input GBuffer textures
        ranges[Slot::EnvironmentMap].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12);  // 1 input environment map texture
        ranges[Slot::FilterWeightSum].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 13);  // 1 input filter weight sum texture
        ranges[Slot::AOFrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 14);  // 1 input AO frame age

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
        rootParameters[Slot::FilterWeightSum].InitAsDescriptorTable(1, &ranges[Slot::FilterWeightSum]);
        rootParameters[Slot::AORayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::AORayHitDistance]);
        rootParameters[Slot::AOFrameAge].InitAsDescriptorTable(1, &ranges[Slot::AOFrameAge]);
        rootParameters[Slot::AORayDirectionOriginDepthHitSRV].InitAsDescriptorTable(1, &ranges[Slot::AORayDirectionOriginDepthHitSRV]);
        rootParameters[Slot::AOSourceToSortedRayIndex].InitAsDescriptorTable(1, &ranges[Slot::AOSourceToSortedRayIndex]);
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        rootParameters[Slot::PartialDepthDerivatives].InitAsDescriptorTable(1, &ranges[Slot::PartialDepthDerivatives]);
#endif
        rootParameters[Slot::MotionVector].InitAsDescriptorTable(1, &ranges[Slot::MotionVector]);
        rootParameters[Slot::ReprojectedNormalDepth].InitAsDescriptorTable(1, &ranges[Slot::ReprojectedNormalDepth]);
        rootParameters[Slot::Color].InitAsDescriptorTable(1, &ranges[Slot::Color]);
        rootParameters[Slot::AOSurfaceAlbedo].InitAsDescriptorTable(1, &ranges[Slot::AOSurfaceAlbedo]);

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
            using namespace LocalRootSignature;

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
            SerializeAndCreateRootSignature(device, localRootSignatureDesc, &m_raytracingLocalRootSignature, L"Local root signature");
        }
    }
}


// DXIL library
// This contains the shaders and their entrypoints for the state object.
// Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
void Pathtracer::CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    auto lib = raytracingPipeline->CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void*)g_pPathtracer, ARRAYSIZE(g_pPathtracer));
    lib->SetDXILLibrary(&libdxil);
    // Use default shader exports for a DXIL library/collection subobject ~ surface all shaders.
}

// Hit groups
// A hit group specifies closest hit, any hit and intersection shaders 
// to be executed when a ray intersects the geometry.
void Pathtracer::CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    // Triangle geometry hit groups
    {
        for (UINT rayType = 0; rayType < RayType::Count; rayType++)
        {
            auto hitGroup = raytracingPipeline->CreateSubobject<CD3DX12_HIT_GROUP_SUBOBJECT>();

            if (c_closestHitShaderNames[rayType])
            {

                hitGroup->SetClosestHitShaderImport(c_closestHitShaderNames[rayType]);
            }
            hitGroup->SetHitGroupExport(c_hitGroupNames[rayType]);
            hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
        }
    }
}

// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void Pathtracer::CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
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
        rootSignatureAssociation->AddExports(c_hitGroupNames);
    }
}

// Create a raytracing pipeline state object (RTPSO).
// An RTPSO represents a full set of shaders reachable by a DispatchRays() call,
// with all configuration options resolved, such as local signatures and other state.
void Pathtracer::CreateRaytracingPipelineStateObject()
{
    auto device = m_deviceResources->GetD3DDevice();
    // Pathracing state object.
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

        // DXIL library
        CreateDxilLibrarySubobject(&raytracingPipeline);

        // Hit groups
        CreateHitGroupSubobjects(&raytracingPipeline);

        // Shader config
        // Defines the maximum sizes in bytes for the ray rayPayload and attribute structure.
        auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
        UINT payloadSize = static_cast<UINT>(max(max(sizeof(RayPayload), sizeof(ShadowRayPayload)), sizeof(GBufferRayPayload)));		// ToDo revise

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
        UINT maxRecursionDepth = MAX_RAY_RECURSION_DEPTH;
        pipelineConfig->Config(maxRecursionDepth);

        PrintStateObjectDesc(raytracingPipeline);

        // Create the state object.
        ThrowIfFailed(device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_dxrStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
    }
}

// Build shader tables.
// This encapsulates all shader records - shaders and the arguments for their local root signatures.
void Pathtracer::BuildShaderTables(Scene& scene)
{
    auto device = m_deviceResources->GetD3DDevice();

    void* rayGenShaderIDs[RayGenShaderType::Count];
    void* missShaderIDs[RayType::Count];
    void* hitGroupShaderIDs_TriangleGeometry[RayType::Count];

    // A shader name look-up table for shader table debug print out.
    unordered_map<void*, wstring> shaderIdToStringMap;

    auto GetShaderIDs = [&](auto* stateObjectProperties)
    {
        for (UINT i = 0; i < RayGenShaderType::Count; i++)
        {
            rayGenShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(c_rayGenShaderNames[i]);
            shaderIdToStringMap[rayGenShaderIDs[i]] = c_rayGenShaderNames[i];
        }

        for (UINT i = 0; i < RayType::Count; i++)
        {
            missShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(c_missShaderNames[i]);
            shaderIdToStringMap[missShaderIDs[i]] = c_missShaderNames[i];
        }

        for (UINT i = 0; i < RayType::Count; i++)
        {
            hitGroupShaderIDs_TriangleGeometry[i] = stateObjectProperties->GetShaderIdentifier(c_hitGroupNames[i]);
            shaderIdToStringMap[hitGroupShaderIDs_TriangleGeometry[i]] = c_hitGroupNames[i];
        }
    };

    // Get shader identifiers.
    UINT shaderIDSize;
    ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
    ThrowIfFailed(m_dxrStateObject.As(&stateObjectProperties));
    GetShaderIDs(stateObjectProperties.Get());
    shaderIDSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;

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
        UINT shaderRecordSize = shaderIDSize;

        for (UINT i = 0; i < RayGenShaderType::Count; i++)
        {
            ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable");
            rayGenShaderTable.push_back(ShaderRecord(rayGenShaderIDs[i], shaderIDSize, nullptr, 0));
            rayGenShaderTable.DebugPrint(shaderIdToStringMap);
            m_rayGenShaderTables[i] = rayGenShaderTable.GetResource();
        }
    }

    // Miss shader table.
    {
        UINT numShaderRecords = RayType::Count;
        UINT shaderRecordSize = shaderIDSize; // No root arguments

        ShaderTable missShaderTable(device, numShaderRecords, shaderRecordSize, L"MissShaderTable");
        for (UINT i = 0; i < RayType::Count; i++)
        {
            missShaderTable.push_back(ShaderRecord(missShaderIDs[i], shaderIDSize, nullptr, 0));
        }
        missShaderTable.DebugPrint(shaderIdToStringMap);
        m_missShaderTableStrideInBytes = missShaderTable.GetShaderRecordSize();
        m_missShaderTable = missShaderTable.GetResource();
    }

    // ToDo remove
    vector<vector<GeometryInstance>*> geometryInstancesArray;

    // ToDo split shader table per unique pass?
    // Hit group shader table.
    {
        auto& bottomLevelASGeometries = scene.BottomLevelASGeometries();
        auto& accelerationStructure = *scene.AccelerationStructure();
        auto& grassPatchVB = scene.GrassPatchVB();

        UINT numShaderRecords = 0;
        for (auto& bottomLevelASGeometryPair : bottomLevelASGeometries)
        {
            auto& bottomLevelASGeometry = bottomLevelASGeometryPair.second;
            numShaderRecords += static_cast<UINT>(bottomLevelASGeometry.m_geometryInstances.size()) * RayType::Count;
        }
        UINT numGrassGeometryShaderRecords = 2 * UIParameters::NumGrassGeometryLODs * 3 * RayType::Count;
        numShaderRecords += numGrassGeometryShaderRecords;

        UINT shaderRecordSize = shaderIDSize + sizeof(LocalRootSignature::RootArguments);
        ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");

        // Triangle geometry hit groups.
        for (auto& bottomLevelASGeometryPair : bottomLevelASGeometries)
        {
            auto& bottomLevelASGeometry = bottomLevelASGeometryPair.second;
            auto& name = bottomLevelASGeometry.GetName();

            UINT shaderRecordOffset = hitGroupShaderTable.GeNumShaderRecords();
            accelerationStructure.GetBottomLevelAS(bottomLevelASGeometryPair.first).SetInstanceContributionToHitGroupIndex(shaderRecordOffset);

            // ToDo cleaner?
            // Grass Patch LOD shader recods
            if (name.find(L"Grass Patch LOD") != wstring::npos)
            {
                UINT LOD = stoi(name.data() + wcsnlen_s(L"Grass Patch LOD", 15));

                // ToDo remove assert
                assert(bottomLevelASGeometry.m_geometryInstances.size() == 1);
                auto& geometryInstance = bottomLevelASGeometry.m_geometryInstances[0];

                LocalRootSignature::RootArguments rootArgs;
                rootArgs.cb.materialID = geometryInstance.materialID;
                rootArgs.cb.isVertexAnimated = geometryInstance.isVertexAnimated;

                memcpy(&rootArgs.indexBufferGPUHandle, &geometryInstance.ib.gpuDescriptorHandle, sizeof(geometryInstance.ib.gpuDescriptorHandle));
                memcpy(&rootArgs.diffuseTextureGPUHandle, &geometryInstance.diffuseTexture, sizeof(geometryInstance.diffuseTexture));
                memcpy(&rootArgs.normalTextureGPUHandle, &geometryInstance.normalTexture, sizeof(geometryInstance.normalTexture));

                // Dynamic geometry with multiple LODs is handled by creating shader records
                // for all cases. Then, on geometry/instance updates, a BLAS instance updates
                // its InstanceContributionToHitGroupIndex to point to the corresponding 
                // shader records for that LOD. 
                // 
                // The LOD selection can change from a frame to frame depending on distance
                // to the camera. For simplicity, we assume the LOD difference from frame to frame 
                // is no greater than 1. This can be false if camera moves fast, but in that case 
                // temporal reprojection would fail for the most part anyway yielding diminishing returns.
                // Consistency checks will prevent blending in from false geometry.
                //
                // Given multiple LODs and LOD delta being 1 at most, we create the records as follows:
                // 2 * 3 Shader Records per LOD
                //  2 - ping-pong frame to frame
                //  3 - transition types
                //      Transition from lower LOD in previous frame
                //      Same LOD as previous frame
                //      Transition from higher LOD in previous frame

                struct VertexBufferHandles {
                    D3D12_GPU_DESCRIPTOR_HANDLE prevFrameVertexBuffer;
                    D3D12_GPU_DESCRIPTOR_HANDLE vertexBuffer;
                };

                VertexBufferHandles vbHandles[2][3];
                for (UINT frameID = 0; frameID < 2; frameID++)
                {
                    UINT prevFrameID = (frameID + 1) % 2;
                   

                    // Transitioning from lower LOD.
                    vbHandles[frameID][0].vertexBuffer = grassPatchVB[LOD][frameID].gpuDescriptorReadAccess;
                    vbHandles[frameID][0].prevFrameVertexBuffer = LOD > 0 ? grassPatchVB[LOD - 1][prevFrameID].gpuDescriptorReadAccess
                        : grassPatchVB[LOD][prevFrameID].gpuDescriptorReadAccess;

                    // Same LOD as previous frame.
                    vbHandles[frameID][1].vertexBuffer = grassPatchVB[LOD][frameID].gpuDescriptorReadAccess;
                    vbHandles[frameID][1].prevFrameVertexBuffer = grassPatchVB[LOD][prevFrameID].gpuDescriptorReadAccess;

                    // Transitioning from higher LOD.
                    vbHandles[frameID][2].vertexBuffer = grassPatchVB[LOD][frameID].gpuDescriptorReadAccess;
                    vbHandles[frameID][2].prevFrameVertexBuffer = LOD < UIParameters::NumGrassGeometryLODs - 1 ? grassPatchVB[LOD + 1][prevFrameID].gpuDescriptorReadAccess
                        : grassPatchVB[LOD][prevFrameID].gpuDescriptorReadAccess;
                }

                for (UINT frameID = 0; frameID < 2; frameID++)
                    for (UINT transitionType = 0; transitionType < 3; transitionType++)
                    {
                        memcpy(&rootArgs.vertexBufferGPUHandle, &vbHandles[frameID][transitionType].vertexBuffer, sizeof(vbHandles[frameID][transitionType].vertexBuffer));
                        memcpy(&rootArgs.previousFrameVertexBufferGPUHandle, &vbHandles[frameID][transitionType].prevFrameVertexBuffer, sizeof(vbHandles[frameID][transitionType].prevFrameVertexBuffer));

                        for (auto& hitGroupShaderID : hitGroupShaderIDs_TriangleGeometry)
                        {
                            hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderID, shaderIDSize, &rootArgs, sizeof(rootArgs)));
                        }
                    }
            }
            else // Non-vertex buffer animated geometry with 1 shader record per ray-type per bottom-level AS
            {
                for (auto& geometryInstance : bottomLevelASGeometry.m_geometryInstances)
                {
                    LocalRootSignature::RootArguments rootArgs;
                    rootArgs.cb.materialID = geometryInstance.materialID;
                    rootArgs.cb.isVertexAnimated = geometryInstance.isVertexAnimated;

                    memcpy(&rootArgs.indexBufferGPUHandle, &geometryInstance.ib.gpuDescriptorHandle, sizeof(geometryInstance.ib.gpuDescriptorHandle));
                    memcpy(&rootArgs.vertexBufferGPUHandle, &geometryInstance.vb.gpuDescriptorHandle, sizeof(geometryInstance.ib.gpuDescriptorHandle));
                    memcpy(&rootArgs.previousFrameVertexBufferGPUHandle, &m_nullVertexBufferGPUhandle, sizeof(m_nullVertexBufferGPUhandle));
                    memcpy(&rootArgs.diffuseTextureGPUHandle, &geometryInstance.diffuseTexture, sizeof(geometryInstance.diffuseTexture));
                    memcpy(&rootArgs.normalTextureGPUHandle, &geometryInstance.normalTexture, sizeof(geometryInstance.normalTexture));


                    for (auto& hitGroupShaderID : hitGroupShaderIDs_TriangleGeometry)
                    {
                        hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderID, shaderIDSize, &rootArgs, sizeof(rootArgs)));
                    }
                }
            }
        }
        hitGroupShaderTable.DebugPrint(shaderIdToStringMap);
        m_hitGroupShaderTableStrideInBytes = hitGroupShaderTable.GetShaderRecordSize();
        m_hitGroupShaderTable = hitGroupShaderTable.GetResource();
    }
}

void Pathtracer::DispatchRays(ID3D12Resource* rayGenShaderTable, UINT width, UINT height)
{
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
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
    dispatchDesc.Width = width != 0 ? width : m_width;
    dispatchDesc.Height = height != 0 ? height : m_height;
    dispatchDesc.Depth = 1;
    commandList->SetPipelineState1(m_dxrStateObject.Get());

    resourceStateTracker->FlushResourceBarriers();
    commandList->DispatchRays(&dispatchDesc);
}

void Pathtracer::SetCamera(const GameCore::Camera& camera)
{
    XMMATRIX view, proj;
    camera.GetProj(&proj, m_quarterResWidth, m_quarterResHeight);

    // Calculate view matrix as if the camera was at (0,0,0) to avoid 
    // precision issues when camera position is too far from (0,0,0).
    // GenerateCameraRay takes this into consideration in the raytracing shader.
    view = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 1), XMVectorSetW(camera.At() - camera.Eye(), 1), camera.Up());
    XMMATRIX viewProj = view * proj;
    m_CB->projectionToWorldWithCameraEyeAtOrigin = XMMatrixInverse(nullptr, viewProj);
    m_CB->Znear = camera.ZMin;
    m_CB->Zfar = camera.ZMax;
}

void Pathtracer::UpdateConstantBuffer(Scene& scene)
{
    // ToDo
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


    XMStoreFloat3(&m_CB->lightPosition, scene.m_lightPosition);
    m_CB->lightColor = scene.m_lightColor;

    // ToDo move
    m_CB->maxRadianceRayRecursionDepth = Pathtracer_Args::MaxRadianceRayRecursionDepth;
    m_CB->maxShadowRayRecursionDepth = Pathtracer_Args::MaxShadowRayRecursionDepth;
    m_CB->useNormalMaps = Pathtracer_Args::RTAOUseNormalMaps;
    m_CB->defaultAmbientIntensity = Pathtracer_Args::DefaultAmbientIntensity;
}

void Pathtracer::Run(Scene& scene)
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();        // ToDo rename to Backbuffer index

    ScopedTimer _prof(L"GenerateGbuffer", commandList);
    UpdateConstantBuffer(scene);

    auto& Scene = scene;
    auto& MaterialBuffer = Scene.MaterialBuffer();
    auto& EnvironmentMap = Scene.EnvironmentMap();
    auto& PrevFrameBottomLevelASInstanceTransforms = Scene.PrevFrameBottomLevelASInstanceTransforms();

    m_CB->useDiffuseFromMaterial = Composition_Args::CompositionMode == CompositionType::Diffuse;

    // ToDo should we use cameraAtPosition0 too and offset the world space pos vector in the shader?
    auto& prevFrameCamera = scene.PrevFrameCamera();
    XMMATRIX prevView, prevProj;
    prevFrameCamera.GetViewProj(&prevView, &prevProj, m_width, m_height);
    m_CB->prevViewProj = prevView * prevProj;
    XMStoreFloat3(&m_CB->prevCameraPosition, prevFrameCamera.Eye());

    // ToDo cleanup
    XMMATRIX prevView0 = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 1), XMVectorSetW(prevFrameCamera.At() - prevFrameCamera.Eye(), 1), prevFrameCamera.Up());
    XMMATRIX viewProj0 = prevView0 * prevProj;
    m_CB->prevProjToWorldWithCameraEyeAtOrigin = XMMatrixInverse(nullptr, viewProj0);

    commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
    commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

    // Copy dynamic buffers to GPU.
    {
        // ToDo copy on change
        m_CB.CopyStagingToGpu(frameIndex);
    }

    // ToDo move this/part(AO,..) of transitions out?
    // Transition all output resources to UAV state.
    {
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Hit], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Material], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::HitPosition], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::SurfaceNormalDepth], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Distance], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Depth], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::PartialDepthDerivatives], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::MotionVector], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::ReprojectedNormalDepth], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Color], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::AOSurfaceAlbedo], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }


    // Bind inputs.
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, scene.AccelerationStructure()->GetTopLevelASResource()->GetGPUVirtualAddress());
    commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_CB.GpuVirtualAddress(frameIndex));
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::MaterialBuffer, Scene.MaterialBuffer().GpuVirtualAddress());
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::EnvironmentMap, EnvironmentMap.gpuDescriptorHandle);
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::PrevFrameBottomLevelASIstanceTransforms, PrevFrameBottomLevelASInstanceTransforms.GpuVirtualAddress(frameIndex));


    // Bind output RTs.
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResources, m_GBufferResources[0].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferDepth, m_GBufferResources[GBufferResource::Depth].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::MotionVector, m_GBufferResources[GBufferResource::MotionVector].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::ReprojectedNormalDepth, m_GBufferResources[GBufferResource::ReprojectedNormalDepth].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::Color, m_GBufferResources[GBufferResource::Color].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOSurfaceAlbedo, m_GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorWriteAccess);

#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::PartialDepthDerivatives, m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess);
#endif	
    // Dispatch Rays.
    DispatchRays(m_rayGenShaderTables[RayGenShaderType::GBuffer].Get());

    // Transition GBuffer resources to shader resource state.
    {
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Hit], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Material], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::HitPosition], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::SurfaceNormalDepth], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Distance], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Depth], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::PartialDepthDerivatives], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
#endif
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::MotionVector], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::ReprojectedNormalDepth], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::Color], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::AOSurfaceAlbedo], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

#if 0  // ToDo Remove?
    // Calculate ray hit counts.
    {
        ScopedTimer _prof(L"CalculateCameraRayHitCount", commandList);
        CalculateCameraRayHitCount();
    }
#endif

#if !CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
    // Calculate partial derivatives.
    {
        ScopedTimer _prof(L"Calculate Partial Depth Derivatives", commandList);
        resourceStateTracker->FlushResourceBarriers();
        m_calculatePartialDerivativesKernel.Run(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            m_width,
            m_height,
            m_GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess,
            m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess);

        resourceStateTracker->TransitionResource(&m_GBufferResources[GBufferResource::PartialDepthDerivatives], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
#endif
    if (RTAO_Args::QuarterResAO)
    {
        DownsampleGBuffer();
    }
}

// ToDo remove
#if 0
void Pathtracer::CalculateRayHitCount()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

    // ToDo make this disabled by default/

    resourceStateTracker->FlushResourceBarriers();
    m_reduceSumKernel.Run(
        commandList,
        m_cbvSrvUavHeap->GetHeap(),
        frameIndex,
        m_GBufferResources[GBufferResource::Hit].gpuDescriptorReadAccess,
        &m_numAORayGeometryHits);
}

void Pathtracer::CreateResolutionDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    CreateTextureResources();
    m_reduceSumKernel.CreateInputResourceSizeDependentResources(
        device,
        m_cbvSrvUavHeap.get(),
        FrameCount,
        m_width,
        m_height);
}
#endif

void Pathtracer::CreateResolutionDependentResources()
{
    CreateTextureResources();
}

void Pathtracer::SetResolution(UINT GBufferWidth, UINT GBufferHeight, UINT RTAOWidth, UINT RTAOHeight)
{
    m_width = GBufferWidth;
    m_height = GBufferHeight;
    m_quarterResWidth = RTAOWidth;
    m_quarterResHeight = RTAOHeight;

    CreateResolutionDependentResources();
}

void Pathtracer::CreateTextureResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

    // ToDo move depth out of normal resource and switch normal to 16bit precision
    DXGI_FORMAT hitPositionFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;// DXGI_FORMAT_R16G16B16A16_FLOAT; // ToDo change to 16bit? or encode as 64bits

    DXGI_FORMAT debugFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;// DXGI_FORMAT_R32G32B32A32_FLOAT;
    // ToDo tune formats
    D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    // ToDo remove obsolete resources, QuarterResAO event triggers this so we may not need all low/gbuffer width AO resources.

    // Full-res GBuffer resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_GBufferResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        m_GBufferResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        for (UINT i = 0; i < GBufferResource::Count; i++)
        {
            m_GBufferResources[i].uavDescriptorHeapIndex = m_GBufferResources[0].uavDescriptorHeapIndex + i;
            m_GBufferResources[i].srvDescriptorHeapIndex = m_GBufferResources[0].srvDescriptorHeapIndex + i;
        }
        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Hit], initialResourceState, L"GBuffer Hit");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_UINT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Material], initialResourceState, L"GBuffer Material");


        CreateRenderTargetResource(device, hitPositionFormat, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::HitPosition], initialResourceState, L"GBuffer HitPosition");

        CreateRenderTargetResource(device, COMPACT_NORMAL_DEPTH_DXGI_FORMAT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::SurfaceNormalDepth], initialResourceState, L"GBuffer Normal Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Distance], initialResourceState, L"GBuffer Distance");
        CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Depth], initialResourceState, L"GBuffer Depth");

        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(Pathtracer_Args::RTAO_PartialDepthDerivativesResourceFormat), m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::PartialDepthDerivatives], initialResourceState, L"GBuffer Partial Depth Derivatives");

        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(Pathtracer_Args::RTAO_MotionVectorResourceFormat), m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::MotionVector], initialResourceState, L"GBuffer Texture Space Motion Vector");

        CreateRenderTargetResource(device, COMPACT_NORMAL_DEPTH_DXGI_FORMAT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::ReprojectedNormalDepth], initialResourceState, L"GBuffer Reprojected Hit Position");

        CreateRenderTargetResource(device, backbufferFormat, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Color], initialResourceState, L"GBuffer Color");

        CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::AOSurfaceAlbedo], initialResourceState, L"GBuffer AO Surface Albedo");
    }

    // Low-res GBuffer resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_GBufferQuarterResResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        m_GBufferQuarterResResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        for (UINT i = 0; i < GBufferResource::Count; i++)
        {
            m_GBufferQuarterResResources[i].uavDescriptorHeapIndex = m_GBufferQuarterResResources[0].uavDescriptorHeapIndex + i;
            m_GBufferQuarterResResources[i].srvDescriptorHeapIndex = m_GBufferQuarterResResources[0].srvDescriptorHeapIndex + i;
        }

        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::Hit], initialResourceState, L"GBuffer LowRes Hit");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_UINT, m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::Material], initialResourceState, L"GBuffer LowRes Material");
        CreateRenderTargetResource(device, hitPositionFormat, m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::HitPosition], initialResourceState, L"GBuffer LowRes HitPosition");
        CreateRenderTargetResource(device, COMPACT_NORMAL_DEPTH_DXGI_FORMAT, m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::SurfaceNormalDepth], initialResourceState, L"GBuffer LowRes Normal");
        CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::Distance], initialResourceState, L"GBuffer LowRes Distance");
        // ToDo are below two used?
        CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::Depth], initialResourceState, L"GBuffer LowRes Depth");
        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(Pathtracer_Args::RTAO_PartialDepthDerivativesResourceFormat), m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::PartialDepthDerivatives], initialResourceState, L"GBuffer LowRes Partial Depth Derivatives");

        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(Pathtracer_Args::RTAO_MotionVectorResourceFormat), m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::MotionVector], initialResourceState, L"GBuffer LowRes Texture Space Motion Vector");

        CreateRenderTargetResource(device, COMPACT_NORMAL_DEPTH_DXGI_FORMAT, m_quarterResWidth, m_quarterResHeight, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::ReprojectedNormalDepth], initialResourceState, L"GBuffer LowRes Reprojected Normal Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_width, m_height, m_cbvSrvUavHeap.get(), &m_GBufferQuarterResResources[GBufferResource::AOSurfaceAlbedo], initialResourceState, L"GBuffer LowRes AO Surface Albedo");

    }
}

void Pathtracer::DownsampleGBuffer()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

    // ToDo move this/part(AO,..) of transitions out?
    // Transition all output resources to UAV state.
    {
        // ToDo move these to the kernels?
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::Hit], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::HitPosition], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::PartialDepthDerivatives], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::MotionVector], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::ReprojectedNormalDepth], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::Depth], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::SurfaceNormalDepth], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::AOSurfaceAlbedo], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    ScopedTimer _prof(L"DownsampleGBuffer", commandList);
    resourceStateTracker->FlushResourceBarriers();
    // ToDo split into per resource downsamples?
    m_downsampleGBufferBilateralFilterKernel.Run(
        commandList,
        m_width,
        m_height,
        m_cbvSrvUavHeap->GetHeap(),
        m_GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::HitPosition].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::Hit].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::MotionVector].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::ReprojectedNormalDepth].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::Depth].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorReadAccess,
        m_GBufferQuarterResResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorWriteAccess,
        m_GBufferQuarterResResources[GBufferResource::HitPosition].gpuDescriptorWriteAccess,
        m_GBufferQuarterResResources[GBufferResource::Hit].gpuDescriptorWriteAccess,
        m_GBufferQuarterResResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess,
        m_GBufferQuarterResResources[GBufferResource::MotionVector].gpuDescriptorWriteAccess,
        m_GBufferQuarterResResources[GBufferResource::ReprojectedNormalDepth].gpuDescriptorWriteAccess,
        m_GBufferQuarterResResources[GBufferResource::Depth].gpuDescriptorWriteAccess,
        m_GBufferQuarterResResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorWriteAccess);

    // Transition GBuffer resources to shader resource state.
    {
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::Hit], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::HitPosition], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::SurfaceNormalDepth], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::PartialDepthDerivatives], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::MotionVector], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::ReprojectedNormalDepth], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::Depth], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_GBufferQuarterResResources[GBufferResource::AOSurfaceAlbedo], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
}