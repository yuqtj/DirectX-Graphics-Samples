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
    const wchar_t* RTAO::c_rayGenShaderNames[] = { L"RayGenShader", L"RayGenShader_sortedRays" };
    const wchar_t* RTAO::c_closestHitShaderName = L"ClosestHitShader";
    const wchar_t* RTAO::c_missShaderName = L"MissShader";

    // Hit groups.
    const wchar_t* RTAO::c_hitGroupName = L"HitGroup_Triangle";

    // ToDo dedupe with below
    void OnRecreateRTAORaytracingResources(void*)
    {
        Sample::instance().RTAO().RequestRecreateRaytracingResources();
    }

    void OnRecreateSampleRaytracingResources(void*)
    {
        Sample::instance().RTAO().RequestRecreateRaytracingResources();
    }

    void OnRecreateSamples(void*)
    {
        Sample::instance().RTAO().RequestRecreateAOSamples();
    }
       
    namespace RTAO_Args
    {

        IntVar AOTileX(L"Render/AO/Tile X", 1, 1, 128, 1);
        IntVar AOTileY(L"Render/AO/Tile Y", 1, 1, 128, 1);

        BoolVar RTAOUseRaySorting(L"Render/AO/RTAO/Ray Sorting/Enabled", true);
        NumVar RTAORayBinDepthSizeMultiplier(L"Render/AO/RTAO/Ray Sorting/Ray bin depth size (multiplier of MaxRayHitTime)", 0.1f, 0.01f, 10.f, 0.01f);
        BoolVar RTAORaySortingUseOctahedralRayDirectionQuantization(L"Render/AO/RTAO/Ray Sorting/Octahedral ray direction quantization", true);


        const WCHAR* RayGenAdaptiveQuadSizeTypes[GpuKernels::AdaptiveRayGenerator::AdaptiveQuadSizeType::Count] = { L"1x1", L"2x2", L"4x4" };
        EnumVar RTAORayGenAdaptiveQuadSize(L"Render/AO/RTAO/Ray Sorting/Adaptive Ray Gen/Ray Count Pixel Window", GpuKernels::AdaptiveRayGenerator::AdaptiveQuadSizeType::Quad2x2, GpuKernels::AdaptiveRayGenerator::AdaptiveQuadSizeType::Count, RayGenAdaptiveQuadSizeTypes);
        IntVar RTAORayGen_MaxFrameAge(L"Render/AO/RTAO/Ray Sorting/Adaptive Ray Gen/Max frame age", 32, 1, 32, 1); // ToDo link this to smoothing factor?
        IntVar RTAORayGen_MinAdaptiveFrameAge(L"Render/AO/RTAO/Ray Sorting/Adaptive Ray Gen/Min frame age for adaptive sampling", 16, 1, 32, 1);
        IntVar RTAORayGen_MaxRaysPerQuad(L"Render/AO/RTAO/Ray Sorting/Adaptive Ray Gen/Max rays per quad", 2, 1, 16, 1);
        IntVar RTAORayGen_MaxFrameAgeToGenerateRaysFor(L"Render/AO/RTAO/Ray Sorting/Adaptive Ray Gen/Max frame age to generate rays for", 32, 1, 64, 1);

        BoolVar RTAORandomFrameSeed(L"Render/AO/RTAO/Random per-frame seed", true);
               

        const WCHAR* FloatingPointFormatsR[TextureResourceFormatR::Count] = { L"R32_FLOAT", L"R16_FLOAT", L"R8_SNORM" };
        EnumVar RTAO_AmbientCoefficientResourceFormat(L"Render/Texture Formats/AO/RTAO/Ambient Coefficient", TextureResourceFormatR::R16_FLOAT, TextureResourceFormatR::Count, FloatingPointFormatsR, OnRecreateSampleRaytracingResources);


        // ToDo cleanup RTAO... vs RTAO_..
        IntVar RTAOAdaptiveSamplingMinSamples(L"Render/AO/RTAO/Adaptive Sampling/Min samples", 1, 1, AO_SPP_N* AO_SPP_N, 1);

        // ToDo remove
        IntVar AOSampleCountPerDimension(L"Render/AO/RTAO/Samples per pixel NxN", AO_SPP_N, 1, AO_SPP_N_MAX, 1, OnRecreateSamples, nullptr);
        IntVar AOSampleSetDistributedAcrossPixels(L"Render/AO/RTAO/Sample set distribution across NxN pixels ", 8, 1, 8, 1, OnRecreateSamples, nullptr);
#if LOAD_PBRT_SCENE
        NumVar RTAOMaxRayHitTime(L"Render/AO/RTAO/Max ray hit time", AO_RAY_T_MAX, 0.0f, 50.0f, 0.2f);
#else
        NumVar RTAOMaxRayHitTime(L"Render/AO/RTAO/Max ray hit time", AO_RAY_T_MAX, 0.0f, 1000.0f, 4);
#endif
        BoolVar RTAOApproximateInterreflections(L"Render/AO/RTAO/Approximate Interreflections/Enabled", true);
        NumVar RTAODiffuseReflectanceScale(L"Render/AO/RTAO/Approximate Interreflections/Diffuse Reflectance Scale", 0.5f, 0.0f, 1.0f, 0.1f);
        NumVar  RTAO_MinimumAmbientIllumination(L"Render/AO/RTAO/Minimum Ambient Illumination", 0.07f, 0.0f, 1.0f, 0.01f);
        BoolVar RTAOIsExponentialFalloffEnabled(L"Render/AO/RTAO/Exponential Falloff", true);
        NumVar RTAO_ExponentialFalloffDecayConstant(L"Render/AO/RTAO/Exponential Falloff Decay Constant", 2.f, 0.0f, 20.f, 0.25f);
        NumVar RTAO_ExponentialFalloffMinOcclusionCutoff(L"Render/AO/RTAO/Exponential Falloff Min Occlusion Cutoff", 0.4f, 0.0f, 1.f, 0.05f);       // ToDo Finetune document perf.
    
        BoolVar QuarterResAO(L"Render/AO/RTAO/Quarter res", true, OnRecreateRaytracingResources, nullptr);
    }

    DXGI_FORMAT RTAO::AOCoefficientFormat()
    {
        return TextureResourceFormatR::ToDXGIFormat(RTAO_Args::RTAO_AmbientCoefficientResourceFormat);
    }

    float RTAO::MaxRayHitTime()
    {
        return RTAO_Args::RTAOMaxRayHitTime;
    }
    void RTAO::SetMaxRayHitTime(float maxRayHitTime)
    {
        return RTAO_Args::RTAOMaxRayHitTime.SetValue(maxRayHitTime);
    }


    RTAO::RTAO()
    {
        for (auto& rayGenShaderTableRecordSizeInBytes : m_rayGenShaderTableRecordSizeInBytes)
        {
            rayGenShaderTableRecordSizeInBytes = UINT_MAX;
        }
        m_generatorURNG.seed(1729);
    }


    void RTAO::Setup(shared_ptr<DeviceResources> deviceResources, shared_ptr<DX::DescriptorHeap> descriptorHeap)
    {
        CreateDeviceDependentResources();
    }

    void RTAO::ReleaseDeviceDependentResources()
    {
        // ToDo 
    }

    // Create resources that depend on the device.
    void RTAO::CreateDeviceDependentResources()
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
        BuildShaderTables();

        // ToDo rename
        CreateSamplesRNG();

#if DEBUG_PRINT_OUT_RTAO_DISPATCH_TIME
        auto device = m_deviceResources->GetD3DDevice();
        dispatchRayTime.RestoreDevice(device, m_deviceResources->GetCommandQueue(), m_deviceResources->GetBackBufferCount());
#endif
    }


    // ToDo rename
    void RTAO::CreateAuxilaryDeviceResources()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto commandQueue = m_deviceResources->GetCommandQueue();
        auto commandList = m_deviceResources->GetCommandList();
        auto FrameCount = m_deviceResources->GetBackBufferCount();

        // ToDo move?
        m_reduceSumKernel.Initialize(device, GpuKernels::ReduceSum::Uint);
        m_rayGen.Initialize(device, FrameCount);
        m_raySorter.Initialize(device, FrameCount);
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
            using namespace RTAOGlobalRootSignature;

            // ToDo use slot index in ranges everywhere
            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count]; // Perfomance TIP: Order from most frequent to least frequent.
                                                          // ToDo reorder
            ranges[Slot::AOResourcesOut].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 10);      // 2 output AO textures
            ranges[Slot::AORayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 15);    // 1 output ray hit distance texture
            ranges[Slot::AORayDirectionOriginDepthHitUAV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 22);  // 1 output AO ray direction and origin depth texture

            ranges[Slot::RayOriginPosition].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7);  // 1 input surface hit position texture
            ranges[Slot::RayOriginSurfaceNormalDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);  // 1 input surface normal depth
            ranges[Slot::FilterWeightSum].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 13);  // 1 input filter weight sum texture
            ranges[Slot::AOFrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 14);  // 1 input AO frame age
            ranges[Slot::AORayDirectionOriginDepthHitSRV].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 22);  // 1 AO ray direction and origin depth texture
            ranges[Slot::AOSortedToSourceRayIndex].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 23);  // 1 input AO ray group thread offsets
            ranges[Slot::AOSurfaceAlbedo].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 24);  // 1 input AO surface diffuse texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::RayOriginPosition].InitAsDescriptorTable(1, &ranges[Slot::RayOriginPosition]);
            rootParameters[Slot::RayOriginSurfaceNormalDepth].InitAsDescriptorTable(1, &ranges[Slot::RayOriginSurfaceNormalDepth]);
            rootParameters[Slot::AOResourcesOut].InitAsDescriptorTable(1, &ranges[Slot::AOResourcesOut]);
            rootParameters[Slot::FilterWeightSum].InitAsDescriptorTable(1, &ranges[Slot::FilterWeightSum]);
            rootParameters[Slot::AORayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::AORayHitDistance]);
            rootParameters[Slot::AOFrameAge].InitAsDescriptorTable(1, &ranges[Slot::AOFrameAge]);
            rootParameters[Slot::AORayDirectionOriginDepthHitSRV].InitAsDescriptorTable(1, &ranges[Slot::AORayDirectionOriginDepthHitSRV]);
            rootParameters[Slot::AOSortedToSourceRayIndex].InitAsDescriptorTable(1, &ranges[Slot::AOSortedToSourceRayIndex]);
            rootParameters[Slot::AORayDirectionOriginDepthHitUAV].InitAsDescriptorTable(1, &ranges[Slot::AORayDirectionOriginDepthHitUAV]);
            rootParameters[Slot::AOSurfaceAlbedo].InitAsDescriptorTable(1, &ranges[Slot::AOSurfaceAlbedo]);

            rootParameters[Slot::AccelerationStructure].InitAsShaderResourceView(0);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);		// ToDo rename to ConstantBuffer
            rootParameters[Slot::SampleBuffers].InitAsShaderResourceView(4);

            CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, globalRootSignatureDesc, &m_raytracingGlobalRootSignature, L"RTAO Global root signature");
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

            // DXIL library
            CreateDxilLibrarySubobject(&raytracingPipeline);

            // Hit groups
            CreateHitGroupSubobjects(&raytracingPipeline);

            // ToDo try 2B float payload
#define AO_4B_RAYPAYLOAD 1
        // Shader config
        // Defines the maximum sizes in bytes for the ray rayPayload and attribute structure.
            auto shaderConfig = raytracingPipeline.CreateSubobject<CD3DX12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
#if AO_4B_RAYPAYLOAD
            UINT payloadSize = static_cast<UINT>(sizeof(ShadowRayPayload));		// ToDo revise
#else
            UINT payloadSize = static_cast<UINT>(max(max(sizeof(RayPayload), sizeof(ShadowRayPayload)), sizeof(GBufferRayPayload)));		// ToDo revise
#endif
            UINT attributeSize = sizeof(XMFLOAT2);  // float2 barycentrics  - ToDo ref the struct directly?
            shaderConfig->Config(payloadSize, attributeSize);

            // Global root signature
            // This is a root signature that is shared across all RTAO shaders invoked during a DispatchRays() call.
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

    void RTAO::CreateTextureResources()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

        // ToDo change this to non-PS resouce since we use CS?
        D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        // ToDo remove obsolete resources, QuarterResAO event triggers this so we may not need all low/gbuffer width AO resources.

        {
            // Preallocate subsequent descriptor indices for both SRV and UAV groups.
            m_AOResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
            m_AOResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
            for (UINT i = 0; i < AOResource::Count; i++)
            {
                m_AOResources[i].uavDescriptorHeapIndex = m_AOResources[0].uavDescriptorHeapIndex + i;
                m_AOResources[i].srvDescriptorHeapIndex = m_AOResources[0].srvDescriptorHeapIndex + i;
            }

            // ToDo pack some resources.

            // ToDo cleanup raytracing resolution - twice for coefficient.
            CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(RTAO_Args::RTAO_AmbientCoefficientResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Coefficient], initialResourceState, L"Render/AO Coefficient");

            // ToDo move outside of RTAO
            CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(RTAO_Args::RTAO_AmbientCoefficientResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Denoised Coefficient");

            // ToDo 8 bit hit count?
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::HitCount], initialResourceState, L"Render/AO Hit Count");

            // ToDo use lower bit float?
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::FilterWeightSum], initialResourceState, L"Render/AO Filter Weight Sum");
            CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::RayHitDistance], initialResourceState, L"Render/AO Hit Distance");
        }


        CreateRenderTargetResource(device, DXGI_FORMAT_R8G8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sourceToSortedRayIndexOffset, initialResourceState, L"Source To Sorted Ray Index");
        CreateRenderTargetResource(device, DXGI_FORMAT_R8G8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sortedToSourceRayIndexOffset, initialResourceState, L"Sorted To Source Ray Index"); // ToDo remove
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32B32A32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sortedRayGroupDebug, initialResourceState, L"Sorted Ray Group Debug");
        CreateRenderTargetResource(device, COMPACT_NORMAL_DEPTH_DXGI_FORMAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AORayDirectionOriginDepth, initialResourceState, L"AO Rays Direction, Origin Depth and Hit");
    }


    // Build shader tables.
    // This encapsulates all shader records - shaders and the arguments for their local root signatures.
    // For AO, the shaders are simple with only one shader type per shader table.
    void RTAO::BuildShaderTables()
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
                // ToDO combine raygens into single table or update the names accordingly
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


        // maxInstanceContributionToHitGroupIndex - since BLAS instances in this sample specify non-zero InstanceContributionToHitGroupIndex for Pathtracer raytracing pass, 
        //  the AO raytracing needs to add as many shader records to all hit group shader tables so that DXR shader addressing lands on a valid shader record for all BLASes.
        UINT maxInstanceContributionToHitGroupIndex = Sample::instance().Scene().AccelerationStructure()->GetMaxInstanceContributionToHitGroupIndex();

        // Hit group shader table.
        {
            // Duplicate the shader records because the TLAS has BLAS instances with non-zero InstanceContributionToHitGroupIndex.
            // For the last offset we need only one more shader record, because RTAO TraceRay always indexes the first shader record
            // of each BLAS instance shader record range due to RTAOTraceRayParameters::HitGroup::GeometryStride of 0.
            UINT numShaderRecords = maxInstanceContributionToHitGroupIndex + 1;
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

        UINT spp = RTAO_Args::AOSampleCountPerDimension * RTAO_Args::AOSampleCountPerDimension;
        UINT samplesPerSet = spp * RTAO_Args::AOSampleSetDistributedAcrossPixels * RTAO_Args::AOSampleSetDistributedAcrossPixels;
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

    float RTAO::GetSpp()
    {
        if (RTAO_Args::RTAOUseRaySorting)
        {
            int MaxRaysPerQuad = RTAO_Args::RTAORayGen_MaxRaysPerQuad;
            switch (RTAO_Args::RTAORayGenAdaptiveQuadSize)
            {
            case GpuKernels::AdaptiveRayGenerator::Quad2x2: return min(MaxRaysPerQuad, 4) / 4.0f;
            case GpuKernels::AdaptiveRayGenerator::Quad4x4: return min(MaxRaysPerQuad, 16) / 16.0f;
            }
        }
        return 1;
    }

    void RTAO::GetRayGenParameters(bool* isCheckerboardSamplingEnabled, bool* checkerboardLoadEvenPixels)
    {
        *isCheckerboardSamplingEnabled = GetSpp() != 1;
        *checkerboardLoadEvenPixels = m_checkerboardGenerateRaysForEvenPixels;
    }

    void RTAO::DispatchRays(ID3D12Resource* rayGenShaderTable, UINT width, UINT height)
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
        dispatchDesc.Width = width != 0 ? width : m_raytracingWidth;
        dispatchDesc.Height = height != 0 ? height : m_raytracingHeight;
        dispatchDesc.Depth = 1;
        commandList->SetPipelineState1(m_dxrStateObject.Get());

        resourceStateTracker->FlushResourceBarriers();
        commandList->DispatchRays(&dispatchDesc);
    }


    void RTAO::OnUpdate()
    {

        if (m_isRecreateAOSamplesRequested)
        {
            m_isRecreateAOSamplesRequested = false;

            // ToDo Update all components at once with 1 wait?
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


        uniform_int_distribution<UINT> seedDistribution(0, UINT_MAX);

#if 0
        if (RTAO_Args::RTAORandomFrameSeed)
        {
            m_CB->seed = seedDistribution(m_generatorURNG);
            RTAO_Args::RTAORandomFrameSeed.Bang();
        }
#elif 0
        m_CB->seed = 287535318;
#else
        if (RTAO_Args::RTAORandomFrameSeed)
        {
            m_CB->seed = seedDistribution(m_generatorURNG);
        }
        else
        {
            m_CB->seed = 1879;
        }
#endif
#if DEBUG_PRINT_OUT_SEED_VALUE
        std::wstringstream wstr;
        static UINT frameIndex = 0;
        m_deviceResources->GetCurrentFrameIndex();
        wstr << L"Frame " << frameIndex++ << L"\n";
        wstr << L" Seed: " << m_CB->seed << L"\n";
        OutputDebugStringW(wstr.str().c_str());
#endif

        m_CB->numSamplesPerSet = m_randomSampler.NumSamples();
        m_CB->numSampleSets = m_randomSampler.NumSampleSets();
        m_CB->numPixelsPerDimPerSet = RTAO_Args::AOSampleSetDistributedAcrossPixels;

        m_CB->RTAO_UseSortedRays = RTAO_Args::RTAOUseRaySorting;

        bool doCheckerboardRayGeneration = GetSpp() != 1;
        m_checkerboardGenerateRaysForEvenPixels = !m_checkerboardGenerateRaysForEvenPixels;
        m_CB->doCheckerboardSampling = doCheckerboardRayGeneration;
        m_CB->areEvenPixelsActive = m_checkerboardGenerateRaysForEvenPixels;
        UINT pixelStepX = doCheckerboardRayGeneration ? 2 : 1;
        m_CB->raytracingDim = XMUINT2(CeilDivide(m_raytracingWidth, pixelStepX), m_raytracingHeight);

        RTAO_Args::RTAOAdaptiveSamplingMinSamples.SetMaxValue(RTAO_Args::AOSampleCountPerDimension * RTAO_Args::AOSampleCountPerDimension);

        // ToDo standardize RTAO RTAO_ prefix, or remove it since this is RTAO class
        m_CB->RTAO_maxShadowRayHitTime = RTAO_Args::RTAOMaxRayHitTime;
        m_CB->RTAO_approximateInterreflections = RTAO_Args::RTAOApproximateInterreflections;
        m_CB->RTAO_diffuseReflectanceScale = RTAO_Args::RTAODiffuseReflectanceScale;
        m_CB->RTAO_MinimumAmbientIllumination = RTAO_Args::RTAO_MinimumAmbientIllumination;
        m_CB->RTAO_IsExponentialFalloffEnabled = RTAO_Args::RTAOIsExponentialFalloffEnabled;
        m_CB->RTAO_exponentialFalloffDecayConstant = RTAO_Args::RTAO_ExponentialFalloffDecayConstant;

        // Calculate a theoretical max ray distance to be used in occlusion factor computation.
        // Occlusion factor of a ray hit is computed based of its ray hit time, falloff exponent and a max ray hit time.
        // By specifying a min occlusion factor of a ray, we can skip tracing rays that would have an occlusion 
        // factor less than the cutoff to save a bit of performance (generally 1-10% perf win without visible AO result impact). // ToDo retest
        // Therefore the sample discerns between true maxRayHitTime, used in TraceRay, 
        // and a theoretical one used in calculating the occlusion factor on a hit.
        {
            float occclusionCutoff = RTAO_Args::RTAO_ExponentialFalloffMinOcclusionCutoff;
            float lambda = RTAO_Args::RTAO_ExponentialFalloffDecayConstant;

            // Invert occlusionFactor = exp(-lambda * t * t), where t is tHit/tMax of a ray.
            float t = sqrt(logf(occclusionCutoff) / -lambda);

            m_CB->RTAO_maxShadowRayHitTime = t * RTAO_Args::RTAOMaxRayHitTime;
            m_CB->RTAO_maxTheoreticalShadowRayHitTime = RTAO_Args::RTAOMaxRayHitTime;
        }
    }

    void RTAO::Run(
        D3D12_GPU_VIRTUAL_ADDRESS accelerationStructure,
        D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceHitPositionResource,
        D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceNormalDepthResource,
        D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceAlbedoResource)
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
        auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
               
        // Copy dynamic buffers to GPU.
        {
            // ToDo copy on change
            m_CB.CopyStagingToGpu(frameIndex);
            m_hemisphereSamplesGPUBuffer.CopyStagingToGpu(frameIndex);
        }

        ScopedTimer _prof(L"CalculateAmbientOcclusion_Root", commandList);

        // Transition AO resources to UAV state.        // ToDo check all comments
        {
            // ToDo remove the if-else
            if (RTAO_Args::RTAOUseRaySorting)
            {
                resourceStateTracker->TransitionResource(&m_AOResources[AOResource::HitCount], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_AOResources[AOResource::Coefficient], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_AOResources[AOResource::RayHitDistance], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_sourceToSortedRayIndexOffset, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_sortedToSourceRayIndexOffset, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_sortedRayGroupDebug, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_AORayDirectionOriginDepth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
            else
            {
                resourceStateTracker->TransitionResource(&m_AOResources[AOResource::HitCount], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_AOResources[AOResource::Coefficient], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->TransitionResource(&m_AOResources[AOResource::RayHitDistance], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            }
        }

        commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());


        // Bind inputs.
        commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::RayOriginPosition, rayOriginSurfaceHitPositionResource);
        commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::RayOriginSurfaceNormalDepth, rayOriginSurfaceNormalDepthResource);
        commandList->SetComputeRootShaderResourceView(RTAOGlobalRootSignature::Slot::SampleBuffers, m_hemisphereSamplesGPUBuffer.GpuVirtualAddress(frameIndex));
        commandList->SetComputeRootConstantBufferView(RTAOGlobalRootSignature::Slot::ConstantBuffer, m_CB.GpuVirtualAddress(frameIndex));
        commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::FilterWeightSum, m_AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);
        commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AOSurfaceAlbedo, rayOriginSurfaceAlbedoResource);


        // Bind output RT.
        // ToDo remove output and rename AOout
        // ToDo use [enum] instead of [0]
        commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AOResourcesOut, m_AOResources[0].gpuDescriptorWriteAccess);
        commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AORayHitDistance, m_AOResources[AOResource::RayHitDistance].gpuDescriptorWriteAccess);
        commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AORayDirectionOriginDepthHitUAV, m_AORayDirectionOriginDepth.gpuDescriptorWriteAccess);

        // Bind the heaps, acceleration structure and dispatch rays. 
        commandList->SetComputeRootShaderResourceView(RTAOGlobalRootSignature::Slot::AccelerationStructure, accelerationStructure);

#if DEBUG_PRINT_OUT_RTAO_DISPATCH_TIME
        dispatchRayTime.BeginFrame(commandList);
        dispatchRayTime.Start(commandList);
#endif


        if (!RTAO_Args::RTAOUseRaySorting)
        {
            ScopedTimer _prof(L"AO DispatchRays 2D", commandList);
            DispatchRays(m_rayGenShaderTables[RTAORayGenShaderType::AOFullRes].Get());
        }
#if DEBUG_PRINT_OUT_RTAO_DISPATCH_TIME
        dispatchRayTime.Stop(commandList);
        dispatchRayTime.EndFrame(commandList);

        std::wstringstream wstr;
        wstr << L"RTAO DispatchRays[FrameCount ago] " << dispatchRayTime.GetElapsedMS() << L"\n";
        OutputDebugStringW(wstr.str().c_str());
#endif
        // ToDo Remove
        //DispatchRays(m_rayGenShaderTables[RTAO_Args::QuarterResAO ? RTAORayGenShaderType::AOQuarterRes : RTAORayGenShaderType::AOFullRes].Get(),
        //    &m_gpuTimers[GpuTimers::Raytracing_AO], m_raytracingWidth, m_raytracingHeight);

        if (RTAO_Args::RTAOUseRaySorting)
        {
            bool doCheckerboardRayGeneration = GetSpp() != 1;

            // Todo verify odd width resolutions when using CB
            UINT activeRaytracingWidth =
                doCheckerboardRayGeneration
                ? CeilDivide(m_raytracingWidth, 2)
                : m_raytracingWidth;
            resourceStateTracker->FlushResourceBarriers();
            m_rayGen.Run(
                commandList,
                activeRaytracingWidth,
                m_raytracingHeight,
                static_cast<GpuKernels::AdaptiveRayGenerator::AdaptiveQuadSizeType>(static_cast<UINT>(RTAO_Args::RTAORayGenAdaptiveQuadSize)),
                RTAO_Args::RTAORayGen_MaxFrameAge,
                RTAO_Args::RTAORayGen_MinAdaptiveFrameAge,
                RTAO_Args::RTAORayGen_MaxFrameAgeToGenerateRaysFor,
                RTAO_Args::RTAORayGen_MaxRaysPerQuad,
                m_CB->seed, // ToDo retrieve from a nonCB variable
                m_randomSampler.NumSamples(),
                m_randomSampler.NumSampleSets(),
                RTAO_Args::AOSampleSetDistributedAcrossPixels,
                doCheckerboardRayGeneration,
                m_checkerboardGenerateRaysForEvenPixels,
                m_cbvSrvUavHeap->GetHeap(),
                rayOriginSurfaceNormalDepthResource,
                rayOriginSurfaceHitPositionResource,
                m_hemisphereSamplesGPUBuffer.GpuVirtualAddress(frameIndex),
                m_AORayDirectionOriginDepth.gpuDescriptorWriteAccess);

            resourceStateTracker->TransitionResource(&m_AORayDirectionOriginDepth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->InsertUAVBarrier(&m_AORayDirectionOriginDepth);

            float rayBinDepthSize = RTAO_Args::RTAORayBinDepthSizeMultiplier * RTAO_Args::RTAOMaxRayHitTime;
            resourceStateTracker->FlushResourceBarriers();
            m_raySorter.Run(
                commandList,
                rayBinDepthSize,
                activeRaytracingWidth,
                m_raytracingHeight,
                GpuKernels::SortRays::FilterType::CountingSort,
                RTAO_Args::RTAORaySortingUseOctahedralRayDirectionQuantization,
                m_cbvSrvUavHeap->GetHeap(),
                m_AORayDirectionOriginDepth.gpuDescriptorReadAccess,
                m_sortedToSourceRayIndexOffset.gpuDescriptorWriteAccess,
                m_sourceToSortedRayIndexOffset.gpuDescriptorWriteAccess,
                m_sortedRayGroupDebug.gpuDescriptorWriteAccess);

            resourceStateTracker->TransitionResource(&m_sourceToSortedRayIndexOffset, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->TransitionResource(&m_sortedToSourceRayIndexOffset, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->TransitionResource(&m_sortedRayGroupDebug, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->InsertUAVBarrier(&m_sourceToSortedRayIndexOffset);
            resourceStateTracker->InsertUAVBarrier(&m_sortedToSourceRayIndexOffset);

            {
                ScopedTimer _prof(L"[Sorted]CalculateAmbientOcclusion", commandList);

                commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());


                // Bind inputs.
                // ToDo use [enum] instead of [0]
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::RayOriginPosition, rayOriginSurfaceHitPositionResource);
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::RayOriginSurfaceNormalDepth, rayOriginSurfaceNormalDepthResource);
                commandList->SetComputeRootShaderResourceView(RTAOGlobalRootSignature::Slot::SampleBuffers, m_hemisphereSamplesGPUBuffer.GpuVirtualAddress(frameIndex));
                commandList->SetComputeRootConstantBufferView(RTAOGlobalRootSignature::Slot::ConstantBuffer, m_CB.GpuVirtualAddress(frameIndex));   // ToDo let AO have its own CB.
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::FilterWeightSum, m_AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AOSurfaceAlbedo, rayOriginSurfaceAlbedoResource);

                // ToDo remove
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AORayDirectionOriginDepthHitSRV, m_AORayDirectionOriginDepth.gpuDescriptorReadAccess);
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AOSortedToSourceRayIndex, m_sortedToSourceRayIndexOffset.gpuDescriptorReadAccess);

                // Bind output RT.
                // ToDo remove output and rename AOout
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AOResourcesOut, m_AOResources[0].gpuDescriptorWriteAccess);
                commandList->SetComputeRootDescriptorTable(RTAOGlobalRootSignature::Slot::AORayHitDistance, m_AOResources[AOResource::RayHitDistance].gpuDescriptorWriteAccess);

                // Bind the heaps, acceleration structure and dispatch rays. 
                // ToDo dedupe calls
                commandList->SetComputeRootShaderResourceView(RTAOGlobalRootSignature::Slot::AccelerationStructure, accelerationStructure);

#if RTAO_RAY_SORT_1DRAYTRACE
                UINT NumRays = activeRaytracingWidth * m_raytracingHeight;
                DispatchRays(m_rayGenShaderTables[RTAORayGenShaderType::AOSortedRays].Get(), NumRays, 1);
#else
                DispatchRays(m_rayGenShaderTables[RTAORayGenShaderType::AOSortedRays].Get(), m_raytracingWidth, m_raytracingHeight);
#endif
            }

        }

        resourceStateTracker->TransitionResource(&m_AOResources[AOResource::HitCount], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_AOResources[AOResource::Coefficient], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->TransitionResource(&m_AOResources[AOResource::RayHitDistance], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resourceStateTracker->InsertUAVBarrier(&m_AOResources[AOResource::Coefficient]);
        resourceStateTracker->InsertUAVBarrier(&m_AOResources[AOResource::RayHitDistance]);

        // Calculate AO ray hit count.
        if (m_calculateRayHitCounts)
        {
            ScopedTimer _prof(L"CalculateAORayHitCount", commandList);
            CalculateRayHitCount();
        }
    }

    void RTAO::CalculateRayHitCount()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

        resourceStateTracker->FlushResourceBarriers();
        m_reduceSumKernel.Run(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            frameIndex,
            m_AOResources[AOResource::HitCount].gpuDescriptorReadAccess,
            &m_numAORayGeometryHits);
    }

    void RTAO::CreateResolutionDependentResources()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto commandQueue = m_deviceResources->GetCommandQueue();
        auto FrameCount = m_deviceResources->GetBackBufferCount();

        CreateTextureResources();
        m_reduceSumKernel.CreateInputResourceSizeDependentResources(
            device,
            m_cbvSrvUavHeap.get(),
            FrameCount,
            m_raytracingWidth,
            m_raytracingHeight);
    }


    void RTAO::SetResolution(UINT width, UINT height)
    {
        m_raytracingWidth = width;
        m_raytracingHeight = height;

        CreateResolutionDependentResources();
    }
}