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
#include "CompiledShaders\RTAO.hlsl.h"


// ToDo prune unused
using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;

// Shader entry points.
const wchar_t* RTAO::c_rayGenShaderNames[] =
{
    L"RayGenShader_AO", L"RayGenShader_AO_sortedRays", L"RayGenShaderQuarterRes_AO"
};
const wchar_t* RTAO::c_closestHitShaderName = L"ClosestHitShader_AORay";
const wchar_t* RTAO::c_missShaderName = L"MissShader_AORay";
// Hit groups.
const wchar_t* RTAO::c_hitGroupName = L"HitGroup_Triangle_AORay";


namespace SceneArgs
{
    void OnRecreateRaytracingResources(void*)
    {
        g_pSample->RequestRecreateRaytracingResources();
    }

    void OnRecreateSamples(void*)
    {
        g_pSample->RequestRecreateAOSamples();
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
    for (UINT i = 0; i < RaytracingType::Count; i++)
    {
        m_missShaderTableStrideInBytes[i] = UINT_MAX;
        m_hitGroupShaderTableStrideInBytes[i] = UINT_MAX;
    }
    m_generatorURNG.seed(1729);
}


void RTAO::Setup(shared_ptr<DeviceResources> deviceResources, shared_ptr<DX::DescriptorHeap> descriptorHeap)
{
    m_deviceResources = deviceResources;
    m_cbvSrvUavHeap = descriptorHeap;

    CreateDeviceDependentResources();
}

// Create resources that depend on the device.
void RTAO::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    
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

    // ToDo move
    CreateSamplesRNG();
}


void RTAO::CreateAuxilaryDeviceResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    auto commandList = m_deviceResources->GetCommandList();

    EngineProfiling::RestoreDevice(device, commandQueue, FrameCount);
    ResourceUploadBatch resourceUpload(device);
    resourceUpload.Begin();

    // ToDo move?
    m_reduceSumKernel.Initialize(device, GpuKernels::ReduceSum::Uint);
    m_atrousWaveletTransformFilter.Initialize(device, ATROUS_DENOISER_MAX_PASSES, FrameCount, MaxAtrousWaveletTransformFilterInvocationsPerFrame);
    m_calculateVarianceKernel.Initialize(device, FrameCount, MaxCalculateVarianceKernelInvocationsPerFrame);
    m_calculateMeanVarianceKernel.Initialize(device, FrameCount, 5 * MaxCalculateVarianceKernelInvocationsPerFrame);
    m_calculatePartialDerivativesKernel.Initialize(device, FrameCount);
    m_gaussianSmoothingKernel.Initialize(device, FrameCount, MaxGaussianSmoothingKernelInvocationsPerFrame);
    m_downsampleBoxFilter2x2Kernel.Initialize(device, FrameCount);
    m_downsampleGaussian9TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap9, FrameCount);
    m_downsampleGaussian25TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap25, FrameCount); // ToDo Dedupe 9 and 25
    m_downsampleGBufferBilateralFilterKernel.Initialize(device, GpuKernels::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::FilterDepthAware2x2);
    m_downsampleValueNormalDepthBilateralFilterKernel.Initialize(device, static_cast<GpuKernels::DownsampleValueNormalDepthBilateralFilter::Type>(static_cast<UINT>(SceneArgs::DownsamplingBilateralFilter)));
    m_upsampleBilateralFilterKernel.Initialize(device, GpuKernels::UpsampleBilateralFilter::Filter2x2, FrameCount);
    m_multiScale_upsampleBilateralFilterAndCombineKernel.Initialize(device, GpuKernels::MultiScale_UpsampleBilateralFilterAndCombine::Filter2x2);
    m_temporalCacheReverseReprojectKernel.Initialize(device, FrameCount);
    m_writeValueToTexture.Initialize(device, m_cbvSrvUavHeap.get());
    m_grassGeometryGenerator.Initialize(device, L"Assets\\wind\\wind2.jpg", m_cbvSrvUavHeap.get(), &resourceUpload, FrameCount, UIParameters::NumGrassGeometryLODs);
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

    m_sceneCB.Create(device, FrameCount, L"Scene Constant Buffer");
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
        CreateLocalRootSignatureSubobjects(&raytracingPipeline;

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

    // ToDo tune formats
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

    // ToDo pass formats via params shared across AO, GBuffer, TC

    // Full-res Temporal Cache resources.
    {
        for (UINT i = 0; i < 2; i++)
        {
            // Preallocate subsequent descriptor indices for both SRV and UAV groups.
            m_temporalCache[i][0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(TemporalCache::Count);
            m_temporalCache[i][0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(TemporalCache::Count);
            for (UINT j = 0; j < TemporalCache::Count; j++)
            {
                m_temporalCache[i][j].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
                m_temporalCache[i][j].uavDescriptorHeapIndex = m_temporalCache[i][0].uavDescriptorHeapIndex + j;
                m_temporalCache[i][j].srvDescriptorHeapIndex = m_temporalCache[i][0].srvDescriptorHeapIndex + j;
            }

            // ToDo cleanup raytracing resolution - twice for coefficient.
            CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_temporalCache[i][TemporalCache::FrameAge], initialResourceState, L"Temporal Cache: Disocclusion Map");

            m_AOTSSCoefficient[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOTSSCoefficient[i], initialResourceState, L"Render/AO Temporally Supersampled Coefficient");

            m_lowResAOTSSCoefficient[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, TextureResourceFormatR::ToDXGIFormat(SceneArgs::RTAO_AmbientCoefficientResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_lowResAOTSSCoefficient[i], initialResourceState, L"Render/AO LowRes Temporally Supersampled Coefficient");


        }
    }

    // ToDo remove
    // Debug resources
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_debugOutput[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(ARRAYSIZE(m_debugOutput));
        m_debugOutput[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(ARRAYSIZE(m_debugOutput));
        for (UINT i = 0; i < ARRAYSIZE(m_debugOutput); i++)
        {
            m_debugOutput[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_debugOutput[i].uavDescriptorHeapIndex = m_debugOutput[0].uavDescriptorHeapIndex + i;
            m_debugOutput[i].srvDescriptorHeapIndex = m_debugOutput[0].srvDescriptorHeapIndex + i;
            CreateRenderTargetResource(device, DXGI_FORMAT_R32G32B32A32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_debugOutput[i], initialResourceState, L"Debug");
        }
    }


    // ToDo move
    // ToDo render shadows at raytracing dim?
    m_VisibilityResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, texFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_VisibilityResource, initialResourceState, L"Visibility");


    m_sourceToSortedRayIndex.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R8G8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sourceToSortedRayIndex, initialResourceState, L"Source To Sorted Ray Index");

    m_sortedToSourceRayIndex.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R8G8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sortedToSourceRayIndex, initialResourceState, L"Sorted To Source Ray Index");


    m_sortedRayGroupDebug.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R32G32B32A32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_sortedRayGroupDebug, initialResourceState, L"Sorted Ray Group Debug");


    // ToDo use 8 bit format
    m_AORayDirectionOriginDepth.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AORayDirectionOriginDepth, initialResourceState, L"AO Rays Direction, Origin Depth and Hit");


    m_ShadowMapResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, c_shadowMapDim.x, c_shadowMapDim.y, m_cbvSrvUavHeap.get(), &m_ShadowMapResource, initialResourceState, L"Shadow Map");


    // ToDo specialize formats instead of using a common one?

    DXGI_FORMAT varianceTexFormat = DXGI_FORMAT_R16_FLOAT;       // ToDo 8 bit suffers from loss of precision and clamps too much.
    m_varianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, varianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_varianceResource, initialResourceState, L"Variance");
    m_smoothedVarianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, varianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_smoothedVarianceResource, initialResourceState, L"Smoothed Variance");
    m_meanResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, varianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_meanResource, initialResourceState, L"Mean");
    m_smoothedMeanResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, varianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_smoothedMeanResource, initialResourceState, L"Smoothed Mean");

    DXGI_FORMAT meanVarianceTexFormat = DXGI_FORMAT_R16G16_FLOAT;       // ToDo 8 bit suffers from loss of precision and clamps too much.
    m_meanVarianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, meanVarianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_meanVarianceResource, initialResourceState, L"Mean Variance");
    m_smoothedMeanVarianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, meanVarianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_smoothedMeanVarianceResource, initialResourceState, L"Smoothed Mean Variance");


    // ToDo move
    for (UINT i = 0; i < c_MaxDenoisingScaleLevels; i++)
    {
        MultiScaleDenoisingResource& msResource = m_multiScaleDenoisingResources[i];
        msResource.m_value.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_normalDepth.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_partialDistanceDerivatives.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_smoothedValue.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_downsampledSmoothedValue.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_downsampledNormalDepthValue.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_downsampledPartialDistanceDerivatives.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_varianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
        msResource.m_smoothedVarianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;

        RWGpuResource m_varianceResource;
        RWGpuResource m_smoothedVarianceResource;
        UINT width = CeilDivide(m_raytracingWidth, 1 << i);
        UINT height = CeilDivide(m_raytracingHeight, 1 << i);
        CreateRenderTargetResource(device, texFormat, width, height, m_cbvSrvUavHeap.get(), &msResource.m_value, initialResourceState, L"MultiScaleDenoisingResource Value");
        CreateRenderTargetResource(device, normalFormat, width, height, m_cbvSrvUavHeap.get(), &msResource.m_normalDepth, initialResourceState, L"MultiScaleDenoisingResource Normal and Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_FLOAT, width, height, m_cbvSrvUavHeap.get(), &msResource.m_partialDistanceDerivatives, initialResourceState, L"MultiScaleDenoisingResource Partial Distance Derivatives");
        CreateRenderTargetResource(device, texFormat, width, height, m_cbvSrvUavHeap.get(), &msResource.m_smoothedValue, initialResourceState, L"MultiScaleDenoisingResource Smoothed");
        CreateRenderTargetResource(device, texFormat, width, height, m_cbvSrvUavHeap.get(), &msResource.m_varianceResource, initialResourceState, L"MultiScaleDenoisingResource Variance");
        CreateRenderTargetResource(device, texFormat, width, height, m_cbvSrvUavHeap.get(), &msResource.m_smoothedVarianceResource, initialResourceState, L"MultiScaleDenoisingResource SmoothedVariance");

        UINT downsampledWidth = CeilDivide(width, 2);
        UINT downsampledHeight = CeilDivide(height, 2);
        CreateRenderTargetResource(device, texFormat, downsampledWidth, downsampledHeight, m_cbvSrvUavHeap.get(), &msResource.m_downsampledSmoothedValue, initialResourceState, L"MultiScaleDenoisingResource Downsampled Smoothed");
        CreateRenderTargetResource(device, normalFormat, downsampledWidth, downsampledHeight, m_cbvSrvUavHeap.get(), &msResource.m_downsampledNormalDepthValue, initialResourceState, L"MultiScaleDenoisingResource Downsampled Normal and Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_FLOAT, downsampledWidth, downsampledHeight, m_cbvSrvUavHeap.get(), &msResource.m_downsampledPartialDistanceDerivatives, initialResourceState, L"MultiScaleDenoisingResource Downsampled Partial Distance Derivatives");
    }

    // ToDo
    // Describe and create the point clamping sampler used for reading from the GBuffer resources.
    //CD3DX12_CPU_DESCRIPTOR_HANDLE samplerHandle(m_samplerHeap->GetHeap()->GetCPUDescriptorHandleForHeapStart());
    //D3D12_SAMPLER_DESC clampSamplerDesc = {};
    //clampSamplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    //clampSamplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    //clampSamplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    //clampSamplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    //clampSamplerDesc.MipLODBias = 0.0f;
    //clampSamplerDesc.MaxAnisotropy = 1;
    //clampSamplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    //clampSamplerDesc.MinLOD = 0;
    //clampSamplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
    //device->CreateSampler(&clampSamplerDesc, samplerHandle);
}


// Build shader tables.
// This encapsulates all shader records - shaders and the arguments for their local root signatures.
void RTAO::BuildShaderTables()
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
            // ToDo cleanup
            if (raytracingType == RaytracingType::Pathtracing ||
                i == RayGenShaderType::AOFullRes ||
                i == RayGenShaderType::AOSortedRays ||
                i == RayGenShaderType::ShadowMap ||
                i == RayGenShaderType::Visibility)
            {
                rayGenShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(c_rayGenShaderNames[i]);
                shaderIdToStringMap[rayGenShaderIDs[i]] = c_rayGenShaderNames[i];
            }
            else
            {
                rayGenShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(c_rayGenShaderNames[RayGenShaderType::AOFullRes]);
                shaderIdToStringMap[rayGenShaderIDs[i]] = c_rayGenShaderNames[RayGenShaderType::AOFullRes];
            }
        }

        for (UINT i = 0; i < RayType::Count; i++)
        {
            if (raytracingType == RaytracingType::Pathtracing ||
                i == RayType::Shadow)
            {
                missShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(c_missShaderNames[i]);
                shaderIdToStringMap[missShaderIDs[i]] = c_missShaderNames[i];
            }
            else
            {
                missShaderIDs[i] = stateObjectProperties->GetShaderIdentifier(c_missShaderNames[RayType::Shadow]);
                shaderIdToStringMap[missShaderIDs[i]] = c_missShaderNames[RayType::Shadow];
            }
        }

        for (UINT i = 0; i < RayType::Count; i++)
        {
            if (raytracingType == RaytracingType::Pathtracing ||
                i == RayType::Shadow)
            {
                hitGroupShaderIDs_TriangleGeometry[i] = stateObjectProperties->GetShaderIdentifier(c_hitGroupNames_TriangleGeometry[i]);
                shaderIdToStringMap[hitGroupShaderIDs_TriangleGeometry[i]] = c_hitGroupNames_TriangleGeometry[i];
            }
            else
            {
                hitGroupShaderIDs_TriangleGeometry[i] = stateObjectProperties->GetShaderIdentifier(c_hitGroupNames_TriangleGeometry[RayType::Shadow]);
                shaderIdToStringMap[hitGroupShaderIDs_TriangleGeometry[i]] = c_hitGroupNames_TriangleGeometry[RayType::Shadow];
            }
        }
    };

    // Get shader identifiers.
    UINT shaderIDSize;
    ComPtr<ID3D12StateObjectProperties> stateObjectProperties;
    ThrowIfFailed(m_dxrStateObjects.As(&stateObjectProperties));
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
        UINT numShaderRecords = 0;
        for (auto& bottomLevelASGeometryPair : m_bottomLevelASGeometries)
        {
            auto& bottomLevelASGeometry = bottomLevelASGeometryPair.second;
            numShaderRecords += static_cast<UINT>(bottomLevelASGeometry.m_geometryInstances.size()) * RayType::Count;
        }
        UINT numGrassGeometryShaderRecords = 2 * UIParameters::NumGrassGeometryLODs * 3 * RayType::Count;
        numShaderRecords += numGrassGeometryShaderRecords;

        UINT shaderRecordSize = shaderIDSize + LocalRootSignature::MaxRootArgumentsSize();
        ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");

        // Triangle geometry hit groups.
        for (auto& bottomLevelASGeometryPair : m_bottomLevelASGeometries)
        {
            auto& bottomLevelASGeometry = bottomLevelASGeometryPair.second;
            auto& name = bottomLevelASGeometry.GetName();

            UINT shaderRecordOffset = hitGroupShaderTable.GeNumShaderRecords();
            m_accelerationStructure->GetBottomLevelAS(bottomLevelASGeometryPair.first).SetInstanceContributionToHitGroupIndex(shaderRecordOffset);

            // ToDo cleaner?
            // Grass Patch LOD shader recods
            if (name.find(L"Grass Patch LOD") != wstring::npos)
            {
                UINT LOD = stoi(name.data() + 15);

                // ToDo remove assert
                assert(bottomLevelASGeometry.m_geometryInstances.size() == 1);
                auto& geometryInstance = bottomLevelASGeometry.m_geometryInstances[0];

                LocalRootSignature::Triangle::RootArguments rootArgs;
                rootArgs.cb.materialID = geometryInstance.materialID;
                rootArgs.cb.isVertexAnimated = geometryInstance.isVertexAnimated;

                memcpy(&rootArgs.indexBufferGPUHandle, &geometryInstance.ib.gpuDescriptorHandle, sizeof(geometryInstance.ib.gpuDescriptorHandle));
                memcpy(&rootArgs.diffuseTextureGPUHandle, &geometryInstance.diffuseTexture, sizeof(geometryInstance.diffuseTexture));
                memcpy(&rootArgs.normalTextureGPUHandle, &geometryInstance.normalTexture, sizeof(geometryInstance.normalTexture));

                // Create three variants:


                struct VertexBufferHandles {
                    D3D12_GPU_DESCRIPTOR_HANDLE prevFrameVertexBuffer;
                    D3D12_GPU_DESCRIPTOR_HANDLE vertexBuffer;
                };

                // 2 * 3 Shader Records per LOD
                //  2 - ping-pong frame to frame
                //  3 - transition types
                //      Transition from lower LOD in previous frame
                //      Same LOD as previous frame
                //      Transition from higher LOD in previous

                VertexBufferHandles vbHandles[2][3];
                for (UINT frameID = 0; frameID < 2; frameID++)
                {
                    UINT prevFrameID = (frameID + 1) % 2;

                    // For simplicity, we assume the LOD difference from frame to frame is no greater than 1.
                    // ToDo explain why multiple LODs somewhere.s
                    // This can be false if camera moves fast, but in that case temporal reprojection 
                    // would fail for the most part anyway, and consistency checks will prevent blending in from false geometry.

                    // Transitioning from lower LOD.
                    vbHandles[frameID][0].vertexBuffer = m_grassPatchVB[LOD][frameID].gpuDescriptorReadAccess;
                    vbHandles[frameID][0].prevFrameVertexBuffer = LOD > 0 ? m_grassPatchVB[LOD - 1][prevFrameID].gpuDescriptorReadAccess
                        : m_grassPatchVB[LOD][prevFrameID].gpuDescriptorReadAccess;

                    // Same LOD as previous frame.
                    vbHandles[frameID][1].vertexBuffer = m_grassPatchVB[LOD][frameID].gpuDescriptorReadAccess;
                    vbHandles[frameID][1].prevFrameVertexBuffer = m_grassPatchVB[LOD][prevFrameID].gpuDescriptorReadAccess;

                    // Transitioning from higher LOD.
                    vbHandles[frameID][2].vertexBuffer = m_grassPatchVB[LOD][frameID].gpuDescriptorReadAccess;
                    vbHandles[frameID][2].prevFrameVertexBuffer = LOD < UIParameters::NumGrassGeometryLODs - 1 ? m_grassPatchVB[LOD + 1][prevFrameID].gpuDescriptorReadAccess
                        : m_grassPatchVB[LOD][prevFrameID].gpuDescriptorReadAccess;
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
                    LocalRootSignature::Triangle::RootArguments rootArgs;
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


// ToDo rename, move
void RTAO::CreateSamplesRNG()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    m_sppAO = SceneArgs::AOSampleCountPerDimension * SceneArgs::AOSampleCountPerDimension;
    UINT samplesPerSet = m_sppAO * SceneArgs::AOSampleSetDistributedAcrossPixels * SceneArgs::AOSampleSetDistributedAcrossPixels;
    m_randomSampler.Reset(samplesPerSet, 83, Samplers::HemisphereDistribution::Cosine);

    // Create root signature.
    {
        using namespace ComputeShader::RootSignature::HemisphereSampleSetVisualization;

        CD3DX12_DESCRIPTOR_RANGE ranges[1]; // Perfomance TIP: Order from most frequent to least frequent.
        ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

        CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
        rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[0]);
        rootParameters[Slot::SampleBuffers].InitAsShaderResourceView(1);
        rootParameters[Slot::SceneConstant].InitAsConstantBufferView(0);

        CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
        SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_computeRootSigs[CSType::HemisphereSampleSetVisualization], L"Root signature: CS hemisphere sample set visualization");
    }

    // Create compute pipeline state.
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
        descComputePSO.pRootSignature = m_computeRootSigs[CSType::HemisphereSampleSetVisualization].Get();
        descComputePSO.CS = CD3DX12_SHADER_BYTECODE((void*)g_pRNGVisualizerCS, ARRAYSIZE(g_pRNGVisualizerCS));

        ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSOs[CSType::HemisphereSampleSetVisualization])));
        m_computePSOs[CSType::HemisphereSampleSetVisualization]->SetName(L"PSO: CS hemisphere sample set visualization");
    }


    // Create shader resources
    {
        // ToDo rename GPU from resource names?
        m_csHemisphereVisualizationCB.Create(device, FrameCount, L"GPU CB: RNG");
        m_samplesGPUBuffer.Create(device, m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(), FrameCount, L"GPU buffer: Random unit square samples");
        m_hemisphereSamplesGPUBuffer.Create(device, m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(), FrameCount, L"GPU buffer: Random hemisphere samples");

        for (UINT i = 0; i < m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(); i++)
        {
            //sample.value = m_randomSampler.GetSample2D();
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
};


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

        // ToDo do we need to recreate resoruces?
        m_deviceResources->WaitForGpu();
        CreateSamplesRNG();
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
    commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));   // ToDo let AO have its own CB.
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

    DispatchRays(m_rayGenShaderTables[RayGenShaderType::AOFullRes].Get(), m_raytracingWidth, m_raytracingHeight);

    // ToDo Remove
    //DispatchRays(m_rayGenShaderTables[SceneArgs::QuarterResAO ? RayGenShaderType::AOQuarterRes : RayGenShaderType::AOFullRes].Get(),
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
            commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));   // ToDo let AO have its own CB.
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
            DispatchRays(m_rayGenShaderTables[RayGenShaderType::AOSortedRays].Get(), m_raytracingWidth * m_raytracingHeight, 1);
#else
            DispatchRays(m_rayGenShaderTables[RayGenShaderType::AOSortedRays].Get(), m_raytracingWidth, m_raytracingHeight);
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
