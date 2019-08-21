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
#include "Composition.h"
#include "GameInput.h"
#include "EngineTuning.h"
#include "EngineProfiling.h"
#include "GpuTimeManager.h"
#include "Composition.h"
#include "CompiledShaders\RNGVisualizerCS.hlsl.h"
#include "CompiledShaders\ComposeRenderPassesCS.hlsl.h"

// ToDo prune unused
using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;

namespace Composition
{
    namespace Args
    {
        // ToDo don't render redundant passes?
        // ToDo Modularize parameters?
        // ToDO standardize capitalization

        const WCHAR* CompositionModes[CompositionType::Count] = {
            L"Phong Lighting",
            L"Denoised Ambient Occlusion",
            L"Temporally Supersampled Ambient Occlusion",
            L"Raw one-frame Ambient Occlusion",
            L"AO and Disocclusion Map",
            L"AO Variance",
            L"AO Local Variance",
            L"Render/AO Minimum Hit Distance",
            L"Normal Map",
            L"Depth Buffer",
            L"Diffuse",
            L"Disocclusion Map" };
        EnumVar CompositionMode(L"Render/Render composition/Mode", CompositionType::AmbientOcclusionOnly_TemporallySupersampled, CompositionType::Count, CompositionModes);
        BoolVar Compose_VarianceVisualizeStdDeviation(L"Render/Render composition/Variance/Visualize std deviation", true);
        NumVar Compose_VarianceScale(L"Render/Render composition/Variance/Variance scale", 1.0f, 0, 10, 0.1f);


        const WCHAR* AntialiasingModes[DownsampleFilter::Count] = { L"OFF", L"SSAA 4x (BoxFilter2x2)", L"SSAA 4x (GaussianFilter9Tap)", L"SSAA 4x (GaussianFilter25Tap)" };
#if REPRO_BLOCKY_ARTIFACTS_NONUNIFORM_CB_REFERENCE_SSAO // Disable SSAA as the blockiness gets smaller with higher resoltuion 
        EnumVar AntialiasingMode(L"Render/Antialiasing", DownsampleFilter::None, DownsampleFilter::Count, AntialiasingModes, OnRecreateRaytracingResources, nullptr);
#else
        EnumVar AntialiasingMode(L"Render/Antialiasing", DownsampleFilter::None, DownsampleFilter::Count, AntialiasingModes, OnRecreateRaytracingResources, nullptr);
#endif
    }

    Composition::Composition()
    {
    }

    void Composition::Setup(shared_ptr<DeviceResources> deviceResources, shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT maxInstanceContributionToHitGroupIndex)
    {
        m_deviceResources = deviceResources;
        m_cbvSrvUavHeap = descriptorHeap;

        CreateDeviceDependentResources(maxInstanceContributionToHitGroupIndex);
    }

    void Composition::ReleaseDeviceDependentResources()
    {
        m_csHemisphereVisualizationCB.Release();
    }

    // Create resources that depend on the device.
    void Composition::CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex)
    {
        CreateAuxilaryDeviceResources();

        // ToDo move/rename
        CreateComposeRenderPassesCSResources();

    }


    // ToDo rename
    void Composition::CreateAuxilaryDeviceResources()
    {
    }


    void Composition::OnUpdate()
    {

    }

    void Composition::CreateResolutionDependentResources()
    {
    }


    void Composition::SetResolution(UINT width, UINT height)
    {
    }



    void Composition::CreateTextureResources()
    {
    }



    void Composition::CreateComposeRenderPassesCSResources()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto FrameCount = m_deviceResources->GetBackBufferCount();

        // Create root signature.
        {
            using namespace CSRootSignature::ComposeRenderPassesCS;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count]; // Perfomance TIP: Order from most frequent to least frequent.
            ranges[Slot::Output].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
            ranges[Slot::GBufferResources].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0);  // 5 input GBuffer textures
            ranges[Slot::AO].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // 1 input AO texture
            ranges[Slot::FilterWeightSum].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);  // 1 input filterWeightSum texture
            ranges[Slot::AORayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 9);  // 1 input AO ray hit distance texture
            ranges[Slot::FrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 10); // 1 input disocclusion map texture
            ranges[Slot::Color].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 11); // 1 input color texture
            ranges[Slot::AOSurfaceAlbedo].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12); // 1 input AO diffuse texture
            ranges[Slot::Variance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 13);
            ranges[Slot::LocalMeanVariance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 14);

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[Slot::Output]);
            rootParameters[Slot::GBufferResources].InitAsDescriptorTable(1, &ranges[Slot::GBufferResources]);
            rootParameters[Slot::AO].InitAsDescriptorTable(1, &ranges[Slot::AO]);
            rootParameters[Slot::FilterWeightSum].InitAsDescriptorTable(1, &ranges[Slot::FilterWeightSum]);
            rootParameters[Slot::AORayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::AORayHitDistance]);
            rootParameters[Slot::FrameAge].InitAsDescriptorTable(1, &ranges[Slot::FrameAge]);
            rootParameters[Slot::Color].InitAsDescriptorTable(1, &ranges[Slot::Color]);
            rootParameters[Slot::AOSurfaceAlbedo].InitAsDescriptorTable(1, &ranges[Slot::AOSurfaceAlbedo]);
            rootParameters[Slot::Variance].InitAsDescriptorTable(1, &ranges[Slot::Variance]);
            rootParameters[Slot::LocalMeanVariance].InitAsDescriptorTable(1, &ranges[Slot::LocalMeanVariance]);
            rootParameters[Slot::MaterialBuffer].InitAsShaderResourceView(7);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);


            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_computeRootSigs[CSType::ComposeRenderPassesCS], L"Root signature: ComposeRenderPassesCS");
        }

        // Create shader resources
        {
            m_csComposeRenderPassesCB.Create(device, FrameCount, L"Constant Buffer: ComposeRenderPassesCS");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_computeRootSigs[CSType::ComposeRenderPassesCS].Get();
            descComputePSO.CS = CD3DX12_SHADER_BYTECODE((void*)g_pComposeRenderPassesCS, ARRAYSIZE(g_pComposeRenderPassesCS));

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSOs[CSType::ComposeRenderPassesCS])));
            m_computePSOs[CSType::ComposeRenderPassesCS]->SetName(L"PSO: ComposeRenderPassesCS");
        }
    }



    // Upsample quarter resources
    void Composition::UpsampleResourcesForRenderComposePass()
    {
        auto commandList = m_deviceResources->GetCommandList();
        GpuResource* inputLowResValueResource = nullptr;
        GpuResource* outputHiResValueResource = nullptr;
        wstring passName;
        GpuKernels::UpsampleBilateralFilter::FilterType filterType = GpuKernels::UpsampleBilateralFilter::Filter2x2R;

        switch (Args::CompositionMode)
        {
            // ToDo Cleanup
        case CompositionType::PhongLighting:
        case CompositionType::AmbientOcclusionOnly_Denoised:
        case CompositionType::AmbientOcclusionOnly_TemporallySupersampled:
        case CompositionType::AmbientOcclusionOnly_RawOneFrame:
        {
            passName = L"Upsample AO";
            if (Args::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
            {
                outputHiResValueResource = &m_AOResources[AOResource::Coefficient];
            }
            else// if (Args::CompositionMode == CompositionType::AmbientOcclusionOnly_TemporallySupersampled)
            {
                outputHiResValueResource = &m_TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex];
            }
            //else
            //{
            //    outputHiResValueResource = &m_AOResources[AOResource::Smoothed];
            //}

            if (Args::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
            {
                inputLowResValueResource = &m_RTAO.AOResources()[AOResource::Coefficient];
            }
            else //(Args::CompositionMode == CompositionType::AmbientOcclusionOnly_TemporallySupersampled)
            {
                inputLowResValueResource = &m_lowResTSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex];
            }
            //else
            //{
            //    inputLowResValueResource = &m_RTAO.AOResources()[AOResource::Smoothed];
            //}
            break;
        }
        case CompositionType::RTAOHitDistance:
        {
            passName = L"Upsample AO ray hit distance";
            inputLowResValueResource = &m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance];
            outputHiResValueResource = &m_AOResources[AOResource::RayHitDistance];
            break;
        }
        case CompositionType::AmbientOcclusionVariance:
        {
            passName = L"Upsample AO variance";
            inputLowResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_lowResVarianceResource[AOVarianceResource::Smoothed] : &m_lowResVarianceResource[AOVarianceResource::Raw];
            outputHiResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_varianceResources[AOVarianceResource::Smoothed] : &m_varianceResources[AOVarianceResource::Raw];
            break;
        }
        case CompositionType::AmbientOcclusionLocalVariance:
        {
            passName = L"Upsample AO local variance";
            filterType = GpuKernels::UpsampleBilateralFilter::Filter2x2RG;
            inputLowResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_lowResLocalMeanVarianceResource[AOVarianceResource::Smoothed] : &m_lowResLocalMeanVarianceResource[AOVarianceResource::Raw];
            outputHiResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_localMeanVarianceResources[AOVarianceResource::Smoothed] : &m_localMeanVarianceResources[AOVarianceResource::Raw];
            break;
        }
        default:
            break;
        }

        if (inputLowResValueResource)
        {
            // ToDo move this within BilateralUpsample().
            GpuResource* GBufferQuarterResResources = m_pathtracer.GetGBufferResources(true);
            GpuResource* GBufferResources = m_pathtracer.GetGBufferResources();

            BilateralUpsample(
                m_GBufferWidth,
                m_GBufferHeight,
                filterType,
                inputLowResValueResource->gpuDescriptorReadAccess,
                GBufferQuarterResResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
                GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
                GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
                outputHiResValueResource,
                passName.c_str());
        }
    }



    // ToDo standardize naming AO vs AmbientOcclusion
    void Composition::BilateralUpsample(
        UINT hiResWidth,
        UINT hiResHeight,
        GpuKernels::UpsampleBilateralFilter::FilterType filterType,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDepthDerivativesResourceHandle,
        GpuResource* outputHiResValueResource,
        LPCWCHAR passName)
    {
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

        ScopedTimer _prof(passName, commandList);

        resourceStateTracker->TransitionResource(outputHiResValueResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        resourceStateTracker->FlushResourceBarriers();
        m_upsampleBilateralFilterKernel.Run(
            commandList,
            hiResWidth,
            hiResHeight,
            filterType,
            m_cbvSrvUavHeap->GetHeap(),
            inputLowResValueResourceHandle,
            inputLowResNormalDepthResourceHandle,
            inputHiResNormalDepthResourceHandle,
            inputHiResPartialDepthDerivativesResourceHandle,
            outputHiResValueResource->gpuDescriptorWriteAccess,
            Args::DownAndUpsamplingUseBilinearWeights,
            Args::DownAndUpsamplingUseDepthWeights,
            Args::DownAndUpsamplingUseNormalWeights,
            Args::DownAndUpsamplingUseDynamicDepthThreshold
        );

        resourceStateTracker->TransitionResource(outputHiResValueResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }


    // ToDo remove?
    void Composition::RenderRNGVisualizations()
    {
#if 0
        auto device = m_deviceResources->GetD3DDevice();
        auto commandList = m_deviceResources->GetCommandList();
        auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

        // Update constant buffer.
        XMUINT2 rngWindowSize(256, 256);
        {
            m_csHemisphereVisualizationCB->dispatchDimensions = rngWindowSize;

            static UINT seed = 0;
            UINT NumFramesPerIter = 400;
            static UINT frameID = NumFramesPerIter * 4;
            m_csHemisphereVisualizationCB->numSamplesToShow = m_randomSampler.NumSamples();// (frameID++ / NumFramesPerIter) % m_randomSampler.NumSamples();
            m_csHemisphereVisualizationCB->sampleSetBase = ((seed++ / NumFramesPerIter) % m_randomSampler.NumSampleSets()) * m_randomSampler.NumSamples();
            m_csHemisphereVisualizationCB->stratums = XMUINT2(static_cast<UINT>(sqrt(m_randomSampler.NumSamples())),
                static_cast<UINT>(sqrt(m_randomSampler.NumSamples())));
            m_csHemisphereVisualizationCB->grid = XMUINT2(m_randomSampler.NumSamples(), m_randomSampler.NumSamples());
            m_csHemisphereVisualizationCB->uavOffset = XMUINT2(0 /*ToDo remove m_width - rngWindowSize.x*/, m_height - rngWindowSize.y);
            m_csHemisphereVisualizationCB->numSamples = m_randomSampler.NumSamples();
            m_csHemisphereVisualizationCB->numSampleSets = m_randomSampler.NumSampleSets();
        }

        // Copy dynamic buffers to GPU
        {
            m_csHemisphereVisualizationCB.CopyStagingToGpu(frameIndex);
            m_samplesGPUBuffer.CopyStagingToGpu(frameIndex);
        }

        // Set pipeline state.
        {
            using namespace ComputeShader::RootSignature::HemisphereSampleSetVisualization;

            commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
            commandList->SetComputeRootSignature(m_computeRootSigs[CSType::HemisphereSampleSetVisualization].Get());
            commandList->SetPipelineState(m_computePSOs[CSType::HemisphereSampleSetVisualization].Get());

            commandList->SetComputeRootConstantBufferView(Slot::SceneConstant, m_csHemisphereVisualizationCB.GpuVirtualAddress(frameIndex));
            commandList->SetComputeRootShaderResourceView(Slot::SampleBuffers, m_samplesGPUBuffer.GpuVirtualAddress(frameIndex));
            commandList->SetComputeRootDescriptorTable(Slot::Output, m_raytracingOutput.gpuDescriptorWriteAccess);
        }

        // Dispatch.
        resourceStateTracker->FlushResourceBarriers();
        commandList->Dispatch(rngWindowSize.x, rngWindowSize.y, 1);
#endif
    }


    void Composition::CreateSamplesRNGVisualization()
    {
#if 0
        auto device = m_deviceResources->GetD3DDevice();
        auto FrameCount = m_deviceResources->GetBackBufferCount();

        UINT samplesPerSet = m_sppAO * Args::AOSampleSetDistributedAcrossPixels * Args::AOSampleSetDistributedAcrossPixels;
        UINT NumSampleSets = 83;
        m_randomSampler.Reset(samplesPerSet, NumSampleSets, Samplers::HemisphereDistribution::Cosine);

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
#endif
    }



    // Composite results from multiple passed into a final image.
    void Composition::Render()
    {
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
        auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

        ScopedTimer _prof(L"ComposeRenderPassesCS", commandList);


        // ToDo
        GpuResource* AOResources = RTAO::Args::QuarterResAO ? m_AOResources : m_RTAO.GetAOResources();
        D3D12_GPU_DESCRIPTOR_HANDLE AOSRV = Args::AOMode == Args::AOType::RTAO ? AOResources[AOResource::Smoothed].gpuDescriptorReadAccess : SSAOgpuDescriptorReadAccess;

        if (Args::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
        {
            AOSRV = AOResources[AOResource::Coefficient].gpuDescriptorReadAccess;
        }
        else //if (Args::CompositionMode == CompositionType::AmbientOcclusionOnly_TemporallySupersampled)
        {
            AOSRV = m_TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex].gpuDescriptorReadAccess;
        }

        // Update constant buffer.
        {
            m_csComposeRenderPassesCB->rtDimensions = XMUINT2(m_GBufferWidth, m_GBufferHeight);
            m_csComposeRenderPassesCB->isAOEnabled = Args::AOEnabled;
            m_csComposeRenderPassesCB->compositionType = static_cast<CompositionType>(static_cast<UINT>(Args::CompositionMode));
            m_csComposeRenderPassesCB->defaultAmbientIntensity = Pathtracer::Args::DefaultAmbientIntensity;

            m_csComposeRenderPassesCB->variance_visualizeStdDeviation = Args::Compose_VarianceVisualizeStdDeviation;
            m_csComposeRenderPassesCB->variance_scale = Args::Compose_VarianceScale;
            m_csComposeRenderPassesCB->RTAO_MaxRayHitDistance = m_RTAO.GetMaxRayHitTime();
            m_csComposeRenderPassesCB.CopyStagingToGpu(frameIndex);
        }

        // ToDo cleanup

        // Set pipeline state.
        {
            using namespace ComputeShader::RootSignature::ComposeRenderPassesCS;


            GpuResource* VarianceResource = Args::RTAODenoisingUseSmoothedVariance ? &m_varianceResources[AOVarianceResource::Smoothed] : &m_varianceResources[AOVarianceResource::Raw];
            GpuResource* LocalMeanVarianceResource = &m_localMeanVarianceResources[AOVarianceResource::Raw];
            GpuResource* RayHitDistance = RTAO::Args::QuarterResAO ? &m_AOResources[AOResource::RayHitDistance] : &m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance];


            commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
            commandList->SetComputeRootSignature(m_computeRootSigs[CSType::ComposeRenderPassesCS].Get());
            commandList->SetPipelineState(m_computePSOs[CSType::ComposeRenderPassesCS].Get());

            // Bind outputs.
            commandList->SetComputeRootDescriptorTable(Slot::Output, m_raytracingOutputIntermediate.gpuDescriptorWriteAccess);

            // Bind inputs.
            commandList->SetComputeRootDescriptorTable(Slot::GBufferResources, Pathtracer::m_GBufferResources[0].gpuDescriptorReadAccess);
#if TWO_STAGE_AO_BLUR && !ATROUS_DENOISER
            commandList->SetComputeRootDescriptorTable(Slot::AO, m_AOResources[AOResource::Coefficient].gpuDescriptorReadAccess);
#else
            commandList->SetComputeRootDescriptorTable(Slot::AO, AOSRV);
#endif
            commandList->SetComputeRootShaderResourceView(Slot::MaterialBuffer, m_materialBuffer.GpuVirtualAddress());
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_csComposeRenderPassesCB.GpuVirtualAddress(frameIndex));
            commandList->SetComputeRootDescriptorTable(Slot::Variance, VarianceResource->gpuDescriptorReadAccess);
            commandList->SetComputeRootDescriptorTable(Slot::LocalMeanVariance, LocalMeanVarianceResource->gpuDescriptorReadAccess);

            commandList->SetComputeRootDescriptorTable(Slot::FilterWeightSum, m_AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);
            commandList->SetComputeRootDescriptorTable(Slot::AORayHitDistance, RayHitDistance->gpuDescriptorReadAccess);
            commandList->SetComputeRootDescriptorTable(Slot::Color, Pathtracer::m_GBufferResources[GBufferResource::Color].gpuDescriptorReadAccess);
            commandList->SetComputeRootDescriptorTable(Slot::AOSurfaceAlbedo, Pathtracer::m_GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorReadAccess);


            commandList->SetComputeRootDescriptorTable(Slot::FrameAge, m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorReadAccess);
        }

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(m_GBufferWidth, DefaultComputeShaderParams::ThreadGroup::Width), CeilDivide(m_GBufferHeight, DefaultComputeShaderParams::ThreadGroup::Height));

        resourceStateTracker->FlushResourceBarriers();
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }


}