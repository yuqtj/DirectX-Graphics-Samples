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
#include "Denoiser.h"
#include "GameInput.h"
#include "EngineTuning.h"
#include "EngineProfiling.h"
#include "GpuTimeManager.h"
#include "D3D12RaytracingAmbientOcclusion.h"

// ToDo prune unused
using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;

namespace Denoiser
{
    // Singleton instance.
    Denoiser* g_pPathracer;
    UINT Denoiser::s_numInstances = 0;

    namespace Args
    {
    }
    Denoiser::Denoiser()
    {
        ThrowIfFalse(++s_numInstances == 1, L"There can be only one Denoiser instance.");
        g_pPathracer = this;

        for (auto& rayGenShaderTableRecordSizeInBytes : m_rayGenShaderTableRecordSizeInBytes)
        {
            rayGenShaderTableRecordSizeInBytes = UINT_MAX;
        }
    }

    void Denoiser::Setup(shared_ptr<DeviceResources> deviceResources, shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT maxInstanceContributionToHitGroupIndex)
    {
        m_deviceResources = deviceResources;
        m_cbvSrvUavHeap = descriptorHeap;

        CreateDeviceDependentResources(maxInstanceContributionToHitGroupIndex);
    }

    void Denoiser::ReleaseDeviceDependentResources()
    {
    }

    // Create resources that depend on the device.
    void Denoiser::CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex)
    {
        CreateAuxilaryDeviceResources();

    }


    // ToDo rename
    void Denoiser::CreateAuxilaryDeviceResources()
    {
    }


    void Denoiser::OnUpdate()
    {

    }

    void Denoiser::CreateResolutionDependentResources()
    {
    }


    void Denoiser::SetResolution(UINT width, UINT height)
    {
    }



    void Denoiser::CreateTextureResources()
    {
    }


    void Denoiser::RenderPass_TemporalSupersamplingReverseProjection()
    {
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

        ScopedTimer _prof(L"Temporal Supersampling p1 (Reverse Reprojection)", commandList);

        GpuResource* GBufferResources = m_pathtracer.GetGBufferResources(RTAO::Args::QuarterResAO);

        UINT prevFrameNormalDepthResourceIndex = (m_normalDepthCurrentFrameResourceIndex + 1) % 2;

        // Ping-pong input output indices across frames.
        UINT temporalCachePreviousFrameResourceIndex = m_temporalCacheCurrentFrameResourceIndex;
        m_temporalCacheCurrentFrameResourceIndex = (m_temporalCacheCurrentFrameResourceIndex + 1) % 2;

        GpuResource* TSSAOCoefficient = RTAO::Args::QuarterResAO ? m_lowResTSSAOCoefficient : m_TSSAOCoefficient;
        UINT temporalCachePreviousFrameTSSAOCoeficientResourceIndex = m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex;
        m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex = (m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex + 1) % 2;

        // ToDo zero out caches on resource reset.

        // ToDo
        // Calculate reverse projection transform T to the previous frame's screen space coordinates.
        //  xy(t-1) = xy(t) * T     // ToDo check mul order
        // The reverse projection transform consists:
        //  1) reverse projecting from current's frame screen space coordinates to world space coordinates
        //  2) projecting from world space coordinates to previous frame's screen space coordinates
        //
        //  T = inverse(P(t)) * inverse(V(t)) * V(t-1) * P(t-1) 
        //      where P is a projection transform and V is a view transform. 
        // Ref: ToDo
        XMMATRIX view, proj, prevView, prevProj;

        m_camera.GetProj(&proj, m_raytracingWidth, m_raytracingHeight);
        m_prevFrameCamera.GetProj(&prevProj, m_raytracingWidth, m_raytracingHeight);

        // ToDO can we remove this or document.
        // Calculate view matrix as if the camera was at (0,0,0) to avoid 
        // precision issues when camera position is too far from (0,0,0).
        // GenerateCameraRay takes this into consideration in the raytracing shader.
        view = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 1), XMVectorSetW(m_camera.At() - m_camera.Eye(), 1), m_camera.Up());
        prevView = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 1), XMVectorSetW(m_prevFrameCamera.At() - m_prevFrameCamera.Eye(), 1), m_prevFrameCamera.Up());

        XMMATRIX viewProj = view * proj;
        XMMATRIX prevViewProj = prevView * prevProj;
        XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);
        XMMATRIX prevInvViewProj = XMMatrixInverse(nullptr, prevViewProj);


        // Transition output resource to UAV state.        
        {
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->TransitionResource(&m_cachedFrameAgeValueSquaredValueRayHitDistance, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        UINT maxFrameAge = static_cast<UINT>(1 / Args::RTAO_TemporalSupersampling_MinSmoothingFactor);
        resourceStateTracker->FlushResourceBarriers();
        m_temporalCacheReverseReprojectKernel.Execute(
            commandList,
            m_raytracingWidth,
            m_raytracingHeight,
            m_cbvSrvUavHeap->GetHeap(),
            Pathtracer::g_GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::ReprojectedNormalDepth].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::MotionVector].gpuDescriptorReadAccess,
            TSSAOCoefficient[temporalCachePreviousFrameTSSAOCoeficientResourceIndex].gpuDescriptorReadAccess,
            m_prevFrameGBufferNormalDepth.gpuDescriptorReadAccess,
            m_temporalCache[temporalCachePreviousFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorReadAccess,
            m_temporalCache[temporalCachePreviousFrameResourceIndex][TemporalSupersampling::CoefficientSquaredMean].gpuDescriptorReadAccess,
            m_temporalCache[temporalCachePreviousFrameResourceIndex][TemporalSupersampling::RayHitDistance].gpuDescriptorReadAccess,
            m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorWriteAccess,
            m_cachedFrameAgeValueSquaredValueRayHitDistance.gpuDescriptorWriteAccess,
            Args::RTAO_TemporalSupersampling_MinSmoothingFactor,
            Args::RTAO_TemporalSupersampling_DepthTolerance,
            Args::RTAO_TemporalSupersampling_UseDepthWeights,
            Args::RTAO_TemporalSupersampling_UseNormalWeights,
            Args::RTAO_TemporalSupersampling_ClampCachedValues_AbsoluteDepthTolerance,
            Args::RTAO_TemporalSupersampling_ClampCachedValues_DepthBasedDepthTolerance,
            Args::RTAO_TemporalSupersampling_ClampCachedValues_DepthSigma,
            Args::RTAO_TemporalSupersampling_UseWorldSpaceDistance,
            RTAO::Args::QuarterResAO,
            Args::RTAO_TemporalSupersampling_PerspectiveCorrectDepthInterpolation,
#if !NORMAL_DEPTH_R8G8B16_ENCODING
            static_cast<TextureResourceFormatRGB::Type>(static_cast<UINT>(Args::RTAO_TemporalSupersampling_NormalDepthResourceFormat)),
#endif
            m_debugOutput,
            invViewProj,
            prevInvViewProj,
            maxFrameAge,
            Args::RTAODenoisingExtraRaysToTraceSinceTSSMovement,
            Args::RTAO_TemporalSupersampling_TestFlag);

        // Transition output resources to SRV state.        
        // ToDo use it as UAV in RTAO?
        // Only the frame age is transitioned out of UAV state as it used in RTAO pass. 
        // All the others are used as input/output UAVs in 2nd stage of Temporal Supersampling.
        {
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->TransitionResource(&m_cachedFrameAgeValueSquaredValueRayHitDistance, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->InsertUAVBarrier(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge]);
        }
    }

    void  Denoiser::RenderPass_TemporalSupersamplingBlendWithCurrentFrame()
    {
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

        ScopedTimer _prof(L"RenderPass_TemporalSupersamplingBlendWithCurrentFrame", commandList);

        GpuResource* GBufferResources = m_pathtracer.GetGBufferResources(RTAO::Args::QuarterResAO);
        GpuResource* AOResources = m_RTAO.AOResources();

        GpuResource* TSSAOCoefficient = RTAO::Args::QuarterResAO ? m_lowResTSSAOCoefficient : m_TSSAOCoefficient;

        GpuResource* VarianceResources = RTAO::Args::QuarterResAO ? m_lowResVarianceResource : m_varianceResource;
        GpuResource* LocalMeanVarianceResources = RTAO::Args::QuarterResAO ? m_lowResLocalMeanVarianceResource : m_localMeanVarianceResource;

        // ToDo remove
        if (Args::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
        {
            // ToDo
            //m_temporalCacheFrameAge = 0;
        }

        // ToDo zero out caches on resource reset.

        // ToDo reuse calculated variance for both TAO and denoising.
        // Transition all output resources to UAV state.
        {
            resourceStateTracker->TransitionResource(&LocalMeanVarianceResources[AOVarianceResource::Raw], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->InsertUAVBarrier(&AOResources[AOResource::Coefficient]);
        }

        bool isCheckerboardSamplingEnabled;
        bool checkerboardLoadEvenPixels;
        m_RTAO.GetRayGenParameters(&isCheckerboardSamplingEnabled, &checkerboardLoadEvenPixels);

        // ToDO Should use separable box filter instead?. Bilateral doesn't work for pixels that don't
        // have anycontribution with bilateral - their variance will be zero. Or set a variance to non-zero in that case?
        // Calculate local mean and variance.
        {
            // ToDo add Separable Bilateral and Square bilateral support how it affects image quality.
            // ToDo checkerboard is same perf ?
            ScopedTimer _prof(L"Calculate Mean and Variance", commandList);
            resourceStateTracker->FlushResourceBarriers();
            m_calculateMeanVarianceKernel.Execute(
                commandList,
                m_cbvSrvUavHeap->GetHeap(),
                m_raytracingWidth,
                m_raytracingHeight,
                //GpuKernels::CalculateMeanVariance::FilterType::Separable_AnyToAnyWaveReadLaneAt,
                GpuKernels::CalculateMeanVariance::FilterType::Separable_CheckerboardSampling_AnyToAnyWaveReadLaneAt,
                AOResources[AOResource::Coefficient].gpuDescriptorReadAccess,
                LocalMeanVarianceResources[AOVarianceResource::Raw].gpuDescriptorWriteAccess,
                Args::VarianceBilateralFilterKernelWidth,
                isCheckerboardSamplingEnabled,
                checkerboardLoadEvenPixels);

            // Interpolate the variance for the inactive cells from the valid checherkboard cells.
            if (isCheckerboardSamplingEnabled)
            {
                bool fillEvenPixels = !checkerboardLoadEvenPixels;
                resourceStateTracker->FlushResourceBarriers();
                m_fillInCheckerboardKernel.Execute(
                    commandList,
                    m_cbvSrvUavHeap->GetHeap(),
                    m_raytracingWidth,
                    m_raytracingHeight,
                    GpuKernels::FillInCheckerboard::FilterType::CrossBox4TapFilter,
                    LocalMeanVarianceResources[AOVarianceResource::Smoothed].gpuDescriptorReadAccess,
                    LocalMeanVarianceResources[AOVarianceResource::Raw].gpuDescriptorWriteAccess,
                    fillEvenPixels);
            }

            resourceStateTracker->TransitionResource(&LocalMeanVarianceResources[AOVarianceResource::Raw], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->InsertUAVBarrier(&LocalMeanVarianceResources[AOVarianceResource::Raw]);
        }
#if 0 // !VARIABLE_RATE_RAYTRACING
        // ToDo - the filter needs to check for invalid values...
        // ToDo should we be smoothing before temporal?
        // Smoothen the local variance which is prone to error due to undersampled input.
        {
            {
                ScopedTimer _prof(L"Mean Variance Smoothing", commandList);
                resourceStateTracker->FlushResourceBarriers();
                m_gaussianSmoothingKernel.Execute(
                    commandList,
                    m_raytracingWidth,
                    m_raytracingHeight,
                    GpuKernels::GaussianFilter::Filter3x3RG,
                    m_cbvSrvUavHeap->GetHeap(),
                    LocalMeanVarianceResources[AOVarianceResource::Raw].gpuDescriptorReadAccess,
                    LocalMeanVarianceResources[AOVarianceResource::Smoothed].gpuDescriptorWriteAccess);
            }
        }
#endif

        {
            resourceStateTracker->InsertUAVBarrier(&LocalMeanVarianceResources[AOVarianceResource::Smoothed]);
        }


        bool fillInMissingValues = false;   // ToDo fix up barriers if changing this to true
#if 0
        // ToDo?
        Args::RTAODenoisingLowTsppFillMissingValues
            && m_RTAO.GetSpp() < 1;
#endif
        GpuResource* TSSOutCoefficient = fillInMissingValues ? &m_temporalSupersampling_blendedAOCoefficient[0] : &TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex];

        // Transition output resource to UAV state.      
        {
            resourceStateTracker->TransitionResource(TSSOutCoefficient, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::CoefficientSquaredMean], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->TransitionResource(&VarianceResources[AOVarianceResource::Raw], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->TransitionResource(&m_multiPassDenoisingBlurStrength, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->InsertUAVBarrier(&m_cachedFrameAgeValueSquaredValueRayHitDistance);
        }

        resourceStateTracker->FlushResourceBarriers();
        m_temporalCacheBlendWithCurrentFrameKernel.Execute(
            commandList,
            m_raytracingWidth,
            m_raytracingHeight,
            m_cbvSrvUavHeap->GetHeap(),
            AOResources[AOResource::Coefficient].gpuDescriptorReadAccess,
#if VARIABLE_RATE_RAYTRACING
            LocalMeanVarianceResources[AOVarianceResource::Raw].gpuDescriptorReadAccess,
#else
            m_smoothedLocalMeanVarianceResource.gpuDescriptorReadAccess,
#endif
            AOResources[AOResource::RayHitDistance].gpuDescriptorReadAccess,
            TSSOutCoefficient->gpuDescriptorWriteAccess,
            m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorWriteAccess,
            m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::CoefficientSquaredMean].gpuDescriptorWriteAccess,
            m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance].gpuDescriptorWriteAccess,
            m_cachedFrameAgeValueSquaredValueRayHitDistance.gpuDescriptorReadAccess,
            VarianceResources[AOVarianceResource::Raw].gpuDescriptorWriteAccess,
            m_multiPassDenoisingBlurStrength.gpuDescriptorWriteAccess,
            Args::RTAO_TemporalSupersampling_MinSmoothingFactor,
            Args::RTAO_TemporalSupersampling_ForceUseMinSmoothingFactor,
            Args::RTAO_TemporalSupersampling_ClampCachedValues_UseClamping,
            Args::RTAO_TemporalSupersampling_ClampCachedValues_StdDevGamma,
            Args::RTAO_TemporalSupersampling_ClampCachedValues_MinStdDevTolerance,
            Args::RTAODenoising_MinFrameAgeToUseTemporalVariance,
            Args::RTAO_TemporalSupersampling_ClampDifferenceToFrameAgeScale,
            m_debugOutput,
            Args::RTAODenoisingnumFramesToDenoiseAfterLastTracedRay,
            Args::RTAODenoisingLowTsppMaxFrameAge,
            Args::RTAODenoisingLowTsppDecayConstant,
            isCheckerboardSamplingEnabled,
            checkerboardLoadEvenPixels);

        // Transition output resource to SRV state.        
        {
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::CoefficientSquaredMean], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->TransitionResource(&m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->TransitionResource(&VarianceResources[AOVarianceResource::Raw], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->TransitionResource(&m_multiPassDenoisingBlurStrength, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        // ToDo remove make condiotional
        // Smoothen the variance.
        {
            {
                resourceStateTracker->TransitionResource(&VarianceResources[AOVarianceResource::Smoothed], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                resourceStateTracker->InsertUAVBarrier(&VarianceResources[AOVarianceResource::Raw]);
            }

            // ToDo should we be smoothing before temporal?
            // Smoothen the local variance which is prone to error due to undersampled input.
            {
                {
                    ScopedTimer _prof(L"Mean Variance Smoothing", commandList);
                    resourceStateTracker->FlushResourceBarriers();
                    m_gaussianSmoothingKernel.Execute(
                        commandList,
                        m_raytracingWidth,
                        m_raytracingHeight,
                        GpuKernels::GaussianFilter::Filter3x3,
                        m_cbvSrvUavHeap->GetHeap(),
                        VarianceResources[AOVarianceResource::Raw].gpuDescriptorReadAccess,
                        VarianceResources[AOVarianceResource::Smoothed].gpuDescriptorWriteAccess);
                }
            }

            resourceStateTracker->TransitionResource(&VarianceResources[AOVarianceResource::Smoothed], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }

        // ToDo?
        if (fillInMissingValues)
        {
            // Fill in missing/disoccluded values.
            {
#if 1
                // ToDo should we use a wider filter?
                if (isCheckerboardSamplingEnabled)
                {
                    bool fillEvenPixels = !checkerboardLoadEvenPixels;
                    resourceStateTracker->FlushResourceBarriers();
                    m_fillInCheckerboardKernel.Execute(
                        commandList,
                        m_cbvSrvUavHeap->GetHeap(),
                        m_raytracingWidth,
                        m_raytracingHeight,
                        GpuKernels::FillInCheckerboard::FilterType::CrossBox4TapFilter,
                        LocalMeanVarianceResources[AOVarianceResource::Smoothed].gpuDescriptorReadAccess,
                        TSSOutCoefficient->gpuDescriptorWriteAccess,
                        fillEvenPixels);

                }
#else
                ScopedTimer _prof(L"Fill in missing values filter", commandList);
                {
                    resourceStateTracker->TransitionResource(&TSSAOCoefficient[m_temporalCacheCurrentFrameResourceIndex], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    resourceStateTracker->InsertUAVBarrier(&TSSOutCoefficient->resource.Get()),
                }

                resourceStateTracker->FlushResourceBarriers();
                m_fillInMissingValuesFilterKernel.Execute(
                    commandList,
                    m_raytracingWidth,
                    m_raytracingHeight,
                    GpuKernels::FillInMissingValuesFilter::DepthAware_GaussianFilter7x7,
                    1,
                    isCheckerboardSamplingEnabled,
                    checkerboardLoadEvenPixels,
                    m_cbvSrvUavHeap->GetHeap(),
                    TSSOutCoefficient->gpuDescriptorReadAccess,
                    GBufferResources[GBufferResource::Depth].gpuDescriptorReadAccess,
                    TSSAOCoefficient[m_temporalCacheCurrentFrameResourceIndex].gpuDescriptorWriteAccess);

                {
                    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    D3D12_RESOURCE_BARRIER barriers[] = {
                        resourceStateTracker->TransitionResource(&TSSAOCoefficient[m_temporalCacheCurrentFrameResourceIndex], after);
                    };
                    commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
                }
#endif
            }
        }
        resourceStateTracker->TransitionResource(TSSOutCoefficient, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    void Denoiser::MultiPassBlur()
    {
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

        ScopedTimer _prof(L"Low-Tspp Multi-pass blur", commandList);

        UINT numPasses = static_cast<UINT>(Args::RTAODenoisingLowTspBlurPasses);

        GpuResource* GBufferResources = m_pathtracer.GetGBufferResources(RTAO::Args::QuarterResAO);
        GpuResource* AOResources = m_RTAO.AOResources();

        GpuResource* resources[2] = {
            &m_temporalSupersampling_blendedAOCoefficient[0],
            &m_temporalSupersampling_blendedAOCoefficient[1],
        };

        GpuResource* TSSAOCoefficient = RTAO::Args::QuarterResAO ? m_lowResTSSAOCoefficient : m_TSSAOCoefficient;
        GpuResource* OutResource = &TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex];
        //GpuResource* OutResource = &AOResources[AOResource::Smoothed];

        bool readWriteUAV_and_skipPassthrough = false;// (numPasses % 2) == 1;

        if (Args::RTAODenoisingLowTsppUseUAVReadWrite)
        {
            readWriteUAV_and_skipPassthrough = true;
            resourceStateTracker->TransitionResource(OutResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        GpuKernels::BilateralFilter::FilterType filter =
            Args::RTAODenoisingLowTsppUseNormalWeights
            ? GpuKernels::BilateralFilter::NormalDepthAware_GaussianFilter5x5
            : GpuKernels::BilateralFilter::DepthAware_GaussianFilter5x5;

        GpuResource* depthResource =
            Args::RTAODenoisingLowTsppUseNormalWeights
            ? &GBufferResources[GBufferResource::SurfaceNormalDepth]
            : &GBufferResources[GBufferResource::Depth];

        UINT FilterSteps[4] = {
            1, 4, 8, 16
        };

        UINT filterStep = 1;

        for (UINT i = 0; i < numPasses; i++)
        {
            // filterStep = FilterSteps[i];
            wstring passName = L"Depth Aware Gaussian Blur with a pixel step " + to_wstring(filterStep);
            ScopedTimer _prof(passName.c_str(), commandList);


            if (Args::RTAODenoisingLowTsppUseUAVReadWrite)
            {
                resourceStateTracker->InsertUAVBarrier(OutResource);

                resourceStateTracker->FlushResourceBarriers();
                m_bilateralFilterKernel.Execute(
                    commandList,
                    filter,
                    filterStep,
                    Args::RTAODenoisingLowTsppNormalExponent,
                    Args::RTAODenoisingLowTsppMinNormalWeight,
                    m_cbvSrvUavHeap->GetHeap(),
                    m_temporalSupersampling_blendedAOCoefficient[0].gpuDescriptorReadAccess,
                    depthResource->gpuDescriptorReadAccess,
                    m_multiPassDenoisingBlurStrength.gpuDescriptorReadAccess,
                    OutResource,
                    readWriteUAV_and_skipPassthrough);
            }
            else
            {
                GpuResource* inResource = i > 0 ? resources[i % 2] : OutResource;
                GpuResource* outResource = i < numPasses - 1 ? resources[(i + 1) % 2] : OutResource;

                {
                    resourceStateTracker->TransitionResource(outResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    resourceStateTracker->InsertUAVBarrier(inResource);
                }

                resourceStateTracker->FlushResourceBarriers();
                m_bilateralFilterKernel.Execute(
                    commandList,
                    filter,
                    filterStep,
                    Args::RTAODenoisingLowTsppNormalExponent,
                    Args::RTAODenoisingLowTsppMinNormalWeight,
                    m_cbvSrvUavHeap->GetHeap(),
                    inResource->gpuDescriptorReadAccess,
                    GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
                    m_multiPassDenoisingBlurStrength.gpuDescriptorReadAccess,
                    outResource,
                    readWriteUAV_and_skipPassthrough);

                {
                    resourceStateTracker->TransitionResource(outResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    resourceStateTracker->InsertUAVBarrier(outResource);
                }
            }

            filterStep *= 2;
        }


        if (Args::RTAODenoisingLowTsppUseUAVReadWrite)
        {
            resourceStateTracker->TransitionResource(OutResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->InsertUAVBarrier(OutResource);
        }
    }

}