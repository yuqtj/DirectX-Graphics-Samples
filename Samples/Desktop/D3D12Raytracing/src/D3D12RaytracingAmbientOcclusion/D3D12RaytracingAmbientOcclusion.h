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

// ToDo move some to cpp or stdafx?
#include "DXSample.h"
#include "StepTimer.h"  // ToDo remove
#include "RaytracingSceneDefines.h"
#include "DirectXRaytracingHelper.h"
#include "RaytracingAccelerationStructure.h"
#include "CameraController.h"
#include "PerformanceTimers.h"
#include "GpuTimeManager.h"
#include "Sampler.h"
#include "UILayer.h"
#include "GpuKernels.h"
#include "PBRTParser.h"
#include "SSAO\SSAO.h"
#include "SceneParameters.h"
#include "RTAO\RTAO.h"
#include "Pathtracer\Pathtracer.h"
#include "Denoiser\Denoiser.h"
#include "Composition.h"
#include "Scene.h"
#include "EngineTuning.h"


namespace Sample
{
    namespace Args
    {
        extern EnumVar CompositionMode;
    }

    class D3D12RaytracingAmbientOcclusion;
    extern D3D12RaytracingAmbientOcclusion* g_pSample;
    static const UINT FrameCount = 3;

    GpuResource g_debugOutput[2];

    class D3D12RaytracingAmbientOcclusion : public DXSample
    {

    public:
        D3D12RaytracingAmbientOcclusion(UINT width, UINT height, std::wstring name);
        ~D3D12RaytracingAmbientOcclusion();
        // IDeviceNotify
        virtual void OnReleaseWindowSizeDependentResources() override { ReleaseWindowSizeDependentResources(); };
        virtual void OnCreateWindowSizeDependentResources() override { CreateWindowSizeDependentResources(); };

        // Messages
        virtual void OnInit();
        virtual void OnKeyDown(UINT8 key);
        virtual void OnUpdate();
        virtual void OnRender();
        virtual void OnSizeChanged(UINT width, UINT height, bool minimized);
        virtual IDXGISwapChain* GetSwapchain() { return m_deviceResources->GetSwapChain(); }

        const DX::DeviceResources& GetDeviceResources() { return *m_deviceResources; }

        void RequestGeometryInitialization(bool bRequest) { m_isGeometryInitializationRequested = bRequest; }
        void RequestASInitialization(bool bRequest) { m_isASinitializationRequested = bRequest; }
        void RequestSceneInitialization() { m_isSceneInitializationRequested = true; }
        void RequestRecreateRaytracingResources() { m_isRecreateRaytracingResourcesRequested = true; }


        static const UINT NumGrassPatchesX = 30;
        static const UINT NumGrassPatchesZ = 30;
        static const UINT MaxBLAS = 10 + NumGrassPatchesX * NumGrassPatchesZ;   // ToDo enumerate all instances in the comment

    private:

        // ToDo change ID3D12Resourcs with views to GpuResource

        std::mt19937 m_generatorURNG;

        int m_numFramesSinceASBuild;
#if TESSELATED_GEOMETRY_BOX
#if TESSELATED_GEOMETRY_THIN
        const XMFLOAT3 m_boxSize = XMFLOAT3(0.01f, 0.1f, 0.01f);
#else
        const XMFLOAT3 m_boxSize = XMFLOAT3(1, 1, 1);
#endif
        const float m_geometryRadius = 2.0f;
#else
        const float m_geometryRadius = 3.0f;
#endif

        const UINT MaxGeometryTransforms = 10000;       // ToDo lower / remove?


        ConstantBuffer<ComposeRenderPassesConstantBuffer>   m_csComposeRenderPassesCB;
        
        // ToDo cleanup - ReduceSum objects are in m_reduceSumKernel.
        ComPtr<ID3D12PipelineState>         m_computePSOs[ComputeShader::Type::Count];
        ComPtr<ID3D12RootSignature>         m_computeRootSigs[ComputeShader::Type::Count];

        GpuKernels::ReduceSum				m_reduceSumKernel;

        GpuKernels::FillInCheckerboard      m_fillInCheckerboardKernel;
        GpuKernels::GaussianFilter          m_gaussianSmoothingKernel;
        const UINT                          MaxGaussianSmoothingKernelInvocationsPerFrame = c_MaxDenoisingScaleLevels + 1; // +1 for TAO 

        // ToDo combine kernels to an array
        GpuKernels::DownsampleBoxFilter2x2	m_downsampleBoxFilter2x2Kernel;
        GpuKernels::DownsampleGaussianFilter	m_downsampleGaussian9TapFilterKernel;
        GpuKernels::DownsampleGaussianFilter	m_downsampleGaussian25TapFilterKernel;
        GpuKernels::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter m_downsampleGBufferBilateralFilterKernel; //ToDo rename?
        GpuKernels::DownsampleValueNormalDepthBilateralFilter m_downsampleValueNormalDepthBilateralFilterKernel;
        GpuKernels::UpsampleBilateralFilter	    m_upsampleBilateralFilterKernel;
        GpuKernels::MultiScale_UpsampleBilateralFilterAndCombine	    m_multiScale_upsampleBilateralFilterAndCombineKernel;
        GpuKernels::FillInMissingValuesFilter m_fillInMissingValuesFilterKernel;
        GpuKernels::BilateralFilter m_bilateralFilterKernel;

        const UINT c_SupersamplingScale = 2;    // ToDo UI parameter
        UINT								m_numCameraRayGeometryHits;

        GpuKernels::WriteValueToTexture     m_writeValueToTexture;
        GpuKernels::GenerateGrassPatch      m_grassGeometryGenerator;


        UINT                                m_animatedCarInstanceIndex;
        UINT                                m_grassInstanceIndices[NumGrassPatchesX * NumGrassPatchesZ];
        UINT                                m_currentGrassPatchVBIndex = 0;
        D3DBuffer                           m_nullVB;               // Null vertex Buffer - used for geometries that don't animate and don't need double buffering for motion vector calculation.
        UINT                                m_grassInstanceShaderRecordOffsets[2];
        UINT                                m_prevFrameLODs[NumGrassPatchesX * NumGrassPatchesZ];

        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;

        ComPtr<ID3D12Fence>                 m_fence;
        UINT64                              m_fenceValues[FrameCount];
        Microsoft::WRL::Wrappers::Event     m_fenceEvent;

        // ToDo move to deviceResources
        std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;
        std::unique_ptr<DX::DescriptorHeap> m_samplerHeap;

        // Raytracing scene
        std::vector<PrimitiveMaterialBuffer> m_materials;	// ToDO dedupe mats - hash materials
        StructuredBuffer<PrimitiveMaterialBuffer> m_materialBuffer;

        D3DTexture m_nullTexture;

        // SSAO
        SSAO        m_SSAO;
        ConstantBuffer<SSAOSceneConstantBuffer> m_SSAOCB;
        UINT m_SSAOsrvDescriptorHeapIndex = UINT_MAX;
        D3D12_GPU_DESCRIPTOR_HANDLE SSAOgpuDescriptorReadAccess = { UINT64_MAX };

        bool m_isProfiling = false;
        UINT m_numRemainingFramesToProfile = 0;
        std::map<std::wstring, std::list<std::wstring>> m_profilingResults;

        // ToDo clean up buffer management
        // SquidRoom buffers
        ComPtr<ID3D12Resource> m_vertexBuffer;
        ComPtr<ID3D12Resource> m_vertexBufferUpload;
        D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
        ComPtr<ID3D12Resource> m_indexBuffer;
        ComPtr<ID3D12Resource> m_indexBufferUpload;
        D3D12_INDEX_BUFFER_VIEW m_indexBufferView;

        struct PBRTScene
        {
            std::wstring name;
            std::string path;    // ToDo switch to wstring 
        };

        D3DTexture m_environmentMap;

        StructuredBuffer<XMFLOAT3X4> m_prevFrameBottomLevelASInstanceTransforms;        // Bottom-Level AS Instance transforms used for previous frame. Used for Temporal Reprojection.
        UINT m_maxInstanceContributionToHitGroupIndex;

        const UINT MaxNumBottomLevelInstances = 10100;           // ToDo tighten this to only what needed or add support a copy of whats used from StructuredBuffers to GPU.

        StructuredBuffer<AlignedGeometryTransform3x4> m_geometryTransforms;

        Pathtracer::Pathtracer m_pathtracer;
        RTAO::RTAO m_RTAO;
        Denoiser::Denoiser m_denoiser;
        Composition::Composition m_composition;
        Scene::Scene m_scene;

        // Raytracing output
        // ToDo use the struct
        GpuResource m_raytracingOutput;
        GpuResource m_raytracingOutputIntermediate;   // ToDo, low res res too?


        GpuResource m_AOResources[AOResource::Count];


        // Multi-scale
        // ToDo Cleanup
    public:
        static const UINT c_MaxDenoisingScaleLevels = 8;
    private:

        UINT m_GBufferWidth;
        UINT m_GBufferHeight;

        UINT m_raytracingWidth;
        UINT m_raytracingHeight;

        // Application state
        StepTimer m_timer;
        bool m_animateCamera;
        bool m_animateLight;
        bool m_animateScene;
        bool m_isCameraFrozen;
        int m_cameraChangedIndex = 0;
        bool m_hasCameraChanged = true;
        GameCore::Camera m_camera;
        float m_manualCameraRotationAngle = 0; // ToDo remove
        GameCore::Camera m_prevFrameCamera;
        std::unique_ptr<GameCore::CameraController> m_cameraController;

        // UI
        std::unique_ptr<UILayer> m_uiLayer;

        float m_fps;
        UINT m_numTriangles;
        UINT m_numInstancedTriangles;

        bool m_isGeometryInitializationRequested;
        bool m_isASinitializationRequested;
        bool m_isSceneInitializationRequested;
        bool m_isRecreateRaytracingResourcesRequested;


        static UINT             s_numInstances;

        // Render passes
        void RenderPass_GenerateGBuffers();
        void RenderPass_CalculateAmbientOcclusion();
        void RenderPass_ComposeRenderPassesCS(D3D12_GPU_DESCRIPTOR_HANDLE AOSRV);

        // ToDo cleanup
        // Utility functions
        void CreateComposeRenderPassesCSResources();
        void ParseCommandLineArgs(WCHAR* argv[], int argc);
        void RecreateD3D();
        void DownsampleRaytracingOutput();

        void UpsampleResourcesForRenderComposePass();
        // ToDo standardize const& vs *
        void BilateralUpsample(
            UINT hiResWidth,
            UINT hiResHeight,
            GpuKernels::UpsampleBilateralFilter::FilterType filterType,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDepthDerivativesResourceHandle,
            GpuResource* outputHiResValueResource,
            LPCWCHAR passName);

        void CreateConstantBuffers();
        void UpdateUI();
        void CreateDeviceDependentResources();
        void CreateWindowSizeDependentResources();
        void ReleaseDeviceDependentResources();
        void ReleaseWindowSizeDependentResources();
        void RenderRNGVisualizations();
        void CreateSamplesRNGVisualization();
        void CreateDescriptorHeaps();
        void CreateRaytracingOutputResource();
        void CreateGBufferResources();
        void CreateAuxilaryDeviceResources();
        void CopyRaytracingOutputToBackbuffer(D3D12_RESOURCE_STATES outRenderTargetState = D3D12_RESOURCE_STATE_PRESENT);
        void CalculateFrameStats();
        void WriteProfilingResultsToFile();
        //float NumCameraRaysPerSecondNumCameraRaysPerSecond() { return NumMPixelsPerSecond(m_gpuTimeManager.GetAverageMS(GpuTimers::Raytracing_GBuffer), m_raytracingWidth, m_raytracingHeight); }
    };

}