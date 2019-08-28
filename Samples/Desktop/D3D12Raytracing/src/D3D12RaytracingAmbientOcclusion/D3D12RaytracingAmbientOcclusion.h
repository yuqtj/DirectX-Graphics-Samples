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
#include "DirectXRaytracingHelper.h"
#include "RaytracingAccelerationStructure.h"
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
#include "Composition\Composition.h"
#include "Scene.h"
#include "EngineTuning.h"


namespace Sample_Args
{
    extern EnumVar CompositionMode;
}

namespace Sample
{
    class D3D12RaytracingAmbientOcclusion;
    D3D12RaytracingAmbientOcclusion& instance();

    extern void OnRecreateRaytracingResources(void*);

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

        void RequestSceneInitialization() { m_isSceneInitializationRequested = true; }
        void RequestRecreateRaytracingResources() { m_isRecreateRaytracingResourcesRequested = true; }

    private:

        // ToDo change ID3D12Resourcs with views to GpuResource

        std::mt19937 m_generatorURNG;

        // ToDo combine kernels to an array

        const UINT c_SupersamplingScale = 2;
        UINT								m_numCameraRayGeometryHits;
                
        ComPtr<ID3D12Fence>                 m_fence;
        UINT64                              m_fenceValues[FrameCount];
        Microsoft::WRL::Wrappers::Event     m_fenceEvent;

        // ToDo move to deviceResources
        std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;
        std::unique_ptr<DX::DescriptorHeap> m_samplerHeap;

        // Raytracing scene


        // ToDo move to SSAO
        // SSAO
        ConstantBuffer<SSAOSceneConstantBuffer> m_SSAOCB;
        UINT m_SSAOsrvDescriptorHeapIndex = UINT_MAX;
        D3D12_GPU_DESCRIPTOR_HANDLE SSAOgpuDescriptorReadAccess = { UINT64_MAX };

        bool m_isProfiling = false;
        UINT m_numRemainingFramesToProfile = 0;
        std::map<std::wstring, std::list<std::wstring>> m_profilingResults;
        
        Pathtracer m_pathtracer;
        RTAO m_RTAO;
        Denoiser m_denoiser;
        Composition m_composition;
        Scene m_scene;
#if ENABLE_SSAO
        SSAO::SSAO        m_SSAO;
#endif
        // Raytracing output
        // ToDo use the struct
        GpuResource m_raytracingOutput;
        GpuResource m_raytracingOutputIntermediate;   // ToDo, low res res too?
        GpuResource m_AOResources[AOResource::Count];

        GpuKernels::DownsampleBoxFilter2x2	m_downsampleBoxFilter2x2Kernel;
        GpuKernels::DownsampleGaussianFilter	m_downsampleGaussian9TapFilterKernel;
        GpuKernels::DownsampleGaussianFilter	m_downsampleGaussian25TapFilterKernel;
    private:

        UINT m_GBufferWidth;
        UINT m_GBufferHeight;

        UINT m_raytracingWidth;
        UINT m_raytracingHeight;

        // Application state

        // UI
        std::unique_ptr<UILayer> m_uiLayer;

        float m_fps;

        bool m_isSceneInitializationRequested;
        bool m_isRecreateRaytracingResourcesRequested;

        // ToDo cleanup
        // Utility functions
        void DownsampleRaytracingOutput();        // ToDo standardize const& vs *
        void ParseCommandLineArgs(WCHAR* argv[], int argc);
        void RecreateD3D();
#if ENABLE_SSAO
        void UpdateCameraMatrices();
        void CreateConstantBuffers();
#endif
        void UpdateUI();
        void CreateDeviceDependentResources();
        void CreateWindowSizeDependentResources();
        void ReleaseDeviceDependentResources();
        void ReleaseWindowSizeDependentResources();
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