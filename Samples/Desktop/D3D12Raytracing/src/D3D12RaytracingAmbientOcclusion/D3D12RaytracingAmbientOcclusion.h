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


namespace Sample_GPUTime {
    enum { Pathtracing = 0, AOraytracing, AOdenoising, Count };
}

namespace Sample
{
    class D3D12RaytracingAmbientOcclusion;
    D3D12RaytracingAmbientOcclusion& instance();

    extern void OnRecreateRaytracingResources(void*);

    static const UINT FrameCount = 3;

    extern GpuResource g_debugOutput[2];

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
        bool m_isProfiling = false;
        UINT m_numRemainingFramesToProfile = 0;
        std::map<std::wstring, std::list<std::wstring>> m_profilingResults;

        // Game components
        Pathtracer m_pathtracer;
        RTAO m_RTAO;
        Denoiser m_denoiser;
        Composition m_composition;
        Scene m_scene;

        // Raytracing output
        // ToDo use the struct
        GpuResource m_raytracingOutput;
    private:

        UINT m_raytracingWidth;
        UINT m_raytracingHeight;

        // Application state
        DX::GPUTimer m_sampleGpuTimes[Sample_GPUTime::Count];

        // UI
        std::unique_ptr<UILayer> m_uiLayer;

        float m_fps;

        bool m_isSceneInitializationRequested;
        bool m_isRecreateRaytracingResourcesRequested;

        // ToDo cleanup
        // Utility functions      // ToDo standardize const& vs *
        void ParseCommandLineArgs(WCHAR* argv[], int argc);
        void RecreateD3D();
        void UpdateUI();
        void CreateDeviceDependentResources();
        void CreateWindowSizeDependentResources();
        void ReleaseDeviceDependentResources();
        void ReleaseWindowSizeDependentResources();
        void CreateDescriptorHeaps();
        void CreateRaytracingOutputResource();
        void CreateDebugResources();
        void CreateAuxilaryDeviceResources();
        void CopyRaytracingOutputToBackbuffer(D3D12_RESOURCE_STATES outRenderTargetState = D3D12_RESOURCE_STATE_PRESENT);
        void CalculateFrameStats();
        void WriteProfilingResultsToFile();
        //float NumCameraRaysPerSecondNumCameraRaysPerSecond() { return NumMPixelsPerSecond(m_gpuTimeManager.GetAverageMS(GpuTimers::Raytracing_GBuffer), m_raytracingWidth, m_raytracingHeight); }
    };

}