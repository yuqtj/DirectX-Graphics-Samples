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

// ToDo move assets to common sample directory?

#include "stdafx.h"
#include "D3D12RaytracingAmbientOcclusion.h"
#include "GameInput.h"
#include "EngineTuning.h"
#include "EngineProfiling.h"
#include "RTAO\RTAO.h"

using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;
using namespace GameCore;

// ToDo tighten shader visibility in Root Sigs - CS + DXR

#define TWO_PASS_DENOISE 0

namespace Sample_Args
{

    //**********************************************************************************************************************************
    // Ambient Occlusion
    // TODo standardize naming in options
    namespace AOType {
        enum Enum { RTAO = 0, SSAO, Count };
    }
    const WCHAR* AOTypes[AOType::Count] = { L"Raytraced (RTAO)", L"Screen-space (MiniEngine SSAO)" };
#if REPRO_BLOCKY_ARTIFACTS_NONUNIFORM_CB_REFERENCE_SSAO
    EnumVar AOMode(L"Render/AO/Mode", AOType::SSAO, AOType::Count, AOTypes);
#else
    EnumVar AOMode(L"Render/AO/Mode", AOType::RTAO, AOType::Count, AOTypes);
#endif
}

namespace Sample
{
    HWND g_hWnd = 0;
    UIParameters g_UIparameters;    // ToDo move
    D3D12RaytracingAmbientOcclusion* g_pSample = nullptr;
    D3D12RaytracingAmbientOcclusion& instance()
    {
        return *g_pSample;
    }

    std::map<std::wstring, BottomLevelAccelerationStructureGeometry> m_bottomLevelASGeometries;
    std::unique_ptr<RaytracingAccelerationStructureManager> m_accelerationStructure;
    GpuResource m_grassPatchVB[UIParameters::NumGrassGeometryLODs][2];      // Two VBs: current and previous frame.


    void OnRecreateRaytracingResources(void*)
    {
        g_pSample->RequestRecreateRaytracingResources();
    }

    /*
RTAO - Titan XP 1440p Quarter Res
- Min kernel width 20
- Depth Sigma 0.5, Cutoff 0.9
- Low tspp 8 frames, decay 0.6, 4 blurs
- 1/2 spp

*/



    D3D12RaytracingAmbientOcclusion::D3D12RaytracingAmbientOcclusion(UINT width, UINT height, wstring name) :
        DXSample(width, height, name),
        m_isSceneInitializationRequested(false),
        m_isRecreateRaytracingResourcesRequested(false)
    {
        ThrowIfFalse(g_pSample == nullptr, L"There can be only one D3D12RaytracingAmbientOcclusion instance.");
        g_pSample = this;

        g_pSample = this;
        UpdateForSizeChange(width, height);
        m_generatorURNG.seed(1729);

    }

    // ToDo worth moving some common member vars and fncs to DxSampleRaytracing base class?
    void D3D12RaytracingAmbientOcclusion::OnInit()
    {
        m_deviceResources = make_shared<DeviceResources>(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_UNKNOWN,
            FrameCount,
            D3D_FEATURE_LEVEL_11_0,
#if ENABLE_VSYNC
            0,
#else
            // Sample shows handling of use cases with tearing support, which is OS dependent and has been supported since TH2.
            // Since the DXR requires October 2018 update, we don't need to handle non-tearing cases.
            DeviceResources::c_RequireTearingSupport,
#endif
            m_adapterIDoverride
            );
        m_deviceResources->RegisterDeviceNotify(this);
        m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);

        g_hWnd = Win32Application::GetHwnd();
        GameInput::Initialize();
        EngineTuning::Initialize();

        m_deviceResources->InitializeDXGIAdapter();

        ThrowIfFalse(IsDirectXRaytracingSupported(m_deviceResources->GetAdapter()),
            L"ERROR: DirectX Raytracing is not supported by your GPU and driver.\n\n");

        // ToDo cleanup
        m_deviceResources->CreateDeviceResources();

        // Initialize scene ToDo

        CreateDeviceDependentResources();


        m_deviceResources->CreateWindowSizeDependentResources();
    }

    D3D12RaytracingAmbientOcclusion::~D3D12RaytracingAmbientOcclusion()
    {
        GameInput::Shutdown();
    }

    void D3D12RaytracingAmbientOcclusion::WriteProfilingResultsToFile()
    {
        std::wofstream outputFile(L"Profile.csv", std::ofstream::trunc);

        // Column headers.
        size_t maxNumResults = 0;
        for (auto& column : m_profilingResults)
        {
            outputFile << column.first << L",";
            maxNumResults = max(maxNumResults, column.second.size());
        }
        outputFile << L"\n";

        // Column results.

        for (size_t i = 0; i < maxNumResults; i++)
        {
            for (auto& column : m_profilingResults)
            {
                if (column.second.size())
                {
                    outputFile << column.second.front();
                    column.second.pop_front();
                }
                outputFile << L",";
            }
            outputFile << L"\n";
        }
        outputFile.close();
    }

#if ENABLE_SSAO
    // Update camera matrices passed into the shader.
    void D3D12RaytracingAmbientOcclusion::UpdateCameraMatrices()
    {
        // SSAO.
        {
            XMMATRIX view, proj;
            m_camera.GetProj(&proj, m_GBufferWidth, m_GBufferHeight);
            view = XMMatrixLookAtLH(m_camera.Eye(), m_camera.At(), m_camera.Up());
            XMMATRIX viewProj = view * proj;

            m_SSAO.OnCameraChanged(proj);
            m_SSAOCB->cameraPosition = m_camera.Eye();
            // ToDo why transpose? Because DirectXMath uses row-major and hlsl is column-major
            m_SSAOCB->worldView = XMMatrixTranspose(view);
            m_SSAOCB->worldViewProjection = XMMatrixTranspose(viewProj);
            m_SSAOCB->projectionToWorld = XMMatrixInverse(nullptr, viewProj);

            // Update frustum.
            {
                BoundingFrustum bf;
                BoundingFrustum::CreateFromMatrix(bf, proj);

                XMMATRIX viewToWorld = XMMatrixInverse(nullptr, view);

                XMFLOAT3 corners[BoundingFrustum::CORNER_COUNT];
                bf.GetCorners(corners);

                auto lowerLeft = XMVector3Transform(
                    XMLoadFloat3(&corners[7]),
                    viewToWorld
                );
                auto lowerRight = XMVector3Transform(
                    XMLoadFloat3(&corners[6]),
                    viewToWorld
                );
                auto topLeft = XMVector3Transform(
                    XMLoadFloat3(&corners[4]),
                    viewToWorld
                );

                XMVECTOR point = XMVectorSubtract(topLeft, m_camera.Eye());
                XMVECTOR horizDelta = XMVectorSubtract(lowerRight, lowerLeft);
                XMVECTOR vertDelta = XMVectorSubtract(lowerLeft, topLeft);

                m_SSAOCB->frustumPoint = point;
                m_SSAOCB->frustumHDelta = horizDelta;
                m_SSAOCB->frustumVDelta = vertDelta;
            }
        }
    }

    void D3D12RaytracingAmbientOcclusion::CreateConstantBuffers()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto FrameCount = m_deviceResources->GetBackBufferCount();

        m_SSAOCB.Create(device, FrameCount, L"SSAO Constant Buffer");
    }
#endif

    // Create resources that depend on the device.
    void D3D12RaytracingAmbientOcclusion::CreateDeviceDependentResources()
    {
        auto device = m_deviceResources->GetD3DDevice();

        // Create a heap for descriptors.
        CreateDescriptorHeaps();

        CreateAuxilaryDeviceResources();


        m_scene.Setup(m_deviceResources, m_cbvSrvUavHeap);
        m_pathtracer.Setup(m_deviceResources, m_cbvSrvUavHeap);
        // ToDo add a note the RTAO setup has to be called after pathtracer built its shader tables and updated instanceContributionToHitGroupIndices.
        m_RTAO.Setup(m_deviceResources, m_cbvSrvUavHeap);
        m_composition.Setup(m_deviceResources, m_cbvSrvUavHeap);
#if ENABLE_SSAO
        CreateConstantBuffers();
        m_SSAO.Setup(m_deviceResources);
#endif
    }


    // Create a 2D output texture for raytracing.
    void D3D12RaytracingAmbientOcclusion::CreateRaytracingOutputResource()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

        CreateRenderTargetResource(device, backbufferFormat, m_width, m_height, m_cbvSrvUavHeap.get(), &m_raytracingOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CreateRenderTargetResource(device, backbufferFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_raytracingOutputIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }


    // ToDo move remove
#if FLOAT_TEXTURE_AS_R8_UNORM_1BYTE_FORMAT
    This has issue with small variance geting rounded to 0...
        DXGI_FORMAT texFormat = DXGI_FORMAT_R8_UNORM;       // ToDo rename to coefficient or avoid using same variable for different types.
    UINT texFormatByteSize = 1;
#elif FLOAT_TEXTURE_AS_R16_FLOAT_2BYTE_FORMAT
    DXGI_FORMAT texFormat = DXGI_FORMAT_R16_FLOAT;       // ToDo rename to coefficient or avoid using same variable for different types.
    UINT texFormatByteSize = 1;
#else
    this has issues with variance going negative
        DXGI_FORMAT texFormat = DXGI_FORMAT_R32_FLOAT;
    UINT texFormatByteSize = 4;
#endif

    void D3D12RaytracingAmbientOcclusion::CreateGBufferResources()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

        // ToDo move depth out of normal resource and switch normal to 16bit precision
        DXGI_FORMAT hitPositionFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;// DXGI_FORMAT_R16G16B16A16_FLOAT; // ToDo change to 16bit? or encode as 64bits

        DXGI_FORMAT debugFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;// DXGI_FORMAT_R32G32B32A32_FLOAT;
        // ToDo tune formats
        D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;


#if ENABLE_SSAO
        // ToDo
        m_SSAO.BindGBufferResources(Pathtracer::GBufferResources()[GBufferResource::SurfaceNormalDepth].GetResource(), Pathtracer::GBufferResources()[GBufferResource::Depth].GetResource());
#endif

        // ToDo remove unneeded ones
        // Full-res AO resources.
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
            CreateRenderTargetResource(device, m_RTAO.AOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Coefficient], initialResourceState, L"Render/AO Coefficient");

#if ATROUS_DENOISER
            CreateRenderTargetResource(device, m_RTAO.AOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
#else
            CreateRenderTargetResource(device, m_RTAO.AOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
#endif
            // ToDo 8 bit hit count?
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_UINT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::HitCount], initialResourceState, L"Render/AO Hit Count");

            // ToDo use lower bit float?
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::FilterWeightSum], initialResourceState, L"Render/AO Filter Weight Sum");
            CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::RayHitDistance], initialResourceState, L"Render/AO Hit Distance");
        }

        // ToDo remove unnecessary copies for 2 resolutions. Only keep one where possible and recreate on change.
        // ToDo pass formats via params shared across AO, GBuffer, TC


        // Debug resources
        {
            // Preallocate subsequent descriptor indices for both SRV and UAV groups.
            g_debugOutput[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(ARRAYSIZE(g_debugOutput));
            g_debugOutput[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(ARRAYSIZE(g_debugOutput));
            for (UINT i = 0; i < ARRAYSIZE(g_debugOutput); i++)
            {
                g_debugOutput[i].uavDescriptorHeapIndex = g_debugOutput[0].uavDescriptorHeapIndex + i;
                g_debugOutput[i].srvDescriptorHeapIndex = g_debugOutput[0].srvDescriptorHeapIndex + i;
                CreateRenderTargetResource(device, debugFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &g_debugOutput[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"Debug");
            }
        }

    }

    void D3D12RaytracingAmbientOcclusion::CreateAuxilaryDeviceResources()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto commandQueue = m_deviceResources->GetCommandQueue();
        auto commandList = m_deviceResources->GetCommandList();

        EngineProfiling::RestoreDevice(device, commandQueue, FrameCount);

        // ToDo move?
        m_downsampleBoxFilter2x2Kernel.Initialize(device, FrameCount);
        m_downsampleGaussian9TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap9, FrameCount);
        m_downsampleGaussian25TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap25, FrameCount); // ToDo Dedupe 9 and 25
    }

    void D3D12RaytracingAmbientOcclusion::CreateDescriptorHeaps()
    {
        auto device = m_deviceResources->GetD3DDevice();

        // ToDo use exact number?
        // 2 * GeometryType::Count + 1 + 2 * MaxBLAS + 1 + ARRAYSIZE(SquidRoomAssets::Draws) * 2 + ARRAYSIZE(SquidRoomAssets::Textures) + 1;
        // Allocate large number of descriptors.
        UINT NumDescriptors = 10000;
        m_cbvSrvUavHeap = make_shared<DX::DescriptorHeap>(device, NumDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // TodO remove
        // Sampler heap.
        {
            UINT NumDescriptors = 1;
            m_samplerHeap = make_unique<DX::DescriptorHeap>(device, NumDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        }
    }


    void D3D12RaytracingAmbientOcclusion::OnKeyDown(UINT8 key)
    {
        float fValue;
        // ToDo 
        switch (key)
        {
        case VK_ESCAPE:
            throw HrException(E_APPLICATION_EXITING);
        case 'L':
            m_scene.ToggleAnimateLight();
            break;
        case 'C':
            m_scene.ToggleAnimateCamera();
            break;
            // ToDO
#if 0
        case 'A':
            //m_animateScene = !m_animateScene;
            break;
        case 'V':
            Args::TAO_LazyRender.Bang();// TODo remove
            break;
        case 'J':
            m_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(0, 5, 0, 0)));
            m_cameraChangedIndex = 2;
            break;
        case 'M':
            m_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(0, -5, 0, 0)));
            m_cameraChangedIndex = 2;
            break;
        case 'U':
            m_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(5, 0, 0, 0)));
            m_cameraChangedIndex = 2;
            break;
        case 'Y':
            m_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(-5, 0, 0, 0)));
            m_cameraChangedIndex = 2;
            break;
        case 'O':
            m_manualCameraRotationAngle = -10;
            break;
        case 'P':
            m_manualCameraRotationAngle = 10;
            break;
        case 'B':
            m_cameraChangedIndex = 2;
            break;
#if ENABLE_PROFILING
        case VK_F9:
            if (m_isProfiling)
                WriteProfilingResultsToFile();
            else
            {
                m_numRemainingFramesToProfile = 1000;
                float perFrameSeconds = Args::CameraRotationDuration / m_numRemainingFramesToProfile;
                m_timer.SetTargetElapsedSeconds(perFrameSeconds);
                m_timer.ResetElapsedTime();
                m_animateCamera = true;
            }
            m_isProfiling = !m_isProfiling;
            m_timer.SetFixedTimeStep(m_isProfiling);
            break;
#endif
        case VK_NUMPAD1:
            Args::CompositionMode.SetValue(CompositionType::AmbientOcclusionOnly_RawOneFrame);
            break;
        case VK_NUMPAD2:
            Args::CompositionMode.SetValue(CompositionType::AmbientOcclusionOnly_Denoised);
            break;
        case VK_NUMPAD3:
            Args::CompositionMode.SetValue(CompositionType::PBRShading);
            break;
        case VK_NUMPAD0:
            Args::AOEnabled.Bang();
            break;
        case VK_NUMPAD9:
            fValue = IsInRange(m_RTAO.GetMaxRayHitTime(), 3.9f, 4.1f) ? 22.f : 4.f;
            m_RTAO.SetMaxRayHitTime(fValue);
            break;
#endif
        default:
            break;
        }
    }

    // Update frame-based values.
    void D3D12RaytracingAmbientOcclusion::OnUpdate()
    {
#if ENABLE_PROFILING
        if (m_isProfiling)
        {
            if (m_numRemainingFramesToProfile == 0)
            {
                m_isProfiling = false;
                m_timer.SetFixedTimeStep(false);
                WriteProfilingResultsToFile();
                m_animateCamera = false;
            }
            else
            {
                m_numRemainingFramesToProfile--;
            }
        }
#endif

        auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
        auto prevFrameIndex = m_deviceResources->GetPreviousFrameIndex();

        if (m_isSceneInitializationRequested)
        {
            m_isSceneInitializationRequested = false;
            m_deviceResources->WaitForGpu();
            OnInit();
        }

        if (m_isRecreateRaytracingResourcesRequested)
        {
            // ToDo what if scenargs change during rendering? race condition??
            m_isRecreateRaytracingResourcesRequested = false;
            m_deviceResources->WaitForGpu();

            // ToDo split to recreate only whats needed?
            OnCreateWindowSizeDependentResources();
            CreateAuxilaryDeviceResources();

            m_RTAO.RequestRecreateRaytracingResources();
        }


        CalculateFrameStats();

        float elapsedTime = static_cast<float>(m_scene.Timer().GetElapsedSeconds());
        GameInput::Update(elapsedTime);
        EngineTuning::Update(elapsedTime);
        EngineProfiling::Update();

        m_scene.OnUpdate();
        m_RTAO.OnUpdate();


#if ENABLE_SSAO
        // ToDo move
        // SSAO
        {
            m_SSAOCB->noiseTile = { float(m_width) / float(SSAO_NOISE_W), float(m_height) / float(SSAO_NOISE_W), 0, 0 };
            m_SSAO.SetParameters(Args::SSAONoiseFilterTolerance, Args::SSAOBlurTolerance, Args::SSAOUpsampleTolerance, Args::SSAONormalMultiply);

        }
#endif
        if (m_enableUI)
        {
            UpdateUI();
        }

    }

    // ToDo extend or remove
    // Parse supplied command line args.
    void D3D12RaytracingAmbientOcclusion::ParseCommandLineArgs(WCHAR* argv[], int argc)
    {
        DXSample::ParseCommandLineArgs(argv, argc);
    }


    // Copy the raytracing output to the backbuffer.
    void D3D12RaytracingAmbientOcclusion::CopyRaytracingOutputToBackbuffer(D3D12_RESOURCE_STATES outRenderTargetState)
    {
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
        auto renderTarget = m_deviceResources->GetRenderTarget();

        ID3D12Resource* raytracingOutput = nullptr;
        if (m_GBufferWidth == m_width && m_GBufferHeight == m_height)
        {
            raytracingOutput = m_raytracingOutputIntermediate.GetResource();
        }
        else
        {
            raytracingOutput = m_raytracingOutput.GetResource();
        }

        resourceStateTracker->FlushResourceBarriers();
        CopyResource(
            commandList,
            raytracingOutput,
            renderTarget,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            outRenderTargetState);
    }

    void D3D12RaytracingAmbientOcclusion::UpdateUI()
    {
        // ToDo average/smoothen numbers of 1/4 second.
        vector<wstring> labels;

#if 0
        // Main runtime information.
        {
            wstringstream wLabel;
            wLabel.precision(1);
            wLabel << L" GPU[" << m_deviceResources->GetAdapterID() << L"]: "
                << m_deviceResources->GetAdapterDescription() << L"\n";
            wLabel << fixed << L" FPS: " << m_fps << L"\n";
            wLabel.precision(2);
            wLabel << fixed << L" CameraRay DispatchRays: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_GBuffer) << L"ms  ~" <<
                0.001f * NumMPixelsPerSecond(GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_GBuffer), m_GBufferWidth, m_GBufferHeight) << " GigaRay/s\n";
            // ToDo use profiler from MiniEngine
            float numAOGigaRays = 1e-6f * m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits] * (RTAO::Args::QuarterResAO ? 0.25f : 1) * m_sppAO / GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_AO);
            wLabel << fixed << L" AORay DispatchRays: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_AO) << L"ms  ~" << numAOGigaRays << " GigaRay/s\n";
            wLabel << fixed << L" - AORay Adaptive Sampling ImportanceMap: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_FilterWeightSum) << L"ms  ~" << numAOGigaRays << " GigaRay/s\n";
            wLabel << fixed << L" AO Denoising: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Denoising) << L"ms\n";
            wLabel << fixed << L" - AO Blurring: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_BlurAO) << L"ms\n";
            wLabel << fixed << L" - Variance: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_Variance) << L"ms\n";
            wLabel << fixed << L" - Var Smoothing: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_VarianceSmoothing) << L"ms\n";
            wLabel << fixed << L" - AO downsample: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::DownsampleGBuffer) << L"ms\n";
            wLabel << fixed << L" - AO upsample: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::UpsampleAOBilateral) << L"ms\n";

            float numVisibilityRays = 1e-6f * m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits] / GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_Visibility);
            //wLabel << fixed << L" VisibilityRay DispatchRays: " << m_gpuTimers[GpuTimers::Raytracing_Visibility].GetAverageMS() << L"ms  ~" << numVisibilityRays << " GigaRay/s\n";
            //wLabel << fixed << L" Shading: " << m_gpuTimers[GpuTimers::ComposeRenderPassesCS].GetAverageMS() << L"ms\n";


            wLabel << fixed << L" Downsample SSAA: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::DownsampleToBackbuffer) << L"ms\n";
            wLabel.precision(1);
            /*
                    wLabel << fixed << L" AS update (BLAS / TLAS / Total): "
                           << m_gpuTimers[GpuTimers::UpdateBLAS].GetElapsedMS() << L"ms / "
                           << m_gpuTimers[GpuTimers::UpdateTLAS].GetElapsedMS() << L"ms / "
                           << m_gpuTimers[GpuTimers::UpdateBLAS].GetElapsedMS() +
                              m_gpuTimers[GpuTimers::UpdateTLAS].GetElapsedMS() << L"ms\n";
                    wLabel << fixed << L" CameraRayGeometryHits: #/%%/time "
                           << m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits] << "/"
                           << ((m_GBufferWidth * m_GBufferHeight) > 0 ? (100.f * m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits]) / (m_GBufferWidth * m_GBufferHeight) : 0) << "%%/"
                           << 1000.0f * m_gpuTimers[GpuTimers::ReduceSum].GetAverageMS(ReduceSumCalculations::CameraRayHits) << L"us \n";
                    wLabel << fixed << L" AORayGeometryHits: #/%%/time "
                           << m_numCameraRayGeometryHits[ReduceSumCalculations::AORayHits] << "/"
                        // ToDo fix up for raytracing at quarter res
                           << ((m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits] * m_sppAO) > 0 ?
                               (100.0f * m_numCameraRayGeometryHits[ReduceSumCalculations::AORayHits]) / (m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits] * m_sppAO) : 0) << "%%/"
                           << 1000.0f * m_gpuTimers[GpuTimers::ReduceSum].GetAverageMS(ReduceSumCalculations::AORayHits) << L"us \n";
                */
            labels.push_back(wLabel.str());
        }

        // Parameters.
        labels.push_back(L"\n");
        {
            wstringstream wLabel;
            wLabel << L"Scene:" << L"\n";
            wLabel << L" " << L"AS update mode: " << Args::ASUpdateMode << L"\n";
            wLabel.precision(3);
            wLabel << L" " << L"AS memory footprint: " << static_cast<double>(m_ASmemoryFootprint) / (1024 * 1024) << L"MB\n";
            // wLabel << L" " << L" # triangles per geometry: " << m_numTrianglesInTheScene << L"\n";
             //wLabel << L" " << L" # geometries per BLAS: " << Args::NumGeometriesPerBLAS << L"\n";
            // wLabel << L" " << L" # Sphere BLAS: " << Args::NumSphereBLAS << L"\n";	// ToDo fix
            wLabel << L" " << L" # total triangles: " << m_numTrianglesInTheScene << L"\n";// Args::NumSphereBLAS * Args::NumGeometriesPerBLAS* m_numTriangles[Args::SceneType] << L"\n";
            // ToDo AS memory
            labels.push_back(wLabel.str());
        }
#endif

        // ToDo fix Window Tab and UI showing the same FPS.

        // Header information
        {
            // ToDo make default resolutions round to 0
            wstringstream wLabel;
            wLabel << L"GBuffer resolution: " << m_GBufferWidth << "x" << m_GBufferHeight << L"\n";
            wLabel << L"AO raytracing resolution: " << m_raytracingWidth << "x" << m_raytracingHeight << L"\n";
            labels.push_back(wLabel.str());
        }
        // Engine tuning.
        {
            wstringstream wLabel;
            EngineTuning::Display(&wLabel, m_isProfiling);
            labels.push_back(wLabel.str());



            if (m_isProfiling)
            {
                set<wstring> profileMarkers = {
                       L"DownsampleGBuffer",
                       L"RTAO_Root",
                       L"TemporalReverseReproject",
                       L"[Sorted]CalculateAmbientOcclusion",
                       L"CalculateAmbientOcclusion_Root",
                       L"Adaptive Ray Gen",
                       L"Sort Rays",
                       L"AO DispatchRays 2D",
                       L"RenderPass_TemporalSupersamplingBlendWithCurrentFrame",
                       L"DenoiseAO",
                       L"Upsample AO",
                       L"Low-Tspp Multi-pass blur"
                };

                wstring line;
                while (getline(wLabel, line))
                {
                    std::wstringstream ss(line);
                    wstring name;
                    wstring time;
                    getline(ss, name, L':');
                    getline(ss, time);
                    for (auto& profileMarker : profileMarkers)
                    {
                        if (name.find(profileMarker) != wstring::npos)
                        {
                            m_profilingResults[profileMarker].push_back(time);
                            break;
                        }
                    }
                }
            }
        }

#if 0 // ToDo
        // Sampling info:
        {
            wstringstream wLabel;
            wLabel << L"\n";
            wLabel << L"Num samples: " << m_randomSampler.NumSamples() << L"\n";
            wLabel << L"Sample set: " << m_csHemisphereVisualizationCB->sampleSetBase / m_randomSampler.NumSamples() << " / " << m_randomSampler.NumSampleSets() << L"\n";

            labels.push_back(wLabel.str());
        }
#endif
        wstring uiText = L"";
        for (auto s : labels)
        {
            uiText += s;
        }

        m_uiLayer->UpdateLabels(uiText);
    }

    // Create resources that are dependent on the size of the main window.
    void D3D12RaytracingAmbientOcclusion::CreateWindowSizeDependentResources()
    {
        auto device = m_deviceResources->GetD3DDevice();
        auto commandQueue = m_deviceResources->GetCommandQueue();
        auto renderTargets = m_deviceResources->GetRenderTargets();

        UINT GBufferWidth, GBufferHeight;
        switch (Composition_Args::AntialiasingMode)
        {
        case DownsampleFilter::None:
            GBufferWidth = m_width;
            GBufferHeight = m_height;
            break;
        case DownsampleFilter::BoxFilter2x2:
            GBufferWidth = c_SupersamplingScale * m_width;
            GBufferHeight = c_SupersamplingScale * m_height;
            break;
        case DownsampleFilter::GaussianFilter9Tap:
        case DownsampleFilter::GaussianFilter25Tap:
            GBufferWidth = c_SupersamplingScale * m_width;
            GBufferHeight = c_SupersamplingScale * m_height;
            break;
        }

        if (RTAO_Args::QuarterResAO)
        {
            // Handle odd resolution.
            m_raytracingWidth = CeilDivide(GBufferWidth, 2);
            m_raytracingHeight = CeilDivide(GBufferHeight, 2);
        }
        else
        {
            m_raytracingWidth = GBufferWidth;
            m_raytracingHeight = GBufferHeight;
        }

        m_pathtracer.SetResolution(GBufferWidth, GBufferHeight, m_raytracingWidth, m_raytracingHeight);
        m_RTAO.SetResolution(m_raytracingWidth, m_raytracingHeight);
        m_denoiser.SetResolution(m_raytracingWidth, m_raytracingHeight);

        // Create an output 2D texture to store the raytracing result to.
        CreateRaytracingOutputResource();

        CreateGBufferResources();


#if ENABLE_SSAO
        // SSAO
        {
            m_SSAO.OnSizeChanged(GBufferWidth, GBufferHeight);
            ID3D12Resource* SSAOoutputResource = m_SSAO.GetSSAOOutputResource();
            D3D12_CPU_DESCRIPTOR_HANDLE dummyHandle;
            CreateTextureSRV(device, SSAOoutputResource, m_cbvSrvUavHeap.get(), &m_SSAOsrvDescriptorHeapIndex, &dummyHandle, &SSAOgpuDescriptorReadAccess);
        }
#endif

        if (m_enableUI)
        {
            if (!m_uiLayer)
            {
                m_uiLayer = make_unique<UILayer>(FrameCount, device, commandQueue);
            }
            m_uiLayer->Resize(renderTargets, m_width, m_height);
        }
    }

    // Release resources that are dependent on the size of the main window.
    void D3D12RaytracingAmbientOcclusion::ReleaseWindowSizeDependentResources()
    {
        if (m_enableUI)
        {
            m_uiLayer.reset();
        }
        m_raytracingOutput.resource.Reset();
    }

    // Release all resources that depend on the device.
    void D3D12RaytracingAmbientOcclusion::ReleaseDeviceDependentResources()
    {
        EngineProfiling::ReleaseDevice();

        if (m_enableUI)
        {
            m_uiLayer.reset();
        }

        m_cbvSrvUavHeap.reset();


        m_pathtracer.ReleaseDeviceDependentResources();
        m_RTAO.ReleaseDeviceDependentResources();
        m_denoiser.ReleaseDeviceDependentResources();
        m_composition.ReleaseDeviceDependentResources();
        m_scene.ReleaseDeviceDependentResources();

        m_raytracingOutput.resource.Reset();
    }

    void D3D12RaytracingAmbientOcclusion::RecreateD3D()
    {
        // Give GPU a chance to finish its execution in progress.
        try
        {
            m_deviceResources->WaitForGpu();
        }
        catch (HrException&)
        {
            // Do nothing, currently attached adapter is unresponsive.
        }
        m_deviceResources->HandleDeviceLost();
    }


    // Render the scene.
    void D3D12RaytracingAmbientOcclusion::OnRender()
    {
        if (!m_deviceResources->IsWindowVisible())
        {
            return;
        }

#if ENABLE_LAZY_RENDER
        if (!(!(Args::TAO_LazyRender && m_cameraChangedIndex <= 0)))
        {
            return;
        }
#endif

        auto commandList = m_deviceResources->GetCommandList();

        // Begin frame.
        m_deviceResources->Prepare();

        EngineProfiling::BeginFrame(commandList);

        {
            // ToDo fix - this dummy and make sure the children are properly enumerated as children in the UI output.
            ScopedTimer _prof(L"Dummy", commandList);
            {

#if ENABLE_LAZY_RENDER
                if (!(Args::TAO_LazyRender && m_cameraChangedIndex <= 0))
#endif
                {

                    m_scene.OnRender();

                    m_pathtracer.Run(m_scene);

                    if (Sample_Args::AOMode == Sample_Args::AOType::RTAO)
                    {
                        ScopedTimer _prof(L"RTAO_Root", commandList);

                        GpuResource* GBufferResources = m_pathtracer.GBufferResources();

                        m_denoiser.Run();
                        m_RTAO.Run(
                            m_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress(),
                            GBufferResources[GBufferResource::HitPosition].gpuDescriptorReadAccess,
                            GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
                            GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorReadAccess);

                        m_denoiser.Run();

                    }
#if ENABLE_SSAO
                    else // SSAO
                    {
                        ScopedTimer _prof(L"SSAO", commandList);
                        // Copy dynamic buffers to GPU.
                        {
                            auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
                            m_SSAOCB.CopyStagingToGpu(frameIndex);
                        }

                        m_SSAO.ChangeScreenScale(1.f);
                        m_SSAO.Run(m_SSAOCB.GetResource());
                    }
#endif
                }
                m_composition.Render(&m_raytracingOutput, m_scene, m_pathtracer, m_RTAO, m_denoiser, m_GBufferWidth, m_GBufferHeight);

                if (m_GBufferWidth != m_width || m_GBufferHeight != m_height)
                {
                    DownsampleRaytracingOutput();
                }

#if RENDER_RNG_SAMPLE_VISUALIZATION
                RenderRNGVisualizations();
#endif
                // UILayer will transition backbuffer to a present state.
                CopyRaytracingOutputToBackbuffer(m_enableUI ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_PRESENT);
            }
        }

        // End frame.
        EngineProfiling::EndFrame(commandList);
        m_deviceResources->ExecuteCommandList();

        // UI overlay.
        if (m_enableUI)
        {
            m_uiLayer->Render(m_deviceResources->GetCurrentFrameIndex());
        }

#if ENABLE_VSYNC
        m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT, VSYNC_PRESENT_INTERVAL);
#else
        m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT, 0);
#endif
    }


    // ToDo rename
    void D3D12RaytracingAmbientOcclusion::DownsampleRaytracingOutput()
    {
#if ENABLE_SSAA
        auto commandList = m_deviceResources->GetCommandList();
        auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

        ScopedTimer _prof(L"DownsampleToBackbuffer", commandList);

        resourceStateTracker->TransitionResource(&m_raytracingOutputIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        // ToDo pass the filter to the kernel instead of using 3 different instances
        resourceStateTracker->FlushResourceBarriers();
        switch (Args::AntialiasingMode)
        {
        case DownsampleFilter::BoxFilter2x2:
            m_downsampleBoxFilter2x2Kernel.Run(
                commandList,
                m_GBufferWidth,
                m_GBufferHeight,
                m_cbvSrvUavHeap->GetHeap(),
                m_raytracingOutputIntermediate.gpuDescriptorReadAccess,
                m_raytracingOutput.gpuDescriptorWriteAccess);
            break;
        case DownsampleFilter::GaussianFilter9Tap:
            m_downsampleGaussian9TapFilterKernel.Run(
                commandList,
                m_GBufferWidth,
                m_GBufferHeight,
                m_cbvSrvUavHeap->GetHeap(),
                m_raytracingOutputIntermediate.gpuDescriptorReadAccess,
                m_raytracingOutput.gpuDescriptorWriteAccess);
            break;
        case DownsampleFilter::GaussianFilter25Tap:
            m_downsampleGaussian25TapFilterKernel.Run(
                commandList,
                m_GBufferWidth,
                m_GBufferHeight,
                m_cbvSrvUavHeap->GetHeap(),
                m_raytracingOutputIntermediate.gpuDescriptorReadAccess,
                m_raytracingOutput.gpuDescriptorWriteAccess);
            break;
        }

        resourceStateTracker->TransitionResource(&m_raytracingOutputIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
#else
        ThrowIfFalse(0, L"ToDo");
#endif
    }


    // Compute the average frames per second and million rays per second.
    void D3D12RaytracingAmbientOcclusion::CalculateFrameStats()
    {
        static int frameCnt = 0;
        static double prevTime = 0.0f;
        double totalTime = m_scene.Timer().GetTotalSeconds();

        frameCnt++;

        // Compute averages over one second period.
        if ((totalTime - prevTime) >= 1.0f)
        {
            float diff = static_cast<float>(totalTime - prevTime);
            m_fps = static_cast<float>(frameCnt) / diff; // Normalize to an exact second.

            frameCnt = 0;
            prevTime = totalTime;

            // Display partial UI on the window title bar if UI is disabled.
            if (1)//!m_enableUI)
            {
                wstringstream windowText;
                windowText << setprecision(2) << fixed
                    << L"    fps: " << m_fps //<< L"     ~Million Primary Rays/s: " << NumCameraRaysPerSecond()
                    << L"    GPU[" << m_deviceResources->GetAdapterID() << L"]: " << m_deviceResources->GetAdapterDescription();
                SetCustomWindowText(windowText.str().c_str());
            }
        }
    }

    // Handle OnSizeChanged message event.
    void D3D12RaytracingAmbientOcclusion::OnSizeChanged(UINT width, UINT height, bool minimized)
    {
        UpdateForSizeChange(width, height);

        if (!m_deviceResources->WindowSizeChanged(width, height, minimized))
        {
            return;
        }
    }
}