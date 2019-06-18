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
	void RequestRecreateAOSamples() { m_isRecreateAOSamplesRequested = true; }

    static const UINT NumGrassPatchesX = 30;
    static const UINT NumGrassPatchesZ = 30;
    static const UINT MaxBLAS = 10 + NumGrassPatchesX * NumGrassPatchesZ;   // ToDo enumerate all instances in the comment

private:
	static const UINT FrameCount = 3;

	// ToDo change ID3D12Resourcs with views to RWGpuResource

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

	// DirectX Raytracing (DXR) attributes
	ComPtr<ID3D12StateObject> m_dxrStateObjects[RaytracingType::Count];

	// Compute resources.
	Samplers::MultiJittered m_randomSampler;

	ConstantBuffer<ComposeRenderPassesConstantBuffer>   m_csComposeRenderPassesCB;
    ConstantBuffer<AoBlurConstantBuffer> m_csAoBlurCB;
	ConstantBuffer<RNGConstantBuffer>   m_csHemisphereVisualizationCB;
	// ToDo cleanup - ReduceSum objects are in m_reduceSumKernel.
	ComPtr<ID3D12PipelineState>         m_computePSOs[ComputeShader::Type::Count];
	ComPtr<ID3D12RootSignature>         m_computeRootSigs[ComputeShader::Type::Count];

	GpuKernels::ReduceSum				m_reduceSumKernel;

    GpuKernels::AtrousWaveletTransformCrossBilateralFilter m_atrousWaveletTransformFilter;
    const UINT                          MaxAtrousWaveletTransformFilterInvocationsPerFrame = c_MaxDenoisingScaleLevels + 1; // +1 for calculating ImportanceMap

    GpuKernels::CalculateVariance       m_calculateVarianceKernel;
    GpuKernels::CalculateMeanVariance   m_calculateMeanVarianceKernel;
    const UINT                          MaxCalculateVarianceKernelInvocationsPerFrame = 
                                            MaxAtrousWaveletTransformFilterInvocationsPerFrame 
                                            + 1; // Temporal Super-Sampling.

    GpuKernels::GaussianFilter          m_gaussianSmoothingKernel;
    const UINT                          MaxGaussianSmoothingKernelInvocationsPerFrame = c_MaxDenoisingScaleLevels + 1; // +1 for TAO 

	// ToDo combine kernels to an array
    GpuKernels::RTAO_TemporalCache_ReverseReproject m_temporalCacheReverseReprojectKernel;
    GpuKernels::DownsampleBoxFilter2x2	m_downsampleBoxFilter2x2Kernel;
	GpuKernels::DownsampleGaussianFilter	m_downsampleGaussian9TapFilterKernel;
	GpuKernels::DownsampleGaussianFilter	m_downsampleGaussian25TapFilterKernel;
    GpuKernels::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter m_downsampleGBufferBilateralFilterKernel; //ToDo rename?
    GpuKernels::DownsampleValueNormalDepthBilateralFilter m_downsampleValueNormalDepthBilateralFilterKernel;
    GpuKernels::UpsampleBilateralFilter	    m_upsampleBilateralFilterKernel;
    GpuKernels::MultiScale_UpsampleBilateralFilterAndCombine	    m_multiScale_upsampleBilateralFilterAndCombineKernel;
	const UINT c_SupersamplingScale = 2;    // ToDo UI parameter
	UINT								m_numRayGeometryHits[ReduceSumCalculations::Count];

    GpuKernels::WriteValueToTexture     m_writeValueToTexture;
    GpuKernels::GenerateGrassPatch      m_grassGeometryGenerator;
    GpuKernels::SortRays                m_raySorter;


    D3D12_GPU_DESCRIPTOR_HANDLE m_nullVertexBufferGPUhandle;
    GpuKernels::CalculatePartialDerivatives  m_calculatePartialDerivativesKernel;

    UINT                                m_grassInstanceIndices[NumGrassPatchesX * NumGrassPatchesZ];
    UINT                                m_currentGrassPatchVBIndex = 0;
    RWGpuResource                       m_grassPatchVB[UIParameters::NumGrassGeometryLODs][2];      // Two VBs: current and previous frame.
    D3DBuffer                           m_nullVB;               // Null vertex Buffer - used for geometries that don't animate and don't need double buffering for motion vector calculation.
    UINT                                m_grassInstanceShaderRecordOffsets[2];
    UINT                                m_prevFrameLODs[NumGrassPatchesX * NumGrassPatchesZ];

	ComPtr<ID3D12RootSignature>         m_rootSignature;
	ComPtr<ID3D12PipelineState>         m_pipelineStateObject;

	ComPtr<ID3D12Fence>                 m_fence;
	UINT64                              m_fenceValues[FrameCount];
    Microsoft::WRL::Wrappers::Event     m_fenceEvent;
	// Root signatures
	ComPtr<ID3D12RootSignature> m_raytracingGlobalRootSignature;
	ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature[LocalRootSignature::Type::Count];

	// ToDo move to deviceResources
	std::unique_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;
	std::unique_ptr<DX::DescriptorHeap> m_samplerHeap;

	// Raytracing scene
	ConstantBuffer<SceneConstantBuffer> m_sceneCB;
	std::vector<PrimitiveMaterialBuffer> m_materials;	// ToDO dedupe mats - hash materials
	StructuredBuffer<PrimitiveMaterialBuffer> m_materialBuffer;

    D3DTexture m_nullTexture;
	
    // SSAO
    SSAO        m_SSAO;
    ConstantBuffer<SSAOSceneConstantBuffer> m_SSAOCB;
    UINT m_SSAOsrvDescriptorHeapIndex = UINT_MAX;
    D3D12_GPU_DESCRIPTOR_HANDLE SSAOgpuDescriptorReadAccess = { UINT64_MAX };
    
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

    std::map<std::wstring, BottomLevelAccelerationStructureGeometry>	m_bottomLevelASGeometries;
    D3DTexture m_environmentMap;

    std::unique_ptr<RaytracingAccelerationStructureManager> m_accelerationStructure;
    StructuredBuffer<XMFLOAT3X4> m_prevFrameBottomLevelASInstanceTransforms;        // Bottom-Level AS Instance transforms used for previous frame. Used for Temporal Reprojection.

    const UINT MaxNumBottomLevelInstances = 10100;           // ToDo tighten this to only what needed or add support a copy of whats used from StructuredBuffers to GPU.

	StructuredBuffer<AlignedGeometryTransform3x4> m_geometryTransforms;

	StructuredBuffer<AlignedUnitSquareSample2D> m_samplesGPUBuffer;
	StructuredBuffer<AlignedHemisphereSample3D> m_hemisphereSamplesGPUBuffer;

    RWGpuResource m_debugOutput[2];

	// Raytracing output
	// ToDo use the struct
	RWGpuResource m_raytracingOutput;
    RWGpuResource m_raytracingOutputIntermediate;   // ToDo, low res res too?
    RWGpuResource m_normalDepthLowPrecision[2];
    RWGpuResource m_normalDepthLowResLowPrecision[2];
	RWGpuResource m_GBufferResources[GBufferResource::Count];
    RWGpuResource m_GBufferLowResResources[GBufferResource::Count]; // ToDo remove unused
    
	RWGpuResource m_AOResources[AOResource::Count];
    RWGpuResource m_AOLowResResources[AOResource::Count];   // ToDo remove unused

    RWGpuResource m_AOTSSCoefficient[2];
    RWGpuResource m_lowResAOTSSCoefficient[2];
	RWGpuResource m_VisibilityResource;


    RWGpuResource m_AORayDirectionOriginDepth;
    RWGpuResource m_sourceToSortedRayIndex;                 // Index of the ray in the sorted array given a source index.
    RWGpuResource m_sortedToSourceRayIndex;     // Index of the ray in the source (screen space) array given a sorted index.
    RWGpuResource m_sortedRayGroupDebug;

    XMUINT2 c_shadowMapDim = XMUINT2(1024, 1024);
    RWGpuResource m_ShadowMapResource;
    bool m_updateShadowMap = true;

    // ToDo dedupe resources. Does dpeth need to have 2 instances?
    RWGpuResource m_temporalCache[2][TemporalCache::Count]; // ~array[Read/Write ping pong resource][Resources].
    
    // ToDo use a common ping-pong index?
    UINT          m_temporalCacheReadResourceIndex = 0;
    UINT          m_normalDepthCurrentFrameResourceIndex = 0;

    RWGpuResource m_varianceResource;
    RWGpuResource m_smoothedVarianceResource;
    RWGpuResource m_meanResource;
    RWGpuResource m_smoothedMeanResource;
    RWGpuResource m_meanVarianceResource;
    RWGpuResource m_smoothedMeanVarianceResource;

    // Multi-scale
    // ToDo Cleanup
    public:
        static const UINT c_MaxDenoisingScaleLevels = 8;
    private:

    struct MultiScaleDenoisingResource
    {
        RWGpuResource m_value;
        RWGpuResource m_normalDepth;
        RWGpuResource m_partialDistanceDerivatives;

        RWGpuResource m_smoothedValue;              // ToDo rename smoothed to denoised

        RWGpuResource m_downsampledSmoothedValue;   // ToDo could be removed and reuse m_value from the higher i of ms_resources.
        RWGpuResource m_downsampledNormalDepthValue;   // ToDo could be removed and reuse m_value from the higher i of ms_resources.
        RWGpuResource m_downsampledPartialDistanceDerivatives;   // ToDo could be removed and reuse m_value from the higher i of ms_resources.

        RWGpuResource m_varianceResource;
        RWGpuResource m_smoothedVarianceResource;
    };
    MultiScaleDenoisingResource m_multiScaleDenoisingResources[c_MaxDenoisingScaleLevels];
    
	UINT m_GBufferWidth;
	UINT m_GBufferHeight;

    UINT m_raytracingWidth;
    UINT m_raytracingHeight;

	// Shader tables
	static const wchar_t* c_hitGroupNames_TriangleGeometry[RayType::Count];
	static const wchar_t* c_rayGenShaderNames[RayGenShaderType::Count];
	static const wchar_t* c_closestHitShaderNames[RayType::Count];
	static const wchar_t* c_missShaderNames[RayType::Count];

	ComPtr<ID3D12Resource> m_rayGenShaderTables[RaytracingType::Count][RayGenShaderType::Count];
	UINT m_rayGenShaderTableRecordSizeInBytes[RaytracingType::Count];
	ComPtr<ID3D12Resource> m_hitGroupShaderTable[RaytracingType::Count];
	UINT m_hitGroupShaderTableStrideInBytes[RaytracingType::Count];
	ComPtr<ID3D12Resource> m_missShaderTable[RaytracingType::Count];
	UINT m_missShaderTableStrideInBytes[RaytracingType::Count];

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
	
	// AO
	// ToDo fix artifacts at 4. Looks like selfshadowing on some AOrays in SquidScene
	UINT m_sppAO;	// Samples per pixel for Ambient Occlusion.

	// UI
	std::unique_ptr<UILayer> m_uiLayer;
	
	float m_fps;
	UINT m_numTriangles;
    UINT m_numInstancedTriangles;

	bool m_isGeometryInitializationRequested;
	bool m_isASinitializationRequested;
	bool m_isSceneInitializationRequested;
	bool m_isRecreateRaytracingResourcesRequested;
	bool m_isRecreateAOSamplesRequested;

	// Render passes
	void RenderPass_GenerateGBuffers();
	void RenderPass_CalculateVisibility();
    void RenderPass_GenerateShadowMap();
	void RenderPass_CalculateAmbientOcclusion();
    void RenderPass_BlurAmbientOcclusion();
	void RenderPass_ComposeRenderPassesCS(D3D12_GPU_DESCRIPTOR_HANDLE AOSRV);
    void RenderPass_TestEarlyExitOVerhead();

	// ToDo cleanup
	// Utility functions
    void GenerateGrassGeometry();
	void CreateComposeRenderPassesCSResources();
    void CreateAoBlurCSResources();
	void ParseCommandLineArgs(WCHAR* argv[], int argc);
	void RecreateD3D();
    void LoadSquidRoom();
    void CreateIndexAndVertexBuffers(const GeometryDescriptor& desc, D3DGeometry* geometry);
	void LoadPBRTScene();
	void LoadSceneGeometry();
    void UpdateCameraMatrices();
	void UpdateBottomLevelASTransforms();
	void UpdateSphereGeometryTransforms();
    void UpdateGridGeometryTransforms();
    void InitializeScene();
	void UpdateAccelerationStructure();
	void DispatchRays(RaytracingType::Enum raytracingType, ID3D12Resource* rayGenShaderTable, uint32_t width=0, uint32_t height=0);
	void CalculateRayHitCount(ReduceSumCalculations::Enum type);
    void ApplyAtrousWaveletTransformFilter();
    void ApplyAtrousWaveletTransformFilter(const  RWGpuResource& inValueResource, const  RWGpuResource& inNormalDepthResource, const  RWGpuResource& inDepthResource, const  RWGpuResource& inRayHitDistanceResource, const  RWGpuResource& inPartialDistanceDerivativesResource, RWGpuResource* outSmoothedValueResource, RWGpuResource* varianceResource, RWGpuResource* smoothedVarianceResource, UINT calculateVarianceTimerId, UINT smoothVarianceTimerId, UINT atrousFilterTimerId);
    void ApplyMultiScaleAtrousWaveletTransformFilter();
    void CalculateAdaptiveSamplingCounts();
    void RenderPass_TemporalCacheReverseProjection();

	void DownsampleRaytracingOutput();
    void DownsampleGBuffer();

    void UpsampleResourcesForRenderComposePass();
    // ToDo standardize const& vs *
    void BilateralUpsample(
        UINT hiResWidth,
        UINT hiResHeight,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDepthDerivativesResourceHandle,
        RWGpuResource* outputHiResValueResource,
        LPCWCHAR passName);

    void CreateConstantBuffers();
    void CreateSamplesRNG();
	void UpdateUI();
    void CreateDeviceDependentResources();
    void CreateWindowSizeDependentResources();
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources();
    void RenderRNGVisualizations();
    void CreateRootSignatures();
    void CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline, bool shadowOnly = false);
    void CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline, bool shadowOnly = false);
    void CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline, bool shadowOnly = false);
    void CreateRaytracingPipelineStateObject();
    void CreateDescriptorHeaps();
    void CreateRaytracingOutputResource();
	void CreateGBufferResources();
	void CreateAuxilaryDeviceResources();
    void InitializeGrassGeometry();
    void InitializeGeometry();
    void BuildPlaneGeometry();
    void BuildTesselatedGeometry();
	void GenerateBottomLevelASInstanceTransforms();
    void InitializeAllBottomLevelAccelerationStructures();
    void InitializeAccelerationStructures();
    void BuildShaderTables(RaytracingType::Enum raytracingType);
    void CopyRaytracingOutputToBackbuffer(D3D12_RESOURCE_STATES outRenderTargetState = D3D12_RESOURCE_STATE_PRESENT);
    void CalculateFrameStats();
	//float NumCameraRaysPerSecondNumCameraRaysPerSecond() { return NumMPixelsPerSecond(m_gpuTimeManager.GetAverageMS(GpuTimers::Raytracing_GBuffer), m_raytracingWidth, m_raytracingHeight); }
};
