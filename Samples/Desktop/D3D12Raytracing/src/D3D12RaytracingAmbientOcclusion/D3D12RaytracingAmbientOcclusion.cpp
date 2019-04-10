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
#include "CompiledShaders\Raytracing.hlsl.h"
#include "CompiledShaders\RNGVisualizerCS.hlsl.h"
#include "CompiledShaders\ComposeRenderPassesCS.hlsl.h"
#include "CompiledShaders\AoBlurCS.hlsl.h"
#include "CompiledShaders\AoBlurAndUpsampleCS.hlsl.h"
#include "SquidRoom.h"

using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;
using namespace GameCore;

D3D12RaytracingAmbientOcclusion* g_pSample = nullptr;
HWND g_hWnd = 0;

// Shader entry points.
const wchar_t* D3D12RaytracingAmbientOcclusion::c_rayGenShaderNames[] = 
{
	// ToDo reorder
    // ToDo rename visiblity to shadow? standardize naming
	L"MyRayGenShader_GBuffer", L"MyRayGenShader_AO", L"MyRayGenShaderQuarterRes_AO", L"MyRayGenShader_Visibility"
};
const wchar_t* D3D12RaytracingAmbientOcclusion::c_closestHitShaderNames[] =
{
    L"MyClosestHitShader", L"MyClosestHitShader_ShadowRay", L"MyClosestHitShader_GBuffer"
};
const wchar_t* D3D12RaytracingAmbientOcclusion::c_missShaderNames[] =
{
    L"MyMissShader", L"MyMissShader_ShadowRay", L"MyMissShader_GBuffer"
};
// Hit groups.
const wchar_t* D3D12RaytracingAmbientOcclusion::c_hitGroupNames_TriangleGeometry[] = 
{ 
    L"MyHitGroup_Triangle", L"MyHitGroup_Triangle_ShadowRay", L"MyHitGroup_Triangle_GBuffer"
};
namespace SceneArgs
{
    void OnGeometryReinitializationNeeded(void* args)
    {
        g_pSample->RequestGeometryInitialization(true);
        g_pSample->RequestASInitialization(true);
    }

    void OnASReinitializationNeeded(void* args)
    {
        g_pSample->RequestASInitialization(true);
    }
    function<void(void*)> OnGeometryChange = OnGeometryReinitializationNeeded;
    function<void(void*)> OnASChange = OnASReinitializationNeeded;
	
	void OnSceneChange(void*)
	{
		g_pSample->RequestSceneInitialization();
	}

	void OnRecreateRaytracingResources(void*)
	{
		g_pSample->RequestRecreateRaytracingResources();
	}

	void OnRecreateSamples(void*)
	{
		g_pSample->RequestRecreateAOSamples();
	}
    BoolVar EnableGeometryAndASBuildsAndUpdates(L"Render/Acceleration structure/Enable geometry & AS builds and updates", true);

#if ONLY_SQUID_SCENE_BLAS
	EnumVar SceneType(L"Scene", Scene::Type::SquidRoom, Scene::Type::Count, Scene::Type::Names, OnSceneChange, nullptr);
#else
	EnumVar SceneType(L"Scene", Scene::Type::SingleObject, Scene::Type::Count, Scene::Type::Names, OnSceneChange, nullptr);
#endif

    // ToDo add an interface so that new UI values get applied on start of the frame, not in mid-flight
    enum UpdateMode { Build = 0, Update, Update_BuildEveryXFrames, Count };
    const WCHAR* UpdateModes[UpdateMode::Count] = { L"Build only", L"Update only", L"Update + build every X frames" };
    EnumVar ASUpdateMode(L"Render/Acceleration structure/Update mode", Build, UpdateMode::Count, UpdateModes);
    IntVar ASBuildFrequency(L"Render/Acceleration structure/Rebuild frame frequency", 1, 1, 1200, 1);
    BoolVar ASMinimizeMemory(L"Render/Acceleration structure/Minimize memory", false, OnASChange, nullptr);
    BoolVar ASAllowUpdate(L"Render/Acceleration structure/Allow update", true, OnASChange, nullptr);

	const WCHAR* AntialiasingModes[DownsampleFilter::Count] = { L"OFF", L"SSAA 4x (BoxFilter2x2)", L"SSAA 4x (GaussianFilter9Tap)", L"SSAA 4x (GaussianFilter25Tap)" };
 #if REPRO_BLOCKY_ARTIFACTS_NONUNIFORM_CB_REFERENCE_SSAO // Disable SSAA as the blockiness gets smaller with higher resoltuion 
	EnumVar AntialiasingMode(L"Render/Antialiasing", DownsampleFilter::None, DownsampleFilter::Count, AntialiasingModes, OnRecreateRaytracingResources, nullptr);
#else
    EnumVar AntialiasingMode(L"Render/Antialiasing", DownsampleFilter::None, DownsampleFilter::Count, AntialiasingModes, OnRecreateRaytracingResources, nullptr);
#endif

    // ToDo test tessFactor 16
    // ToDo fix alias on TessFactor 2
    IntVar GeometryTesselationFactor(L"Render/Geometry/Tesselation factor", 0/*14*/, 0, 80, 1, OnGeometryChange, nullptr);
    IntVar NumGeometriesPerBLAS(L"Render/Geometry/# geometries per BLAS", // ToDo
        NUM_GEOMETRIES, 1, 1000000, 1, OnGeometryChange, nullptr);
    IntVar NumSphereBLAS(L"Render/Geometry/# Sphere BLAS", 1, 1, D3D12RaytracingAmbientOcclusion::MaxBLAS, 1, OnASChange, nullptr);

    // ToDo don't render redundant passes?
    // ToDo Modularize parameters?
    // ToDO standardize capitalization

    const WCHAR* CompositionModes[CompositionType::Count] = { L"Phong Lighting", L"Render/AO", L"Raw one-frame AO", L"Render/AO Sampling Importance Map", L"Render/AO Average Hit Distance", L"Normal Map", L"Depth Buffer" };
    EnumVar CompositionMode(L"Render/Render composition mode", CompositionType::AmbientOcclusionOnly, CompositionType::Count, CompositionModes);

    const UINT DefaultSpp = 4;  // ToDo Cleanup

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
    BoolVar AOEnabled(L"Render/AO/Enabled", true);
    

    // ToDo standardize capitalization
    // ToDo naming down/ up
    const WCHAR* DownsamplingBilateralFilters[GpuKernels::DownsampleValueNormalDepthBilateralFilter::Count] = { L"Point Sampling", L"Depth Weighted", L"Depth Normal Weighted" };
    EnumVar DownsamplingBilateralFilter(L"Render/AO/RTAO/Down\\Upsampling/Downsampled Value Filter", GpuKernels::DownsampleValueNormalDepthBilateralFilter::FilterDepthNormalWeighted2x2, GpuKernels::DownsampleValueNormalDepthBilateralFilter::Count, DownsamplingBilateralFilters, OnRecreateRaytracingResources, nullptr);
    BoolVar DownAndUpsamplingUseBilinearWeights(L"Render/AO/RTAO/Down\\Upsampling/Bilinear weighted", true);
    BoolVar DownAndUpsamplingUseDepthWeights(L"Render/AO/RTAO/Down\\Upsampling/Depth weighted", true);
    BoolVar DownAndUpsamplingUseNormalWeights(L"Render/AO/RTAO/Down\\Upsampling/Normal weighted", true);
    BoolVar DownAndUpsamplingUseDynamicDepthThreshold(L"Render/AO/RTAO/Down\\Upsampling/Dynamic depth threshold", true);        // ToDO rename to adaptive


    // RTAO
    // Adaptive Sampling.
    BoolVar QuarterResAO(L"Render/AO/RTAO/Quarter res", true, OnRecreateRaytracingResources, nullptr);
    BoolVar RTAOAdaptiveSampling(L"Render/AO/RTAO/Adaptive Sampling/Enabled", true);
    BoolVar RTAOUseNormalMaps(L"Render/AO/RTAO/Normal maps", false);
    NumVar RTAOAdaptiveSamplingMaxFilterWeight(L"Render/AO/RTAO/Adaptive Sampling/Filter weight cutoff for max sampling", 0.995f, 0.0f, 1.f, 0.005f);
    BoolVar RTAOAdaptiveSamplingMinMaxSampling(L"Render/AO/RTAO/Adaptive Sampling/Only min\\max sampling", false );
    NumVar RTAOAdaptiveSamplingScaleExponent(L"Render/AO/RTAO/Adaptive Sampling/Sampling scale exponent", 0.3f, 0.0f, 10, 0.1f);
    BoolVar RTAORandomFrameSeed(L"Render/AO/RTAO/Random per-frame seed", true);

    NumVar RTAOTraceRayOffsetAlongNormal(L"Render/AO/RTAO/TraceRay/Ray origin offset along surface normal", 0.001f, 0, 0.1f, 0.0001f);
    NumVar RTAOTraceRayOffsetAlongRayDirection(L"Render/AO/RTAO/TraceRay/Ray origin offset fudge along ray direction", 0, 0, 0.1f, 0.0001f);


    // Temporal Cache.
    // ToDo rename cache to accumulation?
    BoolVar RTAO_TemporalCache_CacheRawAOValue(L"Render/AO/RTAO/Temporal Cache/Cache Raw AO Value", true);
    NumVar RTAO_TemporalCache_MinSmoothingFactor(L"Render/AO/RTAO/Temporal Cache/Min Smoothing Factor", 0.1f, 0, 1.f, 0.01f);
    NumVar RTAO_TemporalCache_DepthTolerance(L"Render/AO/RTAO/Temporal Cache/Depth tolerance [%%]", 0.1f, 0, 1.f, 0.001f);
    BoolVar RTAO_TemporalCache_UseDepthWeights(L"Render/AO/RTAO/Temporal Cache/Use depth weights", true);    // ToDo remove
    BoolVar RTAO_TemporalCache_UseNormalWeights(L"Render/AO/RTAO/Temporal Cache/Use normal weights", true);
    
    // ToDo cleanup RTAO... vs RTAO_..
    IntVar RTAOAdaptiveSamplingMinSamples(L"Render/AO/RTAO/Adaptive Sampling/Min samples", 1, 1, AO_SPP_N * AO_SPP_N, 1);
    IntVar RTAO_KernelStepShift0(L"Render/AO/RTAO/Kernel Step Shifts/0", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift1(L"Render/AO/RTAO/Kernel Step Shifts/1", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift2(L"Render/AO/RTAO/Kernel Step Shifts/2", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift3(L"Render/AO/RTAO/Kernel Step Shifts/3", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift4(L"Render/AO/RTAO/Kernel Step Shifts/4", 0, 0, 10, 1);

    IntVar AOSampleCountPerDimension(L"Render/AO/RTAO/Samples per pixel NxN", AO_SPP_N, 1, 32, 1, OnRecreateSamples, nullptr);
    IntVar AOSampleSetDistributedAcrossPixels(L"Render/AO/RTAO/Sample set distribution across NxN pixels ", 4, 1, 8, 1, OnRecreateSamples, nullptr);
#if PBRT_SCENE
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

    // ToDo standardixe AO RTAO prefix

    // ToDo add Weights On/OFF - 
    // RTAO Denoising
    const WCHAR* VarianceFilterModes[GpuKernels::CalculateVariance::FilterType::Count] = { L"Bilateral5x5", L"Bilateral7x7" };
    EnumVar VarianceFilterMode(L"Render/AO/RTAO/Denoising/Variance filter", GpuKernels::CalculateVariance::FilterType::Bilateral5x5, GpuKernels::CalculateVariance::FilterType::Count, VarianceFilterModes);
    BoolVar UseSpatialVariance(L"Render/AO/RTAO/Denoising/Use spatial variance", true);
    BoolVar ApproximateSpatialVariance(L"Render/AO/RTAO/Denoising/Approximate spatial variance", false);
    BoolVar RTAODenoisingUseMultiscale(L"Render/AO/RTAO/Denoising/Multi-scale/Enabled", true);
    IntVar RTAODenoisingMultiscaleLevels(L"Render/AO/RTAO/Denoising/Multi-scale/Levels", 1, 1, D3D12RaytracingAmbientOcclusion::c_MaxDenoisingScaleLevels);
    BoolVar RTAODenoisingMultiscaleDenoisedAsInput(L"Render/AO/RTAO/Denoising/Multi-scale/Denoised as input", true);
    
    BoolVar RTAODenoisingPerspectiveCorrectDepthInterpolation(L"Render/AO/RTAO/Denoising/Pespective Correct Depth Interpolation", true); // ToDo test perf impact / visual quality gain at the end. Document.
    BoolVar RTAODenoisingUseAdaptiveKernelSize(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Enabled", true);
    IntVar RTAODenoisingFilterMinKernelWidth(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Min kernel width", 5, 1, 101);
    NumVar RTAODenoisingFilterMaxKernelWidthPercentage(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Max kernel width [%% of screen width]", 2.5f, 0, 100, 0.1f);
    NumVar RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Variance sigma scale on small kernels", 2.0f, 1.0f, 20.f, 0.5f); 
    NumVar RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Hit distance scale factor", 1.0f, 0.1f, 10.f, 0.1f);


    const WCHAR* DenoisingModes[GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::Count] = { L"EdgeStoppingBox3x3", L"EdgeStoppingGaussian3x3", L"EdgeStoppingGaussian5x5" };
    EnumVar DenoisingMode(L"Render/AO/RTAO/Denoising/Mode", GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::EdgeStoppingGaussian3x3, GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::Count, DenoisingModes);
    IntVar AtrousFilterPasses(L"Render/AO/RTAO/Denoising/Num passes", 4, 1, 8, 1);
    BoolVar ReverseFilterOrder(L"Render/AO/RTAO/Denoising/Reverse filter order", false);
    NumVar AODenoiseValueSigma(L"Render/AO/RTAO/Denoising/Value Sigma", 6, 0.0f, 30.0f, 0.1f);

    // ToDo why large depth sigma is needed?
    // ToDo the values don't scale to QuarterRes - see ImportaceMap viz
    NumVar AODenoiseDepthSigma(L"Render/AO/RTAO/Denoising/Depth Sigma", 1.2f, 0.0f, 10.0f, 0.02f); // ToDo Fine tune. 1 causes moire patterns at angle under the car

    NumVar AODenoiseNormalSigma(L"Render/AO/RTAO/Denoising/Normal Sigma", 64, 0, 256, 4);
    

    // ToDo dedupe
    BoolVar g_QuarterResAO(L"Misc/QuarterRes AO", false);
    NumVar g_DistanceTolerance(L"Misc/AO Distance Tolerance (log10)", -2.5f, -32.0f, 32.0f, 0.25f);

    // SSAO
    NumVar SSAONoiseFilterTolerance(L"Render/AO/SSAO/Noise Threshold (log10)", -3.f, -8.f, .0f, .1f);
    NumVar SSAOBlurTolerance(L"Render/AO/SSAO/Blur Tolerance (log10)", -5.f, -8.f, -1.f, .1f);
    NumVar SSAOUpsampleTolerance(L"Render/AO/SSAO/Upsample Tolerance (log10)", -7.f, -12.f, -1.f, .1f);
    NumVar SSAONormalMultiply(L"Render/AO/SSAO/Normal Factor", 1.f, .0f, 5.f, .125f);
    //**********************************************************************************************************************************

};

// ToDo move
void D3D12RaytracingAmbientOcclusion::LoadPBRTScene()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto commandQueue = m_deviceResources->GetCommandQueue();
	auto commandAllocator = m_deviceResources->GetCommandAllocator();

	auto Vec3ToXMFLOAT3 = [](SceneParser::Vector3 v)
	{
		return XMFLOAT3(v.x, v.y, v.z);
	};


	auto Vec3ToXMVECTOR = [](SceneParser::Vector3 v)
	{
		return XMLoadFloat3(&XMFLOAT3(v.x, v.y, v.z));
	};

	auto Vec3ToXMFLOAT2 = [](SceneParser::Vector2 v)
	{
		return XMFLOAT2(v.x, v.y);
	};

	// Work
#if 1
    PBRTParser::PBRTParser().Parse("Assets\\spaceship\\scene.pbrt", m_pbrtScene);
    // ToDo move plane from car2 to house
	PBRTParser::PBRTParser().Parse("Assets\\car2\\scene.pbrt", m_pbrtScene);
	PBRTParser::PBRTParser().Parse("Assets\\dragon\\scene.pbrt", m_pbrtScene);		// ToDo model is mirrored
	PBRTParser::PBRTParser().Parse("Assets\\house\\scene.pbrt", m_pbrtScene);
#else
	PBRTParser::PBRTParser().Parse("Assets\\bedroom\\scene.pbrt", m_pbrtScene);
	//PBRTParser::PBRTParser().Parse("Assets\\spaceship\\scene.pbrt", m_pbrtScene);
	//PBRTParser::PBRTParser().Parse("Assets\\bmw-m6\\scene.pbrt", m_pbrtScene);
	//PBRTParser::PBRTParser().Parse("Assets\\staircase2\\scene.pbrt", m_pbrtScene);
	//PBRTParser::PBRTParser().Parse("Assets\\classroom\\scene.pbrt", m_pbrtScene);
	//PBRTParser::PBRTParser().Parse("Assets\\living-room-3\\scene.pbrt", m_pbrtScene); //rug geometry skipped
	//PBRTParser::PBRTParser().Parse("Assets\\staircase\\scene.pbrt", m_pbrtScene);
	//PBRTParser::PBRTParser().Parse("Assets\\living-room\\scene.pbrt", m_pbrtScene);
	//PBRTParser::PBRTParser().Parse("Assets\\living-room-2\\scene.pbrt", m_pbrtScene); // incorrect normals on the backwall.
	//PBRTParser::PBRTParser().Parse("Assets\\kitchen\\scene.pbrt", m_pbrtScene); // incorrect normals on the backwall.
#endif

    // ToDo
	//m_camera.Set(
	//	Vec3ToXMVECTOR(m_pbrtScene.m_Camera.m_Position),
	//	Vec3ToXMVECTOR(m_pbrtScene.m_Camera.m_LookAt),
	//	Vec3ToXMVECTOR(m_pbrtScene.m_Camera.m_Up));
	//m_camera.fov = 2 * m_pbrtScene.m_Camera.m_FieldOfView;   
	UINT numGeometries = static_cast<UINT>(m_pbrtScene.m_Meshes.size());

	auto& geometries = m_geometries[GeometryType::PBRT];
	geometries.resize(numGeometries);

	auto& geometryInstances = m_geometryInstances[GeometryType::PBRT];
	geometryInstances.reserve(numGeometries);

    auto& textures = m_geometryTextures[GeometryType::PBRT];

    ResourceUploadBatch resourceUpload(device);
    resourceUpload.Begin();

	XMVECTOR test = XMVector3TransformNormal(XMVectorSet(0, -1, 0, 0), m_pbrtScene.m_transform);
	// ToDo
	m_numTriangles[GeometryType::PBRT] = 0;
	for (UINT i = 0; i < m_pbrtScene.m_Meshes.size(); i++)
	{
		auto &mesh = m_pbrtScene.m_Meshes[i];
		if (mesh.m_VertexBuffer.size() == 0 || mesh.m_IndexBuffer.size() == 0)
		{
			continue;
		}
		vector<VertexPositionNormalTextureTangent> vertexBuffer;
		vector<Index> indexBuffer;
		vertexBuffer.reserve(mesh.m_VertexBuffer.size());
		indexBuffer.reserve(mesh.m_IndexBuffer.size());

		GeometryDescriptor desc;
		desc.ib.count = static_cast<UINT>(mesh.m_IndexBuffer.size());
		desc.vb.count = static_cast<UINT>(mesh.m_VertexBuffer.size());

		for (auto &parseIndex : mesh.m_IndexBuffer)
		{
			Index index = parseIndex;
			indexBuffer.push_back(index);
		}
		desc.ib.indices = indexBuffer.data();
		
		for (auto &parseVertex : mesh.m_VertexBuffer)
		{
			VertexPositionNormalTextureTangent vertex;
#if PBRT_APPLY_INITIAL_TRANSFORM_TO_VB_ATTRIBUTES
			XMStoreFloat3(&vertex.normal, XMVector3TransformNormal(parseVertex.Normal.GetXMVECTOR(), mesh.m_transform));
			XMStoreFloat3(&vertex.position, XMVector3TransformCoord(parseVertex.Position.GetXMVECTOR(), mesh.m_transform));
#else
			vertex.normal = parseVertex.Normal.xmFloat3;
			vertex.position = parseVertex.Position.xmFloat3;
#endif
			vertex.tangent = parseVertex.Tangents.xmFloat3;
			vertex.textureCoordinate = parseVertex.UV.xmFloat2;
			vertexBuffer.push_back(vertex);
		}
		desc.vb.vertices = vertexBuffer.data();

		auto& geometry = geometries[i];
		CreateGeometry(device, commandList, m_cbvSrvUavHeap.get(), desc, &geometry);
		ThrowIfFalse(geometry.vb.buffer.heapIndex == geometry.ib.buffer.heapIndex + 1, L"Vertex Buffer descriptor index must follow that of Index Buffer descriptor index");

		PrimitiveMaterialBuffer cb;
		cb.diffuse = mesh.m_pMaterial->m_Diffuse.xmFloat3;
		cb.specular = mesh.m_pMaterial->m_Specular.xmFloat3;
        cb.specularPower = 50;
		cb.isMirror = mesh.m_pMaterial->m_Opacity.r < 0.5f ? 1 : 0;
        cb.hasDiffuseTexture = !mesh.m_pMaterial->m_DiffuseTextureFilename.empty();
        cb.hasNormalTexture = !mesh.m_pMaterial->m_NormalMapTextureFilename.empty();
        cb.hasPerVertexTangents = true;


        auto LoadPBRTTexture = [&](auto** ppOutTexture, auto& textureFilename)
        {
            auto& srcString = mesh.m_pMaterial->m_DiffuseTextureFilename;
            wstring filename(textureFilename.begin(), textureFilename.end());
            textures.push_back(D3DTexture());
            auto* texture = &textures.back();
            LoadWICTexture(device, &resourceUpload, filename.c_str(), m_cbvSrvUavHeap.get(), &texture->resource, &texture->heapIndex, &texture->cpuDescriptorHandle, &texture->gpuDescriptorHandle, true);
            *ppOutTexture = texture;
        };

        D3DTexture* diffuseTexture = &m_nullTexture;
        if (cb.hasDiffuseTexture)
        {
            LoadPBRTTexture(&diffuseTexture, mesh.m_pMaterial->m_DiffuseTextureFilename);
        }

        D3DTexture* normalTexture = &m_nullTexture;
        if (cb.hasNormalTexture)
        {
            LoadPBRTTexture(&normalTexture, mesh.m_pMaterial->m_NormalMapTextureFilename);
        }

		UINT materialID = static_cast<UINT>(m_materials.size());
		m_materials.push_back(cb);
		geometryInstances.push_back(GeometryInstance(geometry, materialID, diffuseTexture->gpuDescriptorHandle, normalTexture->gpuDescriptorHandle));
#if !PBRT_APPLY_INITIAL_TRANSFORM_TO_VB_ATTRIBUTES
		XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(m_geometryTransforms[i].transform3x4), mesh.m_transform);
		geometryInstances.back().transform = m_geometryTransforms.GpuVirtualAddress(0, i);
#endif
		m_numTriangles[GeometryType::PBRT] += desc.ib.count / 3;
	}

    // Upload the resources to the GPU.
    auto finish = resourceUpload.End(commandQueue);

    // Wait for the upload thread to terminate
    finish.wait();
}


D3D12RaytracingAmbientOcclusion::D3D12RaytracingAmbientOcclusion(UINT width, UINT height, wstring name) :
    DXSample(width, height, name),
    m_animateCamera(false),
    m_animateLight(false),
    m_animateScene(true),
    m_missShaderTableStrideInBytes(UINT_MAX),
    m_hitGroupShaderTableStrideInBytes(UINT_MAX),
    m_isGeometryInitializationRequested(true),
    m_isASinitializationRequested(true),
    m_isASrebuildRequested(true),
	m_isSceneInitializationRequested(false),
	m_isRecreateRaytracingResourcesRequested(false),
	m_isRecreateAOSamplesRequested(false),
    m_ASmemoryFootprint(0),
    m_numFramesSinceASBuild(0),
	m_isCameraFrozen(false)
{
    g_pSample = this;
    UpdateForSizeChange(width, height);
    m_bottomLevelASdescritorHeapIndices.resize(MaxBLAS, UINT_MAX);
    m_bottomLevelASinstanceDescsDescritorHeapIndices.resize(MaxBLAS, UINT_MAX);
	m_generatorURNG.seed(1729);
}

// ToDo worth moving some common member vars and fncs to DxSampleRaytracing base class?
void D3D12RaytracingAmbientOcclusion::OnInit()
{
    m_deviceResources = make_unique<DeviceResources>(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_UNKNOWN,
        FrameCount,
        D3D_FEATURE_LEVEL_11_0,
        // Sample shows handling of use cases with tearing support, which is OS dependent and has been supported since TH2.
        // Since the DXR requires October 2018 update, we don't need to handle non-tearing cases.
        DeviceResources::c_RequireTearingSupport,
        m_adapterIDoverride
        );
    m_deviceResources->RegisterDeviceNotify(this);
    m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);

    g_hWnd = Win32Application::GetHwnd();
    GameInput::Initialize();
    EngineTuning::Initialize();

    m_deviceResources->InitializeDXGIAdapter();
#if ENABLE_RAYTRACING
	ThrowIfFalse(IsDirectXRaytracingSupported(m_deviceResources->GetAdapter()),
		L"ERROR: DirectX Raytracing is not supported by your GPU and driver.\n\n");
#endif
    // ToDo cleanup
    m_deviceResources->CreateDeviceResources();
	// Initialize scene ToDo
    InitializeScene();
    CreateDeviceDependentResources();
    m_deviceResources->CreateWindowSizeDependentResources();
}

D3D12RaytracingAmbientOcclusion::~D3D12RaytracingAmbientOcclusion()
{
    GameInput::Shutdown();
}

// Update camera matrices passed into the shader.
void D3D12RaytracingAmbientOcclusion::UpdateCameraMatrices()
{
    // Main scene.
    {
        m_sceneCB->cameraPosition = m_camera.Eye();
        XMStoreFloat3(&m_csComposeRenderPassesCB->cameraPosition, m_camera.Eye());

        XMMATRIX view, proj;
        m_camera.GetProj(&proj, m_GBufferWidth, m_GBufferHeight);

        // Calculate view matrix as if the camera was at (0,0,0) to avoid 
        // precision issues when camera position is too far from (0,0,0).
        // GenerateCameraRay takes this into consideration in the raytracing shader.
        view = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 1), XMVectorSetW(m_camera.At() - m_camera.Eye(), 1), m_camera.Up());
        XMMATRIX viewProj = view * proj;
        m_sceneCB->projectionToWorldWithCameraEyeAtOrigin = XMMatrixInverse(nullptr, viewProj);
        // ToDo switch to default column major in hlsl and do a transpose before passing matrices to HLSL.
        m_sceneCB->viewProjection = viewProj;
        m_sceneCB->Zmin = m_camera.ZMin;
        m_sceneCB->Zmax = m_camera.ZMax;

        m_sceneCB->cameraAt = m_camera.At();
        m_sceneCB->cameraUp = m_camera.Up();
        m_sceneCB->cameraRight = XMVector3Normalize(XMVector3Cross(m_camera.Up(), m_camera.At() - m_camera.Eye()));
    }

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

void D3D12RaytracingAmbientOcclusion::UpdateBottomLevelASTransforms()
{
    float animationDuration = 24.0f;
    float curTime = static_cast<float>(m_timer.GetTotalSeconds());
    float t = CalculateAnimationInterpolant(curTime, animationDuration);
    t += -0.5f;
    //ToDo
    t = 0.0f;

    float baseAmplitude = 16.0f;
    for (auto& bottomLevelAS : m_vBottomLevelAS)
    {
        // Animate along Y coordinate.
        XMMATRIX transform = bottomLevelAS.GetTransform();
        float distFromOrigin = XMVectorGetX(XMVector4Length(transform.r[3]));
        float posY = t * (baseAmplitude + 0.35f * distFromOrigin);

        transform.r[3] = XMVectorSetByIndex(transform.r[3], posY, 1);
        bottomLevelAS.SetTransform(transform);
    }
}

void D3D12RaytracingAmbientOcclusion::UpdateSphereGeometryTransforms()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	// Generate geometry desc transforms;
	int dim = static_cast<int>(ceil(cbrt(static_cast<double>(SceneArgs::NumGeometriesPerBLAS))));
    float distanceBetweenGeometry = m_geometryRadius;
    float geometryWidth = 2 * m_geometryRadius;
    float stepDistance = geometryWidth + distanceBetweenGeometry;

    float animationDuration = 12.0f;
    float curTime = static_cast<float>(m_timer.GetTotalSeconds());
    float t = CalculateAnimationInterpolant(curTime, animationDuration);
    //ToDo
    t = 0.0f;
    float rotAngle = XMConvertToRadians(t * 360.0f);

    // Rotate around offset center.
    XMMATRIX localTranslation = XMMatrixTranslation(0.0f, m_geometryRadius, 0.5f * m_geometryRadius);
    XMMATRIX localRotation = XMMatrixRotationY(XMConvertToRadians(rotAngle));
    XMMATRIX localTransform = localTranslation * localRotation;
    
    // ToDo
    localTransform = XMMatrixTranslation(0.0f, m_geometryRadius, 0.0f);

    for (int iY = 0, i = 0; iY < dim; iY++)
        for (int iX = 0; iX < dim; iX++)
            for (int iZ = 0; iZ < dim; iZ++, i++)
            {
                if (i >= SceneArgs::NumGeometriesPerBLAS)
                {
                    break;
                }

                // Translate within BLAS.
                XMFLOAT4 translationVector = XMFLOAT4(
                    static_cast<float>(iX - dim / 2),
                    static_cast<float>(iY - dim / 2),
                    static_cast<float>(iZ - dim / 2),
                    0.0f);
                XMMATRIX transformWithinBLAS= XMMatrixTranslationFromVector(stepDistance * XMLoadFloat4(&translationVector));
                XMMATRIX transform = localTransform * transformWithinBLAS;

                for (UINT j = BottomLevelASType::Sphere; j < m_vBottomLevelAS.size(); j++)
                {
                    UINT transformIndex = j + 1;	// + plane which is first. ToDo break geometries apart.
        			XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(m_geometryTransforms[transformIndex].transform3x4), transform);
                }
            }
}

// ToDo move to CS.
void D3D12RaytracingAmbientOcclusion::UpdateGridGeometryTransforms()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	// Generate geometry desc transforms;
#if TESSELATED_GEOMETRY_ASPECT_RATIO_DIMENSIONS
	int dimX =static_cast<int>(ceil(sqrt(SceneArgs::NumGeometriesPerBLAS * m_aspectRatio)));
#else
	int dimX = static_cast<int>(ceil(sqrt(static_cast<double>(SceneArgs::NumGeometriesPerBLAS))));
#endif
	XMUINT3 dim(dimX, 1, CeilDivide(SceneArgs::NumGeometriesPerBLAS, dimX));

	float spacing = 0.4f * max(m_boxSize.x, m_boxSize.z);
	XMVECTOR stepDistance = XMLoadFloat3(&m_boxSize) + XMVectorSet(spacing, spacing, spacing, 0);
	XMVECTOR offset = - XMLoadUInt3(&dim) / 2 * stepDistance;
	offset = XMVectorSetY(offset, m_boxSize.y / 2);

	// ToDo

	uniform_real_distribution<float> elevationDistribution(-0.4f*m_boxSize.y, 0);
	uniform_real_distribution<float> jitterDistribution(-spacing, spacing);
	uniform_real_distribution<float> rotationDistribution(-XM_PI, 180);

	for (UINT iY = 0, i = 0; iY < dim.y; iY++)
		for (UINT iX = 0; iX < dim.x; iX++)
			for (UINT iZ = 0; iZ < dim.z; iZ++, i++)
			{
				if (static_cast<int>(i) >= SceneArgs::NumGeometriesPerBLAS )
				{
					break;
				}
				const UINT X_TILE_WIDTH = 20;
				const UINT X_TILE_SPACING = X_TILE_WIDTH * 2;
				const UINT Z_TILE_WIDTH = 6;
				const UINT Z_TILE_SPACING = Z_TILE_WIDTH * 2;

				XMVECTOR translationVector = offset + stepDistance * 
					XMVectorSet(
#if TESSELATED_GEOMETRY_TILES
						static_cast<float>((iX / X_TILE_WIDTH) * X_TILE_SPACING + iX % X_TILE_WIDTH),
						static_cast<float>(iY),
						static_cast<float>((iZ/ Z_TILE_WIDTH) * Z_TILE_SPACING + iZ % Z_TILE_WIDTH),
#else
						static_cast<float>(iX),
						static_cast<float>(iY),
						static_cast<float>(iZ),
#endif
						0);
				// Break up Moire alias patterns by jittering the position.
				translationVector += XMVectorSet(
					jitterDistribution(m_generatorURNG),
					elevationDistribution(m_generatorURNG), 
					jitterDistribution(m_generatorURNG),
					0);
				XMMATRIX translation = XMMatrixTranslationFromVector(translationVector);
				XMMATRIX rotation = XMMatrixIdentity();// ToDo - need to rotate normals too XMMatrixRotationY(rotationDistribution(m_generatorURNG));
				XMMATRIX transform = rotation * translation;
				
				// ToDO remove - skip past plane transform
				XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(m_geometryTransforms[i + 1].transform3x4), transform);
			}

	// Update the plane transform.
	XMVECTOR size = XMVectorSetY(1.1f*XMLoadUInt3(&dim) * stepDistance, 1);
	XMMATRIX scale = XMMatrixScalingFromVector(size);
	XMMATRIX translation = XMMatrixTranslationFromVector(XMVectorSetY (- size / 2, 0));
	XMMATRIX transform = scale * translation;
	XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(m_geometryTransforms[0].transform3x4), transform);
}

// Initialize scene rendering parameters.
void D3D12RaytracingAmbientOcclusion::InitializeScene()
{
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    // Setup materials.
    {
        auto SetAttributes = [&](
            UINT primitiveIndex, 
            const XMFLOAT4& albedo, 
            float reflectanceCoef = 0.0f,
            float diffuseCoef = 0.9f,
            float specularCoef = 0.7f,
            float specularPower = 50.0f,
            float stepScale = 1.0f )
        {
			// ToDo
            //auto& attributes = m_aabbMaterialCB[primitiveIndex];
            //attributes.albedo = albedo;
            //attributes.reflectanceCoef = reflectanceCoef;
            //attributes.diffuseCoef = diffuseCoef;
            //attributes.specularCoef = specularCoef;
            //attributes.specularPower = specularPower;
            //attributes.stepScale = stepScale;
        };
		
        // Albedos
        XMFLOAT4 green = XMFLOAT4(0.1f, 1.0f, 0.5f, 1.0f);
        XMFLOAT4 red = XMFLOAT4(1.0f, 0.5f, 0.5f, 1.0f);
        XMFLOAT4 yellow = XMFLOAT4(1.0f, 1.0f, 0.5f, 1.0f);
    }

    // Setup camera.
	{
		// Initialize the view and projection inverse matrices.
		auto& camera = Scene::args[SceneArgs::SceneType].camera;
		m_camera.Set(camera.position.eye, camera.position.at, camera.position.up);
		m_cameraController = make_unique<CameraController>(m_camera);
		m_cameraController->SetBoundaries(camera.boundaries.min, camera.boundaries.max);
	}

    // Setup lights.
    {
        // Initialize the lighting parameters.
		// ToDo remove
		m_csComposeRenderPassesCB->lightPosition = XMFLOAT3(-20.0f, 40.0f, 20.0f);
		m_sceneCB->lightPosition = XMLoadFloat3(&m_csComposeRenderPassesCB->lightPosition);

		m_csComposeRenderPassesCB->lightAmbientColor = XMFLOAT3(0.45f, 0.45f, 0.45f);

        float d = 0.6f;
		m_csComposeRenderPassesCB->lightDiffuseColor = XMFLOAT3(d, d, d);
    }
}

// Create constant buffers.
void D3D12RaytracingAmbientOcclusion::CreateConstantBuffers()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto frameCount = m_deviceResources->GetBackBufferCount();

    m_sceneCB.Create(device, frameCount, L"Scene Constant Buffer");

    m_SSAOCB.Create(device, frameCount, L"SSAO Constant Buffer");
}

// ToDo rename, move
void D3D12RaytracingAmbientOcclusion::CreateSamplesRNG()
{
    auto device = m_deviceResources->GetD3DDevice(); 
    auto frameCount = m_deviceResources->GetBackBufferCount();

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
        descComputePSO.CS = CD3DX12_SHADER_BYTECODE((void *)g_pRNGVisualizerCS, ARRAYSIZE(g_pRNGVisualizerCS));

        ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSOs[CSType::HemisphereSampleSetVisualization])));
        m_computePSOs[CSType::HemisphereSampleSetVisualization]->SetName(L"PSO: CS hemisphere sample set visualization");
    }


    // Create shader resources
    {
        m_csHemisphereVisualizationCB.Create(device, frameCount, L"GPU CB: RNG");
        m_samplesGPUBuffer.Create(device, m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(), frameCount, L"GPU buffer: Random unit square samples");
        m_hemisphereSamplesGPUBuffer.Create(device, m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(), frameCount, L"GPU buffer: Random hemisphere samples");

        for (UINT i = 0; i < m_randomSampler.NumSamples() * m_randomSampler.NumSampleSets(); i++)
        {
            //sample.value = m_randomSampler.GetSample2D();
            XMFLOAT3 p = m_randomSampler.GetHemisphereSample3D();
			// Convert [-1,1] to [0,1].
            m_samplesGPUBuffer[i].value = XMFLOAT2(p.x*0.5f + 0.5f, p.y*0.5f + 0.5f);
            m_hemisphereSamplesGPUBuffer[i].value = p;
        }
    }
}


void D3D12RaytracingAmbientOcclusion::CreateComposeRenderPassesCSResources()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameCount = m_deviceResources->GetBackBufferCount();

	// Create root signature.
	{
		using namespace CSRootSignature::ComposeRenderPassesCS;

		CD3DX12_DESCRIPTOR_RANGE ranges[6]; // Perfomance TIP: Order from most frequent to least frequent.
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 0);  // 5 input GBuffer textures
		ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // 1 input AO texture
		ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6);  // 1 input Visibility texture
        ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);  // 1 input filterWeightSum texture
        ranges[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 9);  // 1 input AO ray hit distance texture

		CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
		rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[0]);
		rootParameters[Slot::GBufferResources].InitAsDescriptorTable(1, &ranges[1]);
		rootParameters[Slot::AO].InitAsDescriptorTable(1, &ranges[2]);
		rootParameters[Slot::Visibility].InitAsDescriptorTable(1, &ranges[3]);
        rootParameters[Slot::FilterWeightSum].InitAsDescriptorTable(1, &ranges[4]);
        rootParameters[Slot::AORayHitDistance].InitAsDescriptorTable(1, &ranges[5]);
		rootParameters[Slot::MaterialBuffer].InitAsShaderResourceView(7);
		rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

        CD3DX12_STATIC_SAMPLER_DESC staticSampler(0, SAMPLER_FILTER);

		CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, 1, &staticSampler);
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
		descComputePSO.CS = CD3DX12_SHADER_BYTECODE((void *)g_pComposeRenderPassesCS, ARRAYSIZE(g_pComposeRenderPassesCS));

		ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSOs[CSType::ComposeRenderPassesCS])));
		m_computePSOs[CSType::ComposeRenderPassesCS]->SetName(L"PSO: ComposeRenderPassesCS");
	}
}

void D3D12RaytracingAmbientOcclusion::CreateAoBlurCSResources()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameCount = m_deviceResources->GetBackBufferCount();

	// Create root signature.
	{
		using namespace CSRootSignature::AoBlurCS;

		CD3DX12_DESCRIPTOR_RANGE ranges[4]; // Perfomance TIP: Order from most frequent to least frequent.
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // smooth AO output
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // normal texture
        ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // distance texture
        ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // noisy AO texture

		CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
		rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[0]);
		rootParameters[Slot::Normal].InitAsDescriptorTable(1, &ranges[1]);
        rootParameters[Slot::Distance].InitAsDescriptorTable(1, &ranges[2]);
        rootParameters[Slot::InputAO].InitAsDescriptorTable(1, &ranges[3]);
		rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

        // ToDo test aniso perf impact.
        CD3DX12_STATIC_SAMPLER_DESC staticSampler(0, SAMPLER_FILTER);

		CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, 1, &staticSampler);
		SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_computeRootSigs[CSType::AoBlurCS], L"Root signature: AoBlurCS");
	}

	// Create shader resources
	{
		m_csAoBlurCB.Create(device, FrameCount, L"Constant Buffer: AoBlurCS");
	}

	// Create compute pipeline state.
	{
		D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
		descComputePSO.pRootSignature = m_computeRootSigs[CSType::AoBlurCS].Get();

        descComputePSO.CS = CD3DX12_SHADER_BYTECODE((void*)g_pAoBlurCS, ARRAYSIZE(g_pAoBlurCS));
		ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSOs[CSType::AoBlurCS])));
		m_computePSOs[CSType::AoBlurCS]->SetName(L"PSO: AoBlurCS");

		descComputePSO.CS = CD3DX12_SHADER_BYTECODE((void*)g_pAoBlurAndUpsampleCS, ARRAYSIZE(g_pAoBlurAndUpsampleCS));
		ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSOs[CSType::AoBlurAndUpsampleCS])));
		m_computePSOs[CSType::AoBlurAndUpsampleCS]->SetName(L"PSO: AoBlurAndUpsampleCS");
}
}

// Create resources that depend on the device.
void D3D12RaytracingAmbientOcclusion::CreateDeviceDependentResources()
{
	auto device = m_deviceResources->GetD3DDevice();

    CreateAuxilaryDeviceResources();

	// ToDo move
	m_geometryTransforms.Create(device, MaxGeometryTransforms, 1, L"Structured buffer: Geometry desc transforms");

	// Create a heap for descriptors.
	CreateDescriptorHeaps();

    // Initialize raytracing pipeline.

    // Create raytracing interfaces: raytracing device and commandlist.
    CreateRaytracingInterfaces();

    // Build geometry to be used in the sample.
    // ToDO
    m_isGeometryInitializationRequested = true;
    InitializeGeometry();
    m_isGeometryInitializationRequested = false;

    // Build raytracing acceleration structures from the generated geometry.
    m_isASinitializationRequested = true;

#if ENABLE_RAYTRACING
    // Create root signatures for the shaders.
    CreateRootSignatures();

    // Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
    CreateRaytracingPipelineStateObject();
#endif
    // Create constant buffers for the geometry and the scene.
    CreateConstantBuffers();

#if ENABLE_RAYTRACING
	// Build shader tables, which define shaders and their local root arguments.
    BuildShaderTables();
#endif

	// ToDo move
    CreateSamplesRNG();
	CreateComposeRenderPassesCSResources();

    CreateAoBlurCSResources();

    m_SSAO.Setup(m_deviceResources);
}

void D3D12RaytracingAmbientOcclusion::CreateRootSignatures()
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
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        ranges[Slot::PartialDepthDerivatives].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 16);  // 1 output partial depth derivative texture
#endif
        ranges[Slot::GBufferResourcesIn].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 4, 5);  // 4 input GBuffer textures
        ranges[Slot::EnvironmentMap].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12);  // 1 input environment map texture
        ranges[Slot::FilterWeightSum].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 13);  // 1 input filter weight sum texture

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
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
        rootParameters[Slot::PartialDepthDerivatives].InitAsDescriptorTable(1, &ranges[Slot::PartialDepthDerivatives]);
#endif
        rootParameters[Slot::AccelerationStructure].InitAsShaderResourceView(0);
        rootParameters[Slot::SceneConstant].InitAsConstantBufferView(0);		// ToDo rename to ConstantBuffer
        rootParameters[Slot::MaterialBuffer].InitAsShaderResourceView(3);
        rootParameters[Slot::SampleBuffers].InitAsShaderResourceView(4);

        CD3DX12_STATIC_SAMPLER_DESC staticSampler(0, SAMPLER_FILTER);

        CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, 1, &staticSampler);
		SerializeAndCreateRootSignature(device, globalRootSignatureDesc, &m_raytracingGlobalRootSignature, L"Global root signature");
    }

    // Local Root Signature
    // This is a root signature that enables a shader to have unique arguments that come from shader tables.
    {
        // Triangle geometry
        {
			using namespace LocalRootSignature::Triangle;

            CD3DX12_DESCRIPTOR_RANGE ranges[3]; // Perfomance TIP: Order from most frequent to least frequent.
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1);  // 2 static index and vertex buffers.
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 10);  // 1 diffuse texture
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 11);  // 1 normal texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::ConstantBuffer].InitAsConstants(SizeOfInUint32(PrimitiveConstantBuffer), 1);
            rootParameters[Slot::VertexBuffers].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::DiffuseTexture].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::NormalTexture].InitAsDescriptorTable(1, &ranges[2]);

            CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
			SerializeAndCreateRootSignature(device, localRootSignatureDesc, &m_raytracingLocalRootSignature[LocalRootSignature::Type::Triangle], L"Local root signature: triangle geometry");
        }
    }
}

// Create raytracing device and command list.
void D3D12RaytracingAmbientOcclusion::CreateRaytracingInterfaces()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

   ThrowIfFailed(device->QueryInterface(IID_PPV_ARGS(&m_dxrDevice)), L"Couldn't get DirectX Raytracing interface for the device.\n");
}

// DXIL library
// This contains the shaders and their entrypoints for the state object.
// Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
void D3D12RaytracingAmbientOcclusion::CreateDxilLibrarySubobject(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    auto lib = raytracingPipeline->CreateSubobject<CD3DX12_DXIL_LIBRARY_SUBOBJECT>();
    D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void *)g_pRaytracing, ARRAYSIZE(g_pRaytracing));
    lib->SetDXILLibrary(&libdxil);
    // Use default shader exports for a DXIL library/collection subobject ~ surface all shaders.
}

// Hit groups
// A hit group specifies closest hit, any hit and intersection shaders 
// to be executed when a ray intersects the geometry.
void D3D12RaytracingAmbientOcclusion::CreateHitGroupSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
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
            hitGroup->SetHitGroupExport(c_hitGroupNames_TriangleGeometry[rayType]);
            hitGroup->SetHitGroupType(D3D12_HIT_GROUP_TYPE_TRIANGLES);
        }
    }
}

// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void D3D12RaytracingAmbientOcclusion::CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline)
{
    // Ray gen and miss shaders in this sample are not using a local root signature and thus one is not associated with them.

    // Hit groups
    // Triangle geometry
    {
        auto localRootSignature = raytracingPipeline->CreateSubobject<CD3DX12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
        localRootSignature->SetRootSignature(m_raytracingLocalRootSignature[LocalRootSignature::Type::Triangle].Get());
        // Shader association
        auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3DX12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
        rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
        rootSignatureAssociation->AddExports(c_hitGroupNames_TriangleGeometry);
    }
}

// Create a raytracing pipeline state object (RTPSO).
// An RTPSO represents a full set of shaders reachable by a DispatchRays() call,
// with all configuration options resolved, such as local signatures and other state.
void D3D12RaytracingAmbientOcclusion::CreateRaytracingPipelineStateObject()
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
    ThrowIfFailed(m_dxrDevice->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_dxrStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
}

// Create a 2D output texture for raytracing.
void D3D12RaytracingAmbientOcclusion::CreateRaytracingOutputResource()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto backbufferFormat = m_deviceResources->GetBackBufferFormat();
	m_raytracingOutput.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
	CreateRenderTargetResource(device, backbufferFormat, m_width, m_height, m_cbvSrvUavHeap.get(), &m_raytracingOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	m_raytracingOutputIntermediate.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
	CreateRenderTargetResource(device, backbufferFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_raytracingOutputIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}


// ToDo move remove
#if FLOAT_TEXTURE_AS_R8_UNORM_1BYTE_FORMAT
DXGI_FORMAT texFormat = DXGI_FORMAT_R8_UNORM;       // ToDo rename to coefficient or avoid using same variable for different types.
UINT texFormatByteSize = 1;
#else
DXGI_FORMAT texFormat = DXGI_FORMAT_R32_FLOAT;
UINT texFormatByteSize = 4;
#endif

void D3D12RaytracingAmbientOcclusion::CreateGBufferResources()
{
	auto device = m_deviceResources->GetD3DDevice();

    // ToDo move depth out of normal resource and switch normal to 16bit precision
    DXGI_FORMAT normalFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;       // ToDo rename to coefficient or avoid using same variable for different types.

	// ToDo tune formats
    // ToDo change this to non-PS resouce since we use CS?
	D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    // ToDo remove obsolete resources, QuarterResAO event triggers this so we may not need all low/gbuffer width AO resources.

    // Full-res GBuffer resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_GBufferResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count, m_GBufferResources[0].uavDescriptorHeapIndex);
        m_GBufferResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count, m_GBufferResources[0].srvDescriptorHeapIndex);
        for (UINT i = 0; i < GBufferResource::Count; i++)
        {
            m_GBufferResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_GBufferResources[i].uavDescriptorHeapIndex = m_GBufferResources[0].uavDescriptorHeapIndex + i;
            m_GBufferResources[i].srvDescriptorHeapIndex = m_GBufferResources[0].srvDescriptorHeapIndex + i;
        }
        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Hit], initialResourceState, L"GBuffer Hit");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_UINT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Material], initialResourceState, L"GBuffer Material");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32B32A32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::HitPosition], initialResourceState, L"GBuffer HitPosition");

        CreateRenderTargetResource(device, normalFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::SurfaceNormal], initialResourceState, L"GBuffer Normal");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Distance], initialResourceState, L"GBuffer Distance");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Depth], initialResourceState, L"GBuffer Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::SurfaceNormalRGB], initialResourceState, L"GBuffer Normal RGB");

        // ToDo use R16G16?
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::PartialDepthDerivatives], initialResourceState, L"GBuffer Partial Depth Derivatives");
    }
    
    // Low-res GBuffer resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_GBufferLowResResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count, m_GBufferLowResResources[0].uavDescriptorHeapIndex);
        m_GBufferLowResResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count, m_GBufferLowResResources[0].srvDescriptorHeapIndex);
        for (UINT i = 0; i < GBufferResource::Count; i++)
        {
            m_GBufferLowResResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_GBufferLowResResources[i].uavDescriptorHeapIndex = m_GBufferLowResResources[0].uavDescriptorHeapIndex + i;
            m_GBufferLowResResources[i].srvDescriptorHeapIndex = m_GBufferLowResResources[0].srvDescriptorHeapIndex + i;
        }
 
        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Hit], initialResourceState, L"GBuffer LowRes Hit");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Material], initialResourceState, L"GBuffer LowRes Material");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32B32A32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::HitPosition], initialResourceState, L"GBuffer LowRes HitPosition");
        CreateRenderTargetResource(device, normalFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::SurfaceNormal], initialResourceState, L"GBuffer LowRes Normal");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Distance], initialResourceState, L"GBuffer LowRes Distance");
        // ToDo are below two used?
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Depth], initialResourceState, L"GBuffer LowRes Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::SurfaceNormalRGB], initialResourceState, L"GBuffer LowRes Normal RGB");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives], initialResourceState, L"GBuffer Partial Depth Derivatives");

    }

    m_SSAO.BindGBufferResources(m_GBufferResources[GBufferResource::SurfaceNormalRGB].resource.Get(), m_GBufferResources[GBufferResource::Depth].resource.Get());



    // Full-res AO resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_AOResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count, m_AOResources[0].uavDescriptorHeapIndex);
        m_AOResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count, m_AOResources[0].srvDescriptorHeapIndex);
        for (UINT i = 0; i < AOResource::Count; i++)
        {
            m_AOResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_AOResources[i].uavDescriptorHeapIndex = m_AOResources[0].uavDescriptorHeapIndex + i;
            m_AOResources[i].srvDescriptorHeapIndex = m_AOResources[0].srvDescriptorHeapIndex + i;
        }

        // ToDo pack some resources.

        // ToDo cleanup raytracing resolution - twice for coefficient.
        CreateRenderTargetResource(device, texFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Coefficient], initialResourceState, L"Render/AO Coefficient");
#if ATROUS_DENOISER
        CreateRenderTargetResource(device, texFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
#else
        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UNORM, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
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
        m_AOLowResResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count, m_AOLowResResources[0].uavDescriptorHeapIndex);
        m_AOLowResResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count, m_AOLowResResources[0].srvDescriptorHeapIndex);
        for (UINT i = 0; i < AOResource::Count; i++)
        {
            m_AOLowResResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_AOLowResResources[i].uavDescriptorHeapIndex = m_AOLowResResources[0].uavDescriptorHeapIndex + i;
            m_AOLowResResources[i].srvDescriptorHeapIndex = m_AOLowResResources[0].srvDescriptorHeapIndex + i;
        }

        CreateRenderTargetResource(device, texFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::Coefficient], initialResourceState, L"Render/AO LowRes Coefficient");
#if ATROUS_DENOISER
        CreateRenderTargetResource(device, texFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::Smoothed], initialResourceState, L"Render/AO LowRes Smoothed");
#else
        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UNORM, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::Smoothed], initialResourceState, L"Render/AO LowRes Smoothed");
#endif
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::HitCount], initialResourceState, L"Render/AO LowRes Hit Count");

        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::FilterWeightSum], initialResourceState, L"Render/AO LowRes Filter Weight Sum");
 
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_AOLowResResources[AOResource::RayHitDistance], initialResourceState, L"Render/AO Hit Distance");
    }

    // ToDo pass formats via params shared across AO, GBuffer, TC

    // Full-res Temporal Cache resources.
    {
        for (UINT i = 0; i < 2; i++)
        {
            // Preallocate subsequent descriptor indices for both SRV and UAV groups.
            m_temporalCache[i][0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(TemporalCache::Count, m_temporalCache[i][0].uavDescriptorHeapIndex);
            m_temporalCache[i][0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(TemporalCache::Count, m_temporalCache[i][0].srvDescriptorHeapIndex);
            for (UINT j = 0; j < TemporalCache::Count; j++)
            {
                m_temporalCache[i][j].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
                m_temporalCache[i][j].uavDescriptorHeapIndex = m_temporalCache[i][0].uavDescriptorHeapIndex + j;
                m_temporalCache[i][j].srvDescriptorHeapIndex = m_temporalCache[i][0].srvDescriptorHeapIndex + j;
            }

            // ToDo cleanup raytracing resolution - twice for coefficient.
            CreateRenderTargetResource(device, texFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_temporalCache[i][TemporalCache::AO], initialResourceState, L"Temporal Cache: AO");
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_temporalCache[i][TemporalCache::Depth], initialResourceState, L"Temporal Cache: Depth");
            CreateRenderTargetResource(device, normalFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_temporalCache[i][TemporalCache::Normal], initialResourceState, L"Temporal Cache: Normal");
        }
     }

    // ToDo move
    // ToDo render shadows at raytracing dim?
	m_VisibilityResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
	CreateRenderTargetResource(device, texFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_VisibilityResource, initialResourceState, L"Visibility");

    m_varianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, texFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_varianceResource, initialResourceState, L"Variance");
    m_smoothedVarianceResource.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
    CreateRenderTargetResource(device, texFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_smoothedVarianceResource, initialResourceState, L"SmoothedVariance");

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

void D3D12RaytracingAmbientOcclusion::CreateAuxilaryDeviceResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();
	auto commandList = m_deviceResources->GetCommandList();

    EngineProfiling::RestoreDevice(device, commandQueue, FrameCount);

	// ToDo move?
	m_reduceSumKernel.Initialize(device, GpuKernels::ReduceSum::Uint);
    m_atrousWaveletTransformFilter.Initialize(device, ATROUS_DENOISER_MAX_PASSES, MaxAtrousWaveletTransformFilterInvocationsPerFrame);
    m_calculateVarianceKernel.Initialize(device, MaxCalculateVarianceKernelInvocationsPerFrame);
    m_calculatePartialDerivativesKernel.Initialize(device);
    m_gaussianSmoothingKernel.Initialize(device, MaxGaussianSmoothingKernelInvocationsPerFrame);
	m_downsampleBoxFilter2x2Kernel.Initialize(device);
	m_downsampleGaussian9TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap9);
	m_downsampleGaussian25TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap25);
    m_downsampleGBufferBilateralFilterKernel.Initialize(device, GpuKernels::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::FilterDepthAware2x2);
    m_downsampleValueNormalDepthBilateralFilterKernel.Initialize(device, static_cast<GpuKernels::DownsampleValueNormalDepthBilateralFilter::Type>(static_cast<UINT>(SceneArgs::DownsamplingBilateralFilter)));
    m_upsampleBilateralFilterKernel.Initialize(device, GpuKernels::UpsampleBilateralFilter::Filter2x2);
    m_multiScale_upsampleBilateralFilterAndCombineKernel.Initialize(device, GpuKernels::MultiScale_UpsampleBilateralFilterAndCombine::Filter2x2);
    m_temporalCacheReverseReprojectKernel.Initialize(device);
    
}

void D3D12RaytracingAmbientOcclusion::CreateDescriptorHeaps()
{
    auto device = m_deviceResources->GetD3DDevice();

    // ToDo revise
	// CBV SRV UAV heap.
	{
		// Allocate a heap for descriptors:
		// 2 per geometry - vertex and index  buffer SRVs
		// 1 - raytracing output texture SRV
		// 2 per BLAS - one for the acceleration structure and one for its instance desc 
		// 1 - top level acceleration structure
        // 1 per texture
        // 1 for a null diffuse texture.
		//ToDo
		UINT NumDescriptors = 2 * GeometryType::Count + 1 + 2 * MaxBLAS + 1 + ARRAYSIZE(SquidRoomAssets::Draws) * 2 + ARRAYSIZE(SquidRoomAssets::Textures) + 1;
		m_cbvSrvUavHeap = make_unique<DX::DescriptorHeap>(device, NumDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

	// Sampler heap.
	{
		UINT NumDescriptors = 1;
		m_samplerHeap = make_unique<DX::DescriptorHeap>(device, NumDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
	}

}

void D3D12RaytracingAmbientOcclusion::BuildPlaneGeometry()
{
    auto device = m_deviceResources->GetD3DDevice();

	m_geometries[GeometryType::Plane].resize(1);

	auto& geometry = m_geometries[GeometryType::Plane][0];

    // Plane indices.
    Index indices[] =
    {
        3, 1, 0,
        2, 1, 3

    };

    // Cube vertices positions and corresponding triangle normals.
    DirectX::VertexPositionNormalTexture vertices[] =
    {
        { XMFLOAT3(0.0f, 0.0f, 0.0f), XMFLOAT3(0.0f, 1.0f, 0.0f), XMFLOAT2(0.0f, 0.0f) },
        { XMFLOAT3(1.0f, 0.0f, 0.0f), XMFLOAT3(0.0f, 1.0f, 0.0f), XMFLOAT2(1.0f, 0.0f) },
        { XMFLOAT3(1.0f, 0.0f, 1.0f), XMFLOAT3(0.0f, 1.0f, 0.0f), XMFLOAT2(1.0f, 0.0f) },
        { XMFLOAT3(0.0f, 0.0f, 1.0f), XMFLOAT3(0.0f, 1.0f, 0.0f), XMFLOAT2(0.0f, 1.0f) }
    };
	
	// A ByteAddressBuffer SRV is created with a ElementSize = 0 and NumElements = number of 32 - bit words.
	UINT indexBufferSize = CeilDivide(sizeof(indices), sizeof(UINT)) * sizeof(UINT);	// Pad the buffer to fit NumElements of 32bit words.
	UINT numIndexBufferElements = indexBufferSize / sizeof(UINT);

    AllocateUploadBuffer(device, indices, indexBufferSize, &geometry.ib.buffer.resource);
    AllocateUploadBuffer(device, vertices, sizeof(vertices), &geometry.vb.buffer.resource);

    // Vertex buffer is passed to the shader along with index buffer as a descriptor range.

    // ToDo revise numElements calculation
	CreateBufferSRV(device, numIndexBufferElements, 0, m_cbvSrvUavHeap.get(), &geometry.ib.buffer);
    CreateBufferSRV(device, ARRAYSIZE(vertices), sizeof(vertices[0]), m_cbvSrvUavHeap.get(), &geometry.vb.buffer);
    ThrowIfFalse(geometry.vb.buffer.heapIndex == geometry.ib.buffer.heapIndex + 1, L"Vertex Buffer descriptor index must follow that of Index Buffer descriptor index");

	PrimitiveMaterialBuffer planeMaterialCB = { XMFLOAT3(0.24f, 0.4f, 0.4f), XMFLOAT3(1, 1, 1), 50, false, false, false, false };
	UINT materialID = static_cast<UINT>(m_materials.size());
	m_materials.push_back(planeMaterialCB);
	m_geometryInstances[GeometryType::Plane].push_back(GeometryInstance(geometry, materialID, m_nullTexture.gpuDescriptorHandle, m_nullTexture.gpuDescriptorHandle));
}

void D3D12RaytracingAmbientOcclusion::BuildTesselatedGeometry()
{
    auto device = m_deviceResources->GetD3DDevice();

    const bool RhCoords = false;

    // ToDo option to reuse multiple geometries
	m_geometries[GeometryType::Sphere].resize(1);
    auto& geometry = m_geometries[GeometryType::Sphere][0];

#if	TESSELATED_GEOMETRY_BOX_TETRAHEDRON
	// Plane indices.
	array<Index, 12> indices =
	{ {
		0, 3, 1,
		1, 3, 2,
		2, 3, 0,

#if !TESSELATED_GEOMETRY_BOX_TETRAHEDRON_REMOVE_BOTTOM_TRIANGLE
		0, 1, 2
#endif
	} };

	const float edgeLength = 0.707f;
	const float e2 = edgeLength / 2;
	// Cube vertices positions and corresponding triangle normals.
	array<DirectX::VertexPositionNormalTexture, 4> vertices =
	{ {
#if 1
		{ XMFLOAT3(-e2, -e2, -e2), XMFLOAT3(0, 0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) },
		{ XMFLOAT3(e2, -e2, -e2), XMFLOAT3(0, 0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) },
		{ XMFLOAT3(0, -e2, e2), XMFLOAT3(0, 0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) },
		{ XMFLOAT3(0, e2, 0), XMFLOAT3(0, 0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) }
#else
		{ XMFLOAT3(e2, -e2, -e2), XMFLOAT3(1, 0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) },
		{ XMFLOAT3(e2, -e2, e2), XMFLOAT3(1, 0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) },
		{ XMFLOAT3(-e2, -e2, e2), XMFLOAT3(1, 0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) },
		{ XMFLOAT3(e2, e2, e2), XMFLOAT3(1,0, 0), XMFLOAT2(FLT_MAX, FLT_MAX) }
#endif
		} };

#if 1
	for (auto& vertex : vertices)
	{
		auto& scale = m_boxSize;
		vertex.position.x *= (m_boxSize.x / e2);
		vertex.position.y *= (m_boxSize.y / e2);
		vertex.position.z *= (m_boxSize.z / e2);
	}
#endif

	auto Edge = [&](UINT v0, UINT v1)
	{
		return XMLoadFloat3(&vertices[v1].position) - XMLoadFloat3(&vertices[v0].position);
	};

	XMVECTOR faceNormals[4] =
	{
		XMVector3Cross(Edge(0, 3), Edge(0, 1)),
		XMVector3Cross(Edge(1, 3), Edge(1, 2)),
		XMVector3Cross(Edge(2, 3), Edge(2, 0)),
		XMVector3Cross(Edge(0, 1), Edge(0, 2))
	};

	
#if 1 // ToDo
	XMStoreFloat3(&vertices[0].normal, XMVector3Normalize(faceNormals[0] + faceNormals[2] + faceNormals[3]));
	XMStoreFloat3(&vertices[1].normal, XMVector3Normalize(faceNormals[0] + faceNormals[1] + faceNormals[3]));
	XMStoreFloat3(&vertices[2].normal, XMVector3Normalize(faceNormals[1] + faceNormals[2] + faceNormals[3]));
#if AO_OVERDOSE_BEND_NORMALS_DOWN
	XMStoreFloat3(&vertices[3].normal, XMVector3Normalize(faceNormals[0] + faceNormals[1] + faceNormals[2]) * XMVectorSet(1, 0.01f, 1, 0));
#else
	XMStoreFloat3(&vertices[3].normal, XMVector3Normalize(faceNormals[0] + faceNormals[1] + faceNormals[2]));
#endif
	float a = 2;
#endif
#else
	vector<GeometricPrimitive::VertexType> vertices;
	vector<Index> indices;
	switch (SceneArgs::GeometryTesselationFactor)
    {
    case 0:
        // 24 indices
#if TESSELATED_GEOMETRY_BOX_TETRAHEDRON
		GeometricPrimitive::CreateTetrahedron(vertices, indices, m_boxSize.x, RhCoords);
#elif TESSELATED_GEOMETRY_TEAPOT
		GeometricPrimitive::CreateTeapot(vertices, indices, m_geometryRadius, 10, RhCoords);
#elif TESSELATED_GEOMETRY_BOX
		GeometricPrimitive::CreateBox(vertices, indices, m_boxSize, RhCoords);
#else
		GeometricPrimitive::CreateOctahedron(vertices, indices, m_geometryRadius, RhCoords);
#endif
		break;
    case 1:
        // 36 indices
        GeometricPrimitive::CreateDodecahedron(vertices, indices, m_geometryRadius, RhCoords);
        break;
    case 2:
        // 60 indices
        GeometricPrimitive::CreateIcosahedron(vertices, indices, m_geometryRadius, RhCoords);
        break;
    default:
        // Tesselation Factor - # Indices:
        // o 3  - 126
        // o 4  - 216
        // o 5  - 330
        // o 10 - 1260
        // o 16 - 3681
        // o 20 - 4920
        const float Diameter = 2 * m_geometryRadius;
        GeometricPrimitive::CreateSphere(vertices, indices, Diameter, SceneArgs::GeometryTesselationFactor, RhCoords);
    }
#endif

#if TESSELATED_GEOMETRY_TEAPOT
	XMMATRIX rotation = XMMatrixIdentity();// XMMatrixRotationY(XM_PIDIV2);
	for (auto& vertex : vertices)
	{
		XMStoreFloat3(&vertex.position, XMVector3TransformCoord(XMLoadFloat3(&vertex.position), rotation));
		XMStoreFloat3(&vertex.normal, XMVector3TransformNormal(XMLoadFloat3(&vertex.normal), rotation));
	}
#endif
#if TESSELATED_GEOMETRY_THIN
#if TESSELATED_GEOMETRY_BOX_TETRAHEDRON
	for (auto& vertex : vertices)
	{
		if (vertex.position.y > 0)
		{
			//vertex.position.y = m_boxSize.y;
		}
	}
#else
	for (auto& vertex : vertices)
	{
		if (vertex.position.y > 0)
		{
			vertex.position.x *= 0;
			vertex.position.z *= 0;
		}
	}
#endif
#endif

	// Index buffer is created with a ByteAddressBuffer SRV. 
	// ByteAddressBuffer SRV is created with an ElementSize = 0 and NumElements = number of 32 - bit words.
	UINT indexBufferSize = CeilDivide(static_cast<UINT>(indices.size() * sizeof(indices[0])), sizeof(UINT)) * sizeof(UINT);	// Pad the buffer to fit NumElements of 32bit words.
	UINT numIndexBufferElements = indexBufferSize / sizeof(UINT);

    AllocateUploadBuffer(device, indices.data(), indexBufferSize, &geometry.ib.buffer.resource);
    AllocateUploadBuffer(device, vertices.data(), vertices.size() * sizeof(vertices[0]), &geometry.vb.buffer.resource);

    // Vertex buffer is passed to the shader along with index buffer as a descriptor table.
    CreateBufferSRV(device, numIndexBufferElements, 0, m_cbvSrvUavHeap.get(), &geometry.ib.buffer);
    CreateBufferSRV(device, static_cast<UINT>(vertices.size()), sizeof(vertices[0]), m_cbvSrvUavHeap.get(), &geometry.vb.buffer);
    ThrowIfFalse(geometry.vb.buffer.heapIndex == geometry.ib.buffer.heapIndex + 1, L"Vertex Buffer descriptor index must follow that of Index Buffer descriptor index");


	PrimitiveMaterialBuffer materialCB = { XMFLOAT3(0.75f, 0.75f, 0.75f), XMFLOAT3(1, 1, 1),  50, false, false, false, false };
	UINT materialID = static_cast<UINT>(m_materials.size());
	m_materials.push_back(materialCB);
	m_geometryInstances[GeometryType::Sphere].resize(SceneArgs::NumGeometriesPerBLAS, GeometryInstance(geometry, materialID, m_nullTexture.gpuDescriptorHandle, m_nullTexture.gpuDescriptorHandle));

	m_numTriangles[GeometryType::Sphere] = static_cast<UINT>(indices.size()) / 3;
}

// ToDo move this out as a helper
void D3D12RaytracingAmbientOcclusion::LoadSceneGeometry()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	
	m_geometries[GeometryType::SquidRoom].resize(1);
	auto& geometry = m_geometries[GeometryType::SquidRoom][0];
    auto& textures = m_geometryTextures[GeometryType::SquidRoom];

	SquidRoomAssets::LoadGeometry(
		device,
		commandList,
		m_cbvSrvUavHeap.get(),
		GetAssetFullPath(SquidRoomAssets::DataFileName).c_str(),
		&geometry,
        &textures,
        &m_materials,
		&m_geometryInstances[GeometryType::SquidRoom]);

	m_numTriangles[GeometryType::SquidRoom] = 0;
	for (auto& geometry : m_geometryInstances[GeometryType::SquidRoom])
	{
		m_numTriangles[GeometryType::SquidRoom] += geometry.ib.count / 3;
	}
#if PBRT_SCENE
	LoadPBRTScene();
#endif
}

// Build geometry used in the sample.
void D3D12RaytracingAmbientOcclusion::InitializeGeometry()
{
	auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

    // Create a null SRV for geometries with no diffuse texture.
    // Null descriptors are needed in order to achieve the effect of an "unbound" resource.
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC nullSrvDesc = {};
        nullSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        nullSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        nullSrvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        nullSrvDesc.Texture2D.MipLevels = 1;
        nullSrvDesc.Texture2D.MostDetailedMip = 0;
        nullSrvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
        
        m_nullTexture.heapIndex = m_cbvSrvUavHeap->AllocateDescriptor(&m_nullTexture.cpuDescriptorHandle, m_nullTexture.heapIndex);
        device->CreateShaderResourceView(nullptr, &nullSrvDesc, m_nullTexture.cpuDescriptorHandle);
        m_nullTexture.gpuDescriptorHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(m_cbvSrvUavHeap->GetHeap()->GetGPUDescriptorHandleForHeapStart(),
            m_nullTexture.heapIndex, m_cbvSrvUavHeap->DescriptorSize());
    }

    BuildTesselatedGeometry();
    BuildPlaneGeometry();   

	// Begin frame.
	m_deviceResources->ResetCommandAllocatorAndCommandlist();
#if ONLY_SQUID_SCENE_BLAS
	LoadSceneGeometry();
#endif

	m_materialBuffer.Create(device, static_cast<UINT>(m_materials.size()), 1, L"Structured buffer: materials");
	copy(m_materials.begin(), m_materials.end(), m_materialBuffer.begin());

    // ToDo move
    LoadDDSTexture(device, commandList, L"Assets\\Textures\\FlowerRoad\\flower_road_8khdri_1kcubemap.BC7.dds", m_cbvSrvUavHeap.get(), &m_environmentMap, D3D12_SRV_DIMENSION_TEXTURECUBE);
    //LoadDDSTexture(device, commandList, L"Assets\\Textures\\cloud_layers_8khdri_1kcubemap.BC7.dds", m_cbvSrvUavHeap.get(), &m_environmentMap, D3D12_SRV_DIMENSION_TEXTURECUBE);

#if !RUNTIME_AS_UPDATES
	InitializeAccelerationStructures();

#if ONLY_SQUID_SCENE_BLAS
#else
#if TESSELATED_GEOMETRY_BOX
	UpdateGridGeometryTransforms();
#else 
	UpdateSphereGeometryTransforms();
#endif
#endif
	UpdateBottomLevelASTransforms();

	UpdateAccelerationStructures(m_isASrebuildRequested);
#endif
	m_materialBuffer.CopyStagingToGpu();
	m_deviceResources->ExecuteCommandList();
}

void D3D12RaytracingAmbientOcclusion::GenerateBottomLevelASInstanceTransforms()
{
#if ONLY_SQUID_SCENE_BLAS
	// Bottom-level AS with a single plane.
	int BLASindex = 0;
	{
		m_vBottomLevelAS[0].SetTransform(XMMatrixIdentity());
	}
#else
    // Bottom-level AS with a single plane.
    int BLASindex = 0;
    {
        // Scale in XZ dimensions.
#if 0
        float width = 50.0f;
        XMMATRIX mScale = XMMatrixScaling(width, 1.0f, width);
        XMMATRIX mTranslation = XMMatrixTranslationFromVector(XMLoadFloat3(&XMFLOAT3(-width/2.0f, 0.0f, -width/2.0f)));
        XMMATRIX mTransform = mScale * mTranslation;
        m_vBottomLevelAS[BLASindex].SetTransform(mTransform);
#endif
		BLASindex += 1;
    }
#if DEBUG_AS
	return;
#endif 

    // Bottom-level AS with one or more spheres.
    {
        int geometryDim = static_cast<int>(ceil(cbrt(static_cast<double>(SceneArgs::NumGeometriesPerBLAS))));
        float distanceBetweenGeometry = m_geometryRadius;
        float geometryWidth = 2 * m_geometryRadius;

        int dim = static_cast<int>(ceil(sqrt(static_cast<double>(SceneArgs::NumSphereBLAS))));
        float blasWidth = geometryDim * geometryWidth + (geometryDim - 1) * distanceBetweenGeometry;
        float distanceBetweenBLAS = 3 * distanceBetweenGeometry;
        float stepDistance = blasWidth + distanceBetweenBLAS;

        for (int iX = 0; iX < dim; iX++)
            for (int iZ = 0; iZ < dim; iZ++, BLASindex++)
            {
                if (BLASindex - 1 >= SceneArgs::NumSphereBLAS)
                {
                    break;
                }

                XMFLOAT4 translationVector = XMFLOAT4(
                    static_cast<float>(iX),
                    0.0f,
                    static_cast<float>(iZ),
                    0.0f);
                XMMATRIX transform = XMMatrixTranslationFromVector(stepDistance * XMLoadFloat4(&translationVector));
                m_vBottomLevelAS[BLASindex].SetTransform(transform);
            }
    }
#endif

}

// Build acceleration structure needed for raytracing.
void D3D12RaytracingAmbientOcclusion::InitializeAccelerationStructures()
{
    auto device = m_deviceResources->GetD3DDevice();
    
    // Build flags.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    // Initialize bottom-level AS.
	UINT64 maxScratchResourceSize = 0;
    m_ASmemoryFootprint = 0;
    {
#if ONLY_SQUID_SCENE_BLAS
		m_vBottomLevelAS.resize(1);
		// ToDo apply scale transform to make all scenes using same spatial unit lengths.
#if PBRT_SCENE
		auto geometryType = GeometryType::PBRT;
#else
		auto geometryType = GeometryType::SquidRoom;
#endif
		m_vBottomLevelAS.resize(1);
		m_vBottomLevelAS[0].Initialize(device, buildFlags, SquidRoomAssets::StandardIndexFormat, SquidRoomAssets::StandardIndexStride, SquidRoomAssets::StandardVertexStride, m_geometryInstances[geometryType]);
		m_numTrianglesInTheScene = m_numTriangles[geometryType];
		
		m_vBottomLevelAS[0].SetInstanceContributionToHitGroupIndex(0);	// ToDo fix hack
		maxScratchResourceSize = max(m_vBottomLevelAS[0].RequiredScratchSize(), maxScratchResourceSize);
		m_ASmemoryFootprint += m_vBottomLevelAS[0].RequiredResultDataSizeInBytes();

		UINT numGeometryTransforms = static_cast<UINT>(m_geometryInstances[geometryType].size());
#else
		m_numTrianglesInTheScene = 0;
#if DEBUG_AS
		m_vBottomLevelAS.resize(1);
		for (UINT i = 0; i < 1; i++)
#else
		m_vBottomLevelAS.resize(2);
		for (UINT i = 0; i < m_vBottomLevelAS.size(); i++)
#endif
		{
			UINT instanceContributionHitGroupIndex;
			GeometryType::Enum geometryType;
            switch (i) 
            {
			case 0: geometryType = GeometryType::Plane;
				instanceContributionHitGroupIndex = 0;
				break;
			case 1: geometryType = GeometryType::Sphere;
				instanceContributionHitGroupIndex = static_cast<UINT>(m_geometryInstances[GeometryType::Plane].size()) * RayType::Count;
				break;
			default:
				assert(0);
				break;
            };
			auto& geometryInstances = m_geometryInstances[geometryType];

			// ToDo pass IB stride from a geometryInstance object
			m_vBottomLevelAS[i].Initialize(device, buildFlags, DXGI_FORMAT_R16_UINT, sizeof(Index), sizeof(DirectX::GeometricPrimitive::VertexType), geometryInstances);
			m_numTrianglesInTheScene += m_numTriangles[geometryType];
			
			m_vBottomLevelAS[i].SetInstanceContributionToHitGroupIndex(instanceContributionHitGroupIndex);
            maxScratchResourceSize = max(m_vBottomLevelAS[i].RequiredScratchSize(), maxScratchResourceSize);
            m_ASmemoryFootprint += m_vBottomLevelAS[i].RequiredResultDataSizeInBytes();
        }
		UINT numGeometryTransforms = 1 + SceneArgs::NumSphereBLAS * SceneArgs::NumGeometriesPerBLAS;
#endif

		// ToDo Allocate exactly as much is needed?
		ThrowIfFalse(numGeometryTransforms <= MaxGeometryTransforms, L"Scene requires more transform space.");
    }

    GenerateBottomLevelASInstanceTransforms();

    // Initialize top-level AS.
    {
        m_topLevelAS.Initialize(device, m_vBottomLevelAS, buildFlags, &m_bottomLevelASinstanceDescsDescritorHeapIndices);
        maxScratchResourceSize = max(m_topLevelAS.RequiredScratchSize(), maxScratchResourceSize);
        m_ASmemoryFootprint += m_topLevelAS.RequiredResultDataSizeInBytes();
    }

    // Create a scratch buffer.
    // ToDo: Compare build perf vs using per AS scratch
    AllocateUAVBuffer(device, maxScratchResourceSize, &m_accelerationStructureScratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"Acceleration structure scratch resource");

    m_isASrebuildRequested = true;
}

// Build shader tables.
// This encapsulates all shader records - shaders and the arguments for their local root signatures.
void D3D12RaytracingAmbientOcclusion::BuildShaderTables()
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
			hitGroupShaderIDs_TriangleGeometry[i] = stateObjectProperties->GetShaderIdentifier(c_hitGroupNames_TriangleGeometry[i]);
			shaderIdToStringMap[hitGroupShaderIDs_TriangleGeometry[i]] = c_hitGroupNames_TriangleGeometry[i];
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
#if ONLY_SQUID_SCENE_BLAS
#if PBRT_SCENE
	geometryInstancesArray.push_back(&m_geometryInstances[GeometryType::PBRT]);
#else
	geometryInstancesArray.push_back(&m_geometryInstances[GeometryType::SquidRoom]);
#endif
#else

	geometryInstancesArray.push_back(&m_geometryInstances[GeometryType::Plane]);

#if !DEBUG_AS
	geometryInstancesArray.push_back(&m_geometryInstances[GeometryType::Sphere]);
#endif
#endif

	// Hit group shader table.
	{
		UINT numShaderRecords = 0;
		for (auto& geometryInstances : geometryInstancesArray)
		{
			numShaderRecords += static_cast<UINT>(geometryInstances->size()) * RayType::Count;
		}
		UINT shaderRecordSize = shaderIDSize + LocalRootSignature::MaxRootArgumentsSize();
		ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");

		// Triangle geometry hit groups.
		for (auto& geometryInstances : geometryInstancesArray)
			for (auto& geometryInstance: *geometryInstances)
			{
				LocalRootSignature::Triangle::RootArguments rootArgs;
				rootArgs.cb.materialID = geometryInstance.materialID;

                // ToDo rename to ibGPUHandle, otherwise its confusing why copy ib to vb handle.
				memcpy(&rootArgs.vertexBufferGPUHandle, &geometryInstance.ib.gpuDescriptorHandle, sizeof(geometryInstance.ib.gpuDescriptorHandle));
                memcpy(&rootArgs.diffuseTextureGPUHandle, &geometryInstance.diffuseTexture, sizeof(geometryInstance.diffuseTexture));
                memcpy(&rootArgs.normalTextureGPUHandle, &geometryInstance.normalTexture, sizeof(geometryInstance.normalTexture));


				for (auto& hitGroupShaderID : hitGroupShaderIDs_TriangleGeometry)
				{
					hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderID, shaderIDSize, &rootArgs, sizeof(rootArgs)));
				}
			}

        hitGroupShaderTable.DebugPrint(shaderIdToStringMap);
        m_hitGroupShaderTableStrideInBytes = hitGroupShaderTable.GetShaderRecordSize();
        m_hitGroupShaderTable = hitGroupShaderTable.GetResource();
    }
}

void D3D12RaytracingAmbientOcclusion::OnKeyDown(UINT8 key)
{
	// ToDo 
    switch (key)
    {
    case VK_ESCAPE:
        throw HrException(E_APPLICATION_EXITING);
    case 'L':
        m_animateLight = !m_animateLight;
        break;
    case 'C':
        m_animateCamera = !m_animateCamera;
        break;
    case 'A':
        m_animateScene = !m_animateScene;
        break;
    case 'R':
        m_temporalCacheFrameAge = 0;
    default:
        break;
    }
}

// Update frame-based values.
void D3D12RaytracingAmbientOcclusion::OnUpdate()
{
    m_timer.Tick();

    float elapsedTime = static_cast<float>(m_timer.GetElapsedSeconds());
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
	}

	if (m_isRecreateAOSamplesRequested)
	{
		m_isRecreateAOSamplesRequested = false;
		m_deviceResources->WaitForGpu();
		CreateSamplesRNG();
	}

    CalculateFrameStats();

    GameInput::Update(elapsedTime);
    EngineTuning::Update(elapsedTime);
    EngineProfiling::Update();
	
	if (GameInput::IsFirstPressed(GameInput::kKey_f))
	{
		m_isCameraFrozen = !m_isCameraFrozen;
	}

    m_cameraChangedIndex--;
    m_hasCameraChanged = false;
	if (!m_isCameraFrozen)
	{
        m_prevFrameCamera = m_camera;
        m_hasCameraChanged = m_cameraController->Update(elapsedTime);
        // ToDo
        // if (CameraChanged)
        //m_bClearTemporalCache = true;

	}


    // Rotate the camera around Y axis.
    if (m_animateCamera)
    {
        m_hasCameraChanged = true;
		// ToDo
        float secondsToRotateAround = 48.0f;
        float angleToRotateBy = 360.0f * (elapsedTime / secondsToRotateAround);
        XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
		XMVECTOR eye =  m_camera.Eye();
		XMVECTOR at = m_camera.At();
		XMVECTOR up = m_camera.Up();		
		at = XMVector3TransformCoord(at, rotate);
		eye = XMVector3TransformCoord(eye, rotate);
		up = XMVector3TransformNormal(up, rotate);
		m_camera.Set(eye, at, up);
    }

    if (m_hasCameraChanged)
    {
        m_cameraChangedIndex = 5;
        OutputDebugString(L"CameraChanged\n");
    }

	UpdateCameraMatrices();

    // Rotate the second light around Y axis.
    if (m_animateLight)
    {
        float secondsToRotateAround = 8.0f;
        float angleToRotateBy = -360.0f * (elapsedTime / secondsToRotateAround);
        XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
		XMVECTOR prevLightPosition = XMLoadFloat3(&m_csComposeRenderPassesCB->lightPosition);
		XMStoreFloat3(&m_csComposeRenderPassesCB->lightPosition, XMVector3Transform(prevLightPosition, rotate));
		m_sceneCB->lightPosition = XMLoadFloat3(&m_csComposeRenderPassesCB->lightPosition);
    }
    m_sceneCB->elapsedTime = static_cast<float>(m_timer.GetTotalSeconds());

#if ENABLE_RAYTRACING
#if RUNTIME_AS_UPDATES
    // Lazy initialize and update geometries and acceleration structures.
    if (SceneArgs::EnableGeometryAndASBuildsAndUpdates &&
        (m_isGeometryInitializationRequested || m_isASinitializationRequested))
    {
        // Since we'll be recreating D3D resources, GPU needs to be done with the current ones.
		// ToDo
        m_deviceResources->WaitForGpu();

        m_deviceResources->ResetCommandAllocatorAndCommandlist();
        if (m_isGeometryInitializationRequested)
        {
            InitializeGeometry();
        }
        if (m_isASinitializationRequested)
        {
            InitializeAccelerationStructures();
        }

        m_isGeometryInitializationRequested = false;
        m_isASinitializationRequested = false;
        m_deviceResources->ExecuteCommandList();

		// ToDo remove CPU-GPU syncs
		m_deviceResources->WaitForGpu();
    }
    if (m_animateScene)
    {
#if TESSELATED_GEOMETRY_BOX
		UpdateGridGeometryTransforms();
#else
        UpdateSphereGeometryTransforms();
#endif
		UpdateBottomLevelASTransforms();
    }
#endif
#endif

    // ToDo move
    // SSAO
    {   
        m_SSAOCB->noiseTile = { float(m_width) / float(SSAO_NOISE_W), float(m_height) / float(SSAO_NOISE_W), 0, 0};
        m_SSAO.SetParameters(SceneArgs::SSAONoiseFilterTolerance, SceneArgs::SSAOBlurTolerance, SceneArgs::SSAOUpsampleTolerance, SceneArgs::SSAONormalMultiply);
    
    }
	if (m_enableUI)
    {
        UpdateUI();
    }

    // ToDo move
    m_sceneCB->RTAO_UseAdaptiveSampling = SceneArgs::RTAOAdaptiveSampling;
    m_sceneCB->RTAO_AdaptiveSamplingMaxWeightSum = SceneArgs::RTAOAdaptiveSamplingMaxFilterWeight;
    m_sceneCB->RTAO_UseNormalMaps = SceneArgs::RTAOUseNormalMaps;
    m_sceneCB->RTAO_AdaptiveSamplingMinMaxSampling = SceneArgs::RTAOAdaptiveSamplingMinMaxSampling;
    m_sceneCB->RTAO_AdaptiveSamplingScaleExponent = SceneArgs::RTAOAdaptiveSamplingScaleExponent;
    m_sceneCB->RTAO_AdaptiveSamplingMinSamples = SceneArgs::RTAOAdaptiveSamplingMinSamples;
    m_sceneCB->RTAO_TraceRayOffsetAlongNormal = SceneArgs::RTAOTraceRayOffsetAlongNormal;
    m_sceneCB->RTAO_TraceRayOffsetAlongRayDirection = SceneArgs::RTAOTraceRayOffsetAlongRayDirection;


    SceneArgs::RTAOAdaptiveSamplingMinSamples.SetMaxValue(SceneArgs::AOSampleCountPerDimension * SceneArgs::AOSampleCountPerDimension);
 }

// Parse supplied command line args.
void D3D12RaytracingAmbientOcclusion::ParseCommandLineArgs(WCHAR* argv[], int argc)
{
    DXSample::ParseCommandLineArgs(argv, argc);
}

void D3D12RaytracingAmbientOcclusion::UpdateAccelerationStructures(bool forceBuild)
{
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
    m_numFramesSinceASBuild++;

    ScopedTimer _prof(L"BuildAccelerationStructure", commandList);

    // ToDo move this next to TLAS build? But BLAS update resets its dirty flag
    m_topLevelAS.UpdateInstanceDescTransforms(m_vBottomLevelAS);
    
    BOOL bUpdate = false;    // ~ build or update
    if (!forceBuild)
    {
        switch (SceneArgs::ASUpdateMode)
        {
        case SceneArgs::Update:
            bUpdate = true;
            break;
        case SceneArgs::Build:
            bUpdate = false;
            break;
        case SceneArgs::Update_BuildEveryXFrames:
            bUpdate = m_numFramesSinceASBuild < SceneArgs::ASBuildFrequency;
        default: 
            break;
        };
    }


    // Build BLAS.
	{
        ScopedTimer _prof(L"BLAS", commandList);

		m_geometryTransforms.CopyStagingToGpu(frameIndex);
#if ONLY_SQUID_SCENE_BLAS
		// ToDo this should be per scene
		// SquidRoom
		{
			// ToDo Heuristic to do an update based on transform amplitude
			D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGpuAddress = 0;
			m_vBottomLevelAS[0].Build(commandList, m_accelerationStructureScratch.Get(), m_cbvSrvUavHeap->GetHeap(), baseGeometryTransformGpuAddress, bUpdate);
		}
#else
		// Plane
		{
			D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGpuAddress = 0;
#if USE_GPU_TRANSFORM // ToDo either place in same blas or move transform to blas?
			baseGeometryTransformGpuAddress = m_geometryTransforms.GpuVirtualAddress(frameIndex);
#endif
			m_vBottomLevelAS[BottomLevelASType::Plane].Build(commandList, m_accelerationStructureScratch.Get(), m_cbvSrvUavHeap->GetHeap(), baseGeometryTransformGpuAddress, bUpdate);
		}
#if DEBUG_AS
		if (0)
#endif
		// Sphere
		{
            D3D12_GPU_VIRTUAL_ADDRESS baseGeometryTransformGpuAddress = 0;     
#if USE_GPU_TRANSFORM
			// ToDo - remove: skip past plane transform
            baseGeometryTransformGpuAddress = m_geometryTransforms.GpuVirtualAddress(frameIndex, 1);
#endif
			m_vBottomLevelAS[BottomLevelASType::Sphere].Build(commandList, m_accelerationStructureScratch.Get(), m_cbvSrvUavHeap->GetHeap(), baseGeometryTransformGpuAddress, bUpdate);
        }
#endif
    }

    // Build TLAS.
    {
        ScopedTimer _prof(L"TLAS", commandList);
        m_topLevelAS.Build(commandList, m_accelerationStructureScratch.Get(), m_cbvSrvUavHeap->GetHeap(), bUpdate);
    }

    if (!bUpdate)
    {
        m_numFramesSinceASBuild = 0;
    }
}

void D3D12RaytracingAmbientOcclusion::DispatchRays(ID3D12Resource* rayGenShaderTable, uint32_t width, uint32_t height)
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
	dispatchDesc.Width = width != 0 ? width : m_GBufferWidth;
	dispatchDesc.Height = height != 0 ? height : m_GBufferHeight;
	dispatchDesc.Depth = 1;
	commandList->SetPipelineState1(m_dxrStateObject.Get());

	commandList->DispatchRays(&dispatchDesc);
};

void D3D12RaytracingAmbientOcclusion::CalculateRayHitCount(ReduceSumCalculations::Enum type)
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
	auto commandList = m_deviceResources->GetCommandList();

	RWGpuResource* inputResource = nullptr;
	switch (type)
	{
	case ReduceSumCalculations::CameraRayHits: inputResource = &m_GBufferResources[GBufferResource::Hit]; break;
	case ReduceSumCalculations::AORayHits: inputResource = &m_AOResources[AOResource::HitCount]; break;
	}


    // Todo
    if (SceneArgs::QuarterResAO)
    {
        return;
    }

	m_reduceSumKernel.Execute(
		commandList,
		m_cbvSrvUavHeap->GetHeap(),
		frameIndex,
		type,
        inputResource->gpuDescriptorReadAccess,
		&m_numRayGeometryHits[type]);
};

void D3D12RaytracingAmbientOcclusion::ApplyAtrousWaveletTransformFilter()
{
    auto commandList = m_deviceResources->GetCommandList();

    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;

    ScopedTimer _prof(L"DenoiseAO", commandList);

    // Calculate local variance.
    {
        ScopedTimer _prof(L"CalculateVariance", commandList);
        m_calculateVarianceKernel.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            static_cast<GpuKernels::CalculateVariance::FilterType>(static_cast<UINT>(SceneArgs::VarianceFilterMode)),
            m_raytracingWidth,
            m_raytracingHeight,
            AOResources[AOResource::Coefficient].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess,
            m_varianceResource.gpuDescriptorWriteAccess,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            SceneArgs::ApproximateSpatialVariance,
            SceneArgs::RTAODenoisingPerspectiveCorrectDepthInterpolation,
            m_calculateVarianceKernelInstancePerFrameInstanceId++);

        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_varianceResource.resource.Get(), before, after));
    }

    // ToDo, should the smoothing be applied after each pass?
    // Smoothen the local variance which is prone to error due to undersampled input.
    {
        ScopedTimer _prof(L"VarianceSmoothing", commandList);
        m_gaussianSmoothingKernel.Execute(
            commandList,
            m_raytracingWidth,
            m_raytracingHeight,
            GpuKernels::GaussianFilter::Filter3X3,
            m_cbvSrvUavHeap->GetHeap(),
            m_varianceResource.gpuDescriptorReadAccess,
            m_smoothedVarianceResource.gpuDescriptorWriteAccess,
            m_gaussianSmoothingKernelPerFrameInstanceId++);
    }

    // Transition Variance resource to shader resource state.
    // Also prepare smoothed AO resource for the next pass and transition it to UAV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_smoothedVarianceResource.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Smoothed].resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }
    
#if RAYTRACING_MANUAL_KERNEL_STEP_SHIFTS
    UINT offsets[5] = {
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift0),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift1),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift2),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift3),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift4)};

    UINT newStartId = 0;
    for (UINT i = 0; i < 5; i++)
    {
        offsets[i] = newStartId + offsets[i];
        newStartId = offsets[i] + 1;
    }
#endif
    // A-trous edge-preserving wavelet tranform filter
    {
        ScopedTimer _prof(L"AtrousWaveletTransformFilter", commandList);
        m_atrousWaveletTransformFilter.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            static_cast<GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType>(static_cast<UINT>(SceneArgs::DenoisingMode)),
            AOResources[AOResource::Coefficient].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess,
            m_varianceResource.gpuDescriptorReadAccess,
            m_smoothedVarianceResource.gpuDescriptorReadAccess,
            AOResources[AOResource::RayHitDistance].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            &AOResources[AOResource::Smoothed],
            SceneArgs::AODenoiseValueSigma,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            offsets,
            SceneArgs::AtrousFilterPasses,
            GpuKernels::AtrousWaveletTransformCrossBilateralFilter::Mode::OutputFilteredValue,
            SceneArgs::ReverseFilterOrder,
            SceneArgs::UseSpatialVariance,
            SceneArgs::RTAODenoisingPerspectiveCorrectDepthInterpolation,
            SceneArgs::RTAODenoisingUseAdaptiveKernelSize,
            SceneArgs::RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor,
            SceneArgs::RTAODenoisingFilterMinKernelWidth,
            static_cast<UINT>((SceneArgs::RTAODenoisingFilterMaxKernelWidthPercentage / 100) * m_raytracingWidth),
            SceneArgs::RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels,
            SceneArgs::QuarterResAO,
            m_atrousWaveletTransformFilterPerFrameInstanceId++);
    }

    // Transition the output resource to SRV.
    {
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_AOLowResResources[AOResource::Smoothed].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
    }
};


// Apply multi scale denoising to denoise low frequencies.
// Ref: Delbracio et al. 2014, Boosting Monte Carlo Rendering by Ray Histogram Fusion
void D3D12RaytracingAmbientOcclusion::ApplyMultiScaleAtrousWaveletTransformFilter()
{
    auto commandList = m_deviceResources->GetCommandList();

    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    
    ScopedTimer _prof(L"MultiScaleAtrousWaveletTransform", commandList);

    // ToDo remove this since the next iter is downsampling.
    // Downsample input value.
    {
        ScopedTimer _prof(L"DownsampleValueBuffers", commandList);

        RWGpuResource* inputResources[] = { &AOResources[AOResource::Coefficient], &GBufferResources[GBufferResource::SurfaceNormal] };
      
        for (int i = 0; i < SceneArgs::RTAODenoisingMultiscaleLevels; i++)
        {
            MultiScaleDenoisingResource& msResource = m_multiScaleDenoisingResources[i];
            
            ScopedTimer _prof(L"Scale Level", i, commandList);

            // Transition all output resources to UAV state.
            {
                D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                D3D12_RESOURCE_BARRIER barriers[] = {
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), before, after),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_normalDepth.resource.Get(), before, after),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_partialDistanceDerivatives.resource.Get(), before, after)
                };
                commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
            }
            
            // ToDo combine barriers
            // ToDo avoid the copy and use source directly?
            // Copy the inputs to the first level
            if (i == 0)
            {
                D3D12_RESOURCE_BARRIER preCopyBarriers[] = {
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_normalDepth.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_partialDistanceDerivatives.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
                    CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Coefficient].resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    CD3DX12_RESOURCE_BARRIER::Transition(GBufferResources[GBufferResource::SurfaceNormal].resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    CD3DX12_RESOURCE_BARRIER::Transition(GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE)
                };
                commandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

                commandList->CopyResource(msResource.m_value.resource.Get(), AOResources[AOResource::Coefficient].resource.Get());
                commandList->CopyResource(msResource.m_normalDepth.resource.Get(), GBufferResources[GBufferResource::SurfaceNormal].resource.Get());
                commandList->CopyResource(msResource.m_partialDistanceDerivatives.resource.Get(), GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get());

                D3D12_RESOURCE_BARRIER postCopyBarriers[] = {
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_normalDepth.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_partialDistanceDerivatives.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Coefficient].resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                    CD3DX12_RESOURCE_BARRIER::Transition(GBufferResources[GBufferResource::SurfaceNormal].resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                    CD3DX12_RESOURCE_BARRIER::Transition(GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
                };
                commandList->ResourceBarrier(ARRAYSIZE(postCopyBarriers), postCopyBarriers);
            }
            // ToDo Cleanup - don't need to downsample values
            else
            {
                MultiScaleDenoisingResource& msPrevIterResource = m_multiScaleDenoisingResources[i - 1];
                D3D12_RESOURCE_DESC desc = msPrevIterResource.m_value.resource.Get()->GetDesc();
                UINT width = static_cast<UINT>(desc.Width);
                UINT height = static_cast<UINT>(desc.Height);
                m_downsampleValueNormalDepthBilateralFilterKernel.Execute(
                    commandList,
                    width,
                    height,
                    m_cbvSrvUavHeap->GetHeap(),
                    msPrevIterResource.m_value.gpuDescriptorReadAccess,
                    msPrevIterResource.m_normalDepth.gpuDescriptorReadAccess,
                    msPrevIterResource.m_partialDistanceDerivatives.gpuDescriptorReadAccess,
                    msResource.m_value.gpuDescriptorWriteAccess,
                    msResource.m_normalDepth.gpuDescriptorWriteAccess,
                    msResource.m_partialDistanceDerivatives.gpuDescriptorWriteAccess);
            }

            // Transition the output resources to shader resource state.
            {
                D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                D3D12_RESOURCE_BARRIER barriers[] = {
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), before, after),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_normalDepth.resource.Get(), before, after),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_partialDistanceDerivatives.resource.Get(), before, after)
                };
                commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
            }
        }
    }

    // Denoise and downsample each level.
    {
        ScopedTimer _prof(L"DenoiseBuffers", commandList);
        for (int i = 0; i < SceneArgs::RTAODenoisingMultiscaleLevels; i++)
        {
            ScopedTimer _prof(L"Scale Level", i, commandList);

            MultiScaleDenoisingResource& msResource = m_multiScaleDenoisingResources[i];

            // Transition all output resources to UAV state.
            {
                D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                D3D12_RESOURCE_BARRIER barriers[] = {
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_smoothedValue.resource.Get(), before, after),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_varianceResource.resource.Get(), before, after),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_smoothedVarianceResource.resource.Get(), before, after)
                };
                commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
            }

            // ToDo skip some downsampling above due to following?
            // Denoise value resource on the 1st iteration and then propagate down the levels the downsampled denoised resource.
            RWGpuResource* valueResource;
            if (SceneArgs::RTAODenoisingMultiscaleDenoisedAsInput)
            {
                valueResource = i == 0 ? &msResource.m_value : &m_multiScaleDenoisingResources[i - 1].m_downsampledSmoothedValue;
            }
            else
            {
                valueResource = &msResource.m_value;
            }

            ApplyAtrousWaveletTransformFilter(
                *valueResource,
                msResource.m_normalDepth,
                // ToDo remove
                GBufferResources[GBufferResource::Distance],
                AOResources[AOResource::RayHitDistance],
                msResource.m_partialDistanceDerivatives,

                &msResource.m_smoothedValue,
                &msResource.m_varianceResource,
                &msResource.m_smoothedVarianceResource,
                UINT_MAX,
                UINT_MAX,
                UINT_MAX);

            // Downsample the denoised value.
            {
                // Transition input resource to SRV and output resources to UAV state.
                {
                    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                    D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    D3D12_RESOURCE_BARRIER barriers[] = {
                        CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_smoothedValue.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                        CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_downsampledSmoothedValue.resource.Get(), before, after),
                        CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_downsampledNormalDepthValue.resource.Get(), before, after)
                    };
                    commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
                }

                D3D12_RESOURCE_DESC desc = msResource.m_smoothedValue.resource.Get()->GetDesc();
                UINT width = static_cast<UINT>(desc.Width);
                UINT height = static_cast<UINT>(desc.Height);

                // Last denoised scale doesn't need to be downsampled as the denoised result itself is propagated upstream.
                if (i < SceneArgs::RTAODenoisingMultiscaleLevels - 1)
                {

                    ScopedTimer _prof(L"Downsample", i, commandList);
                    m_downsampleValueNormalDepthBilateralFilterKernel.Execute(
                        commandList,
                        width,
                        height,
                        m_cbvSrvUavHeap->GetHeap(),
                        msResource.m_smoothedValue.gpuDescriptorReadAccess,
                        msResource.m_normalDepth.gpuDescriptorReadAccess,
                        msResource.m_partialDistanceDerivatives.gpuDescriptorReadAccess,
                        msResource.m_downsampledSmoothedValue.gpuDescriptorWriteAccess,
                        msResource.m_downsampledNormalDepthValue.gpuDescriptorWriteAccess,
                        msResource.m_downsampledPartialDistanceDerivatives.gpuDescriptorWriteAccess);
                }

                // Transition the output resources to shader resource state.    // ToDo say SRV instead to match UAV wording
                {
                    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                    D3D12_RESOURCE_BARRIER barriers[] = {
                        CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_downsampledSmoothedValue.resource.Get(), before, after),
                        CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_downsampledNormalDepthValue.resource.Get(), before, after)
                    };
                    commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
                }
            }
        }
    }

    // For single scale level, copy the denoised scale level 0 result.
    if (SceneArgs::RTAODenoisingMultiscaleLevels < 2)
    {
        MultiScaleDenoisingResource& msResource = m_multiScaleDenoisingResources[0];

        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER preCopyBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_smoothedValue.resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST)
        };
        commandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

        commandList->CopyResource(msResource.m_value.resource.Get(), msResource.m_smoothedValue.resource.Get());

        D3D12_RESOURCE_BARRIER postCopyBarriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_smoothedValue.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
        };
        commandList->ResourceBarrier(ARRAYSIZE(postCopyBarriers), postCopyBarriers);
    }
    // Reconstruct the final result from all denoised scales.
    else 
    {
        ScopedTimer _prof(L"CombineAllLevels", commandList);

        for (int i = SceneArgs::RTAODenoisingMultiscaleLevels - 2; i >= 0; i--)
        {
            ScopedTimer _prof(L"Scale Level", i, commandList);

            MultiScaleDenoisingResource& msResource = m_multiScaleDenoisingResources[i];
            MultiScaleDenoisingResource& msLowResLevelResource = m_multiScaleDenoisingResources[i + 1];

            // Transition output resource to UAV state.
            {
                D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), before, after));
            }

            // m_value resource is being used as an output resource going up the chain.
            // However, lowest resolution level is skipped and its m_value is m_smoothedValue.
            // Thus, on first iteration, use lower resource's m_smoothedValue instead.
            const bool isFirstIter = i == SceneArgs::RTAODenoisingMultiscaleLevels - 2;
            RWGpuResource *inLowResResource = isFirstIter ? &msLowResLevelResource.m_smoothedValue : &msLowResLevelResource.m_value;

            D3D12_RESOURCE_DESC desc = msResource.m_value.resource.Get()->GetDesc();
            UINT width = static_cast<UINT>(desc.Width);
            UINT height = static_cast<UINT>(desc.Height);
            m_multiScale_upsampleBilateralFilterAndCombineKernel.Execute(
                commandList,
                width,
                height,
                m_cbvSrvUavHeap->GetHeap(),
                msResource.m_downsampledSmoothedValue.gpuDescriptorReadAccess,
                inLowResResource->gpuDescriptorReadAccess,
                msResource.m_downsampledNormalDepthValue.gpuDescriptorReadAccess,
                msResource.m_smoothedValue.gpuDescriptorReadAccess,
                msResource.m_normalDepth.gpuDescriptorReadAccess,
                msResource.m_partialDistanceDerivatives.gpuDescriptorReadAccess,
                msResource.m_value.gpuDescriptorWriteAccess);

            // Transition the output resource to shader resource state.
            {
                D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), before, after));
            }
        }
    }
}

// ToDo move out
void D3D12RaytracingAmbientOcclusion::ApplyAtrousWaveletTransformFilter(
    const RWGpuResource& inValueResource,
    const RWGpuResource& inNormalDepthResource,
    const RWGpuResource& inDepthResource,
    const RWGpuResource& inRayHitDistanceResource, 
    const RWGpuResource& inPartialDistanceDerivativesResource,
    RWGpuResource* outSmoothedValueResource,
    RWGpuResource* varianceResource,
    RWGpuResource* smoothedVarianceResource,
    UINT calculateVarianceTimerId,
    UINT smoothVarianceTimerId,
    UINT atrousFilterTimerId
)
{
    auto commandList = m_deviceResources->GetCommandList();
    
    auto desc = inValueResource.resource.Get()->GetDesc();
    // ToDo cleanup widths on GPU kernels, it should be the one of input resource.
    UINT width = static_cast<UINT>(desc.Width);
    UINT height = static_cast<UINT>(desc.Height);

    // Calculate local variance.
    {
        ScopedTimer _prof(L"CalculateVariance", commandList);
        m_calculateVarianceKernel.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            static_cast<GpuKernels::CalculateVariance::FilterType>(static_cast<UINT>(SceneArgs::VarianceFilterMode)),
            width,
            height,
            inValueResource.gpuDescriptorReadAccess,
            inNormalDepthResource.gpuDescriptorReadAccess,
            inDepthResource.gpuDescriptorReadAccess,
            varianceResource->gpuDescriptorWriteAccess,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            SceneArgs::ApproximateSpatialVariance,
            SceneArgs::RTAODenoisingPerspectiveCorrectDepthInterpolation,
            m_calculateVarianceKernelInstancePerFrameInstanceId++);

        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(varianceResource->resource.Get(), before, after));
    }

    // ToDo, should the smoothing be applied after each pass?
    // Smoothen the local variance which is prone to error due to undersampled input.
    {
        ScopedTimer _prof(L"VarianceSmoothing", commandList);
        m_gaussianSmoothingKernel.Execute(
            commandList,
            width,
            height,
            GpuKernels::GaussianFilter::Filter3X3,
            m_cbvSrvUavHeap->GetHeap(),
            varianceResource->gpuDescriptorReadAccess,
            smoothedVarianceResource->gpuDescriptorWriteAccess,
            m_gaussianSmoothingKernelPerFrameInstanceId++);
    }

    // Transition Variance resource to shader resource state.
    // Also prepare smoothed AO resource for the next pass and transition it to UAV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(smoothedVarianceResource->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
      // ToDo Remove      CD3DX12_RESOURCE_BARRIER::Transition(outSmoothedValueResource->resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

#if RAYTRACING_MANUAL_KERNEL_STEP_SHIFTS
    UINT offsets[5] = {
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift0),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift1),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift2),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift3),
        static_cast<UINT>(SceneArgs::RTAO_KernelStepShift4) };

    UINT newStartId = 0;
    for (UINT i = 0; i < 5; i++)
    {
        offsets[i] = newStartId + offsets[i];
        newStartId = offsets[i] + 1;
    }
#endif
    // A-trous edge-preserving wavelet tranform filter.
    {
        ScopedTimer _prof(L"AtrousWaveletTransformFilter", commandList);
        m_atrousWaveletTransformFilter.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            static_cast<GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType>(static_cast<UINT>(SceneArgs::DenoisingMode)),
            inValueResource.gpuDescriptorReadAccess,
            inNormalDepthResource.gpuDescriptorReadAccess,
            inDepthResource.gpuDescriptorReadAccess,
            varianceResource->gpuDescriptorReadAccess,
            smoothedVarianceResource->gpuDescriptorReadAccess,
            inRayHitDistanceResource.gpuDescriptorReadAccess,
            inPartialDistanceDerivativesResource.gpuDescriptorReadAccess,
            outSmoothedValueResource,
            SceneArgs::AODenoiseValueSigma,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            offsets,
            SceneArgs::AtrousFilterPasses,
            GpuKernels::AtrousWaveletTransformCrossBilateralFilter::Mode::OutputFilteredValue,
            SceneArgs::ReverseFilterOrder,
            SceneArgs::UseSpatialVariance,
            SceneArgs::RTAODenoisingPerspectiveCorrectDepthInterpolation,
            SceneArgs::RTAODenoisingUseAdaptiveKernelSize,
            SceneArgs::RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor,
            SceneArgs::RTAODenoisingFilterMinKernelWidth,
            static_cast<UINT>((SceneArgs::RTAODenoisingFilterMaxKernelWidthPercentage / 100) * m_raytracingWidth),
            SceneArgs::RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels,
            SceneArgs::QuarterResAO,
            m_atrousWaveletTransformFilterPerFrameInstanceId++);
    }
};



// Calculate adaptive per-pixel sampling counts.
// Ref: Bauszat et al. 2011, Adaptive Sampling for Geometry-Aware Reconstruction Filters
// The per-pixel sampling counts are calculated in two steps:
//  - Computing a per-pixel filter weight. This value represents how much of neighborhood contributes
//    to a pixel when filtering. Pixels with small values benefit from filtering only a little and thus will 
//    benefit from increased sampling the most.
//  - Calculating per-pixel sampling count based on the filter weight;
void D3D12RaytracingAmbientOcclusion::CalculateAdaptiveSamplingCounts()
{
    auto commandList = m_deviceResources->GetCommandList();

    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    
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
            GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess,
            m_varianceResource.gpuDescriptorReadAccess,
            m_smoothedVarianceResource.gpuDescriptorReadAccess,
            AOResources[AOResource::RayHitDistance].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            &AOResources[AOResource::FilterWeightSum],
            SceneArgs::AODenoiseValueSigma,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
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
            SceneArgs::QuarterResAO,
            m_atrousWaveletTransformFilterPerFrameInstanceId++);
    }

    // Transition the output to SRV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::FilterWeightSum].resource.Get(), before, after));
    }
};

void D3D12RaytracingAmbientOcclusion::DownsampleRaytracingOutput()
{
	auto commandList = m_deviceResources->GetCommandList();

	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutputIntermediate.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));

    ScopedTimer _prof(L"DownsampleToBackbuffer", commandList);

    // ToDo pass the filter to the kernel instead of using 3 different instances
	switch (SceneArgs::AntialiasingMode)
	{
	case DownsampleFilter::BoxFilter2x2:
		m_downsampleBoxFilter2x2Kernel.Execute(
			commandList,
			m_GBufferWidth,
			m_GBufferHeight,
			m_cbvSrvUavHeap->GetHeap(),
			m_raytracingOutputIntermediate.gpuDescriptorReadAccess,
			m_raytracingOutput.gpuDescriptorWriteAccess);
		break;
	case DownsampleFilter::GaussianFilter9Tap:
		m_downsampleGaussian9TapFilterKernel.Execute(
			commandList,
			m_GBufferWidth,
			m_GBufferHeight,
			m_cbvSrvUavHeap->GetHeap(),
			m_raytracingOutputIntermediate.gpuDescriptorReadAccess,
			m_raytracingOutput.gpuDescriptorWriteAccess);
		break;
	case DownsampleFilter::GaussianFilter25Tap:
		m_downsampleGaussian25TapFilterKernel.Execute(
			commandList,
			m_GBufferWidth,
			m_GBufferHeight,
			m_cbvSrvUavHeap->GetHeap(),
			m_raytracingOutputIntermediate.gpuDescriptorReadAccess,
			m_raytracingOutput.gpuDescriptorWriteAccess);
		break;
	}

	commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_raytracingOutputIntermediate.resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
};


void D3D12RaytracingAmbientOcclusion::RenderPass_GenerateGBuffers()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"GenerateGbuffer", commandList);

	uniform_int_distribution<UINT> seedDistribution(0, UINT_MAX);
#if AO_RANDOM_SEED_EVERY_FRAME
	m_sceneCB->seed = seedDistribution(m_generatorURNG);
#else
    if (SceneArgs::RTAORandomFrameSeed)
    {
        m_sceneCB->seed = seedDistribution(m_generatorURNG);
    }
    else
    {
        m_sceneCB->seed = 1879;
    }
#endif
	m_sceneCB->numSamplesPerSet = m_randomSampler.NumSamples();
	m_sceneCB->numSampleSets = m_randomSampler.NumSampleSets();
	m_sceneCB->numSamplesToUse = SceneArgs::AOSampleCountPerDimension * SceneArgs::AOSampleCountPerDimension;
    m_sceneCB->numPixelsPerDimPerSet = SceneArgs::AOSampleSetDistributedAcrossPixels;
#if CAMERA_JITTER

	// ToDo remove?
	static UINT seed = 0;
	static UINT counter = 0;
	switch (counter++ % 4)
	{
	case 0: m_sceneCB->cameraJitter = XMFLOAT2(-0.25f, -0.25f); break;
	case 1: m_sceneCB->cameraJitter = XMFLOAT2(0.25f, -0.25f); break;
	case 2: m_sceneCB->cameraJitter = XMFLOAT2(-0.25f, 0.25f); break;
	case 3: m_sceneCB->cameraJitter = XMFLOAT2(0.25f, 0.25f); break;
	};
#endif

    // ToDo move
    m_sceneCB->useShadowRayHitTime = SceneArgs::RTAOIsExponentialFalloffEnabled;
    // ToDo standardize RTAO RTAO_ prefix
    m_sceneCB->RTAO_maxShadowRayHitTime = SceneArgs::RTAOMaxRayHitTime;
    m_sceneCB->RTAO_approximateInterreflections = SceneArgs::RTAOApproximateInterreflections;
    m_sceneCB->RTAO_diffuseReflectanceScale = SceneArgs::RTAODiffuseReflectanceScale;
    m_sceneCB->RTAO_minimumAmbientIllumnination = SceneArgs::minimumAmbientIllumination;
    m_sceneCB->RTAO_IsExponentialFalloffEnabled = SceneArgs::RTAOIsExponentialFalloffEnabled;
    m_sceneCB->RTAO_exponentialFalloffDecayConstant = SceneArgs::RTAO_ExponentialFalloffDecayConstant;

    // Calculate a theoretical max ray distance to be used in occlusion factor computation.
    // Occlusion factor of a ray hit is computed based of its ray hit time, falloff exponent and a max ray hit time.
    // By specifying a min occlusion factor of a ray, we can skip tracing rays that would have an occlusion 
    // factor less than the cutoff to save a bit of performance (generally 1-10% perf win without visible AO result impact). 
    // Therefore we discern between true maxRayHitTime, used in TraceRay, 
    // and a theoretical one used of calculating the occlusion factor on hit.
    {
        float occclusionCutoff = SceneArgs::RTAO_ExponentialFalloffMinOcclusionCutoff;
        float lambda = SceneArgs::RTAO_ExponentialFalloffDecayConstant;

        // Invert occlusionFactor = exp(-lambda * t * t), where t is tHit/tMax of a ray.
        float t = sqrt(logf(occclusionCutoff) / -lambda);

        m_sceneCB->RTAO_maxShadowRayHitTime = t * SceneArgs::RTAOMaxRayHitTime;
        m_sceneCB->RTAO_maxTheoreticalShadowRayHitTime = SceneArgs::RTAOMaxRayHitTime;
    }

	commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
	commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

	// Copy dynamic buffers to GPU.
	{
		// ToDo copy on change
		m_sceneCB.CopyStagingToGpu(frameIndex);
		m_hemisphereSamplesGPUBuffer.CopyStagingToGpu(frameIndex);
	}

    // ToDo move this/part(AO,..) of transitions out?
	// Transition all output resources to UAV state.
	{
		D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
		D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		D3D12_RESOURCE_BARRIER barriers[] = {
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Hit].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Material].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::HitPosition].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::SurfaceNormal].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Distance].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Depth].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::SurfaceNormalRGB].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(m_VisibilityResource.resource.Get(), before, after),
            // ToDo remove
            //CD3DX12_RESOURCE_BARRIER::Transition(m_varianceResource.resource.Get(), before, after) ,
            //CD3DX12_RESOURCE_BARRIER::Transition(m_smoothedVarianceResource.resource.Get(), before, after)
		};
		commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
	}


	// Bind inputs.
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_topLevelAS.GetResource()->GetGPUVirtualAddress());
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::SampleBuffers, m_hemisphereSamplesGPUBuffer.GpuVirtualAddress(frameIndex));
	commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::MaterialBuffer, m_materialBuffer.GpuVirtualAddress());
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::EnvironmentMap, m_environmentMap.gpuDescriptorHandle);
    
	// Bind output RTs.
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResources, m_GBufferResources[0].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferDepth, m_GBufferResources[GBufferResource::Depth].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GbufferNormalRGB, m_GBufferResources[GBufferResource::SurfaceNormalRGB].gpuDescriptorWriteAccess); 
    
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::PartialDepthDerivatives, m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess);
#endif	
	// Dispatch Rays.
    DispatchRays(m_rayGenShaderTables[RayGenShaderType::GBuffer].Get());

	// Transition GBuffer resources to shader resource state.
	{
		D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;   // ToDo should it transition to NON_PIXEL_SHADER_RESOURCE for use in a CS?
		D3D12_RESOURCE_BARRIER barriers[] = {
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Hit].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Material].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::HitPosition].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::SurfaceNormal].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Distance].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Depth].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::SurfaceNormalRGB].resource.Get(), before, after),
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get(), before, after),
#endif
		};
		commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
	}

    // Calculate ray hit counts.
    {
        ScopedTimer _prof(L"CalculateCameraRayHitCount", commandList);
        CalculateRayHitCount(ReduceSumCalculations::CameraRayHits);
    }

#if !CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
    // Calculate partial derivatives.
    {
        ScopedTimer _prof(L"Calculate Partial Depth Derivatives", commandList);
        m_calculatePartialDerivativesKernel.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            m_GBufferWidth,
            m_GBufferHeight,
            m_GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess,
            m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess,
            m_calculateVarianceKernelInstancePerFrameInstanceId++);

        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
    }
#endif
    if (SceneArgs::QuarterResAO)
    {
        DownsampleGBuffer();
    }

	PIXEndEvent(commandList);
}


void D3D12RaytracingAmbientOcclusion::DownsampleGBuffer()
{
    auto commandList = m_deviceResources->GetCommandList();

    // ToDo move this/part(AO,..) of transitions out?
    // Transition all output resources to UAV state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::Hit].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::HitPosition].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::SurfaceNormal].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::Depth].resource.Get(), before, after)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    ScopedTimer _prof(L"DownsampleGBuffer", commandList);
    m_downsampleGBufferBilateralFilterKernel.Execute(
        commandList,
        m_GBufferWidth,
        m_GBufferHeight,
        m_cbvSrvUavHeap->GetHeap(),
        m_GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::HitPosition].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::Hit].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::Depth].gpuDescriptorReadAccess,
        m_GBufferLowResResources[GBufferResource::SurfaceNormal].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::HitPosition].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::Hit].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::Depth].gpuDescriptorWriteAccess);
    // Transition GBuffer resources to shader resource state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::Hit].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::HitPosition].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::SurfaceNormal].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::Depth].resource.Get(), before, after)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }
};

// Upsample quarter resources
void D3D12RaytracingAmbientOcclusion::UpsampleResourcesForRenderComposePass()
{
    auto commandList = m_deviceResources->GetCommandList();
    RWGpuResource* inputLowResValueResource = nullptr;
    RWGpuResource* outputHiResValueResource = nullptr;
    wstring passName;

    switch (SceneArgs::CompositionMode)
    {
        // ToDo Cleanup
    case CompositionType::PhongLighting:
    case CompositionType::AmbientOcclusionOnly:
    case CompositionType::AmbientOcclusionOnly_RawOneFrame:
    {
        passName = L"Upsample AO";
        AOResource::Enum AOResouceIndex;
        if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
        {
            AOResouceIndex = AOResource::Coefficient;
        }
        else
        {
            AOResouceIndex = AOResource::Smoothed;
        }
        outputHiResValueResource = &m_AOResources[AOResouceIndex];
        if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
        {
            inputLowResValueResource = &m_AOLowResResources[AOResource::Coefficient];
        }
        else if (SceneArgs::RTAODenoisingUseMultiscale)
        {
            inputLowResValueResource = &m_multiScaleDenoisingResources[0].m_value;
        }
        else
        {
            inputLowResValueResource = &m_AOLowResResources[AOResource::Smoothed];
        }
        break;
    }
    case CompositionType::AmbientOcclusionHighResSamplingPixels:
    {
        // ToDo rename all to importance map
        passName = L"Upsample AO sampling importance map";
        inputLowResValueResource = &m_AOLowResResources[AOResource::FilterWeightSum];
        outputHiResValueResource = &m_AOResources[AOResource::FilterWeightSum];
        break;
    }
    case CompositionType::RTAOHitDistance:
    {
        passName = L"Upsample AO ray hit distance";
        inputLowResValueResource = &m_AOLowResResources[AOResource::RayHitDistance];
        outputHiResValueResource = &m_AOResources[AOResource::RayHitDistance];
        break;
    }
    }

    if (inputLowResValueResource)
    {
        BilateralUpsample(
            m_GBufferWidth,
            m_GBufferHeight,
            inputLowResValueResource->gpuDescriptorReadAccess,
            m_GBufferLowResResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
            m_GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
            m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            outputHiResValueResource,
            passName.c_str());
    }
}

// ToDo standardize naming AO vs AmbientOcclusion
void D3D12RaytracingAmbientOcclusion::BilateralUpsample(
    UINT hiResWidth,
    UINT hiResHeight,
    const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValueResourceHandle,
    const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalDepthResourceHandle,
    const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalDepthResourceHandle,
    const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDepthDerivativesResourceHandle,
    RWGpuResource* outputHiResValueResource,
    LPCWCHAR passName)
{
    auto commandList = m_deviceResources->GetCommandList();
    ScopedTimer _prof(passName, commandList);

    // Transition the output resource to UAV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(outputHiResValueResource->resource.Get(), before, after));
    }

    m_upsampleBilateralFilterKernel.Execute(
        commandList,
        hiResWidth,
        hiResHeight,
        m_cbvSrvUavHeap->GetHeap(),
        inputLowResValueResourceHandle,
        inputLowResNormalDepthResourceHandle,
        inputHiResNormalDepthResourceHandle,
        inputHiResPartialDepthDerivativesResourceHandle,
        outputHiResValueResource->gpuDescriptorWriteAccess,
        0,
        SceneArgs::DownAndUpsamplingUseBilinearWeights,
        SceneArgs::DownAndUpsamplingUseDepthWeights,
        SceneArgs::DownAndUpsamplingUseNormalWeights,
        SceneArgs::DownAndUpsamplingUseDynamicDepthThreshold
    );

    // Transition the output resource to SRV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(outputHiResValueResource->resource.Get(), before, after));
    }
};


// ToDo - rename to hardshadows?
void D3D12RaytracingAmbientOcclusion::RenderPass_CalculateVisibility()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"Shadows", commandList);

    commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
	commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

	// Bind inputs.
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResourcesIn, m_GBufferResources[0].gpuDescriptorReadAccess);
	commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));
	
	// Bind output RT.
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::VisibilityResource, m_VisibilityResource.gpuDescriptorWriteAccess);

	// Bind the heaps, acceleration structure and dispatch rays. 
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_topLevelAS.GetResource()->GetGPUVirtualAddress());

    DispatchRays(m_rayGenShaderTables[RayGenShaderType::Visibility].Get());
    
	// Transition shadow resources to shader resource state.
	{
		D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_VisibilityResource.resource.Get(), before, after));
	}

	PIXEndEvent(commandList);
}

void D3D12RaytracingAmbientOcclusion::RenderPass_CalculateAmbientOcclusion()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"CalculateAmbientOcclusion", commandList);

	commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;

    // Transition AO resources to UAV state.        // ToDo check all comments
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::HitCount].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Coefficient].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::RayHitDistance].resource.Get(), before, after)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }


	// Bind inputs.
    // ToDo use [enum] instead of [0]
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResourcesIn, GBufferResources[0].gpuDescriptorReadAccess);
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::SampleBuffers, m_hemisphereSamplesGPUBuffer.GpuVirtualAddress(frameIndex));
	commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));   // ToDo let AO have its own CB.
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::FilterWeightSum, AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);

	// Bind output RT.
	// ToDo remove output and rename AOout
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOResourcesOut, AOResources[0].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AORayHitDistance, AOResources[AOResource::RayHitDistance].gpuDescriptorReadAccess);

	// Bind the heaps, acceleration structure and dispatch rays. 
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_topLevelAS.GetResource()->GetGPUVirtualAddress());

    DispatchRays(m_rayGenShaderTables[RayGenShaderType::AOFullRes].Get(), m_raytracingWidth, m_raytracingHeight);

    // ToDo Remove
    //DispatchRays(m_rayGenShaderTables[SceneArgs::QuarterResAO ? RayGenShaderType::AOQuarterRes : RayGenShaderType::AOFullRes].Get(),
    //    &m_gpuTimers[GpuTimers::Raytracing_AO], m_raytracingWidth, m_raytracingHeight);

	// Transition AO resources to shader resource state.
	{
		D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
		D3D12_RESOURCE_BARRIER barriers[] = {
			CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::HitCount].resource.Get(), before, after),
			CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Coefficient].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::RayHitDistance].resource.Get(), before, after)
		};
		commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
	}

    // Calculate AO ray hit count.
    {
        ScopedTimer _prof(L"CalculateAORayHitCount", commandList);
        CalculateRayHitCount(ReduceSumCalculations::AORayHits);
    }

	PIXEndEvent(commandList);
}

void D3D12RaytracingAmbientOcclusion::RenderPass_BlurAmbientOcclusion()
{
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"BlurAO", commandList);
    
    m_csAoBlurCB->kRcpBufferDim.x = 1.0f / m_raytracingWidth;
    m_csAoBlurCB->kRcpBufferDim.y = 1.0f / m_raytracingHeight;
    m_csAoBlurCB->kDistanceTolerance = powf(10.0f, SceneArgs::g_DistanceTolerance);
	m_csAoBlurCB.CopyStagingToGpu(frameIndex);

	// Set common pipeline state
	using namespace ComputeShader::RootSignature::AoBlurCS;

    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;

	commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
	commandList->SetComputeRootSignature(m_computeRootSigs[CSType::AoBlurCS].Get());
	commandList->SetPipelineState(m_computePSOs[SceneArgs::QuarterResAO ? CSType::AoBlurAndUpsampleCS : CSType::AoBlurCS].Get());
	commandList->SetComputeRootDescriptorTable(Slot::Normal, GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess);
    commandList->SetComputeRootDescriptorTable(Slot::Distance, GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess);
	commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_csAoBlurCB.GpuVirtualAddress(frameIndex));
	XMUINT2 groupCount;
    groupCount.x = CeilDivide(m_raytracingWidth, AoBlurCS::ThreadGroup::Width);
    groupCount.y = CeilDivide(m_raytracingHeight, AoBlurCS::ThreadGroup::Height);

    // Begin timing actual work

    {
	    D3D12_RESOURCE_BARRIER barriers = CD3DX12_RESOURCE_BARRIER::Transition(m_AOResources[AOResource::Smoothed].resource.Get(),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	    commandList->ResourceBarrier(1, &barriers);
    }

    // Pass 1:  Blurs once to "Smoothed" buffer
	commandList->SetComputeRootDescriptorTable(Slot::Output, m_AOResources[AOResource::Smoothed].gpuDescriptorWriteAccess);
	commandList->SetComputeRootDescriptorTable(Slot::InputAO, m_AOResources[AOResource::Coefficient].gpuDescriptorReadAccess);
	commandList->Dispatch(groupCount.x, groupCount.y, 1);

#if TWO_STAGE_AO_BLUR
	D3D12_RESOURCE_BARRIER barriers[] =
    {
		CD3DX12_RESOURCE_BARRIER::Transition(m_AOResources[AOResource::Smoothed].resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
		CD3DX12_RESOURCE_BARRIER::Transition(m_AOResources[AOResource::Coefficient].resource.Get(),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
	};
	commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);

    // Pass 2:  Blurs a second time back to "Coefficient" buffer
	commandList->SetComputeRootDescriptorTable(Slot::Output, m_AOResources[AOResource::Coefficient].gpuDescriptorWriteAccess);
	commandList->SetComputeRootDescriptorTable(Slot::InputAO, m_AOResources[AOResource::Smoothed].gpuDescriptorReadAccess);
	commandList->Dispatch(groupCount.x, groupCount.y, 1);

    {
	    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_AOResources[AOResource::Coefficient].resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	    commandList->ResourceBarrier(1, &barrier);
    }
#else
    {
	    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_AOResources[AOResource::Smoothed].resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
	    commandList->ResourceBarrier(1, &barrier);
    }
#endif
}


// Composite results from multiple passed into a final image.
void D3D12RaytracingAmbientOcclusion::RenderPass_ComposeRenderPassesCS(D3D12_GPU_DESCRIPTOR_HANDLE AOSRV)
{
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"ComposeRenderPassesCS", commandList);
    
	// Update constant buffer.
	{
		m_csComposeRenderPassesCB->rtDimensions = XMUINT2(m_GBufferWidth, m_GBufferHeight);
        m_csComposeRenderPassesCB->enableAO = SceneArgs::AOEnabled;
        m_csComposeRenderPassesCB->compositionType = static_cast<CompositionType>(static_cast<UINT>(SceneArgs::CompositionMode));

        // ToDo use a unique CB for compose passes?
        m_csComposeRenderPassesCB->RTAO_UseAdaptiveSampling = SceneArgs::RTAOAdaptiveSampling;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMaxWeightSum = SceneArgs::RTAOAdaptiveSamplingMaxFilterWeight;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMinMaxSampling = SceneArgs::RTAOAdaptiveSamplingMinMaxSampling;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingScaleExponent = SceneArgs::RTAOAdaptiveSamplingScaleExponent;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMinSamples = SceneArgs::RTAOAdaptiveSamplingMinSamples;
        m_csComposeRenderPassesCB->RTAO_MaxRayHitDistance = SceneArgs::RTAOMaxRayHitTime;

        m_csComposeRenderPassesCB->RTAO_MaxSPP = SceneArgs::AOSampleCountPerDimension * SceneArgs::AOSampleCountPerDimension;

        m_csComposeRenderPassesCB.CopyStagingToGpu(frameIndex);
	}


	// Set pipeline state.
	{
		using namespace ComputeShader::RootSignature::ComposeRenderPassesCS;

		commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
		commandList->SetComputeRootSignature(m_computeRootSigs[CSType::ComposeRenderPassesCS].Get());
		commandList->SetPipelineState(m_computePSOs[CSType::ComposeRenderPassesCS].Get());

		// Bind outputs.
		commandList->SetComputeRootDescriptorTable(Slot::Output, m_raytracingOutputIntermediate.gpuDescriptorWriteAccess);
		
		// Bind inputs.
		commandList->SetComputeRootDescriptorTable(Slot::GBufferResources, m_GBufferResources[0].gpuDescriptorReadAccess);
#if TWO_STAGE_AO_BLUR && !ATROUS_DENOISER
		commandList->SetComputeRootDescriptorTable(Slot::AO, m_AOResources[AOResource::Coefficient].gpuDescriptorReadAccess);
#else
        commandList->SetComputeRootDescriptorTable(Slot::AO, AOSRV);
#endif
		commandList->SetComputeRootDescriptorTable(Slot::Visibility, m_VisibilityResource.gpuDescriptorReadAccess);
		commandList->SetComputeRootShaderResourceView(Slot::MaterialBuffer, m_materialBuffer.GpuVirtualAddress());
		commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_csComposeRenderPassesCB.GpuVirtualAddress(frameIndex));
        commandList->SetComputeRootDescriptorTable(Slot::FilterWeightSum, m_AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);
        commandList->SetComputeRootDescriptorTable(Slot::AORayHitDistance, m_AOResources[AOResource::RayHitDistance].gpuDescriptorReadAccess);
	}

	// Dispatch.
	XMUINT2 groupSize(CeilDivide(m_GBufferWidth, ComposeRenderPassesCS::ThreadGroup::Width), CeilDivide(m_GBufferHeight, ComposeRenderPassesCS::ThreadGroup::Height));

	commandList->Dispatch(groupSize.x, groupSize.y, 1);
}

// Copy the raytracing output to the backbuffer.
void D3D12RaytracingAmbientOcclusion::CopyRaytracingOutputToBackbuffer(D3D12_RESOURCE_STATES outRenderTargetState)
{
    auto commandList = m_deviceResources->GetCommandList();
    auto renderTarget = m_deviceResources->GetRenderTarget();

    ID3D12Resource* raytracingOutput = nullptr;
    if (m_GBufferWidth == m_width && m_GBufferHeight == m_height)
    {
        raytracingOutput = m_raytracingOutputIntermediate.resource.Get();
    }
    else
    {
        raytracingOutput = m_raytracingOutput.resource.Get();
    }

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
			0.001f* NumMPixelsPerSecond(GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_GBuffer), m_GBufferWidth, m_GBufferHeight)  << " GigaRay/s\n";
        // ToDo use profiler from MiniEngine
		float numAOGigaRays = 1e-6f * m_numRayGeometryHits[ReduceSumCalculations::CameraRayHits] * (SceneArgs::QuarterResAO ? 0.25f : 1) * m_sppAO / GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_AO);
		wLabel << fixed << L" AORay DispatchRays: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_AO) << L"ms  ~" <<	numAOGigaRays << " GigaRay/s\n";
        wLabel << fixed << L" - AORay Adaptive Sampling ImportanceMap: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_FilterWeightSum) << L"ms  ~" << numAOGigaRays << " GigaRay/s\n";
        wLabel << fixed << L" AO Denoising: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Denoising) << L"ms\n";
        wLabel << fixed << L" - AO Blurring: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_BlurAO) << L"ms\n";
        wLabel << fixed << L" - Variance: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_Variance) << L"ms\n";
        wLabel << fixed << L" - Var Smoothing: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_VarianceSmoothing) << L"ms\n";
        wLabel << fixed << L" - AO downsample: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::DownsampleGBuffer) << L"ms\n";
        wLabel << fixed << L" - AO upsample: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::UpsampleAOBilateral) << L"ms\n";

		float numVisibilityRays = 1e-6f * m_numRayGeometryHits[ReduceSumCalculations::CameraRayHits] / GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_Visibility);
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
			   << m_numRayGeometryHits[ReduceSumCalculations::CameraRayHits] << "/"
			   << ((m_GBufferWidth * m_GBufferHeight) > 0 ? (100.f * m_numRayGeometryHits[ReduceSumCalculations::CameraRayHits]) / (m_GBufferWidth * m_GBufferHeight) : 0) << "%%/"
			   << 1000.0f * m_gpuTimers[GpuTimers::ReduceSum].GetAverageMS(ReduceSumCalculations::CameraRayHits) << L"us \n";
		wLabel << fixed << L" AORayGeometryHits: #/%%/time "
			   << m_numRayGeometryHits[ReduceSumCalculations::AORayHits] << "/"
            // ToDo fix up for raytracing at quarter res
			   << ((m_numRayGeometryHits[ReduceSumCalculations::CameraRayHits] * m_sppAO) > 0 ?
				   (100.0f * m_numRayGeometryHits[ReduceSumCalculations::AORayHits]) / (m_numRayGeometryHits[ReduceSumCalculations::CameraRayHits] * m_sppAO) : 0) << "%%/"
			   << 1000.0f * m_gpuTimers[GpuTimers::ReduceSum].GetAverageMS(ReduceSumCalculations::AORayHits) << L"us \n";
    */
        labels.push_back(wLabel.str());
    }

    // Parameters.
    labels.push_back(L"\n");
    {
        wstringstream wLabel;
        wLabel << L"Scene:" << L"\n";
        wLabel << L" " << L"AS update mode: " << SceneArgs::ASUpdateMode << L"\n";
        wLabel.precision(3);
        wLabel << L" " << L"AS memory footprint: " << static_cast<double>(m_ASmemoryFootprint)/(1024*1024) << L"MB\n";
       // wLabel << L" " << L" # triangles per geometry: " << m_numTrianglesInTheScene << L"\n";
        //wLabel << L" " << L" # geometries per BLAS: " << SceneArgs::NumGeometriesPerBLAS << L"\n";
       // wLabel << L" " << L" # Sphere BLAS: " << SceneArgs::NumSphereBLAS << L"\n";	// ToDo fix
		wLabel << L" " << L" # total triangles: " << m_numTrianglesInTheScene << L"\n";// SceneArgs::NumSphereBLAS * SceneArgs::NumGeometriesPerBLAS* m_numTriangles[SceneArgs::SceneType] << L"\n";
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
        EngineTuning::Display(&wLabel);
        labels.push_back(wLabel.str());
    }

	// Sampling info:
	{
		wstringstream wLabel;
		wLabel << L"\n";
		wLabel << L"Num samples: " << m_randomSampler.NumSamples() << L"\n";
		wLabel << L"Sample set: " << m_csHemisphereVisualizationCB->sampleSetBase / m_randomSampler.NumSamples() << " / " << m_randomSampler.NumSampleSets() << L"\n";
		
		labels.push_back(wLabel.str());
	}
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

	switch (SceneArgs::AntialiasingMode)
	{
	case DownsampleFilter::None:
        m_GBufferWidth = m_width;
		m_GBufferHeight = m_height;
		break;
	case DownsampleFilter::BoxFilter2x2:
		m_GBufferWidth = c_SupersamplingScale * m_width;
		m_GBufferHeight = c_SupersamplingScale * m_height;
		break;
	case DownsampleFilter::GaussianFilter9Tap:
	case DownsampleFilter::GaussianFilter25Tap:
		m_GBufferWidth = c_SupersamplingScale * m_width + 1;
		m_GBufferHeight = c_SupersamplingScale * m_height + 1;
		break;
	}

    if (SceneArgs::QuarterResAO)
    {
        // Handle odd resolution.
        m_raytracingWidth = CeilDivide(m_GBufferWidth, 2);
        m_raytracingHeight = CeilDivide(m_GBufferHeight, 2);
    }
    else
    {
        m_raytracingWidth = m_GBufferWidth;
        m_raytracingHeight = m_GBufferHeight;
    }

    // Create an output 2D texture to store the raytracing result to.
    CreateRaytracingOutputResource();

	CreateGBufferResources();
	m_reduceSumKernel.CreateInputResourceSizeDependentResources(
		device,
		m_cbvSrvUavHeap.get(), 
		FrameCount, 
		m_GBufferWidth,
		m_GBufferHeight,
		ReduceSumCalculations::Count);
    m_atrousWaveletTransformFilter.CreateInputResourceSizeDependentResources(device, m_cbvSrvUavHeap.get(), m_raytracingWidth, m_raytracingHeight);
	m_downsampleBoxFilter2x2Kernel.CreateInputResourceSizeDependentResources(device, m_GBufferWidth, m_GBufferHeight);
	m_downsampleGaussian9TapFilterKernel.CreateInputResourceSizeDependentResources(device, m_GBufferWidth, m_GBufferHeight);
	m_downsampleGaussian25TapFilterKernel.CreateInputResourceSizeDependentResources(device, m_GBufferWidth, m_GBufferHeight);
    
    // SSAO
    {
        m_SSAO.OnSizeChanged(m_GBufferWidth, m_GBufferHeight);
        ID3D12Resource* SSAOoutputResource = m_SSAO.GetSSAOOutputResource();
        D3D12_CPU_DESCRIPTOR_HANDLE dummyHandle;
        CreateTextureSRV(device, SSAOoutputResource, m_cbvSrvUavHeap.get(), &m_SSAOsrvDescriptorHeapIndex, &dummyHandle, &SSAOgpuDescriptorReadAccess);
    }

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

    m_raytracingGlobalRootSignature.Reset();
    ResetComPtrArray(&m_raytracingLocalRootSignature);

    m_dxrDevice.Reset();
    m_dxrStateObject.Reset();

    m_raytracingGlobalRootSignature.Reset();
    ResetComPtrArray(&m_raytracingLocalRootSignature);

	m_cbvSrvUavHeap.release();

    m_csHemisphereVisualizationCB.Release();

    // ToDo
    for (auto& bottomLevelAS : m_vBottomLevelAS)
    {
        bottomLevelAS.ReleaseD3DResources();
    }
    m_topLevelAS.ReleaseD3DResources();

    m_raytracingOutput.resource.Reset();
	ResetComPtrArray(&m_rayGenShaderTables);
    m_missShaderTable.Reset();
    m_hitGroupShaderTable.Reset();
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

void D3D12RaytracingAmbientOcclusion::RenderRNGVisualizations()
{
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
    commandList->Dispatch(rngWindowSize.x, rngWindowSize.y, 1);
}


void D3D12RaytracingAmbientOcclusion::RenderPass_TemporalCacheReverseProjection()
{
    auto commandList = m_deviceResources->GetCommandList();

    ScopedTimer _prof(L"Temporal Cache Reverse Reprojection", commandList);

    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* valueResource = SceneArgs::RTAO_TemporalCache_CacheRawAOValue ? &AOResources[AOResource::Coefficient] : nullptr;

    UINT TC_readID = m_temporalCacheReadResourceIndex;
    UINT TC_writeID = (m_temporalCacheReadResourceIndex + 1) % 2;

    // ToDo remove
    if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
    {
        m_temporalCacheFrameAge = 0;
    }

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
    XMMATRIX cameraTranslation = XMMatrixTranslationFromVector(m_camera.Eye() - m_prevFrameCamera.Eye());

    XMMATRIX invView = XMMatrixInverse(nullptr, view);
    XMMATRIX invProj = XMMatrixInverse(nullptr, proj);

    XMMATRIX viewProj = view * proj;
    XMMATRIX prevViewProj = prevView * prevProj;
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);
    
#if 1
   // XMMATRIX reverseProjectionTransform = invViewProj * cameraTranslation * prevView * prevProj;
    XMMATRIX reverseProjectionTransform = invViewProj * cameraTranslation * prevView * prevProj;
#else
    XMMATRIX reverseProjectionTransform = cameraTranslation * prevView * prevProj;
#endif

#if PRINT_OUT_TC_MATRICES
    auto MatrixToStr = [&](wstring* out, XMMATRIX& matrix, const WCHAR* name)
    {
        XMFLOAT4X4 fMatrix;
        XMStoreFloat4x4(&fMatrix, matrix);

        wstringstream wstream;
        wstream << L"Matrix:" << name << L"\n";
        for (UINT r = 0; r < 4; r++)
        {
            wstream << L" { ";
            for (UINT c = 0; c < 4; c++)
            {
                wstream << fMatrix(r, c);
                wstream << (c < 3 ? L", " : L"");
            }
            wstream << L" }\n";
        }
        *out += wstream.str();
    };

    wstring dbgText;
    dbgText = L"\n=============================\n";
    MatrixToStr(&dbgText, reverseProjectionTransform, L"ReverseProjectionTransform");
    MatrixToStr(&dbgText, invViewProj, L"invViewProj");
    MatrixToStr(&dbgText, viewProj, L"viewProj");
    dbgText += L"=============================\n";
    OutputDebugStringW(dbgText.c_str());
#endif

    // Transition output resource to UAV state.        
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_temporalCache[TC_writeID][TemporalCache::AO].resource.Get(), before, after)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    m_temporalCacheReverseReprojectKernel.Execute(
        commandList,
        m_raytracingWidth,
        m_raytracingHeight,
        m_cbvSrvUavHeap->GetHeap(),
        valueResource->gpuDescriptorReadAccess,
        GBufferResources[GBufferResource::Depth].gpuDescriptorReadAccess,
        GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
        m_temporalCache[TC_readID][TemporalCache::AO].gpuDescriptorReadAccess,
        m_temporalCache[0][TemporalCache::Depth].gpuDescriptorReadAccess,      // ToDo dedupe depth from the array
        m_temporalCache[0][TemporalCache::Normal].gpuDescriptorReadAccess,     
        m_temporalCache[TC_writeID][TemporalCache::AO].gpuDescriptorWriteAccess,
        m_temporalCacheFrameAge,
        SceneArgs::RTAO_TemporalCache_MinSmoothingFactor,
        invView,
        invProj,
        reverseProjectionTransform,
        m_camera.ZMin,
        m_camera.ZMax,
        SceneArgs::RTAO_TemporalCache_DepthTolerance,
        SceneArgs::RTAO_TemporalCache_UseDepthWeights,
        SceneArgs::RTAO_TemporalCache_UseNormalWeights);

#if 0
    // Transition output resource to SRV state.        
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_temporalCache[TC_writeID][TemporalCache::AO].resource.Get(), before, after));
    }
#else
    // ToDo use the out resource directly.
    CopyTextureRegion(
        commandList,
        m_temporalCache[TC_writeID][TemporalCache::AO].resource.Get(),
        valueResource->resource.Get(),
        &CD3DX12_BOX(0, 0, m_raytracingWidth, m_raytracingHeight),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
#endif

    // Cache current frame's depth buffer.
    CopyTextureRegion(
        commandList,
        GBufferResources[GBufferResource::Depth].resource.Get(),
        m_temporalCache[0][TemporalCache::Depth].resource.Get(),
        &CD3DX12_BOX(0, 0, m_raytracingWidth, m_raytracingHeight),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    // Cache current frame's normal buffer.
    CopyTextureRegion(
        commandList,
        GBufferResources[GBufferResource::SurfaceNormal].resource.Get(),
        m_temporalCache[0][TemporalCache::Normal].resource.Get(),
        &CD3DX12_BOX(0, 0, m_raytracingWidth, m_raytracingHeight),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

// Render the scene.
void D3D12RaytracingAmbientOcclusion::OnRender()
{
    if (!m_deviceResources->IsWindowVisible() || m_cameraChangedIndex <= 0)
    {
        return;
    }

    auto commandList = m_deviceResources->GetCommandList();

    // ToDo cleanup?
    m_atrousWaveletTransformFilterPerFrameInstanceId = 0;
    m_calculateVarianceKernelInstancePerFrameInstanceId = 0;
    m_gaussianSmoothingKernelPerFrameInstanceId = 0;

    // Begin frame.
    m_deviceResources->Prepare();

    EngineProfiling::BeginFrame(commandList);
    {
        // ToDo fix RTAO not being a child of Frame marker
        ScopedTimer _prof(L"Frame", commandList);

#if RUNTIME_AS_UPDATES
        // Update acceleration structures.
        if (m_isASrebuildRequested && SceneArgs::EnableGeometryAndASBuildsAndUpdates)
        {
            UpdateAccelerationStructures(m_isASrebuildRequested);
            m_isASrebuildRequested = false;
        }
#endif

        // Render.
        RenderPass_GenerateGBuffers();

        // AO. 
        if (SceneArgs::AOMode == SceneArgs::AOType::RTAO)
        {
            ScopedTimer _prof(L"RTAO", commandList);

            if (SceneArgs::RTAOAdaptiveSampling)
            {
                // ToDo move to within AO
                CalculateAdaptiveSamplingCounts();
            }
            RenderPass_CalculateAmbientOcclusion();
            
            RenderPass_TemporalCacheReverseProjection();

#if BLUR_AO
#if ATROUS_DENOISER

#if 0
            ApplyAtrousWaveletTransformFilter();
#else
            if (SceneArgs::RTAODenoisingUseMultiscale)
            {
                ApplyMultiScaleAtrousWaveletTransformFilter();
            }
            else
            {
                ApplyAtrousWaveletTransformFilter();
            }
#endif
#else
            ToDo - fix up resources
                RenderPass_BlurAmbientOcclusion();
#endif
#endif
            if (SceneArgs::QuarterResAO)
            {
                UpsampleResourcesForRenderComposePass();
            }
            else // ToDo move this to ApplyAtrousWaveletTransformFilter?
            {
                // Transition AO Smoothed resource to SRV.
                //commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_AOResources[AOResource::Smoothed].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
            }
        }
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

        RenderPass_CalculateVisibility();

        D3D12_GPU_DESCRIPTOR_HANDLE AOSRV = SceneArgs::AOMode == SceneArgs::AOType::RTAO ? m_AOResources[AOResource::Smoothed].gpuDescriptorReadAccess : SSAOgpuDescriptorReadAccess;

        // ToDo cleanup
        if (SceneArgs::AOMode == SceneArgs::AOType::RTAO && SceneArgs::RTAODenoisingUseMultiscale && !SceneArgs::QuarterResAO)
        {
            AOSRV = m_multiScaleDenoisingResources[0].m_value.gpuDescriptorReadAccess;
        }

        if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
        {
            AOSRV = m_AOResources[AOResource::Coefficient].gpuDescriptorReadAccess;
        }
        //UINT TC_writeID = (m_temporalCacheReadResourceIndex + 1) % 2;
        //AOSRV = m_temporalCache[TC_writeID][TemporalCache::AO].gpuDescriptorReadAccess;

        RenderPass_ComposeRenderPassesCS(AOSRV);

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

 
    // End frame.
    EngineProfiling::EndFrame(commandList);

    m_deviceResources->ExecuteCommandList();

    // UI overlay.
    if (m_enableUI)
    {
        m_uiLayer->Render(m_deviceResources->GetCurrentFrameIndex());
    }
    
    m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT);

    // ToDo move
    m_temporalCacheFrameAge = min(m_temporalCacheFrameAge + 1, 999999u);
    m_temporalCacheReadResourceIndex = (m_temporalCacheReadResourceIndex + 1) % 2;
}

// Compute the average frames per second and million rays per second.
void D3D12RaytracingAmbientOcclusion::CalculateFrameStats()
{
    static int frameCnt = 0;
    static double prevTime = 0.0f;
    double totalTime = m_timer.GetTotalSeconds();

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
