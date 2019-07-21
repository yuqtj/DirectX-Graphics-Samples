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
#include "RTAO\RTAO.h"

using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;
using namespace GameCore;

// ToDo tighten shader visibility in Root Sigs - CS + DXR

// Singleton instance.
static D3D12RaytracingAmbientOcclusion* global_pSample;
UINT D3D12RaytracingAmbientOcclusion::s_numInstances = 0;


HWND g_hWnd = 0;
UIParameters g_UIparameters;    // ToDo move

// Shader entry points.
const wchar_t* D3D12RaytracingAmbientOcclusion::c_rayGenShaderNames[] = 
{
	// ToDo reorder
    // ToDo rename visiblity to shadow? standardize naming
	L"MyRayGenShader_GBuffer", L"MyRayGenShader_Visibility", L"MyRayGenShader_ShadowMap"
};
const wchar_t* D3D12RaytracingAmbientOcclusion::c_closestHitShaderNames[] =
{
     L"MyClosestHitShader_GBuffer", L"MyClosestHitShader_ShadowRay"
};
const wchar_t* D3D12RaytracingAmbientOcclusion::c_missShaderNames[] =
{
    L"MyMissShader_GBuffer", L"MyMissShader_ShadowRay",
};
// Hit groups.
const wchar_t* D3D12RaytracingAmbientOcclusion::c_hitGroupNames_TriangleGeometry[] = 
{ 
    L"MyHitGroup_Triangle_GBuffer", L"MyHitGroup_Triangle_ShadowRay",
};
namespace SceneArgs
{
    void OnGeometryReinitializationNeeded(void* args)
    {
        global_pSample->RequestGeometryInitialization(true);
        global_pSample->RequestASInitialization(true);
    }

    void OnASReinitializationNeeded(void* args)
    {
        global_pSample->RequestASInitialization(true);
    }
    function<void(void*)> OnGeometryChange = OnGeometryReinitializationNeeded;
    function<void(void*)> OnASChange = OnASReinitializationNeeded;
	
	void OnSceneChange(void*)
	{
		global_pSample->RequestSceneInitialization();
	}

	void OnRecreateRaytracingResources(void*)
	{
		global_pSample->RequestRecreateRaytracingResources();
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

    const WCHAR* CompositionModes[CompositionType::Count] = { 
        L"Phong Lighting", 
        L"Denoised Ambient Occlusion",
        L"Temporally Supersampled Ambient Occlusion",
        L"Raw one-frame Ambient Occlusion", 
        L"Render/AO Sampling Importance Map",
        L"AO and Disocclusion Map",
        L"Render/AO Minimum Hit Distance", 
        L"Normal Map", 
        L"Depth Buffer", 
        L"Diffuse",
        L"Disocclusion Map" };
    EnumVar CompositionMode(L"Render/Render composition mode", CompositionType::AmbientOcclusionOnly_TemporallySupersampled, CompositionType::Count, CompositionModes);


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

    // Default ambient intensity for hitPositions that don't have a calculated Ambient coefficient.
    // Calculating AO just for a single hitPosition per pixel can cause visible visual differences
    // in bounces off surfaces that have non-zero Albedo, such as reflection on car paint at sharp angles. 
    // With default Ambient coefficient added to every hit along the ray, this visual difference is surpressed.
    NumVar DefaultAmbientIntensity(L"Render/PathTracing/Default ambient intensity", 0.4f, 0, 1, 0.01f);  

    IntVar MaxRadianceRayRecursionDepth(L"Render/PathTracing/Max Radiance Ray recursion depth", 2, 1, MAX_RAY_RECURSION_DEPTH, 1);   // ToDo Replace with 3/4 depth as it adds visible differences on spaceship/car
    IntVar MaxShadowRayRecursionDepth(L"Render/PathTracing/Max Shadow Ray recursion depth", 3, 1, MAX_RAY_RECURSION_DEPTH, 1);
    
    BoolVar UseShadowMap(L"Render/PathTracing/Use shadow map", false);        // ToDO use enumeration

    // Avoid tracing rays where they have close to zero visual impact.
    // todo test perf gain or remove.
    // ToDo remove RTAO from name
    NumVar RTAO_minimumFrBounceCoefficient(L"Render/PathTracing/Minimum BRDF bounce contribution coefficient", 0.03f, 0, 1.01f, 0.01f);        // Minimum BRDF coefficient to cast a ray for.
    NumVar RTAO_minimumFtBounceCoefficient(L"Render/PathTracing/Minimum BTDF bounce contribution coefficient", 0.00f, 0, 1.01f, 0.01f);        // Minimum BTDF coefficient to cast a ray for.

    // ToDo standardize capitalization
    // ToDo naming down/ up
    const WCHAR* DownsamplingBilateralFilters[GpuKernels::DownsampleValueNormalDepthBilateralFilter::Count] = { L"Point Sampling", L"Depth Weighted", L"Depth Normal Weighted" };
    EnumVar DownsamplingBilateralFilter(L"Render/AO/RTAO/Down/Upsampling/Downsampled Value Filter", GpuKernels::DownsampleValueNormalDepthBilateralFilter::FilterDepthNormalWeighted2x2, GpuKernels::DownsampleValueNormalDepthBilateralFilter::Count, DownsamplingBilateralFilters, OnRecreateRaytracingResources, nullptr);
    BoolVar DownAndUpsamplingUseBilinearWeights(L"Render/AO/RTAO/Down/Upsampling/Bilinear weighted", true);
    BoolVar DownAndUpsamplingUseDepthWeights(L"Render/AO/RTAO/Down/Upsampling/Depth weighted", true);
    BoolVar DownAndUpsamplingUseNormalWeights(L"Render/AO/RTAO/Down/Upsampling/Normal weighted", true);
    BoolVar DownAndUpsamplingUseDynamicDepthThreshold(L"Render/AO/RTAO/Down/Upsampling/Dynamic depth threshold", true);        // ToDO rename to adaptive


    NumVar CameraRotationDuration(L"Scene2/Camera rotation time", 48.f, 1.f, 120.f, 1.f);

    BoolVar QuarterResAO(L"Render/AO/RTAO/Quarter res", false, OnRecreateRaytracingResources, nullptr);

    // Temporal Cache.
    // ToDo rename cache to accumulation/supersampling?
    BoolVar RTAO_UseTemporalCache(L"Render/AO/RTAO/Temporal Cache/Enabled", true);
    BoolVar RTAO_TemporalCache_CacheRawAOValue(L"Render/AO/RTAO/Temporal Cache/Cache Raw AO Value", true);
    NumVar RTAO_TemporalCache_MinSmoothingFactor(L"Render/AO/RTAO/Temporal Cache/Min Smoothing Factor", 0.03f, 0, 1.f, 0.01f);
    NumVar RTAO_TemporalCache_DepthTolerance(L"Render/AO/RTAO/Temporal Cache/Depth tolerance [%%]", 0.05f, 0, 1.f, 0.001f);
    BoolVar RTAO_TemporalCache_UseWorldSpaceDistance(L"Render/AO/RTAO/Temporal Cache/Use world space distance", false);    // ToDo remove
    BoolVar RTAO_TemporalCache_UseDepthWeights(L"Render/AO/RTAO/Temporal Cache/Use depth weights", true);    // ToDo remove
    BoolVar RTAO_TemporalCache_UseNormalWeights(L"Render/AO/RTAO/Temporal Cache/Use normal weights", true);
    BoolVar RTAO_TemporalCache_ForceUseMinSmoothingFactor(L"Render/AO/RTAO/Temporal Cache/Force min smoothing factor", false);


    // ToDo remove
    IntVar RTAO_KernelStepShift0(L"Render/AO/RTAO/Kernel Step Shifts/0", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift1(L"Render/AO/RTAO/Kernel Step Shifts/1", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift2(L"Render/AO/RTAO/Kernel Step Shifts/2", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift3(L"Render/AO/RTAO/Kernel Step Shifts/3", 0, 0, 10, 1);
    IntVar RTAO_KernelStepShift4(L"Render/AO/RTAO/Kernel Step Shifts/4", 0, 0, 10, 1);

    const WCHAR* VarianceBilateralFilters[GpuKernels::CalculateVariance::FilterType::Count] = { L"Square Bilateral", L"Separable Bilateral", L"Separable" };
    EnumVar VarianceBilateralFilter(L"Render/GpuKernels/CalculateVariance/Filter", GpuKernels::CalculateVariance::Separable, GpuKernels::CalculateVariance::Count, VarianceBilateralFilters);

    IntVar VarianceBilateralFilterKernelWidth(L"Render/GpuKernels/CalculateVariance/Kernel width", 7, 3, 11, 2);    // ToDo find lowest good enough width


    // ToDo rename to temporal supersampling
    // ToDo address: Clamping causes rejection of samples in low density areas - such as on ground plane at the end of max ray distance from other objects.
    BoolVar RTAO_TemporalCache_ClampCachedValues_UseClamping(L"Render/AO/RTAO/Temporal Cache/Clamping/Enabled", true);
    NumVar RTAO_TemporalCache_ClampCachedValues_StdDevGamma(L"Render/AO/RTAO/Temporal Cache/Clamping/Std.dev gamma", 1.0f, 0.1f, 20.f, 0.1f);
    NumVar RTAO_TemporalCache_ClampCachedValues_MinStdDevTolerance(L"Render/AO/RTAO/Temporal Cache/Clamping/Minimum std.dev", 0.04f, 0.0f, 1.f, 0.01f);   // ToDo finetune
    NumVar RTAO_TemporalCache_ClampCachedValues_AbsoluteDepthTolerance(L"Render/AO/RTAO/Temporal Cache/Depth threshold/Absolute depth tolerance", 1.0f, 0.0f, 100.f, 1.f);
    NumVar RTAO_TemporalCache_ClampCachedValues_DepthBasedDepthTolerance(L"Render/AO/RTAO/Temporal Cache/Depth threshold/Depth based depth tolerance", 1.0f, 0.0f, 100.f, 1.f);

    // Todo revise comment
    // Setting it lower than 0.9 makes cache values to swim...
    NumVar RTAO_TemporalCache_ClampCachedValues_DepthSigma(L"Render/AO/RTAO/Temporal Cache/Depth threshold/Depth sigma", 1.0f, 0.0f, 10.f, 0.01f);   

    const WCHAR* FloatingPointFormatsRGB[TextureResourceFormatRGB::Count] = { L"R32G32B32A32_FLOAT", L"R16G16B16A16_FLOAT", L"R11G11B10_FLOAT" };
    EnumVar RTAO_TemporalCache_NormalDepthResourceFormat(L"Render/Texture Formats/AO/RTAO/Temporal Cache/Encoded Normal (RG) Depth (B) resource", TextureResourceFormatRGB::R11G11B10_FLOAT, TextureResourceFormatRGB::Count, FloatingPointFormatsRGB, OnRecreateRaytracingResources);

    
    const WCHAR* FloatingPointFormatsRG[TextureResourceFormatRG::Count] = { L"R32G32_FLOAT", L"R16G16_FLOAT", L"R8G8_UNORM" };
    // ToDo  ddx needs to be in normalized to use UNORM.
    EnumVar RTAO_PartialDepthDerivativesResourceFormat(L"Render/Texture Formats/PartialDepthDerivatives", TextureResourceFormatRG::R16G16_FLOAT, TextureResourceFormatRG::Count, FloatingPointFormatsRG, OnRecreateRaytracingResources);
    EnumVar RTAO_MotionVectorResourceFormat(L"Render/Texture Formats/AO/RTAO/Temporal Supersampling/Motion Vector", TextureResourceFormatRG::R16G16_FLOAT, TextureResourceFormatRG::Count, FloatingPointFormatsRG, OnRecreateRaytracingResources);
    
    BoolVar TAO_LazyRender(L"TAO/Lazy render", false);
    IntVar RTAO_LazyRenderNumFrames(L"TAO/Lazy render frames", 1, 0, 20, 1);
    BoolVar RTAOUseNormalMaps(L"Render/AO/RTAO/Normal maps", false);


    // ToDo add Weights On/OFF - 
    // RTAO Denoising
    IntVar RTAOVarianceFilterKernelWidth(L"Render/AO/RTAO/Denoising/Variance filter/Kernel width", 7, 3, 11, 2);    // ToDo find lowest good enough width
    BoolVar UseSpatialVariance(L"Render/AO/RTAO/Denoising/Use spatial variance", true);
    BoolVar ApproximateSpatialVariance(L"Render/AO/RTAO/Denoising/Approximate spatial variance", false);
    BoolVar RTAODenoisingUseMultiscale(L"Render/AO/RTAO/Denoising/Multi-scale/Enabled", false);
    IntVar RTAODenoisingMultiscaleLevels(L"Render/AO/RTAO/Denoising/Multi-scale/Levels", 1, 1, D3D12RaytracingAmbientOcclusion::c_MaxDenoisingScaleLevels);
    BoolVar RTAODenoisingMultiscaleDenoisedAsInput(L"Render/AO/RTAO/Denoising/Multi-scale/Denoised as input", true);
    
    BoolVar RTAODenoisingPerspectiveCorrectDepthInterpolation(L"Render/AO/RTAO/Denoising/Pespective Correct Depth Interpolation", true); // ToDo test perf impact / visual quality gain at the end. Document.
    BoolVar RTAODenoisingUseAdaptiveKernelSize(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Enabled", true);
    IntVar RTAODenoisingFilterMinKernelWidth(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Min kernel width", 5, 1, 101);
    NumVar RTAODenoisingFilterMaxKernelWidthPercentage(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Max kernel width [%% of screen width]", 2.5f, 0, 100, 0.1f);
    NumVar RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Variance sigma scale on small kernels", 2.0f, 1.0f, 20.f, 0.5f); 
    NumVar RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Hit distance scale factor", 1.0f, 0.1f, 10.f, 0.1f);
    BoolVar RTAODenoising_Variance_UseDepthWeights(L"Render/AO/RTAO/Denoising/Variance/Use normal weights", true);
    BoolVar RTAODenoising_Variance_UseNormalWeights(L"Render/AO/RTAO/Denoising/Variance/Use normal weights", true); 


    const WCHAR* DenoisingModes[GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::Count] = { L"EdgeStoppingBox3x3", L"EdgeStoppingGaussian3x3", L"EdgeStoppingGaussian5x5" };
    EnumVar DenoisingMode(L"Render/AO/RTAO/Denoising/Mode", GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::EdgeStoppingGaussian3x3, GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::Count, DenoisingModes);
#if    DISABLE_DENOISING
    IntVar AtrousFilterPasses(L"Render/AO/RTAO/Denoising/Num passes", 1, 1, 8, 1);
    NumVar AODenoiseValueSigma(L"Render/AO/RTAO/Denoising/Value Sigma", 0.011f, 0.0f, 30.0f, 0.1f);
#else
    IntVar AtrousFilterPasses(L"Render/AO/RTAO/Denoising/Num passes", 3, 1, 8, 1);
    NumVar AODenoiseValueSigma(L"Render/AO/RTAO/Denoising/Value Sigma", 6, 0.0f, 30.0f, 0.1f);
#endif
    BoolVar ReverseFilterOrder(L"Render/AO/RTAO/Denoising/Reverse filter order", false);

    // ToDo why large depth sigma is needed?
    // ToDo the values don't scale to QuarterRes - see ImportaceMap viz
    NumVar AODenoiseDepthSigma(L"Render/AO/RTAO/Denoising/Depth Sigma", 1.2f, 0.0f, 10.0f, 0.02f); // ToDo Fine tune. 1 causes moire patterns at angle under the car

    NumVar AODenoiseNormalSigma(L"Render/AO/RTAO/Denoising/Normal Sigma", 64, 0, 256, 4);   // ToDo rename sigma as sigma in depth/var means tolernace. here its an exponent.
    

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


void D3D12RaytracingAmbientOcclusion::CreateIndexAndVertexBuffers(
    const GeometryDescriptor& desc,
    D3DGeometry* geometry)
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

    CreateGeometry(device, commandList, m_cbvSrvUavHeap.get(), desc, geometry);
}

// ToDo move
void D3D12RaytracingAmbientOcclusion::LoadPBRTScene()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto commandQueue = m_deviceResources->GetCommandQueue();
    auto commandAllocator = m_deviceResources->GetCommandAllocator();

    // ToDo remove?
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


    // ToDo
    //m_camera.Set(
    //	Vec3ToXMVECTOR(m_pbrtScene.m_Camera.m_Position),
    //	Vec3ToXMVECTOR(m_pbrtScene.m_Camera.m_LookAt),
    //	Vec3ToXMVECTOR(m_pbrtScene.m_Camera.m_Up));
    //m_camera.fov = 2 * m_pbrtScene.m_Camera.m_FieldOfView;   


    PBRTScene pbrtSceneDefinitions[] = {
        {L"Spaceship", "Assets\\spaceship\\scene.pbrt"},
        {L"GroundPlane", "Assets\\groundplane\\scene.pbrt"},
#if !LOAD_ONLY_ONE_PBRT_MESH 
        {L"Car", "Assets\\car2\\scene.pbrt"},
        {L"Dragon", "Assets\\dragon\\scene.pbrt"},
        {L"House", "Assets\\house\\scene.pbrt"},

        {L"MirrorQuad", "Assets\\mirrorquad\\scene.pbrt"},
#endif
    };

    ResourceUploadBatch resourceUpload(device);
    resourceUpload.Begin();

    // ToDo
    bool isVertexAnimated = false;

    for (auto& pbrtSceneDefinition : pbrtSceneDefinitions)
    {
        SceneParser::Scene pbrtScene;
        PBRTParser::PBRTParser().Parse(pbrtSceneDefinition.path, pbrtScene);

        auto& bottomLevelASGeometry = m_bottomLevelASGeometries[pbrtSceneDefinition.name];
        bottomLevelASGeometry.SetName(pbrtSceneDefinition.name);

        // ToDo switch to a common namespace rather than 't reference SquidRoomAssets?
        bottomLevelASGeometry.m_indexFormat = SquidRoomAssets::StandardIndexFormat;
        bottomLevelASGeometry.m_ibStrideInBytes = SquidRoomAssets::StandardIndexStride;
        bottomLevelASGeometry.m_vertexFormat = DXGI_FORMAT_R32G32B32_FLOAT; // ToDo use common or add support to shaders 
        bottomLevelASGeometry.m_vbStrideInBytes = SquidRoomAssets::StandardVertexStride;

        UINT numGeometries = static_cast<UINT>(pbrtScene.m_Meshes.size());
        auto& geometries = bottomLevelASGeometry.m_geometries;
        geometries.resize(numGeometries);

        auto& textures = bottomLevelASGeometry.m_textures;
        auto& numTriangles = bottomLevelASGeometry.m_numTriangles;

        for (UINT i = 0; i < pbrtScene.m_Meshes.size(); i++)
        {
            auto &mesh = pbrtScene.m_Meshes[i];
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
                vertex.tangent = parseVertex.Tangent.xmFloat3;
                vertex.textureCoordinate = parseVertex.UV.xmFloat2;
                vertexBuffer.push_back(vertex);
            }
            desc.vb.vertices = vertexBuffer.data();

            auto& geometry = geometries[i];
            CreateIndexAndVertexBuffers(desc, &geometry);

            PrimitiveMaterialBuffer cb;
            cb.Kd = mesh.m_pMaterial->m_Kd.xmFloat3;
            cb.Ks = mesh.m_pMaterial->m_Ks.xmFloat3;
            cb.Kr = mesh.m_pMaterial->m_Kr.xmFloat3;
            cb.Kt = mesh.m_pMaterial->m_Kt.xmFloat3;
            cb.opacity = mesh.m_pMaterial->m_Opacity.xmFloat3;
            cb.eta = mesh.m_pMaterial->m_Eta.xmFloat3;
            cb.roughness = mesh.m_pMaterial->m_Roughness;
            cb.hasDiffuseTexture = !mesh.m_pMaterial->m_DiffuseTextureFilename.empty();
            cb.hasNormalTexture = !mesh.m_pMaterial->m_NormalMapTextureFilename.empty();
            cb.hasPerVertexTangents = true;
            cb.type = mesh.m_pMaterial->m_Type;


            auto LoadPBRTTexture = [&](auto** ppOutTexture, auto& textureFilename)
            {
                wstring filename(textureFilename.begin(), textureFilename.end());
                D3DTexture texture;
                // ToDo use a hel
                if (filename.find(L".dds") != wstring::npos)
                {
                    LoadDDSTexture(device, commandList, filename.c_str(), m_cbvSrvUavHeap.get(), &texture);
                }
                else
                {
                    LoadWICTexture(device, &resourceUpload, filename.c_str(), m_cbvSrvUavHeap.get(), &texture.resource, &texture.heapIndex, &texture.cpuDescriptorHandle, &texture.gpuDescriptorHandle, true);
                }
                textures.push_back(texture);

                *ppOutTexture = &textures.back();
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

            D3D12_RAYTRACING_GEOMETRY_FLAGS geometryFlags;

            if (cb.opacity.x > 0.99f && cb.opacity.y > 0.99f && cb.opacity.z > 0.99f)
            {
                geometryFlags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            }
            else
            {
                geometryFlags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
            }

            bottomLevelASGeometry.m_geometryInstances.push_back(GeometryInstance(geometry, materialID, diffuseTexture->gpuDescriptorHandle, normalTexture->gpuDescriptorHandle, geometryFlags, isVertexAnimated));
#if !PBRT_APPLY_INITIAL_TRANSFORM_TO_VB_ATTRIBUTES
            XMStoreFloat3x4(reinterpret_cast<XMFLOAT3X4*>(m_geometryTransforms[i].transform3x4), mesh.m_transform);
            geometryInstances.back().transform = m_geometryTransforms.GpuVirtualAddress(0, i);
#endif
            numTriangles += desc.ib.count / 3;
        }
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
    m_animateScene(false),
    m_isGeometryInitializationRequested(true),
    m_isASinitializationRequested(true),
	m_isSceneInitializationRequested(false),
	m_isRecreateRaytracingResourcesRequested(false),
    m_numFramesSinceASBuild(0),
	m_isCameraFrozen(false)
{
    ThrowIfFalse(++s_numInstances == 1, L"There can be only one D3D12RaytracingAmbientOcclusion instance.");
    global_pSample = this;

    for (auto& rayGenShaderTableRecordSizeInBytes : m_rayGenShaderTableRecordSizeInBytes)
    {
        rayGenShaderTableRecordSizeInBytes = UINT_MAX;
    }
    global_pSample = this;
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
#if ENABLE_RAYTRACING
	ThrowIfFalse(IsDirectXRaytracingSupported(m_deviceResources->GetAdapter()),
		L"ERROR: DirectX Raytracing is not supported by your GPU and driver.\n\n");
#endif
    // ToDo cleanup
    m_deviceResources->CreateDeviceResources();

    // Initialize scene ToDo

    CreateDeviceDependentResources();

    InitializeScene();

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
    ThrowIfFalse(false, L"ToDO");
    return;

#if 0
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
#endif
}

void D3D12RaytracingAmbientOcclusion::UpdateSphereGeometryTransforms()
{
    ThrowIfFalse(false, L"ToDO");
    return;

#if 0
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
#endif
}

// ToDo move to CS.
void D3D12RaytracingAmbientOcclusion::UpdateGridGeometryTransforms()
{
    // ToDO remove
    return;
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
        // ToDo
        m_cameraController->EnableMomentum(false);
        m_prevFrameCamera = m_camera;
	}

    // Setup lights.
    {
        // Initialize the lighting parameters.
		// ToDo remove
		m_csComposeRenderPassesCB->lightPosition = XMFLOAT3(-20.0f, 60.0f, 20.0f);
		m_sceneCB->lightPosition = XMLoadFloat3(&m_csComposeRenderPassesCB->lightPosition);

		m_csComposeRenderPassesCB->lightAmbientColor = XMFLOAT3(0.45f, 0.45f, 0.45f);

        float d = 0.6f;
        m_sceneCB->lightColor = XMFLOAT3(d, d, d);
		m_csComposeRenderPassesCB->lightDiffuseColor = XMFLOAT3(d, d, d);
    }
}

// Create constant buffers.
void D3D12RaytracingAmbientOcclusion::CreateConstantBuffers()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    m_sceneCB.Create(device, FrameCount, L"Scene Constant Buffer");

    m_SSAOCB.Create(device, FrameCount, L"SSAO Constant Buffer");
}

void D3D12RaytracingAmbientOcclusion::CreateComposeRenderPassesCSResources()
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
		ranges[Slot::Visibility].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6);  // 1 input Visibility texture
        ranges[Slot::FilterWeightSum].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);  // 1 input filterWeightSum texture
        ranges[Slot::AORayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 9);  // 1 input AO ray hit distance texture
        ranges[Slot::FrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 10); // 1 input disocclusion map texture
        ranges[Slot::Color].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 11); // 1 input color texture
        ranges[Slot::AOSurfaceAlbedo].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 12); // 1 input AO diffuse texture
        

		CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
		rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[Slot::Output]);
		rootParameters[Slot::GBufferResources].InitAsDescriptorTable(1, &ranges[Slot::GBufferResources]);
		rootParameters[Slot::AO].InitAsDescriptorTable(1, &ranges[Slot::AO]);
		rootParameters[Slot::Visibility].InitAsDescriptorTable(1, &ranges[Slot::Visibility]);
        rootParameters[Slot::FilterWeightSum].InitAsDescriptorTable(1, &ranges[Slot::FilterWeightSum]);
        rootParameters[Slot::AORayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::AORayHitDistance]);
        rootParameters[Slot::FrameAge].InitAsDescriptorTable(1, &ranges[Slot::FrameAge]);
        rootParameters[Slot::Color].InitAsDescriptorTable(1, &ranges[Slot::Color]);
        rootParameters[Slot::AOSurfaceAlbedo].InitAsDescriptorTable(1, &ranges[Slot::AOSurfaceAlbedo]);
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
	auto FrameCount = m_deviceResources->GetBackBufferCount();

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

    // ToDo remove
    //GpuTimeManager::instance().SetAvgRefreshPeriodMS(3000);

    // Create a heap for descriptors.
    CreateDescriptorHeaps();

    CreateAuxilaryDeviceResources();

	// ToDo move
	m_geometryTransforms.Create(device, MaxGeometryTransforms, FrameCount, L"Structured buffer: Geometry desc transforms");

    // Initialize raytracing pipeline.

    // Build geometry to be used in the sample.
    InitializeGeometry();

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

 
	// Build shader tables, which define shaders and their local root arguments.
    BuildShaderTables();

    InitializeAccelerationStructures();
   

	// ToDo move
	CreateComposeRenderPassesCSResources();

    CreateAoBlurCSResources();

    m_RTAO.Setup(m_deviceResources, m_cbvSrvUavHeap, m_maxInstanceContributionToHitGroupIndex);
    m_SSAO.Setup(m_deviceResources);

    // 
    m_prevFrameBottomLevelASInstanceTransforms.Create(device, MaxNumBottomLevelInstances, FrameCount, L"GPU buffer: Bottom Level AS Instance transforms for previous frame");
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
        ranges[Slot::NormalDepthLowPrecision].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 23);  // 1 output normal depth texture
        ranges[Slot::AORayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 15);  // 1 output ray hit distance texture
        ranges[Slot::MotionVector].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 17);  // 1 output texture space motion vector.
        ranges[Slot::ReprojectedHitPosition].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 18);  // 1 output texture reprojected hit position
        ranges[Slot::Color].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 19);  // 1 output texture shaded color
        ranges[Slot::AOSurfaceAlbedo].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 20);  // 1 output texture AO diffuse
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
        rootParameters[Slot::AOSurfaceAlbedo].InitAsDescriptorTable(1, &ranges[Slot::AOSurfaceAlbedo]);
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
    auto device = m_deviceResources->GetD3DDevice();
    // Pathracing state object.
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
        ThrowIfFailed(device->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&m_dxrStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
    }
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
    DXGI_FORMAT normalFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;       // ToDo rename to coefficient or avoid using same variable for different types.
    DXGI_FORMAT hitPositionFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;// DXGI_FORMAT_R16G16B16A16_FLOAT; // ToDo change to 16bit? or DXGI_FORMAT_R32G32B32_FLOAT
	// ToDo tune formats
    // ToDo change this to non-PS resouce since we use CS?
	D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

    // ToDo remove obsolete resources, QuarterResAO event triggers this so we may not need all low/gbuffer width AO resources.

    // Full-res GBuffer resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_GBufferResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        m_GBufferResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        for (UINT i = 0; i < GBufferResource::Count; i++)
        {
            m_GBufferResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_GBufferResources[i].uavDescriptorHeapIndex = m_GBufferResources[0].uavDescriptorHeapIndex + i;
            m_GBufferResources[i].srvDescriptorHeapIndex = m_GBufferResources[0].srvDescriptorHeapIndex + i;
        }
        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Hit], initialResourceState, L"GBuffer Hit");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_UINT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Material], initialResourceState, L"GBuffer Material");

        
        CreateRenderTargetResource(device, hitPositionFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::HitPosition], initialResourceState, L"GBuffer HitPosition");

        CreateRenderTargetResource(device, normalFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::SurfaceNormal], initialResourceState, L"GBuffer Normal");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Distance], initialResourceState, L"GBuffer Distance");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Depth], initialResourceState, L"GBuffer Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::SurfaceNormalRGB], initialResourceState, L"GBuffer Normal RGB");
        
        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(SceneArgs::RTAO_PartialDepthDerivativesResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::PartialDepthDerivatives], initialResourceState, L"GBuffer Partial Depth Derivatives");

        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(SceneArgs::RTAO_MotionVectorResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::MotionVector], initialResourceState, L"GBuffer Texture Space Motion Vector");
        
        CreateRenderTargetResource(device, TextureResourceFormatRGB::ToDXGIFormat(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::ReprojectedHitPosition], initialResourceState, L"GBuffer Reprojected Hit Position");
        
        
        CreateRenderTargetResource(device, backbufferFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::Color], initialResourceState, L"GBuffer Color");

        CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_GBufferResources[GBufferResource::AOSurfaceAlbedo], initialResourceState, L"GBuffer AO Surface Albedo");

    }
    
    // Low-res GBuffer resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_GBufferLowResResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        m_GBufferLowResResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(GBufferResource::Count);
        for (UINT i = 0; i < GBufferResource::Count; i++)
        {
            m_GBufferLowResResources[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            m_GBufferLowResResources[i].uavDescriptorHeapIndex = m_GBufferLowResResources[0].uavDescriptorHeapIndex + i;
            m_GBufferLowResResources[i].srvDescriptorHeapIndex = m_GBufferLowResResources[0].srvDescriptorHeapIndex + i;
        }
 
        CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Hit], initialResourceState, L"GBuffer LowRes Hit");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32G32_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Material], initialResourceState, L"GBuffer LowRes Material");
        CreateRenderTargetResource(device, hitPositionFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::HitPosition], initialResourceState, L"GBuffer LowRes HitPosition");
        CreateRenderTargetResource(device, normalFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::SurfaceNormal], initialResourceState, L"GBuffer LowRes Normal");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Distance], initialResourceState, L"GBuffer LowRes Distance");
        // ToDo are below two used?
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::Depth], initialResourceState, L"GBuffer LowRes Depth");
        CreateRenderTargetResource(device, DXGI_FORMAT_R11G11B10_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::SurfaceNormalRGB], initialResourceState, L"GBuffer LowRes Normal RGB");
        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(SceneArgs::RTAO_PartialDepthDerivativesResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives], initialResourceState, L"GBuffer LowRes Partial Depth Derivatives");

        CreateRenderTargetResource(device, TextureResourceFormatRG::ToDXGIFormat(SceneArgs::RTAO_MotionVectorResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::MotionVector], initialResourceState, L"GBuffer LowRes Texture Space Motion Vector");
        CreateRenderTargetResource(device, TextureResourceFormatRGB::ToDXGIFormat(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_GBufferLowResResources[GBufferResource::ReprojectedHitPosition], initialResourceState, L"GBuffer LowRes Reprojected Hit Position");

    }


    // Full-res Normal Depth Low Precision.
    {
        for (UINT i = 0; i < ARRAYSIZE(m_normalDepthLowPrecision); i++)
        {
            m_normalDepthLowPrecision[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, TextureResourceFormatRGB::ToDXGIFormat(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_normalDepthLowPrecision[i], initialResourceState, L"Normal Depth Low Precision");
        }
    }

    // Low-res Normal Depth Low Precision.
    {
        for (UINT i = 0; i < ARRAYSIZE(m_normalDepthLowResLowPrecision); i++)
        {
            m_normalDepthLowResLowPrecision[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, TextureResourceFormatRGB::ToDXGIFormat(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_normalDepthLowResLowPrecision[i], initialResourceState, L"Normal Depth Low Res Low Precision");
        }
    }

    m_SSAO.BindGBufferResources(m_GBufferResources[GBufferResource::SurfaceNormalRGB].resource.Get(), m_GBufferResources[GBufferResource::Depth].resource.Get());

    // ToDo remove unneeded ones
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
        CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Coefficient], initialResourceState, L"Render/AO Coefficient");
        
#if ATROUS_DENOISER
        CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
#else
        CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::Smoothed], initialResourceState, L"Render/AO Smoothed");
#endif
        // ToDo 8 bit hit count?
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_UINT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::HitCount], initialResourceState, L"Render/AO Hit Count");

        // ToDo use lower bit float?
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::FilterWeightSum], initialResourceState, L"Render/AO Filter Weight Sum");
        CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::RayHitDistance], initialResourceState, L"Render/AO Hit Distance");
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
            CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOTSSCoefficient[i], initialResourceState, L"Render/AO Temporally Supersampled Coefficient");

            m_lowResAOTSSCoefficient[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_lowResAOTSSCoefficient[i], initialResourceState, L"Render/AO LowRes Temporally Supersampled Coefficient");


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

void D3D12RaytracingAmbientOcclusion::CreateAuxilaryDeviceResources()
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
    m_calculateMeanVarianceKernel.Initialize(device, FrameCount, 5*MaxCalculateVarianceKernelInvocationsPerFrame);
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

    // Upload the resources to the GPU.
    auto finish = resourceUpload.End(commandQueue);

    // Wait for the upload thread to terminate
    finish.wait();
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
		m_cbvSrvUavHeap = make_shared<DX::DescriptorHeap>(device, NumDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	}

    // TodO remove
	// Sampler heap.
	{
		UINT NumDescriptors = 1;
		m_samplerHeap = make_unique<DX::DescriptorHeap>(device, NumDescriptors, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
	}
}

void D3D12RaytracingAmbientOcclusion::BuildPlaneGeometry()
{
    auto device = m_deviceResources->GetD3DDevice();

    auto& bottomLevelASGeometry = m_bottomLevelASGeometries[L"Plane"];
    bottomLevelASGeometry.SetName(L"Plane");
    bottomLevelASGeometry.m_indexFormat = DXGI_FORMAT_R16_UINT; // ToDo use common or add support to shaders 
    bottomLevelASGeometry.m_ibStrideInBytes = sizeof(Index);
    bottomLevelASGeometry.m_vertexFormat = DXGI_FORMAT_R32G32B32_FLOAT; // ToDo use common or add support to shaders 
    bottomLevelASGeometry.m_vbStrideInBytes = sizeof(DirectX::GeometricPrimitive::VertexType);

    auto& geometries = bottomLevelASGeometry.m_geometries;
    geometries.resize(1);
	auto& geometry = geometries[0];

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
       
    ThrowIfFalse(0 && L"ToDo: fix up null VB SRV");


    PrimitiveMaterialBuffer planeMaterialCB;
    planeMaterialCB.Kd = XMFLOAT3(0.24f, 0.4f, 0.4f);
    planeMaterialCB.opacity = XMFLOAT3(1, 1, 1);
    planeMaterialCB.hasDiffuseTexture = false;
    planeMaterialCB.hasNormalTexture = false;
    planeMaterialCB.hasPerVertexTangents = false;
    planeMaterialCB.roughness = 0.0;
    planeMaterialCB.type = MaterialType::Matte;

	UINT materialID = static_cast<UINT>(m_materials.size());
	m_materials.push_back(planeMaterialCB);

    bottomLevelASGeometry.m_geometryInstances.resize(1);
    bottomLevelASGeometry.m_geometryInstances.push_back(GeometryInstance(geometry, materialID, m_nullTexture.gpuDescriptorHandle, m_nullTexture.gpuDescriptorHandle));
    bottomLevelASGeometry.m_numTriangles = bottomLevelASGeometry.m_geometryInstances.back().ib.count / 3;
}

void D3D12RaytracingAmbientOcclusion::BuildTesselatedGeometry()
{
    auto device = m_deviceResources->GetD3DDevice();

    const bool RhCoords = false;    // ToDo use a global constant

    auto& bottomLevelASGeometry = m_bottomLevelASGeometries[L"Tesselated Geometry"];
    bottomLevelASGeometry.SetName(L"Tesselated Geometry");
    bottomLevelASGeometry.m_indexFormat = SquidRoomAssets::StandardIndexFormat;
    bottomLevelASGeometry.m_ibStrideInBytes = sizeof(Index);
    bottomLevelASGeometry.m_vertexFormat = DXGI_FORMAT_R32G32B32_FLOAT; // ToDo use common or add support to shaders 
    bottomLevelASGeometry.m_vbStrideInBytes = sizeof(VertexPositionNormalTextureTangent);

    const UINT NumTrees = 1;
    auto& geometries = bottomLevelASGeometry.m_geometries;
    geometries.resize(NumTrees);
    
	vector<GeometricPrimitive::VertexType> dxtk_vertices;
	vector<uint16_t> dxtk_indices;

    float diameter = 5;
    float height = 5;
    size_t tesselation = 7;
    GeometricPrimitive::CreateCone(dxtk_vertices, dxtk_indices, diameter, height, tesselation, RhCoords);
    //GeometricPrimitive::CreateTetrahedron(dxtk_vertices, dxtk_indices, diameter, RhCoords);
    //GeometricPrimitive::CreateTorus(dxtk_vertices, dxtk_indices, diameter, 0.5, 4, RhCoords);

    vector<VertexPositionNormalTextureTangent> vertices;
    vector<Index> indices;

    for (auto& dxtk_vertex : dxtk_vertices)
    {
        VertexPositionNormalTextureTangent vertex =
        {
            dxtk_vertex.position,
            dxtk_vertex.normal,
            dxtk_vertex.textureCoordinate,
            XMFLOAT3()
        };
        vertices.push_back(vertex);
    }
    for (auto& dxtk_index : dxtk_indices)
    {
        Index index = dxtk_index;
        indices.push_back(index);
    }
    std::mt19937 m_generatorURNG;  // Uniform random number generator
    m_generatorURNG.seed(1729);
    uniform_real_distribution<float> unitSquareDistributionInclusive(0.f, nextafter(1.f, FLT_MAX));
    function<float()> GetRandomFloat01inclusive = bind(unitSquareDistributionInclusive, ref(m_generatorURNG));

    // Deform the vertices a bit
    float deformDistance = height * 0.01f;
    float radius = diameter / 2;
    for (auto& vertex : vertices)
    {
        // Bottom vertices
        if (vertex.position.y < 0)
        {
            float angle = XM_PIDIV2 + asinf(vertex.position.x / radius);    // <0, XM_PI>
            angle += vertex.position.z < 0 ? XM_PI : 0;                     // <0, XM_2PI>
            float frequency = 5;
            vertex.position.y += deformDistance * sinf(frequency * angle);
        }
    }
    
    auto CalculateNormals = [&](vector<VertexPositionNormalTextureTangent>* pvVertices, vector<Index>& vIndices)
    {
        // Since some vertices may be shared across faces,
        // update a copy of vertex normals while evaluating all the faces.
        vector<UINT> vertexFaceCountContributions;
        vertexFaceCountContributions.resize(pvVertices->size(), 0);
        vector<XMVECTOR> vertexNormalsSum;
        vertexNormalsSum.resize(pvVertices->size(), XMVectorZero());

        for (UINT i = 0; i < vIndices.size(); i += 3)
        {
            UINT indices[3] = { vIndices[i], vIndices[i + 1], vIndices[i + 2] };
            auto& v0 = (*pvVertices)[indices[0]];
            auto& v1 = (*pvVertices)[indices[1]];
            auto& v2 = (*pvVertices)[indices[2]];
            XMVECTOR normals[3] = {
                XMVector3Normalize(XMLoadFloat3(&v0.normal)),
                XMVector3Normalize(XMLoadFloat3(&v1.normal)),
                XMVector3Normalize(XMLoadFloat3(&v2.normal))
            };

            XMVECTOR* nSums[3] = { &vertexNormalsSum[indices[0]], &vertexNormalsSum[indices[1]], &vertexNormalsSum[indices[2]] };

            for (UINT i = 0; i < 3; i++)
            {
                vertexFaceCountContributions[indices[i]]++;
            }

            // Calculate the face normal.
            XMVECTOR v01 = XMLoadFloat3(&v1.position) - XMLoadFloat3(&v0.position);
            XMVECTOR v02 = XMLoadFloat3(&v2.position) - XMLoadFloat3(&v0.position);
            XMVECTOR faceNormal = XMVector3Normalize(XMVector3Cross(v01, v02));

            // Add the face normal contribution to all three vertices.
            for (UINT i = 0; i < 3; i++)
            {
                *nSums[i] += faceNormal;
            }
        }

        // Update the vertices with normalized normals across all contributing faces.
        for (UINT i = 0; i < (*pvVertices).size(); i++)
        {
            XMStoreFloat3(&(*pvVertices)[i].normal, vertexNormalsSum[i] / static_cast<float>(vertexFaceCountContributions[i]));
        }
    };

    CalculateNormals(&vertices, indices);


    auto& geometry = geometries[0];

    // Convert index and vertex buffers to the sample's common format.

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

    PrimitiveMaterialBuffer materialCB;
#if 1
    ThrowIfFalse(false && L"ToDo");
#else
    = { XMFLOAT3(14 / 255.f, 117 / 255.f, 0), XMFLOAT3(1, 1, 1), XMFLOAT3(1, 1, 1), 50, false, false, false, 1, MaterialType::Default };
#endif
    UINT materialID = static_cast<UINT>(m_materials.size());
	m_materials.push_back(materialCB);
    bottomLevelASGeometry.m_geometryInstances.resize(SceneArgs::NumGeometriesPerBLAS, GeometryInstance(geometry, materialID, m_nullTexture.gpuDescriptorHandle, m_nullTexture.gpuDescriptorHandle));

    bottomLevelASGeometry.m_numTriangles = bottomLevelASGeometry.m_geometryInstances.back().ib.count / 3;
}

// ToDo move this out as a helper

void D3D12RaytracingAmbientOcclusion::LoadSquidRoom()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

    auto& bottomLevelASGeometry = m_bottomLevelASGeometries[L"Squid Room"];
    bottomLevelASGeometry.SetName(L"Squid Room");
    bottomLevelASGeometry.m_indexFormat = SquidRoomAssets::StandardIndexFormat; // ToDo use common or add support to shaders 
    bottomLevelASGeometry.m_vertexFormat = DXGI_FORMAT_R32G32B32_FLOAT; // ToDo use common or add support to shaders 
    bottomLevelASGeometry.m_ibStrideInBytes = SquidRoomAssets::StandardIndexStride;
    bottomLevelASGeometry.m_vbStrideInBytes = SquidRoomAssets::StandardVertexStride;

    bottomLevelASGeometry.m_geometries.resize(1);
    auto& geometry = bottomLevelASGeometry.m_geometries[0];
    auto& textures = bottomLevelASGeometry.m_textures;

    SquidRoomAssets::LoadGeometry(
        device,
        commandList,
        m_cbvSrvUavHeap.get(),
        GetAssetFullPath(SquidRoomAssets::DataFileName).c_str(),
        &geometry,
        &textures,
        &m_materials,
        &bottomLevelASGeometry.m_geometryInstances);

    bottomLevelASGeometry.m_numTriangles = 0;
    for (auto& geometryInstance : bottomLevelASGeometry.m_geometryInstances)
    {
        bottomLevelASGeometry.m_numTriangles = geometryInstance.ib.count / 3;
    }
}

void D3D12RaytracingAmbientOcclusion::LoadSceneGeometry()
{
    //BuildTesselatedGeometry();

#if LOAD_PBRT_SCENE
	LoadPBRTScene();
#else
    LoadSquidRoom();
#endif

#if USE_GRASS_GEOMETRY
    InitializeGrassGeometry();
#endif

    auto device = m_deviceResources->GetD3DDevice();

    // Create null resource descriptor for the unused second VB in non-animated geometry.
    D3D12_CPU_DESCRIPTOR_HANDLE nullCPUhandle;
    UINT nullHeapIndex = UINT_MAX;
    CreateBufferSRV(nullptr, device, 0, sizeof(VertexPositionNormalTextureTangent), m_cbvSrvUavHeap.get(), &nullCPUhandle, &m_nullVertexBufferGPUhandle, &nullHeapIndex);
}

// Build geometry used in the sample.
void D3D12RaytracingAmbientOcclusion::InitializeGrassGeometry()
{
#if !GENERATE_GRASS
    return;
#endif
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto commandQueue = m_deviceResources->GetCommandQueue();

    D3DTexture* diffuseTexture = nullptr;
    D3DTexture* normalTexture = &m_nullTexture;

    // Initialize all LOD bottom-level Acceleration Structures for the grass.
    for (UINT i = 0; i < UIParameters::NumGrassGeometryLODs; i++)
    {
        wstring name = L"Grass Patch LOD " + to_wstring(i);
        auto& bottomLevelASGeometry = m_bottomLevelASGeometries[name];
        bottomLevelASGeometry.SetName(name);
        bottomLevelASGeometry.m_indexFormat = SquidRoomAssets::StandardIndexFormat; // ToDo use common or add support to shaders 
        bottomLevelASGeometry.m_vertexFormat = DXGI_FORMAT_R32G32B32_FLOAT; // ToDo use common or add support to shaders 
        bottomLevelASGeometry.m_ibStrideInBytes = SquidRoomAssets::StandardIndexStride;
        bottomLevelASGeometry.m_vbStrideInBytes = SquidRoomAssets::StandardVertexStride;

        // Single patch geometry per bottom-level AS.
        bottomLevelASGeometry.m_geometries.resize(1);
        auto& geometry = bottomLevelASGeometry.m_geometries[0];
        auto& textures = bottomLevelASGeometry.m_textures;

        // Initialize index and vertex buffers.
        {
            const UINT NumStraws = MAX_GRASS_STRAWS_1D * MAX_GRASS_STRAWS_1D;
            const UINT NumTrianglesPerStraw = N_GRASS_TRIANGLES;
            const UINT NumTriangles = NumStraws * NumTrianglesPerStraw;
            const UINT NumVerticesPerStraw = N_GRASS_VERTICES;
            const UINT NumVertices = NumStraws * NumVerticesPerStraw;
            const UINT NumIndicesPerStraw = NumTrianglesPerStraw * 3;
            const UINT NumIndices = NumStraws * NumIndicesPerStraw;
            UINT strawIndices[NumIndicesPerStraw] = { 0, 2, 1, 1, 2, 3, 2, 4, 3, 3, 4, 5, 4, 6, 5 };
            vector<UINT> indices;
            indices.resize(NumIndices);

            UINT indexID = 0;
            for (UINT i = 0, indexID = 0; i < NumStraws; i++)
            {
                UINT baseVertexID = i * NumVerticesPerStraw;
                for (auto index : strawIndices)
                {
                    indices[indexID++] = baseVertexID + index;
                }
            }
            // Preallocate subsequent descriptor indices for both SRV and UAV groups.
            UINT baseSRVHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(3);      // 1 IB + 2 VB
            geometry.ib.buffer.heapIndex = baseSRVHeapIndex;
            m_grassPatchVB[i][0].srvDescriptorHeapIndex = baseSRVHeapIndex + 1;
            m_grassPatchVB[i][1].srvDescriptorHeapIndex = baseSRVHeapIndex + 2;

            UINT baseUAVHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(2);      // 2 VB
            m_grassPatchVB[i][0].uavDescriptorHeapIndex = baseUAVHeapIndex;
            m_grassPatchVB[i][1].uavDescriptorHeapIndex = baseUAVHeapIndex + 1;

            AllocateIndexBuffer(device, NumIndices, sizeof(Index), m_cbvSrvUavHeap.get(), &geometry.ib.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
            UploadDataToBuffer(device, commandList, &indices[0], NumIndices, sizeof(Index), geometry.ib.buffer.resource.Get(), &geometry.ib.upload, D3D12_RESOURCE_STATE_INDEX_BUFFER);

            for (auto& vb : m_grassPatchVB[i])
            {
                vb.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
                AllocateUAVBuffer(device, NumVertices, sizeof(VertexPositionNormalTextureTangent), &vb, DXGI_FORMAT_UNKNOWN, m_cbvSrvUavHeap.get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"Vertex Buffer: Grass geometry");
            }

            // ToDo add comment
            geometry.vb.buffer.resource = m_grassPatchVB[i][0].resource;
            geometry.vb.buffer.gpuDescriptorHandle = m_grassPatchVB[i][0].gpuDescriptorReadAccess;
            geometry.vb.buffer.heapIndex = m_grassPatchVB[i][0].srvDescriptorHeapIndex;
        }

        // Load textures during initialization of the first LOD.
        if (i == 0)
        {
            ResourceUploadBatch resourceUpload(device);
            resourceUpload.Begin();

            auto LoadTexture = [&](auto** ppOutTexture, const wchar_t* textureFilename)
            {
                D3DTexture texture;
                LoadWICTexture(device, &resourceUpload, textureFilename, m_cbvSrvUavHeap.get(), &texture.resource, &texture.heapIndex, &texture.cpuDescriptorHandle, &texture.gpuDescriptorHandle, false);
                textures.push_back(texture);

                *ppOutTexture = &textures.back();
            };
            LoadTexture(&diffuseTexture, L"assets\\grass\\albedo.png");

            // ToDo load everything via single resource upload?
            // Upload the resources to the GPU.
            auto finish = resourceUpload.End(commandQueue);

            // Wait for the upload thread to terminate
            finish.wait();
        }
        else
        {
            textures.push_back(*diffuseTexture);
        }

        UINT materialID;
        {
            PrimitiveMaterialBuffer materialCB;

            switch (i)
            {
            case 0: materialCB.Kd = XMFLOAT3(0.25f, 0.75f, 0.25f); break;
            case 1: materialCB.Kd = XMFLOAT3(0.5f, 0.75f, 0.5f); break;
            case 2: materialCB.Kd = XMFLOAT3(0.25f, 0.5f, 0.5f); break;
            case 3: materialCB.Kd = XMFLOAT3(0.5f, 0.5f, 0.75f); break;
            case 4: materialCB.Kd = XMFLOAT3(0.75f, 0.25f, 0.75f); break;
            }

            materialCB.Ks = XMFLOAT3(0, 0, 0);
            materialCB.Kr = XMFLOAT3(0, 0, 0);
            materialCB.Kt = XMFLOAT3(0, 0, 0);
            materialCB.opacity = XMFLOAT3(1, 1, 1);
            materialCB.eta = XMFLOAT3(1, 1, 1);
            materialCB.roughness = 0.1f; // ToDO  
            materialCB.hasDiffuseTexture = true;
            materialCB.hasNormalTexture = false;
            materialCB.hasPerVertexTangents = false;    // ToDo calculate these when geometry is generated?
            materialCB.type = MaterialType::Matte;

            materialID = static_cast<UINT>(m_materials.size());
            m_materials.push_back(materialCB);
        }


        // Create geometry instance.
        bool isVertexAnimated = true;
        bottomLevelASGeometry.m_geometryInstances.push_back(GeometryInstance(geometry, materialID, diffuseTexture->gpuDescriptorHandle, normalTexture->gpuDescriptorHandle, D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE, isVertexAnimated));

        bottomLevelASGeometry.m_numTriangles = bottomLevelASGeometry.m_geometryInstances[0].ib.count / 3;
    }

    ZeroMemory(m_prevFrameLODs, ARRAYSIZE(m_prevFrameLODs) * sizeof(m_prevFrameLODs[0]));
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

    //BuildPlaneGeometry();   

	// Begin frame.
	m_deviceResources->ResetCommandAllocatorAndCommandlist();

	LoadSceneGeometry();
    InitializeAllBottomLevelAccelerationStructures();

	m_materialBuffer.Create(device, static_cast<UINT>(m_materials.size()), 1, L"Structured buffer: materials");
	copy(m_materials.begin(), m_materials.end(), m_materialBuffer.begin());

    // ToDo move
    LoadDDSTexture(device, commandList, L"Assets\\Textures\\FlowerRoad\\flower_road_8khdri_1kcubemap.BC7.dds", m_cbvSrvUavHeap.get(), &m_environmentMap, D3D12_SRV_DIMENSION_TEXTURECUBE);
    
	m_materialBuffer.CopyStagingToGpu();
	m_deviceResources->ExecuteCommandList();
}

void D3D12RaytracingAmbientOcclusion::GenerateBottomLevelASInstanceTransforms()
{

    ThrowIfFalse(false, L"ToDO");


#if 0

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
        float bottomLevelASWidth = geometryDim * geometryWidth + (geometryDim - 1) * distanceBetweenGeometry;
        float distanceBetweenBLAS = 3 * distanceBetweenGeometry;
        float stepDistance = bottomLevelASWidth + distanceBetweenBLAS;

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
#endif

}

// Build acceleration structure needed for raytracing.
void D3D12RaytracingAmbientOcclusion::InitializeAllBottomLevelAccelerationStructures()
{
    auto device = m_deviceResources->GetD3DDevice();

    m_accelerationStructure = make_unique<RaytracingAccelerationStructureManager>(device, MaxNumBottomLevelInstances, FrameCount);
    
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;    // ToDo specify via SceneArgs
    for (auto& bottomLevelASGeometryPair : m_bottomLevelASGeometries)
    {
        auto& bottomLevelASGeometry = bottomLevelASGeometryPair.second;
        bool updateOnBuild = false;
        bool compactAS = false;
        // ToDO parametrize?
        if (bottomLevelASGeometry.GetName().find(L"Grass Patch LOD") != wstring::npos)
        {
            updateOnBuild = true;
        }
        if (bottomLevelASGeometry.GetName().find(L"Spaceship") != wstring::npos ||
            bottomLevelASGeometry.GetName().find(L"Dragon") != wstring::npos ||
            bottomLevelASGeometry.GetName().find(L"House") != wstring::npos ||
            bottomLevelASGeometry.GetName().find(L"Car") != wstring::npos)
        {
            compactAS = true;
        }
        m_accelerationStructure->AddBottomLevelAS(device, buildFlags, bottomLevelASGeometry, updateOnBuild, updateOnBuild, compactAS);
    }
}


// Build acceleration structure needed for raytracing.
void D3D12RaytracingAmbientOcclusion::InitializeAccelerationStructures()
{
    auto device = m_deviceResources->GetD3DDevice();

    // Initialize bottom-level AS.

#if LOAD_PBRT_SCENE
    wstring bottomLevelASnames[] = {
        L"Spaceship",
        L"GroundPlane",
#if !LOAD_ONLY_ONE_PBRT_MESH
        L"Dragon",
        L"Car",
        L"House",
#endif    
        //L"Tesselated Geometry"
    };
#else
    wstring bottomLevelASnames[] = {
        L"Squid Room" };

#endif

    // Initialize the bottom-level AS instances.
    for (auto& bottomLevelASname : bottomLevelASnames)
    {
        m_accelerationStructure->AddBottomLevelASInstance(bottomLevelASname);
    }
#if GENERATE_GRASS
#if GRASS_NO_DEGENERATE_INSTANCES
    UINT grassInstanceIndex = 0;
    for (int i = 0; i < NumGrassPatchesZ; i++)
        for (int j = 0; j < NumGrassPatchesX; j++)
        {
            int z = i - 15;
            int x = j - 15;

            if ((x < -1 || x > 2 || z < -2 || z > 1) &&
                (IsInRange(x, -2, 3) && IsInRange(z, -3, 2)))

            {
                m_grassInstanceIndices[grassInstanceIndex] = m_accelerationStructure->AddBottomLevelASInstance(L"Grass Patch LOD 0", UINT_MAX, XMMatrixIdentity());
                grassInstanceIndex++;
            }
        }
#else
    for (UINT i = 0; i < NumGrassPatchesX * NumGrassPatchesZ; i++)
    {
        // Initialize all grass patches to be "inactive" by way of making them to contain only degenerate triangles.
        // Triangle is a degenerate if it forms a point or a line after applying all transforms.
        // Degenerate triangles do not generate any intersections.
        XMMATRIX degenerateTransform = XMMatrixSet(
            0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 0.f);
        m_grassInstanceIndices[i] = m_accelerationStructure->AddBottomLevelASInstance(L"Grass Patch LOD 0", UINT_MAX, degenerateTransform);
    }
#endif
#endif

    // Initialize the top-level AS.
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;    // ToDo specify via SceneArgs
    bool allowUpdate = false;
    bool performUpdateOnBuild = false;
    m_accelerationStructure->InitializeTopLevelAS(device, buildFlags, allowUpdate, performUpdateOnBuild, L"Top-Level Acceleration Structure");
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

    // ToDo split shader table per unique pass?

    m_maxInstanceContributionToHitGroupIndex = 0;
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
            m_maxInstanceContributionToHitGroupIndex = shaderRecordOffset;

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
    case 'V':
        SceneArgs::TAO_LazyRender.Bang();// TODo remove
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

        m_RTAO.RequestRecreateRaytracingResources();
    }


    CalculateFrameStats();

    GameInput::Update(elapsedTime);
    EngineTuning::Update(elapsedTime);
    EngineProfiling::Update();

    m_RTAO.OnUpdate();

	
	if (GameInput::IsFirstPressed(GameInput::kKey_f))
	{
		m_isCameraFrozen = !m_isCameraFrozen;
	}
    m_prevFrameCamera = m_camera;

    m_cameraChangedIndex--;
    m_hasCameraChanged = false;
	if (!m_isCameraFrozen)
	{
        m_hasCameraChanged = m_cameraController->Update(elapsedTime);
        // ToDo
        // if (CameraChanged)
        //m_bClearTemporalCache = true;
	}


    if (m_animateScene)
    {
        float animationDuration = 36.0f;
        float t = static_cast<float>(m_timer.GetTotalSeconds());
        float rotAngle1 = XMConvertToRadians(t * 360.0f / animationDuration);
        float rotAngle2 = XMConvertToRadians((t + 12) * 360.0f / animationDuration);
        float rotAngle3 = XMConvertToRadians((t + 24) * 360.0f / animationDuration);
        m_accelerationStructure->GetBottomLevelASInstance(5).SetTransform(XMMatrixRotationAxis(XMVectorSet(0, 1, 0, 0), rotAngle1)
            * XMMatrixTranslationFromVector(XMVectorSet(-10, 4, -10, 0)));
        m_accelerationStructure->GetBottomLevelASInstance(6).SetTransform(XMMatrixRotationAxis(XMVectorSet(0, 1, 0, 0), rotAngle2)
            * XMMatrixTranslationFromVector(XMVectorSet(-15, 4, -10, 0)));
        m_accelerationStructure->GetBottomLevelASInstance(7).SetTransform(XMMatrixRotationAxis(XMVectorSet(0, 1, 0, 0), rotAngle3)
            * XMMatrixTranslationFromVector(XMVectorSet(-5, 4, -10, 0)));

        //m_accelerationStructure->GetBottomLevelASInstance(3).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(-5 + 10 * t, 0, 0, 0)));
        //m_accelerationStructure->GetBottomLevelASInstance(0).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(0, 10 * t, 0, 0)));
        //m_accelerationStructure->GetBottomLevelASInstance(1).SetTransform(XMMatrixRotationX(XMConvertToRadians((t-0.5f) * 20)));
    }

    // Rotate the camera around Y axis.
    if (m_animateCamera)
    {
        m_hasCameraChanged = true;
		// ToDo
        float secondsToRotateAround = SceneArgs::CameraRotationDuration;
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
        m_cameraChangedIndex = SceneArgs::RTAO_LazyRenderNumFrames;
#if DEBUG_CAMERA_POS
        //OutputDebugString(L"CameraChanged\n");
#endif
    }
    // ToDo remove
    if (fabs(m_manualCameraRotationAngle) > 0)
    {
        m_hasCameraChanged = true;
        m_cameraChangedIndex = 2;
        m_camera.RotateAroundYAxis(XMConvertToRadians(m_manualCameraRotationAngle));
        m_manualCameraRotationAngle = 0;
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

        m_updateShadowMap = true;
    }
    m_sceneCB->elapsedTime = static_cast<float>(m_timer.GetTotalSeconds());

    // Lazy initialize and update geometries and acceleration structures.
#if 0
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
    m_sceneCB->maxRadianceRayRecursionDepth = SceneArgs::MaxRadianceRayRecursionDepth;
    m_sceneCB->maxShadowRayRecursionDepth = SceneArgs::MaxShadowRayRecursionDepth;
    m_sceneCB->useShadowMap = SceneArgs::UseShadowMap;
    m_sceneCB->RTAO_UseNormalMaps = SceneArgs::RTAOUseNormalMaps;
    m_sceneCB->defaultAmbientIntensity = SceneArgs::DefaultAmbientIntensity;
 }

// Parse supplied command line args.
void D3D12RaytracingAmbientOcclusion::ParseCommandLineArgs(WCHAR* argv[], int argc)
{
    DXSample::ParseCommandLineArgs(argv, argc);
}

void D3D12RaytracingAmbientOcclusion::UpdateAccelerationStructure()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    if (SceneArgs::EnableGeometryAndASBuildsAndUpdates)
    {
        bool forceBuild = false;    // ToDo

        m_accelerationStructure->Build(commandList, m_cbvSrvUavHeap->GetHeap(), frameIndex, forceBuild);
    }

    // Copy previous frame Bottom Level AS instance transforms to GPU. 
    m_prevFrameBottomLevelASInstanceTransforms.CopyStagingToGpu(frameIndex);

    // Update the CPU staging copy with the current frame transforms.
    const auto& bottomLevelASInstanceDescs = m_accelerationStructure->GetBottomLevelASInstancesBuffer();
    for (UINT i = 0; i < bottomLevelASInstanceDescs.NumElements(); i++)
    {
        m_prevFrameBottomLevelASInstanceTransforms[i] = *reinterpret_cast<const XMFLOAT3X4*>(bottomLevelASInstanceDescs[i].Transform);
    }
}

void D3D12RaytracingAmbientOcclusion::DispatchRays(ID3D12Resource* rayGenShaderTable, UINT width, UINT height)
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

void D3D12RaytracingAmbientOcclusion::CalculateCameraRayHitCount()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
	auto commandList = m_deviceResources->GetCommandList();

	RWGpuResource* inputResource = &m_GBufferResources[GBufferResource::Hit];

    // ToDo make this disabled by default/

    // Todo
    if (SceneArgs::QuarterResAO)
    {
        return;
    }

	m_reduceSumKernel.Execute(
		commandList,
		m_cbvSrvUavHeap->GetHeap(),
		frameIndex,
        inputResource->gpuDescriptorReadAccess,
		&m_numCameraRayGeometryHits);
};

void D3D12RaytracingAmbientOcclusion::ApplyAtrousWaveletTransformFilter()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    RWGpuResource* AOResources = m_RTAO.AOResources();
    RWGpuResource* AOTSSCoefficient = SceneArgs::QuarterResAO ? m_lowResAOTSSCoefficient : m_AOTSSCoefficient;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    
    RWGpuResource& NormalDepthLowPrecisionResource = SceneArgs::QuarterResAO ? 
            m_normalDepthLowResLowPrecision[m_normalDepthCurrentFrameResourceIndex]
        :   m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex];
    
    ScopedTimer _prof(L"DenoiseAO", commandList);

    // Transition Smoothed AO to UAV.
    {
        D3D12_RESOURCE_BARRIER barriers[] = {
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
           AOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex].gpuDescriptorReadAccess,
            NormalDepthLowPrecisionResource.gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::Distance].gpuDescriptorReadAccess,
#if PACK_MEAN_VARIANCE
            m_smoothedMeanVarianceResource.gpuDescriptorReadAccess,
#else
            m_smoothedVarianceResource.gpuDescriptorReadAccess,
#endif
            AOResources[AOResource::RayHitDistance].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            &AOResources[AOResource::Smoothed],
            SceneArgs::AODenoiseValueSigma,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            // ToDo rename this to be global normalDepth
            static_cast<TextureResourceFormatRGB::Type>(static_cast<UINT>(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat)),
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
            SceneArgs::QuarterResAO);
    }

    // Transition the output resource to SRV.
    {
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::Smoothed].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
    }
};


// Apply multi scale denoising to denoise low frequencies.
// Ref: Delbracio et al. 2014, Boosting Monte Carlo Rendering by Ray Histogram Fusion
void D3D12RaytracingAmbientOcclusion::ApplyMultiScaleAtrousWaveletTransformFilter()
{
#if 1 
    ThrowIfFalse(false, L"ToDo");
#else
    auto commandList = m_deviceResources->GetCommandList();

    RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOLowResResources : m_AOResources;
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    
    ScopedTimer _prof(L"MultiScaleAtrousWaveletTransform", commandList);

    // ToDo remove this since the next iter is downsampling.
    // Downsample input value.
    {
        ScopedTimer _prof(L"DownsampleValueBuffers", commandList);

        RWGpuResource* inputResources[] = { &AOResources[AOResource::TSSCoefficient], &NormalDepthLowPrecisionResource };
      
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
                    CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::TSSCoefficient].resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    CD3DX12_RESOURCE_BARRIER::Transition(NormalDepthLowPrecisionResource.resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
                    CD3DX12_RESOURCE_BARRIER::Transition(GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE)
                };
                commandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

                commandList->CopyResource(msResource.m_value.resource.Get(), AOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex].resource.Get());
                commandList->CopyResource(msResource.m_normalDepth.resource.Get(), NormalDepthLowPrecisionResource.resource.Get());
                commandList->CopyResource(msResource.m_partialDistanceDerivatives.resource.Get(), GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get());

                D3D12_RESOURCE_BARRIER postCopyBarriers[] = {
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_value.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_normalDepth.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    CD3DX12_RESOURCE_BARRIER::Transition(msResource.m_partialDistanceDerivatives.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    CD3DX12_RESOURCE_BARRIER::Transition(AOResources[AOResource::TSSCoefficient].resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
                    CD3DX12_RESOURCE_BARRIER::Transition(NormalDepthLowPrecisionResource.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
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
#endif
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
    UINT calculateVarianceTimerId,      // ToDo remove obsolete
    UINT smoothVarianceTimerId,
    UINT atrousFilterTimerId
)
{
    auto commandList = m_deviceResources->GetCommandList();
    
    auto desc = inValueResource.resource.Get()->GetDesc();
    // ToDo cleanup widths on GPU kernels, it should be the one of input resource.
    UINT width = static_cast<UINT>(desc.Width);
    UINT height = static_cast<UINT>(desc.Height);

#if 0
    // Calculate local variance.
    {
        ScopedTimer _prof(L"CalculateVariance", commandList);
        m_calculateVarianceKernel.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            width,
            height,
            static_cast<GpuKernels::CalculateVariance::FilterType>(static_cast<UINT>(SceneArgs::VarianceBilateralFilter)),
            inValueResource.gpuDescriptorReadAccess,
            inNormalDepthResource.gpuDescriptorReadAccess,
            inDepthResource.gpuDescriptorReadAccess,
            varianceResource->gpuDescriptorWriteAccess,
            CD3DX12_GPU_DESCRIPTOR_HANDLE(),    // unused mean resource output
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            false,
            SceneArgs::RTAODenoising_Variance_UseDepthWeights,
            SceneArgs::RTAODenoising_Variance_UseNormalWeights,
            SceneArgs::RTAOVarianceFilterKernelWidth);

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
            smoothedVarianceResource->gpuDescriptorWriteAccess);
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
#endif

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
            smoothedVarianceResource->gpuDescriptorReadAccess,
            inRayHitDistanceResource.gpuDescriptorReadAccess,
            inPartialDistanceDerivativesResource.gpuDescriptorReadAccess,
            outSmoothedValueResource,
            SceneArgs::AODenoiseValueSigma,
            SceneArgs::AODenoiseDepthSigma,
            SceneArgs::AODenoiseNormalSigma,
            static_cast<TextureResourceFormatRGB::Type>(static_cast<UINT>(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat)),
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
            SceneArgs::QuarterResAO);
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
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();        // ToDo rename to Backbuffer index

    ScopedTimer _prof(L"GenerateGbuffer", commandList);


    m_normalDepthCurrentFrameResourceIndex = (m_normalDepthCurrentFrameResourceIndex + 1) % 2;
    RWGpuResource& NormalDepthLowPrecisionResource = m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex];
    
    m_sceneCB->useDiffuseFromMaterial = SceneArgs::CompositionMode == CompositionType::Diffuse;
    m_sceneCB->doShading = SceneArgs::CompositionMode == CompositionType::PhongLighting;
#if CAMERA_JITTER

#if 1
    uniform_real_distribution<float> jitterDistribution(-0.5f, 0.5f);
    m_sceneCB->cameraJitter = XMFLOAT2(jitterDistribution(m_generatorURNG), jitterDistribution(m_generatorURNG));
#else
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
#endif



    // ToDo should we use cameraAtPosition0 too and offset the world space pos vector in the shader?
    XMMATRIX prevView, prevProj;
    m_prevFrameCamera.GetViewProj(&prevView, &prevProj, m_GBufferWidth, m_GBufferHeight);
    m_sceneCB->prevViewProj = prevView * prevProj;
    m_sceneCB->prevCameraPosition = m_prevFrameCamera.Eye();

    // ToDo cleanup
    XMMATRIX prevView0 = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 1), XMVectorSetW(m_prevFrameCamera.At() - m_prevFrameCamera.Eye(), 1), m_prevFrameCamera.Up());
    XMMATRIX viewProj0 = prevView0 * prevProj;
    m_sceneCB->prevProjToWorldWithCameraEyeAtOrigin = XMMatrixInverse(nullptr, viewProj0);


    m_sceneCB->raytracingDim = XMUINT2(m_raytracingWidth, m_raytracingHeight);
	commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
	commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

	// Copy dynamic buffers to GPU.
	{
		// ToDo copy on change
		m_sceneCB.CopyStagingToGpu(frameIndex);
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
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::MotionVector].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::ReprojectedHitPosition].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Color].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::AOSurfaceAlbedo].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(NormalDepthLowPrecisionResource.resource.Get(), before, after),
            // ToDo remove
            //CD3DX12_RESOURCE_BARRIER::Transition(m_varianceResource.resource.Get(), before, after) ,
            //CD3DX12_RESOURCE_BARRIER::Transition(m_smoothedVarianceResource.resource.Get(), before, after)
		};
		commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
	}


	// Bind inputs.
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress());
	commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::MaterialBuffer, m_materialBuffer.GpuVirtualAddress());
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::EnvironmentMap, m_environmentMap.gpuDescriptorHandle);
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::PrevFrameBottomLevelASIstanceTransforms, m_prevFrameBottomLevelASInstanceTransforms.GpuVirtualAddress(frameIndex));
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::ShadowMapSRV, m_ShadowMapResource.gpuDescriptorReadAccess);

    
	// Bind output RTs.
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResources, m_GBufferResources[0].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferDepth, m_GBufferResources[GBufferResource::Depth].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::NormalDepthLowPrecision, NormalDepthLowPrecisionResource.gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GbufferNormalRGB, m_GBufferResources[GBufferResource::SurfaceNormalRGB].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::MotionVector, m_GBufferResources[GBufferResource::MotionVector].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::ReprojectedHitPosition, m_GBufferResources[GBufferResource::ReprojectedHitPosition].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::Color, m_GBufferResources[GBufferResource::Color].gpuDescriptorWriteAccess);
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOSurfaceAlbedo, m_GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorWriteAccess);
    
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
            CD3DX12_RESOURCE_BARRIER::Transition(NormalDepthLowPrecisionResource.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::SurfaceNormalRGB].resource.Get(), before, after),
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::PartialDepthDerivatives].resource.Get(), before, after),
#endif
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::MotionVector].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::ReprojectedHitPosition].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::Color].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferResources[GBufferResource::AOSurfaceAlbedo].resource.Get(), before, after),
		};
		commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
	}

    // Calculate ray hit counts.
    {
        ScopedTimer _prof(L"CalculateCameraRayHitCount", commandList);
        CalculateCameraRayHitCount();
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
            m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess);

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
    RWGpuResource& NormalDepthLowPrecisionResource = m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex];
    RWGpuResource& NormalDeptLowResLowPrecisionResource = m_normalDepthLowResLowPrecision[m_normalDepthCurrentFrameResourceIndex];

    // ToDo move this/part(AO,..) of transitions out?
    // Transition all output resources to UAV state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::Hit].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::HitPosition].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::MotionVector].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::ReprojectedHitPosition].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::Depth].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::SurfaceNormal].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(NormalDeptLowResLowPrecisionResource.resource.Get(), before, after),
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    ScopedTimer _prof(L"DownsampleGBuffer", commandList);
    m_downsampleGBufferBilateralFilterKernel.Execute(
        commandList,
        m_GBufferWidth,
        m_GBufferHeight,
        m_cbvSrvUavHeap->GetHeap(),
        NormalDepthLowPrecisionResource.gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::HitPosition].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::Hit].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::MotionVector].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::ReprojectedHitPosition].gpuDescriptorReadAccess,
        m_GBufferResources[GBufferResource::Depth].gpuDescriptorReadAccess,
        m_GBufferLowResResources[GBufferResource::SurfaceNormal].gpuDescriptorWriteAccess,
        NormalDeptLowResLowPrecisionResource.gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::HitPosition].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::Hit].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::MotionVector].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::ReprojectedHitPosition].gpuDescriptorWriteAccess,
        m_GBufferLowResResources[GBufferResource::Depth].gpuDescriptorWriteAccess);
    // Transition GBuffer resources to shader resource state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::Hit].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::HitPosition].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::SurfaceNormal].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(NormalDeptLowResLowPrecisionResource.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::PartialDepthDerivatives].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::MotionVector].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_GBufferLowResResources[GBufferResource::ReprojectedHitPosition].resource.Get(), before, after),
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
    case CompositionType::AmbientOcclusionOnly_Denoised:
    case CompositionType::AmbientOcclusionOnly_TemporallySupersampled:
    case CompositionType::AmbientOcclusionOnly_RawOneFrame:
    {
        passName = L"Upsample AO";
        if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
        {
            outputHiResValueResource = &m_AOResources[AOResource::Coefficient];
        }
        else if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_TemporallySupersampled)
        {
            outputHiResValueResource = &m_AOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex];
        }
        else
        {
            outputHiResValueResource = &m_AOResources[AOResource::Smoothed];
        }
        
        if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
        {
            inputLowResValueResource = &m_RTAO.AOResources()[AOResource::Coefficient];
        }
        else if (SceneArgs::RTAODenoisingUseMultiscale)
        {
            inputLowResValueResource = &m_multiScaleDenoisingResources[0].m_value;
        }
        else if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_TemporallySupersampled)
        {
            inputLowResValueResource = &m_lowResAOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex];
        }
        else
        {
            inputLowResValueResource = &m_RTAO.AOResources()[AOResource::Smoothed];
        }
        break;
    }
    case CompositionType::AmbientOcclusionHighResSamplingPixels:
    {
        // ToDo rename all to importance map
        passName = L"Upsample AO sampling importance map";
        inputLowResValueResource = &m_RTAO.AOResources()[AOResource::FilterWeightSum];
        outputHiResValueResource = &m_AOResources[AOResource::FilterWeightSum];
        break;
    }
    case CompositionType::RTAOHitDistance:
    {
        passName = L"Upsample AO ray hit distance";
        inputLowResValueResource = &m_RTAO.AOResources()[AOResource::RayHitDistance];
        outputHiResValueResource = &m_AOResources[AOResource::RayHitDistance];
        break;
    }
    default:
        break;
    }

    if (inputLowResValueResource)
    {
        BilateralUpsample(
            m_GBufferWidth,
            m_GBufferHeight,
            inputLowResValueResource->gpuDescriptorReadAccess,
            m_normalDepthLowResLowPrecision[m_normalDepthCurrentFrameResourceIndex].gpuDescriptorReadAccess,
            m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex].gpuDescriptorReadAccess,
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
    ThrowIfFalse(false, L"ToDo/Remove");
#if 0

	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"Shadows", commandList);

    // Transition the shadow resource to UAV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_VisibilityResource.resource.Get(), before, after));
    }

    commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
	commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

	// Bind inputs.
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::GBufferResourcesIn, m_GBufferResources[0].gpuDescriptorReadAccess);
	commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));
	
	// Bind output RT.
	commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::VisibilityResource, m_VisibilityResource.gpuDescriptorWriteAccess);

	// Bind the heaps, acceleration structure and dispatch rays. 
	commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress());

    DispatchRays(m_rayGenShaderTables[RayGenShaderType::Visibility].Get());
    
	// Transition shadow resources to shader resource state.
	{
		D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
		D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
		commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_VisibilityResource.resource.Get(), before, after));
	}

	PIXEndEvent(commandList);
#endif
}


void D3D12RaytracingAmbientOcclusion::RenderPass_GenerateShadowMap()
{
    if (!m_updateShadowMap)
    {
        return;
    }
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"Shadow Map", commandList);
    
    
    // ToDo tighten the fov.
    // reuse the same camera
    GameCore::Camera lightViewCamera = m_camera;

    XMMATRIX proj;
    lightViewCamera.fov = 90;
    lightViewCamera.GetProj(&proj, c_shadowMapDim.x, c_shadowMapDim.y);

    // Calculate view matrix as if the eye was at (0,0,0) to avoid 
    // precision issues when camera position is too far from (0,0,0).
    // GenerateCameraRay takes this into consideration in the raytracing shader.
    XMVECTOR eye = m_sceneCB->lightPosition;
    XMVECTOR at = XMVectorSetY(eye, 0);         // pointing down
    XMVECTOR up = XMVectorSet(1, 0, 0, 0);
    XMMATRIX viewAtOrigin = XMMatrixLookAtLH(XMVectorSet(0, 0, 0, 1), XMVectorSetW(at - eye, 1), up);
    XMMATRIX viewProjAtOrigin = viewAtOrigin * proj;
    m_sceneCB->lightProjectionToWorldWithCameraEyeAtOrigin = XMMatrixInverse(nullptr, viewProjAtOrigin);
    
    // ToDo cleanup
    XMMATRIX view = XMMatrixLookAtLH(eye, at, up);
    XMMATRIX viewProj = view * proj;
    m_sceneCB->lightViewProj = viewProj;

    // Todo same m_sceneCB is copied multiple times.
    m_sceneCB.CopyStagingToGpu(frameIndex);

    // Transition shadow map resourcee to UAV.
    commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_ShadowMapResource.resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ));

    commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
    commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

    // Bind inputs.
    commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));

    // Bind output RT.
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::ShadowMapUAV, m_ShadowMapResource.gpuDescriptorWriteAccess);

    // Bind the heaps, acceleration structure and dispatch rays. 
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress());

    DispatchRays(m_rayGenShaderTables[RayGenShaderType::ShadowMap].Get(), c_shadowMapDim.x, c_shadowMapDim.y);

    // Transition shadow resources to shader resource state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_ShadowMapResource.resource.Get(), before, after),
            // Make sure the resource is done being written to.
            CD3DX12_RESOURCE_BARRIER::UAV(m_ShadowMapResource.resource.Get())   
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);   
    }
    

    // ToDo add UAV barriers

    m_updateShadowMap = false;
    PIXEndEvent(commandList);
}

void D3D12RaytracingAmbientOcclusion::RenderPass_TestEarlyExitOVerhead()
{
#if 1
    ThrowIfFalse(false, L"Dead code");
#else
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();


    D3D12_GPU_DESCRIPTOR_HANDLE resourceHandle;
    m_writeValueToTexture.Execute(commandList, m_cbvSrvUavHeap->GetHeap(), &resourceHandle);

    commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
    commandList->SetComputeRootSignature(m_raytracingGlobalRootSignature.Get());

    m_sceneCB.CopyStagingToGpu(frameIndex);

    // Bind inputs.
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::AOFrameAge, resourceHandle);
    commandList->SetComputeRootConstantBufferView(GlobalRootSignature::Slot::SceneConstant, m_sceneCB.GpuVirtualAddress(frameIndex));

    // Bind output RT.
    commandList->SetComputeRootDescriptorTable(GlobalRootSignature::Slot::VisibilityResource, m_VisibilityResource.gpuDescriptorWriteAccess);

    // Bind the heaps, acceleration structure and dispatch rays. 
    commandList->SetComputeRootShaderResourceView(GlobalRootSignature::Slot::AccelerationStructure, m_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress());

    DispatchRays(m_rayGenShaderTables[RayGenShaderType::Visibility].Get());

    UINT width = SceneArgs::TestEarlyExit_TightScheduling ? 3840 / 2 : 3840;
    UINT height = 2160;

    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    dispatchDesc.HitGroupTable.StartAddress = m_hitGroupShaderTable->GetGPUVirtualAddress();
    dispatchDesc.HitGroupTable.SizeInBytes = m_hitGroupShaderTable->GetDesc().Width;
    dispatchDesc.HitGroupTable.StrideInBytes = m_hitGroupShaderTableStrideInBytes;
    dispatchDesc.MissShaderTable.StartAddress = m_missShaderTable->GetGPUVirtualAddress();
    dispatchDesc.MissShaderTable.SizeInBytes = m_missShaderTable->GetDesc().Width;
    dispatchDesc.MissShaderTable.StrideInBytes = m_missShaderTableStrideInBytes;
    dispatchDesc.RayGenerationShaderRecord.StartAddress = m_rayGenShaderTables[RayGenShaderType::Visibility]->GetGPUVirtualAddress();
    dispatchDesc.RayGenerationShaderRecord.SizeInBytes = m_rayGenShaderTables[RayGenShaderType::Visibility]->GetDesc().Width;
    dispatchDesc.Width = width;
    dispatchDesc.Height = height;
    dispatchDesc.Depth = 1;
    commandList->SetPipelineState1(m_dxrStateObject.Get());
    
    {
        ScopedTimer _prof(L"TestEarlyExit", commandList);
        for (UINT i = 0; i < 1000; i++)
        {
            commandList->DispatchRays(&dispatchDesc);
        }
    }
    // Transition shadow resources to shader resource state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_VisibilityResource.resource.Get(), before, after));
    }

    PIXEndEvent(commandList);
#endif
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

    RWGpuResource* AOResources = m_RTAO.AOResources();
    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;

    RWGpuResource& NormalDepthLowPrecisionResource = SceneArgs::QuarterResAO ?
            m_normalDepthLowResLowPrecision[m_normalDepthCurrentFrameResourceIndex]
        : m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex];

	commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
	commandList->SetComputeRootSignature(m_computeRootSigs[CSType::AoBlurCS].Get());
	commandList->SetPipelineState(m_computePSOs[SceneArgs::QuarterResAO ? CSType::AoBlurAndUpsampleCS : CSType::AoBlurCS].Get());
	commandList->SetComputeRootDescriptorTable(Slot::Normal, NormalDepthLowPrecisionResource.gpuDescriptorReadAccess);
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
        m_csComposeRenderPassesCB->defaultAmbientIntensity = SceneArgs::DefaultAmbientIntensity;

#if 0 // ToDo
        // ToDo use a unique CB for compose passes?
        m_csComposeRenderPassesCB->RTAO_UseAdaptiveSampling = SceneArgs::RTAOAdaptiveSampling;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMaxWeightSum = SceneArgs::RTAOAdaptiveSamplingMaxFilterWeight;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMinMaxSampling = SceneArgs::RTAOAdaptiveSamplingMinMaxSampling;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingScaleExponent = SceneArgs::RTAOAdaptiveSamplingScaleExponent;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMinSamples = SceneArgs::RTAOAdaptiveSamplingMinSamples;
        m_csComposeRenderPassesCB->RTAO_MaxRayHitDistance = SceneArgs::RTAOMaxRayHitTime;
        m_csComposeRenderPassesCB->RTAO_MaxSPP = SceneArgs::AOSampleCountPerDimension * SceneArgs::AOSampleCountPerDimension;
#endif
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
        commandList->SetComputeRootDescriptorTable(Slot::Color, m_GBufferResources[GBufferResource::Color].gpuDescriptorReadAccess);
        commandList->SetComputeRootDescriptorTable(Slot::AOSurfaceAlbedo, m_GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorReadAccess);

        
        commandList->SetComputeRootDescriptorTable(Slot::FrameAge, m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalCache::FrameAge].gpuDescriptorReadAccess);
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
		float numAOGigaRays = 1e-6f * m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits] * (SceneArgs::QuarterResAO ? 0.25f : 1) * m_sppAO / GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_AO);
		wLabel << fixed << L" AORay DispatchRays: " << GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_AO) << L"ms  ~" <<	numAOGigaRays << " GigaRay/s\n";
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
        m_GBufferWidth = c_SupersamplingScale * m_width;// +1;        // ToDo remove +1
        m_GBufferHeight = c_SupersamplingScale * m_height;// +1;
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
    m_RTAO.SetResolution(m_raytracingWidth, m_raytracingHeight);

	m_reduceSumKernel.CreateInputResourceSizeDependentResources(
		device,
		m_cbvSrvUavHeap.get(), 
		FrameCount, 
		m_GBufferWidth,
		m_GBufferHeight);
    m_atrousWaveletTransformFilter.CreateInputResourceSizeDependentResources(device, m_cbvSrvUavHeap.get(), m_raytracingWidth, m_raytracingHeight);
    
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
    m_dxrStateObject.Reset();

    m_raytracingGlobalRootSignature.Reset();
    ResetComPtrArray(&m_raytracingLocalRootSignature);

    m_cbvSrvUavHeap.reset();
    m_csHemisphereVisualizationCB.Release();

    m_RTAO.ReleaseDeviceDependentResources();

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
    commandList->Dispatch(rngWindowSize.x, rngWindowSize.y, 1);
#endif
}



void D3D12RaytracingAmbientOcclusion::CreateSamplesRNGVisualization()
{
#if 0
    auto device = m_deviceResources->GetD3DDevice();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    UINT samplesPerSet = m_sppAO * SceneArgs::AOSampleSetDistributedAcrossPixels * SceneArgs::AOSampleSetDistributedAcrossPixels;
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


void D3D12RaytracingAmbientOcclusion::RenderPass_TemporalCacheReverseProjection()
{
    auto commandList = m_deviceResources->GetCommandList();

    ScopedTimer _prof(L"Temporal Cache Reverse Reprojection", commandList);

    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
    RWGpuResource* AOResources = m_RTAO.AOResources();
    RWGpuResource& NormalDepthLowPrecisionResource = SceneArgs::QuarterResAO ?
        m_normalDepthLowResLowPrecision[m_normalDepthCurrentFrameResourceIndex]
        : m_normalDepthLowPrecision[m_normalDepthCurrentFrameResourceIndex];

    UINT prevFrameNormalDepthResourceIndex = (m_normalDepthCurrentFrameResourceIndex + 1) % 2;
    RWGpuResource& PreviousFrameNormalDeptLowPrecisionResource = SceneArgs::QuarterResAO ?
        m_normalDepthLowResLowPrecision[prevFrameNormalDepthResourceIndex]
        : m_normalDepthLowPrecision[prevFrameNormalDepthResourceIndex];


    
    UINT temporalCachePreviousFrameResourceIndex = m_temporalCacheCurrentFrameResourceIndex;
    m_temporalCacheCurrentFrameResourceIndex = (m_temporalCacheCurrentFrameResourceIndex + 1) % 2;

    RWGpuResource* AOTSSCoefficient = SceneArgs::QuarterResAO ? m_lowResAOTSSCoefficient : m_AOTSSCoefficient;

    // ToDo remove
    if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
    {
        // ToDo
        //m_temporalCacheFrameAge = 0;
    }

    // ToDo zero out caches on resource reset.

    // ToDo reuse calculated variance for both TAO and denoising.
    // Transition all output resources to UAV state.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_meanVarianceResource.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_smoothedMeanVarianceResource.resource.Get(), before, after),
            
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    // ToDO Should use separable box filter instead?. Bilateral doesn't work for pixels that don't
    // have anycontribution with bilateral - their variance will be zero. Or set a variance to non-zero in that case?
    // Calculate local mean and variance.
    {
        // ToDo add Separable Bilateral and Square bilateral support how it affects image quality.
        ScopedTimer _prof(L"Calculate Mean and Variance", commandList);
        m_calculateMeanVarianceKernel.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            m_raytracingWidth,
            m_raytracingHeight,
            GpuKernels::CalculateMeanVariance::FilterType::Separable_AnyToAnyWaveReadLaneAt,
            AOResources[AOResource::Coefficient].gpuDescriptorReadAccess,
            m_meanVarianceResource.gpuDescriptorWriteAccess,
            SceneArgs::VarianceBilateralFilterKernelWidth);

        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER barriers[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_meanVarianceResource.resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::UAV(m_meanVarianceResource.resource.Get())  // ToDo
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }

    // Smoothen the local variance which is prone to error due to undersampled input.
    {
        {
            ScopedTimer _prof(L"Mean Variance Smoothing", commandList);
            m_gaussianSmoothingKernel.Execute(
                commandList,
                m_raytracingWidth,
                m_raytracingHeight,
                GpuKernels::GaussianFilter::FilterRG3X3,
                m_cbvSrvUavHeap->GetHeap(),
                m_meanVarianceResource.gpuDescriptorReadAccess,
                m_smoothedMeanVarianceResource.gpuDescriptorWriteAccess);
        }
    }

    D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    D3D12_RESOURCE_BARRIER barriers[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_smoothedMeanVarianceResource.resource.Get(), before, after),
                CD3DX12_RESOURCE_BARRIER::UAV(m_smoothedMeanVarianceResource.resource.Get())  // ToDo
    };
    commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);


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
    XMMATRIX cameraTranslation = XMMatrixTranslationFromVector(m_camera.Eye() - m_prevFrameCamera.Eye());
    XMVECTOR prevToCurrentFrameCameraTranslation = m_camera.Eye() - m_prevFrameCamera.Eye();

    XMMATRIX invView = XMMatrixInverse(nullptr, view);
    XMMATRIX invProj = XMMatrixInverse(nullptr, proj);

    XMMATRIX viewProj = view * proj;
    XMMATRIX prevViewProj = prevView * prevProj;
    XMMATRIX invViewProj = XMMatrixInverse(nullptr, viewProj);
    XMMATRIX prevInvViewProj = XMMatrixInverse(nullptr, prevViewProj);

#if 1
    XMMATRIX invViewProjAndCameraTranslation = invViewProj * cameraTranslation;
   // XMMATRIX reverseProjectionTransform = invViewProj * cameraTranslation * prevView * prevProj;
    XMMATRIX reverseProjectionTransform = invViewProjAndCameraTranslation * prevView * prevProj;
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
            CD3DX12_RESOURCE_BARRIER::Transition(AOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalCache::FrameAge].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_debugOutput[0].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_debugOutput[1].resource.Get(), before, after)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }


    m_temporalCacheReverseReprojectKernel.Execute(
        commandList,
        m_raytracingWidth,
        m_raytracingHeight,
        m_cbvSrvUavHeap->GetHeap(),
        AOResources[AOResource::Coefficient].gpuDescriptorReadAccess,
        NormalDepthLowPrecisionResource.gpuDescriptorReadAccess,
#if PACK_MEAN_VARIANCE
        m_smoothedMeanVarianceResource.gpuDescriptorReadAccess,
#else
        m_smoothedVarianceResource.gpuDescriptorReadAccess,
        m_smoothedMeanResource.gpuDescriptorReadAccess,
#endif
        GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
        AOTSSCoefficient[temporalCachePreviousFrameResourceIndex].gpuDescriptorReadAccess,
        PreviousFrameNormalDeptLowPrecisionResource.gpuDescriptorReadAccess,
        m_temporalCache[temporalCachePreviousFrameResourceIndex][TemporalCache::FrameAge].gpuDescriptorReadAccess,
        GBufferResources[GBufferResource::MotionVector].gpuDescriptorReadAccess,
        GBufferResources[GBufferResource::ReprojectedHitPosition].gpuDescriptorReadAccess,
        AOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex].gpuDescriptorWriteAccess,
        m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalCache::FrameAge].gpuDescriptorWriteAccess,
        SceneArgs::RTAO_TemporalCache_MinSmoothingFactor,
        invView,
        invProj,
        invViewProjAndCameraTranslation,
        reverseProjectionTransform,
        prevInvViewProj,
        m_camera.ZMin,
        m_camera.ZMax,
        SceneArgs::RTAO_TemporalCache_DepthTolerance,
        SceneArgs::RTAO_TemporalCache_UseDepthWeights,
        SceneArgs::RTAO_TemporalCache_UseNormalWeights,
        SceneArgs::RTAO_TemporalCache_ForceUseMinSmoothingFactor,
        SceneArgs::RTAO_TemporalCache_ClampCachedValues_UseClamping,
        SceneArgs::RTAO_TemporalCache_ClampCachedValues_StdDevGamma,
        SceneArgs::RTAO_TemporalCache_ClampCachedValues_MinStdDevTolerance,
        SceneArgs::RTAO_TemporalCache_ClampCachedValues_AbsoluteDepthTolerance,
        SceneArgs::RTAO_TemporalCache_ClampCachedValues_DepthBasedDepthTolerance,
        SceneArgs::RTAO_TemporalCache_ClampCachedValues_DepthSigma,
        SceneArgs::RTAO_TemporalCache_UseWorldSpaceDistance,
        static_cast<TextureResourceFormatRGB::Type>(static_cast<UINT>(SceneArgs::RTAO_TemporalCache_NormalDepthResourceFormat)),
        m_debugOutput,
        m_camera.Eye(),
        invViewProj,
        prevToCurrentFrameCameraTranslation,
        prevInvViewProj);

    // Transition output resource to SRV state.        
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_BARRIER barriers[] = {
            
            CD3DX12_RESOURCE_BARRIER::Transition(AOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalCache::FrameAge].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_debugOutput[0].resource.Get(), before, after),
            CD3DX12_RESOURCE_BARRIER::Transition(m_debugOutput[1].resource.Get(), before, after)
            // ToDo UAVs
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }
}

void GetGrassParameters(GenerateGrassStrawsConstantBuffer_AppParams* params, UINT LOD, float totalTime)
{
    params->activePatchDim = XMUINT2(
        g_UIparameters.GrassGeometryLOD[LOD].NumberStrawsX,
        g_UIparameters.GrassGeometryLOD[LOD].NumberStrawsZ);
    params->maxPatchDim = XMUINT2(MAX_GRASS_STRAWS_1D, MAX_GRASS_STRAWS_1D);

    params->timeOffset = XMFLOAT2(
        totalTime * g_UIparameters.GrassCommon.WindMapSpeedU * g_UIparameters.GrassGeometryLOD[LOD].WindFrequency,
        totalTime * g_UIparameters.GrassCommon.WindMapSpeedV * g_UIparameters.GrassGeometryLOD[LOD].WindFrequency);

    params->grassHeight = g_UIparameters.GrassGeometryLOD[LOD].StrawHeight;
    params->grassScale = g_UIparameters.GrassGeometryLOD[LOD].StrawScale;
    params->bendStrengthAlongTangent = g_UIparameters.GrassGeometryLOD[LOD].BendStrengthSideways;

    params->patchSize = XMFLOAT3(   // ToDO rename to scale?
        g_UIparameters.GrassCommon.PatchWidth,
        g_UIparameters.GrassCommon.PatchHeight,
        g_UIparameters.GrassCommon.PatchWidth);

    params->grassThickness = g_UIparameters.GrassGeometryLOD[LOD].StrawThickness;
    params->windDirection = XMFLOAT3(0, 0, 0); // ToDo
    params->windStrength = g_UIparameters.GrassGeometryLOD[LOD].WindStrength;
    params->positionJitterStrength = g_UIparameters.GrassGeometryLOD[LOD].RandomPositionJitterStrength;


}

void D3D12RaytracingAmbientOcclusion::GenerateGrassGeometry()
{
#if !GENERATE_GRASS
    return;
#endif
    auto commandList = m_deviceResources->GetCommandList();
    float totalTime = 0;// static_cast<float>(m_timer.GetTotalSeconds());

    m_currentGrassPatchVBIndex = (m_currentGrassPatchVBIndex + 1) % 2;

    // Update all LODs.
    for (UINT i = 0; i < UIParameters::NumGrassGeometryLODs; i++)
    {
        GenerateGrassStrawsConstantBuffer_AppParams params;
        GetGrassParameters(&params, i, totalTime);

        UINT vbID = m_currentGrassPatchVBIndex & 1;
        auto& grassPatchVB = m_grassPatchVB[i][vbID];

        // Transition output vertex buffer to UAV state and make sure the resource is done being read from.      
        {
            D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            D3D12_RESOURCE_BARRIER barriers[] = {
                CD3DX12_RESOURCE_BARRIER::Transition(grassPatchVB.resource.Get(), before, after),
                CD3DX12_RESOURCE_BARRIER::UAV(grassPatchVB.resource.Get())  // ToDo
            };
            commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
        }

        m_grassGeometryGenerator.Execute(commandList, params, m_cbvSrvUavHeap->GetHeap(), grassPatchVB.gpuDescriptorWriteAccess);

        // Transition the output vertex buffer to VB state and make sure the CS is done writing.        
        {
            D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            D3D12_RESOURCE_BARRIER barriers[] = {
                CD3DX12_RESOURCE_BARRIER::Transition(grassPatchVB.resource.Get(), before, after),
                CD3DX12_RESOURCE_BARRIER::UAV(grassPatchVB.resource.Get())
            };
            commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
        }

        // Point bottom-levelAS VB pointer to the updated VB.
        auto& bottomLevelAS = m_accelerationStructure->GetBottomLevelAS(L"Grass Patch LOD " + to_wstring(i));
        auto& geometryDesc = bottomLevelAS.GetGeometryDescs()[0];
        geometryDesc.Triangles.VertexBuffer.StartAddress = grassPatchVB.resource->GetGPUVirtualAddress();
        bottomLevelAS.SetDirty(true);
    }

    // Update bottom-level AS instances.
    {
        // Enumerate all hit contribution indices for grass bottom-level acceleration structures.
        BottomLevelAccelerationStructure* grassBottomLevelAS[UIParameters::NumGrassGeometryLODs];

        for (UINT i = 0; i < UIParameters::NumGrassGeometryLODs; i++)
        {
            grassBottomLevelAS[i] = &m_accelerationStructure->GetBottomLevelAS(L"Grass Patch LOD " + to_wstring(i));
        }


        std::mt19937 m_generatorURNG;  // Uniform random number generator
        m_generatorURNG.seed(1729);
        uniform_real_distribution<float> unitSquareDistributionInclusive(0.f, nextafter(1.f, FLT_MAX));
        function<float()> GetRandomFloat01inclusive = bind(unitSquareDistributionInclusive, ref(m_generatorURNG));

        XMVECTOR baseIndex = XMVectorSet(0, 0, 2, 0);
        XMVECTOR patchOffset = XMLoadFloat3(&g_UIparameters.GrassCommon.PatchOffset);
        float width = g_UIparameters.GrassCommon.PatchWidth;

#if GRASS_NO_DEGENERATE_INSTANCES
        UINT grassInstanceIndex = 0;
#endif
        for (int i = 0; i < NumGrassPatchesZ; i++)
            for (int j = 0; j < NumGrassPatchesX; j++)
            {
                int z = i - 15;
                int x = j - 15;

                if ((x < -1 || x > 2 || z < -2 || z > 1) &&
                    (IsInRange(x, -2, 3) && IsInRange(z, -3, 2)))

                {
#if !GRASS_NO_DEGENERATE_INSTANCES
                    UINT grassInstanceIndex = i * NumGrassPatchesX + j;
#endif

                    auto& BLASinstance = m_accelerationStructure->GetBottomLevelASInstance(m_grassInstanceIndices[grassInstanceIndex]);
                    
                    float jitterX = 2 * GetRandomFloat01inclusive() - 1;
                    float jitterZ = 2 * GetRandomFloat01inclusive() - 1;
                    XMVECTOR position = patchOffset + width * (baseIndex + XMVectorSet(static_cast<float>(x), 0, static_cast<float>(z), 0) + 0.01f * XMVectorSet(jitterX, 0, jitterZ, 0));
                    XMMATRIX transform = XMMatrixTranslationFromVector(position);
                    BLASinstance.SetTransform(transform);

                    // Find the LOD for this instance based on the distance from the camera.
                    XMVECTOR centerPosition = position + XMVectorSet(0.5f * width, 0, 0.5f * width, 0);
                    float approxDistanceToCamera = max(0.f, XMVectorGetX(XMVector3Length((centerPosition - m_camera.Eye()))) - 0.5f * width );
                    UINT LOD = UIParameters::NumGrassGeometryLODs - 1;
                    if (!g_UIparameters.GrassCommon.ForceLOD0)
                    {
                        for (UINT i = 0; i < UIParameters::NumGrassGeometryLODs - 1; i++)
                        {
                            if (approxDistanceToCamera <= g_UIparameters.GrassGeometryLOD[i].MaxLODdistance)
                            {
                                LOD = i;
                                break;
                            }
                        }
                    }
                    else
                    {
                        LOD = 0;
                    }

                    auto GetShaderRecordIndexOffset = [&](UINT* outShaderRecordIndexOffset, UINT instanceIndex, UINT LOD, UINT prevFrameLOD)
                    {
                        UINT baseShaderRecordID = grassBottomLevelAS[LOD]->GetInstanceContributionToHitGroupIndex();

                        UINT NumTransitionTypes = 3;
                        UINT transitionType;
                        if (LOD > prevFrameLOD) transitionType = 0;
                        else if (LOD == prevFrameLOD) transitionType = 1;
                        else transitionType = 2;
                        UINT NumShaderRecordsPerHitGroup = RayType::Count;

                        *outShaderRecordIndexOffset = baseShaderRecordID + (m_currentGrassPatchVBIndex * NumTransitionTypes + transitionType) * NumShaderRecordsPerHitGroup;
                    };

                    UINT shaderRecordIndexOffset;
                    GetShaderRecordIndexOffset(&shaderRecordIndexOffset, grassInstanceIndex, LOD, m_prevFrameLODs[grassInstanceIndex]);

                    // Point the instance at BLAS at the LOD.
                    BLASinstance.InstanceContributionToHitGroupIndex = shaderRecordIndexOffset;
                    BLASinstance.AccelerationStructure = grassBottomLevelAS[LOD]->GetResource()->GetGPUVirtualAddress();

                    m_prevFrameLODs[grassInstanceIndex] = LOD;
#if GRASS_NO_DEGENERATE_INSTANCES
                    grassInstanceIndex++;
#endif
                }
            }
    }
}

// Render the scene.
void D3D12RaytracingAmbientOcclusion::OnRender()
{
    if (!m_deviceResources->IsWindowVisible())
    {
        return;
    }


#if 0
    auto commandList = m_deviceResources->GetCommandList();
    m_deviceResources->Prepare();
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(m_deviceResources->GetRenderTarget(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    commandList->ResourceBarrier(1, &barrier);
    m_deviceResources->ExecuteCommandList();
    m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT, 0);
#else
    auto commandList = m_deviceResources->GetCommandList();
    
    // Begin frame.
    m_deviceResources->Prepare();
    
    EngineProfiling::BeginFrame(commandList);

    {
        // ToDo fix - this dummy and make sure the children are properly enumerated as children in the UI output.
        ScopedTimer _prof(L"Dummy", commandList);
        {

            if (!(SceneArgs::TAO_LazyRender && m_cameraChangedIndex <= 0))
            {

#if USE_GRASS_GEOMETRY
                GenerateGrassGeometry();
#endif

                UpdateAccelerationStructure();

                // Render.
                RenderPass_GenerateShadowMap();

                RenderPass_GenerateGBuffers();
#if 1
                // AO. 
                if (SceneArgs::AOMode == SceneArgs::AOType::RTAO)
                {
                    ScopedTimer _prof(L"RTAO", commandList);

                    RWGpuResource* GBufferResources = SceneArgs::QuarterResAO ? m_GBufferLowResResources : m_GBufferResources;
                    m_RTAO.OnRender(
                        m_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress(),
                        GBufferResources[GBufferResource::HitPosition].gpuDescriptorReadAccess,
                        GBufferResources[GBufferResource::SurfaceNormal].gpuDescriptorReadAccess,
                        GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorReadAccess);

                    RenderPass_TemporalCacheReverseProjection();

#if BLUR_AO
#if ATROUS_DENOISER
                    if (SceneArgs::RTAODenoisingUseMultiscale)
                    {
                        ApplyMultiScaleAtrousWaveletTransformFilter();
                    }
                    else
                    {
                        ApplyAtrousWaveletTransformFilter();
                    }
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
#if 0
#if TEST_EARLY_EXIT
                RenderPass_TestEarlyExitOVerhead();
#else
                RenderPass_CalculateVisibility();
#endif
#endif

#endif
            }
            RWGpuResource* AOResources = SceneArgs::QuarterResAO ? m_AOResources : m_RTAO.GetAOResources();
            D3D12_GPU_DESCRIPTOR_HANDLE AOSRV = SceneArgs::AOMode == SceneArgs::AOType::RTAO ? AOResources[AOResource::Smoothed].gpuDescriptorReadAccess : SSAOgpuDescriptorReadAccess;

            // ToDo cleanup
            if (SceneArgs::AOMode == SceneArgs::AOType::RTAO && SceneArgs::RTAODenoisingUseMultiscale && !SceneArgs::QuarterResAO)
            {
                AOSRV = m_multiScaleDenoisingResources[0].m_value.gpuDescriptorReadAccess;
            }

            if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_RawOneFrame)
            {
                AOSRV = AOResources[AOResource::Coefficient].gpuDescriptorReadAccess;
            }
            else if (SceneArgs::CompositionMode == CompositionType::AmbientOcclusionOnly_TemporallySupersampled)
            {
                AOSRV = m_AOTSSCoefficient[m_temporalCacheCurrentFrameResourceIndex].gpuDescriptorReadAccess;
            }
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

#endif

   // SceneArgs::TAO_LazyRender.Bang();
    //m_cameraChangedIndex = 0;
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
