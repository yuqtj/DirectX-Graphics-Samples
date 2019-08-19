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

#define TWO_PASS_DENOISE 0


namespace Sample
{
    HWND g_hWnd = 0;
    UIParameters g_UIparameters;    // ToDo move
    D3D12RaytracingAmbientOcclusion* g_pSample = nullptr;
    UINT D3D12RaytracingAmbientOcclusion::s_numInstances = 0;

    std::map<std::wstring, BottomLevelAccelerationStructureGeometry> g_bottomLevelASGeometries;
    std::unique_ptr<RaytracingAccelerationStructureManager> g_accelerationStructure;
    GpuResource g_grassPatchVB[UIParameters::NumGrassGeometryLODs][2];      // Two VBs: current and previous frame.

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

    namespace Args
    {
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
        EnumVar DownsamplingBilateralFilter(L"Render/AO/RTAO/Down/Upsampling/Downsampled Value Filter", GpuKernels::DownsampleValueNormalDepthBilateralFilter::FilterDepthNormalWeighted2x2, GpuKernels::DownsampleValueNormalDepthBilateralFilter::Count, DownsamplingBilateralFilters, OnRecreateRaytracingResources, nullptr);
        BoolVar DownAndUpsamplingUseBilinearWeights(L"Render/AO/RTAO/Down/Upsampling/Bilinear weighted", true);
        BoolVar DownAndUpsamplingUseDepthWeights(L"Render/AO/RTAO/Down/Upsampling/Depth weighted", true);
        BoolVar DownAndUpsamplingUseNormalWeights(L"Render/AO/RTAO/Down/Upsampling/Normal weighted", true);
        BoolVar DownAndUpsamplingUseDynamicDepthThreshold(L"Render/AO/RTAO/Down/Upsampling/Dynamic depth threshold", true);        // ToDO rename to adaptive

#if USE_NORMALIZED_Z
        NumVar PathTracing_Znear(L"Render/PathTracing/Znear", 0.0f, 0, 1000.0f, 1.0f);
        NumVar PathTracing_Zfar(L"Render/PathTracing/Zfar", 100.0f, 0, 1000.0f, 1.0f);
#endif

        NumVar CameraRotationDuration(L"Scene2/Camera rotation time", 48.f, 1.f, 120.f, 1.f);
        BoolVar AnimateGrass(L"Scene2/Animate grass", true);



        NumVar DebugVar(L"Render/Debug var", -20, -90, 90, 0.5f);

        // Temporal Cache.
        // ToDo rename cache to accumulation/supersampling?
        BoolVar RTAO_UseTemporalSupersampling(L"Render/AO/RTAO/Temporal Cache/Enabled", true);
        BoolVar RTAO_TemporalSupersampling_CacheRawAOValue(L"Render/AO/RTAO/Temporal Cache/Cache Raw AO Value", true);
        NumVar RTAO_TemporalSupersampling_MinSmoothingFactor(L"Render/AO/RTAO/Temporal Cache/Min Smoothing Factor", 0.03f, 0, 1.f, 0.01f);
        NumVar RTAO_TemporalSupersampling_DepthTolerance(L"Render/AO/RTAO/Temporal Cache/Depth tolerance [%%]", 0.05f, 0, 1.f, 0.001f);
        BoolVar RTAO_TemporalSupersampling_UseWorldSpaceDistance(L"Render/AO/RTAO/Temporal Cache/Use world space distance", false);    // ToDo test / remove
        BoolVar RTAO_TemporalSupersampling_PerspectiveCorrectDepthInterpolation(L"Render/AO/RTAO/Temporal Cache/Depth testing/Use perspective correct depth interpolation", false);    // ToDo remove
        BoolVar RTAO_TemporalSupersampling_UseDepthWeights(L"Render/AO/RTAO/Temporal Cache/Use depth weights", true);    // ToDo remove
        BoolVar RTAO_TemporalSupersampling_UseNormalWeights(L"Render/AO/RTAO/Temporal Cache/Use normal weights", true);
        BoolVar RTAO_TemporalSupersampling_ForceUseMinSmoothingFactor(L"Render/AO/RTAO/Temporal Cache/Force min smoothing factor", false);


        // ToDo remove
        BoolVar RTAO_KernelStepRotateShift0(L"Render/AO/RTAO/Kernel Step Shifts/Rotate 0:", true);
        IntVar RTAO_KernelStepShift0(L"Render/AO/RTAO/Kernel Step Shifts/0", 3, 0, 10, 1);
        IntVar RTAO_KernelStepShift1(L"Render/AO/RTAO/Kernel Step Shifts/1", 1, 0, 10, 1);
        IntVar RTAO_KernelStepShift2(L"Render/AO/RTAO/Kernel Step Shifts/2", 0, 0, 10, 1);
        IntVar RTAO_KernelStepShift3(L"Render/AO/RTAO/Kernel Step Shifts/3", 0, 0, 10, 1);
        IntVar RTAO_KernelStepShift4(L"Render/AO/RTAO/Kernel Step Shifts/4", 0, 0, 10, 1);

        const WCHAR* VarianceBilateralFilters[GpuKernels::CalculateVariance::FilterType::Count] = { L"Square Bilateral", L"Separable Bilateral", L"Separable" };
        EnumVar VarianceBilateralFilter(L"Render/GpuKernels/CalculateVariance/Filter", GpuKernels::CalculateVariance::Separable, GpuKernels::CalculateVariance::Count, VarianceBilateralFilters);

        IntVar VarianceBilateralFilterKernelWidth(L"Render/GpuKernels/CalculateVariance/Kernel width", 9, 3, 11, 2);    // ToDo find lowest good enough width


        // ToDo rename to temporal supersampling
        // ToDo address: Clamping causes rejection of samples in low density areas - such as on ground plane at the end of max ray distance from other objects.
        BoolVar RTAO_TemporalSupersampling_CacheDenoisedOutput(L"Render/AO/RTAO/Temporal Cache/Cache denoised output", true);
        IntVar RTAO_TemporalSupersampling_CacheDenoisedOutputPassNumber(L"Render/AO/RTAO/Temporal Cache/Cache denoised output - pass number", 0, 0, 10, 1);
        BoolVar RTAO_TemporalSupersampling_ClampCachedValues_UseClamping(L"Render/AO/RTAO/Temporal Cache/Clamping/Enabled", true);
        BoolVar RTAO_TemporalSupersampling_CacheSquaredMean(L"Render/AO/RTAO/Temporal Cache/Cached SquaredMean", false);
        NumVar RTAO_TemporalSupersampling_ClampCachedValues_StdDevGamma(L"Render/AO/RTAO/Temporal Cache/Clamping/Std.dev gamma", 1.0f, 0.1f, 20.f, 0.1f);
        NumVar RTAO_TemporalSupersampling_ClampCachedValues_MinStdDevTolerance(L"Render/AO/RTAO/Temporal Cache/Clamping/Minimum std.dev", 0.04f, 0.0f, 1.f, 0.01f);   // ToDo finetune
        NumVar RTAO_TemporalSupersampling_ClampDifferenceToFrameAgeScale(L"Render/AO/RTAO/Temporal Cache/Clamping/Frame Age scale", 4.00f, 0, 10.f, 0.05f);
        NumVar RTAO_TemporalSupersampling_ClampCachedValues_AbsoluteDepthTolerance(L"Render/AO/RTAO/Temporal Cache/Depth threshold/Absolute depth tolerance", 1.0f, 0.0f, 100.f, 1.f);
        NumVar RTAO_TemporalSupersampling_ClampCachedValues_DepthBasedDepthTolerance(L"Render/AO/RTAO/Temporal Cache/Depth threshold/Depth based depth tolerance", 1.0f, 0.0f, 100.f, 1.f);
        BoolVar RTAO_TemporalSupersampling_TestFlag(L"Render/AO/RTAO/Temporal Cache/Test flag", false);

        // Todo revise comment
        // Setting it lower than 0.9 makes cache values to swim...
        NumVar RTAO_TemporalSupersampling_ClampCachedValues_DepthSigma(L"Render/AO/RTAO/Temporal Cache/Depth threshold/Depth sigma", 1.0f, 0.0f, 10.f, 0.01f);

#if !NORMAL_DEPTH_R8G8B16_ENCODING
        const WCHAR* FloatingPointFormatsRGB[TextureResourceFormatRGB::Count] = { L"R32G32B32A32_FLOAT", L"R16G16B16A16_FLOAT", L"R11G11B10_FLOAT" };
        EnumVar RTAO_TemporalSupersampling_NormalDepthResourceFormat(L"Render/Texture Formats/AO/RTAO/Temporal Cache/Encoded Normal (RG) Depth (B) resource", TextureResourceFormatRGB::R16G16B16A16_FLOAT, TextureResourceFormatRGB::Count, FloatingPointFormatsRGB, OnRecreateRaytracingResources);
#endif

        const WCHAR* FloatingPointFormatsRG[TextureResourceFormatRG::Count] = { L"R32G32_FLOAT", L"R16G16_FLOAT", L"R8G8_SNORM" };
        // ToDo  ddx needs to be in normalized to use UNORM.
        EnumVar RTAO_PartialDepthDerivativesResourceFormat(L"Render/Texture Formats/PartialDepthDerivatives", TextureResourceFormatRG::R16G16_FLOAT, TextureResourceFormatRG::Count, FloatingPointFormatsRG, OnRecreateRaytracingResources);
        EnumVar RTAO_MotionVectorResourceFormat(L"Render/Texture Formats/AO/RTAO/Temporal Supersampling/Motion Vector", TextureResourceFormatRG::R16G16_FLOAT, TextureResourceFormatRG::Count, FloatingPointFormatsRG, OnRecreateRaytracingResources);

        BoolVar TAO_LazyRender(L"TAO/Lazy render", false);
        IntVar RTAO_LazyRenderNumFrames(L"TAO/Lazy render frames", 1, 0, 20, 1);


        // ToDo add Weights On/OFF - 
        // RTAO Denoising
        IntVar RTAOVarianceFilterKernelWidth(L"Render/AO/RTAO/Denoising/Variance filter/Kernel width", 7, 3, 11, 2);    // ToDo find lowest good enough width
        BoolVar UseSpatialVariance(L"Render/AO/RTAO/Denoising/Use spatial variance", true);
        //BoolVar ApproximateSpatialVariance(L"Render/AO/RTAO/Denoising/Approximate spatial variance", false);
        BoolVar RTAODenoisingUseMultiscale(L"Render/AO/RTAO/Denoising/Multi-scale/Enabled", false);
        IntVar RTAODenoisingMultiscaleLevels(L"Render/AO/RTAO/Denoising/Multi-scale/Levels", 3, 1, D3D12RaytracingAmbientOcclusion::c_MaxDenoisingScaleLevels);
        BoolVar RTAODenoisingMultiscaleDenoisedAsInput(L"Render/AO/RTAO/Denoising/Multi-scale/Denoised as input", true);

        BoolVar RTAODenoisingPerspectiveCorrectDepthInterpolation(L"Render/AO/RTAO/Denoising/Pespective Correct Depth Interpolation", true); // ToDo test perf impact / visual quality gain at the end. Document.
        BoolVar RTAODenoisingUseAdaptiveKernelSize(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Enabled", true);
        IntVar RTAODenoisingFilterMinKernelWidth(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Min kernel width", 3, 3, 101);
        NumVar RTAODenoisingFilterMaxKernelWidthPercentage(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Max kernel width [%% of screen width]", 1.5f, 0, 100, 0.1f);
        NumVar RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Variance sigma scale on small kernels", 2.0f, 1.0f, 20.f, 0.5f);
        NumVar RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor(L"Render/AO/RTAO/Denoising/AdaptiveKernelSize/Hit distance scale factor", 0.07f, 0.001f, 10.f, 0.005f);
        BoolVar RTAODenoising_Variance_UseDepthWeights(L"Render/AO/RTAO/Denoising/Variance/Use normal weights", true);
        BoolVar RTAODenoising_Variance_UseNormalWeights(L"Render/AO/RTAO/Denoising/Variance/Use normal weights", true);
        BoolVar RTAODenoising_ForceDenoisePass(L"Render/AO/RTAO/Denoising/Force denoise pass", false);
        IntVar RTAODenoising_MinFrameAgeToUseTemporalVariance(L"Render/AO/RTAO/Denoising/Min Temporal Variance Frame Age", 4, 1, 40);
        NumVar RTAODenoisingMinVarianceToDenoise(L"Render/AO/RTAO/Denoising/Min Variance to denoise", 0.0f, 0.0f, 1.f, 0.01f);
        BoolVar RTAODenoisingUseSmoothedVariance(L"Render/AO/RTAO/Denoising/Use smoothed variance", false);
        BoolVar RTAODenoisingUseProjectedDepthTest(L"Render/AO/RTAO/Denoising/Use projected depth test", true);

        BoolVar RTAODenoising_LowerWeightForStaleSamples(L"Render/AO/RTAO/Denoising/Scale down stale samples weight", false);


        // TODo This probalby should be false, otherwise the newly disoccluded samples get too biased?
        BoolVar RTAODenoisingFilterWeightByFrameAge(L"Render/AO/RTAO/Denoising/Filter weight by frame age", false);


#define MIN_NUM_PASSES_LOW_TSPP 2 // THe blur writes to the initial input resource and thus must numPasses must be 2+.
#define MAX_NUM_PASSES_LOW_TSPP 6
        BoolVar RTAODenoisingLowTspp(L"Render/AO/RTAO/Denoising/Low tspp filter/enabled", true);
        IntVar RTAODenoisingLowTsppMaxFrameAge(L"Render/AO/RTAO/Denoising/Low tspp filter/Max frame age", 12, 0, 33);
        IntVar RTAODenoisingLowTspBlurPasses(L"Render/AO/RTAO/Denoising/Low tspp filter/Num blur passes", 3, 2, MAX_NUM_PASSES_LOW_TSPP);
        BoolVar RTAODenoisingLowTsppUseUAVReadWrite(L"Render/AO/RTAO/Denoising/Low tspp filter/Use single UAV resource Read+Write", true);
        NumVar RTAODenoisingLowTsppDecayConstant(L"Render/AO/RTAO/Denoising/Low tspp filter/Decay constant", 1.0f, 0.1f, 32.f, 0.1f);
        BoolVar RTAODenoisingLowTsppFillMissingValues(L"Render/AO/RTAO/Denoising/Low tspp filter/Post-TSS fill in missing values", true);
        BoolVar RTAODenoisingLowTsppUseNormalWeights(L"Render/AO/RTAO/Denoising/Low tspp filter/Normal Weights/Enabled", false);
        NumVar RTAODenoisingLowTsppMinNormalWeight(L"Render/AO/RTAO/Denoising/Low tspp filter/Normal Weights/Min weight", 0.25f, 0.0f, 1.f, 0.05f);
        NumVar RTAODenoisingLowTsppNormalExponent(L"Render/AO/RTAO/Denoising/Low tspp filter/Normal Weights/Exponent", 4.0f, 1.0f, 32.f, 1.0f);

        const WCHAR* DenoisingModes[GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::Count] = { L"EdgeStoppingBox3x3", L"EdgeStoppingGaussian3x3", L"EdgeStoppingGaussian5x5" };
        EnumVar DenoisingMode(L"Render/AO/RTAO/Denoising/Mode", GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::EdgeStoppingGaussian3x3, GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType::Count, DenoisingModes);
#if    DISABLE_DENOISING
        IntVar AtrousFilterPasses(L"Render/AO/RTAO/Denoising/Num passes", 1, 1, 8, 1);
        NumVar AODenoiseValueSigma(L"Render/AO/RTAO/Denoising/Value Sigma", 0.011f, 0.0f, 30.0f, 0.1f);
#else
        IntVar AtrousFilterPasses(L"Render/AO/RTAO/Denoising/Num passes", 1, 1, 8, 1);
        NumVar AODenoiseValueSigma(L"Render/AO/RTAO/Denoising/Value Sigma", 0.3f, 0.0f, 30.0f, 0.1f);
        BoolVar RTAODenoising_2ndPass_UseVariance(L"Render/AO/RTAO/Denoising/2nd+ pass/Use variance", false);
        NumVar RTAODenoising_2ndPass_NormalSigma(L"Render/AO/RTAO/Denoising/2nd+ pass/Normal Sigma", 2, 1, 256, 2);
        NumVar RTAODenoising_2ndPass_DepthSigma(L"Render/AO/RTAO/Denoising/2nd+ pass/Depth Sigma", 1.0f, 0.0f, 10.0f, 0.02f);
#endif
        IntVar RTAODenoising_MaxFrameAgeToDenoiseAfter1stPass(L"Render/AO/RTAO/Denoising/Max Frame Age To Denoise 2nd+ pass", 33, 1, 34, 1);
        IntVar RTAODenoising_MaxFrameAgeToDenoiseOn1stPass(L"Render/AO/RTAO/Denoising/1st pass/Max Frame Age To Denoise", 16, 1, 64, 1);
        IntVar RTAODenoisingExtraRaysToTraceSinceTSSMovement(L"Render/AO/RTAO/Denoising/Heuristics/Num rays to cast since TSS movement", 32, 0, 64);
        IntVar RTAODenoisingnumFramesToDenoiseAfterLastTracedRay(L"Render/AO/RTAO/Denoising/Heuristics/Num frames to denoise after last traced ray", 32, 0, 64);



        BoolVar ReverseFilterOrder(L"Render/AO/RTAO/Denoising/Reverse filter order", false);
        NumVar RTAODenoising_WeightScale(L"Render/AO/RTAO/Denoising/Weight Scale", 1, 0.0f, 5.0f, 0.01f);

        // ToDo why large depth sigma is needed?
        // ToDo the values don't scale to QuarterRes - see ImportaceMap viz
        NumVar AODenoiseDepthSigma(L"Render/AO/RTAO/Denoising/Depth Sigma", 0.5f, 0.0f, 10.0f, 0.02f); // ToDo Fine tune. 1 causes moire patterns at angle under the car

         // ToDo Fine tune. 1 causes moire patterns at angle under the car
        // aT LOW RES 1280X768. causes depth disc lines down to 0.8 cutoff at long ranges
        NumVar AODenoiseDepthWeightCutoff(L"Render/AO/RTAO/Denoising/Depth Weight Cutoff", 0.2f, 0.0f, 2.0f, 0.01f);

        NumVar AODenoiseNormalSigma(L"Render/AO/RTAO/Denoising/Normal Sigma", 64, 0, 256, 4);   // ToDo rename sigma as sigma in depth/var means tolernace. here its an exponent.


        // ToDo dedupe
        BoolVar g_QuarterResAO(L"Misc/QuarterRes AO", false);
        NumVar g_DistanceTolerance(L"Misc/AO Distance Tolerance (log10)", -2.5f, -32.0f, 32.0f, 0.25f);

        // SSAO
        NumVar SSAONoiseFilterTolerance(L"Render/AO/SSAO/Noise Threshold (log10)", -3.f, -8.f, .0f, .1f);
        NumVar SSAOBlurTolerance(L"Render/AO/SSAO/Blur Tolerance (log10)", -5.f, -8.f, -1.f, .1f);
        NumVar SSAOUpsampleTolerance(L"Render/AO/SSAO/Upsample Tolerance (log10)", -7.f, -12.f, -1.f, .1f);
        NumVar SSAONormalMultiply(L"Render/AO/SSAO/Normal Factor", 1.f, .0f, 5.f, .125f);
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
    m_animateCamera(false),
    m_animateLight(false),
    m_animateScene(true),
    m_isGeometryInitializationRequested(true),
    m_isASinitializationRequested(true),
	m_isSceneInitializationRequested(false),
	m_isRecreateRaytracingResourcesRequested(false),
    m_numFramesSinceASBuild(0),
	m_isCameraFrozen(false)
{
    ThrowIfFalse(++s_numInstances == 1, L"There can be only one D3D12RaytracingAmbientOcclusion instance.");
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

// Update camera matrices passed into the shader.
void D3D12RaytracingAmbientOcclusion::UpdateCameraMatrices()
{
    m_pathtracer.SetCamera(m_camera);

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
        ranges[Slot::Variance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 13); 
        ranges[Slot::LocalMeanVariance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 14);
  
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
		descComputePSO.CS = CD3DX12_SHADER_BYTECODE((void *)g_pComposeRenderPassesCS, ARRAYSIZE(g_pComposeRenderPassesCS));

		ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_computePSOs[CSType::ComposeRenderPassesCS])));
		m_computePSOs[CSType::ComposeRenderPassesCS]->SetName(L"PSO: ComposeRenderPassesCS");
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

    // Create constant buffers for the geometry and the scene.
    CreateConstantBuffers();

 

    InitializeAccelerationStructures();
   

	// ToDo move
	CreateComposeRenderPassesCSResources();

    CreateAoBlurCSResources();

    m_pathtracer.Setup(m_deviceResources, m_cbvSrvUavHeap, m_maxInstanceContributionToHitGroupIndex);
    m_RTAO.Setup(m_deviceResources, m_cbvSrvUavHeap, m_maxInstanceContributionToHitGroupIndex);
    m_SSAO.Setup(m_deviceResources);

    // 
    m_prevFrameBottomLevelASInstanceTransforms.Create(device, MaxNumBottomLevelInstances, FrameCount, L"GPU buffer: Bottom Level AS Instance transforms for previous frame");
}


// Create a 2D output texture for raytracing.
void D3D12RaytracingAmbientOcclusion::CreateRaytracingOutputResource()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto backbufferFormat = m_deviceResources->GetBackBufferFormat();
	m_raytracingOutput.rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
	CreateRenderTargetResource(device, backbufferFormat, m_width, m_height, m_cbvSrvUavHeap.get(), &m_raytracingOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
	m_raytracingOutputIntermediate.rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
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

    // ToDo remove obsolete resources, QuarterResAO event triggers this so we may not need all low/gbuffer width AO resources.
    m_multiPassDenoisingBlurStrength.rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R8_UNORM, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_multiPassDenoisingBlurStrength, initialResourceState, L"Multi Pass Denoising Blur Strength");
    
  
    // ToDo
    m_SSAO.BindGBufferResources(Pathtracer::g_GBufferResources[GBufferResource::SurfaceNormalDepth].GetResource(), Pathtracer::g_GBufferResources[GBufferResource::Depth].GetResource());

    // ToDo remove unneeded ones
    // Full-res AO resources.
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_AOResources[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
        m_AOResources[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(AOResource::Count);
        for (UINT i = 0; i < AOResource::Count; i++)
        {
            m_AOResources[i].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
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
        CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_AOResources[AOResource::RayHitDistance], initialResourceState, L"Render/AO Hit Distance");
    }

    // ToDo remove unnecessary copies for 2 resolutions. Only keep one where possible and recreate on change.
    // ToDo pass formats via params shared across AO, GBuffer, TC

    // Full-res Temporal Cache resources.
    {
        for (UINT i = 0; i < 2; i++)
        {
            // Preallocate subsequent descriptor indices for both SRV and UAV groups.
            m_temporalCache[i][0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(TemporalSupersampling::Count);
            m_temporalCache[i][0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(TemporalSupersampling::Count);
            for (UINT j = 0; j < TemporalSupersampling::Count; j++)
            {
                m_temporalCache[i][j].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
                m_temporalCache[i][j].uavDescriptorHeapIndex = m_temporalCache[i][0].uavDescriptorHeapIndex + j;
                m_temporalCache[i][j].srvDescriptorHeapIndex = m_temporalCache[i][0].srvDescriptorHeapIndex + j;
            }

            // ToDo cleanup raytracing resolution - twice for coefficient.
            CreateRenderTargetResource(device, DXGI_FORMAT_R8G8_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_temporalCache[i][TemporalSupersampling::FrameAge], initialResourceState, L"Temporal Cache: Frame Age");
            CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_temporalCache[i][TemporalSupersampling::CoefficientSquaredMean], initialResourceState, L"Temporal Cache: Coefficient Squared Mean");
            CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_temporalCache[i][TemporalSupersampling::RayHitDistance], initialResourceState, L"Temporal Cache: Ray Hit Distance");


            m_TSSAOCoefficient[i].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_TSSAOCoefficient[i], initialResourceState, L"Render/AO Temporally Supersampled Coefficient");
            
                
            m_lowResTSSAOCoefficient[i].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_lowResTSSAOCoefficient[i], initialResourceState, L"Render/AO LowRes Temporally Supersampled Coefficient");        
        }
     }

    for (UINT i = 0; i < 2; i++)
    {
        m_temporalSupersampling_blendedAOCoefficient[i].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
        CreateRenderTargetResource(device, m_RTAO.GetAOCoefficientFormat(), m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_temporalSupersampling_blendedAOCoefficient[i], initialResourceState, L"Temporal Supersampling: AO coefficient current frame blended with the cache.");
    }
    m_cachedFrameAgeValueSquaredValueRayHitDistance.rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R16G16B16A16_UINT, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_cachedFrameAgeValueSquaredValueRayHitDistance, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"Temporal Supersampling intermediate reprojected Frame Age, Value, Squared Mean Value, Ray Hit Distance");


    // ToDo remove
    // Debug resources
    {
        // Preallocate subsequent descriptor indices for both SRV and UAV groups.
        m_debugOutput[0].uavDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(ARRAYSIZE(m_debugOutput));
        m_debugOutput[0].srvDescriptorHeapIndex = m_cbvSrvUavHeap->AllocateDescriptorIndices(ARRAYSIZE(m_debugOutput));
        for (UINT i = 0; i < ARRAYSIZE(m_debugOutput); i++)
        {
            m_debugOutput[i].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            m_debugOutput[i].uavDescriptorHeapIndex = m_debugOutput[0].uavDescriptorHeapIndex + i;
            m_debugOutput[i].srvDescriptorHeapIndex = m_debugOutput[0].srvDescriptorHeapIndex + i;
            CreateRenderTargetResource(device, debugFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_debugOutput[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"Debug");
        }
    }


    // ToDo move
    // ToDo render shadows at raytracing dim?
	m_VisibilityResource.rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
	CreateRenderTargetResource(device, texFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_VisibilityResource, initialResourceState, L"Visibility");
    
       
    m_ShadowMapResource.rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
    CreateRenderTargetResource(device, DXGI_FORMAT_R16_FLOAT, c_shadowMapDim.x, c_shadowMapDim.y, m_cbvSrvUavHeap.get(), &m_ShadowMapResource, initialResourceState, L"Shadow Map");


    // Variance resources
    DXGI_FORMAT varianceTexFormat = m_RTAO.GetAOCoefficientFormat();       // ToDo 8 bit suffers from loss of precision and clamps too much.
    {
        DXGI_FORMAT meanVarianceTexFormat = DXGI_FORMAT_R16G16_FLOAT;       // ToDo 8 bit suffers from loss of precision and clamps too much.
        // HiRes
        // ToDo specialize formats instead of using a common one?
        {
            m_varianceResource[AOVarianceResource::Raw].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, varianceTexFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_varianceResource[AOVarianceResource::Raw], initialResourceState, L"Post Temporal Reprojection Variance");

            m_varianceResource[AOVarianceResource::Smoothed].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, varianceTexFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_varianceResource[AOVarianceResource::Smoothed], initialResourceState, L"Smoothed Post Temporal Reprojection Variance");

            m_localMeanVarianceResource[AOVarianceResource::Raw].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, meanVarianceTexFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_localMeanVarianceResource[AOVarianceResource::Raw], initialResourceState, L"Local Mean Variance");

            m_localMeanVarianceResource[AOVarianceResource::Smoothed].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, meanVarianceTexFormat, m_GBufferWidth, m_GBufferHeight, m_cbvSrvUavHeap.get(), &m_localMeanVarianceResource[AOVarianceResource::Smoothed], initialResourceState, L"Smoothed Local Mean Variance");
        }

        // Low res
        {
            m_lowResVarianceResource[AOVarianceResource::Raw].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, varianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_lowResVarianceResource[AOVarianceResource::Raw], initialResourceState, L"LowRes Post Temporal Reprojection Variance");

            m_lowResVarianceResource[AOVarianceResource::Smoothed].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, varianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_lowResVarianceResource[AOVarianceResource::Smoothed], initialResourceState, L"LowRes Smoothed Post Temporal Reprojection Variance");

            m_lowResLocalMeanVarianceResource[AOVarianceResource::Raw].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, meanVarianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_lowResLocalMeanVarianceResource[AOVarianceResource::Raw], initialResourceState, L"LowRes Local Mean Variance");

            m_lowResLocalMeanVarianceResource[AOVarianceResource::Smoothed].rwFlags = GpuResource::RWFlags::AllowWrite | GpuResource::RWFlags::AllowRead;
            CreateRenderTargetResource(device, meanVarianceTexFormat, m_raytracingWidth, m_raytracingHeight, m_cbvSrvUavHeap.get(), &m_lowResLocalMeanVarianceResource[AOVarianceResource::Smoothed], initialResourceState, L"LowRes Smoothed Local Mean Variance");
        }
    }
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
    m_calculateMeanVarianceKernel.Initialize(device, FrameCount, 5*MaxCalculateVarianceKernelInvocationsPerFrame); // ToDo revise the ount
    m_fillInCheckerboardKernel.Initialize(device, FrameCount);
    m_gaussianSmoothingKernel.Initialize(device, FrameCount, MaxGaussianSmoothingKernelInvocationsPerFrame);
	m_downsampleBoxFilter2x2Kernel.Initialize(device, FrameCount);
	m_downsampleGaussian9TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap9, FrameCount);
	m_downsampleGaussian25TapFilterKernel.Initialize(device, GpuKernels::DownsampleGaussianFilter::Tap25, FrameCount); // ToDo Dedupe 9 and 25
    m_downsampleGBufferBilateralFilterKernel.Initialize(device, GpuKernels::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::FilterDepthAware2x2, FrameCount);
    m_downsampleValueNormalDepthBilateralFilterKernel.Initialize(device, static_cast<GpuKernels::DownsampleValueNormalDepthBilateralFilter::Type>(static_cast<UINT>(Args::DownsamplingBilateralFilter)));
    m_upsampleBilateralFilterKernel.Initialize(device, FrameCount);
    m_multiScale_upsampleBilateralFilterAndCombineKernel.Initialize(device, GpuKernels::MultiScale_UpsampleBilateralFilterAndCombine::Filter2x2);
    m_temporalCacheReverseReprojectKernel.Initialize(device, FrameCount); 
    m_temporalCacheBlendWithCurrentFrameKernel.Initialize(device, FrameCount); 
    m_writeValueToTexture.Initialize(device, m_cbvSrvUavHeap.get());
    m_fillInMissingValuesFilterKernel.Initialize(device, FrameCount, 2);
    m_bilateralFilterKernel.Initialize(device, FrameCount, MAX_NUM_PASSES_LOW_TSPP);
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


void D3D12RaytracingAmbientOcclusion::OnKeyDown(UINT8 key)
{
    float fValue;
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
        //m_animateScene = !m_animateScene;
        break;
    case 'V':
        Args::TAO_LazyRender.Bang();// TODo remove
        break;
    case 'J':
        g_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(0, 5, 0, 0)));
        m_cameraChangedIndex = 2;
        break;
    case 'M':
        g_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(0, -5, 0, 0)));
        m_cameraChangedIndex = 2;
        break;
    case 'U':
        g_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(5, 0, 0, 0)));
        m_cameraChangedIndex = 2;
        break;
    case 'Y':
        g_accelerationStructure->GetBottomLevelASInstance(GeometryType::PBRT).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(-5, 0, 0, 0)));
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
    case VK_NUMPAD1:
        Args::CompositionMode.SetValue(CompositionType::AmbientOcclusionOnly_RawOneFrame);
        break;
    case VK_NUMPAD2:
        Args::CompositionMode.SetValue(CompositionType::AmbientOcclusionOnly_Denoised);
        break;
    case VK_NUMPAD3:
        Args::CompositionMode.SetValue(CompositionType::PhongLighting);
        break;
    case VK_NUMPAD0:
        Args::AOEnabled.Bang();
        break;
    case VK_NUMPAD9:
        fValue = IsInRange(m_RTAO.GetMaxRayHitTime(), 3.9f, 4.1f) ? 22.f : 4.f;
        m_RTAO.SetMaxRayHitTime(fValue);
        break;
    default:
        break;
    }
}

// Update frame-based values.
void D3D12RaytracingAmbientOcclusion::OnUpdate()
{
    m_timer.Tick();

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

    m_scene.OnUpdate();
    m_RTAO.OnUpdate();


    // ToDo move
    // SSAO
    {   
        m_SSAOCB->noiseTile = { float(m_width) / float(SSAO_NOISE_W), float(m_height) / float(SSAO_NOISE_W), 0, 0};
        m_SSAO.SetParameters(Args::SSAONoiseFilterTolerance, Args::SSAOBlurTolerance, Args::SSAOUpsampleTolerance, Args::SSAONormalMultiply);
    
    }
	if (m_enableUI)
    {
        UpdateUI();
    }

 }

// Parse supplied command line args.
void D3D12RaytracingAmbientOcclusion::ParseCommandLineArgs(WCHAR* argv[], int argc)
{
    DXSample::ParseCommandLineArgs(argv, argc);
}

void D3D12RaytracingAmbientOcclusion::UpdateAccelerationStructure()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    if (Args::EnableGeometryAndASBuildsAndUpdates)
    {
        bool forceBuild = false;    // ToDo

        resourceStateTracker->FlushResourceBarriers();
        g_accelerationStructure->Build(commandList, m_cbvSrvUavHeap->GetHeap(), frameIndex, forceBuild);
    }

    // Copy previous frame Bottom Level AS instance transforms to GPU. 
    m_prevFrameBottomLevelASInstanceTransforms.CopyStagingToGpu(frameIndex);

    // Update the CPU staging copy with the current frame transforms.
    const auto& bottomLevelASInstanceDescs = g_accelerationStructure->GetBottomLevelASInstancesBuffer();
    for (UINT i = 0; i < bottomLevelASInstanceDescs.NumElements(); i++)
    {
        m_prevFrameBottomLevelASInstanceTransforms[i] = *reinterpret_cast<const XMFLOAT3X4*>(bottomLevelASInstanceDescs[i].Transform);
    }
}

void D3D12RaytracingAmbientOcclusion::ApplyAtrousWaveletTransformFilter(bool isFirstPass)
{
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    GpuResource* AOResources = m_RTAO.AOResources();
    GpuResource* TSSAOCoefficient = RTAO::Args::QuarterResAO ? m_lowResTSSAOCoefficient : m_TSSAOCoefficient;
    GpuResource* GBufferResources = m_pathtracer.GetGBufferResources(RTAO::Args::QuarterResAO);

    GpuResource* VarianceResources = RTAO::Args::QuarterResAO ? m_lowResVarianceResource : m_varianceResource;
    // ToDO use separate toggles for local and temporal
    GpuResource* VarianceResource = Args::RTAODenoisingUseSmoothedVariance ? &VarianceResources[AOVarianceResource::Smoothed] : &VarianceResources[AOVarianceResource::Raw];

    
    ScopedTimer _prof(L"DenoiseAO", commandList);

    // Transition Resources.
    resourceStateTracker->TransitionResource(&AOResources[AOResource::Smoothed], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    
#if RAYTRACING_MANUAL_KERNEL_STEP_SHIFTS
    static UINT frameID = 0;

    UINT offsets[5] = {
        static_cast<UINT>(Args::RTAO_KernelStepShift0),
        static_cast<UINT>(Args::RTAO_KernelStepShift1),
        static_cast<UINT>(Args::RTAO_KernelStepShift2),
        static_cast<UINT>(Args::RTAO_KernelStepShift3),
        static_cast<UINT>(Args::RTAO_KernelStepShift4)};

    if (isFirstPass)
    {
        offsets[0] = Args::RTAO_KernelStepRotateShift0 ? 1 + (frameID++ % (offsets[0] + 1)) : offsets[0];
    }
    else
    {
        for (UINT i = 1; i < 5; i++)
        {
            offsets[i - 1] = offsets[i];
        }
    }

    UINT newStartId = 0;
    for (UINT i = 1; i < 5; i++)
    {
        offsets[i] = newStartId + offsets[i];
        newStartId = offsets[i] + 1;
    }
#endif

    float ValueSigma;
    float NormalSigma;
    float DepthSigma;
    if (isFirstPass)
    {
        ValueSigma = Args::AODenoiseValueSigma;
        NormalSigma = Args::AODenoiseNormalSigma;
        DepthSigma = Args::AODenoiseDepthSigma;
    }
    else
    {
        ValueSigma = Args::RTAODenoising_2ndPass_UseVariance ? 1.f : 0.f;
        NormalSigma = Args::RTAODenoising_2ndPass_NormalSigma;
        DepthSigma = Args::RTAODenoising_2ndPass_DepthSigma;
    }
    
#if TWO_PASS_DENOISE
    UINT numFilterPasses = Args::AtrousFilterPasses;// isFirstPass ? 1 : Args::AtrousFilterPasses - 1;
#else
    UINT numFilterPasses = Args::AtrousFilterPasses;
#endif
    bool cacheIntermediateDenoiseOutput =
        Args::RTAO_TemporalSupersampling_CacheDenoisedOutput &&
        static_cast<UINT>(Args::RTAO_TemporalSupersampling_CacheDenoisedOutputPassNumber) < numFilterPasses;

    GpuResource* InputAOCoefficientResource = &TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex];
    GpuResource* OutputIntermediateResource = nullptr;
    if (cacheIntermediateDenoiseOutput)
    {
        // ToDo clean this up so that its clear.
        m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex = (m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex + 1) % 2;
        OutputIntermediateResource = &TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex];
    }

    if (OutputIntermediateResource)
    {
        resourceStateTracker->TransitionResource(OutputIntermediateResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    
    float staleNeighborWeightScale = Args::RTAODenoising_LowerWeightForStaleSamples ?  m_RTAO.GetSpp() : 1;
    bool forceDenoisePass = Args::RTAODenoising_ForceDenoisePass;
    
    if (forceDenoisePass)
    {
        Args::RTAODenoising_ForceDenoisePass.Bang();
    }
    // A-trous edge-preserving wavelet tranform filter
    if (numFilterPasses > 0)
    {
        ScopedTimer _prof(L"AtrousWaveletTransformFilter", commandList);
        resourceStateTracker->FlushResourceBarriers();
        m_atrousWaveletTransformFilter.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            static_cast<GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType>(static_cast<UINT>(Args::DenoisingMode)),
            InputAOCoefficientResource->gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
            VarianceResource->gpuDescriptorReadAccess,
            m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorReadAccess,
            &AOResources[AOResource::Smoothed],
            OutputIntermediateResource,
            &m_debugOutput[0],
            &m_debugOutput[1],
            ValueSigma,
            DepthSigma,
            NormalSigma,
            Args::RTAODenoising_WeightScale,
#if !NORMAL_DEPTH_R8G8B16_ENCODING
            // ToDo rename this to be global normalDepth
            static_cast<TextureResourceFormatRGB::Type>(static_cast<UINT>(Args::RTAO_TemporalSupersampling_NormalDepthResourceFormat)),
#endif
            offsets,
            static_cast<UINT>(Args::RTAO_TemporalSupersampling_CacheDenoisedOutputPassNumber),
            numFilterPasses,
            GpuKernels::AtrousWaveletTransformCrossBilateralFilter::Mode::OutputFilteredValue,
            Args::ReverseFilterOrder,
            Args::UseSpatialVariance,
            Args::RTAODenoisingPerspectiveCorrectDepthInterpolation,
            Args::RTAODenoisingUseAdaptiveKernelSize,
            Args::RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor,
            Args::RTAODenoisingFilterMinKernelWidth,
            static_cast<UINT>((Args::RTAODenoisingFilterMaxKernelWidthPercentage / 100) * m_raytracingWidth),
            Args::RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels,
            RTAO::Args::QuarterResAO,
            Args::RTAODenoisingMinVarianceToDenoise,
            staleNeighborWeightScale,
            Args::AODenoiseDepthWeightCutoff,
            Args::RTAODenoisingUseProjectedDepthTest,
            forceDenoisePass,
            Args::RTAODenoisingFilterWeightByFrameAge);
    }

    // ToDo move these right before the call?
    resourceStateTracker->TransitionResource(&AOResources[AOResource::Smoothed], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    resourceStateTracker->TransitionResource(OutputIntermediateResource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}


// ToDo move out
void D3D12RaytracingAmbientOcclusion::ApplyAtrousWaveletTransformFilter(
    const GpuResource& inValueResource,
    const GpuResource& inNormalDepthResource,
    const GpuResource& inDepthResource,
    const GpuResource& inRayHitDistanceResource, 
    const GpuResource& inPartialDistanceDerivativesResource,
    GpuResource* outSmoothedValueResource,
    GpuResource* varianceResource,
    GpuResource* smoothedVarianceResource,
    UINT calculateVarianceTimerId,      // ToDo remove obsolete
    UINT smoothVarianceTimerId,
    UINT atrousFilterTimerId
)
{
    ThrowIfFalse(false, L"ToDo");

#if 0
    auto commandList = m_deviceResources->GetCommandList();
    
    auto desc = inValueResource->GetDesc();
    // ToDo cleanup widths on GPU kernels, it should be the one of input resource.
    UINT width = static_cast<UINT>(desc.Width);
    UINT height = static_cast<UINT>(desc.Height);

#if 0
    // Calculate local variance.
    {
        ScopedTimer _prof(L"CalculateVariance", commandList);
        resourceStateTracker->FlushResourceBarriers();
        m_calculateVarianceKernel.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            width,
            height,
            static_cast<GpuKernels::CalculateVariance::FilterType>(static_cast<UINT>(Args::VarianceBilateralFilter)),
            inValueResource.gpuDescriptorReadAccess,
            inNormalDepthResource.gpuDescriptorReadAccess,
            inDepthResource.gpuDescriptorReadAccess,
            varianceResource->gpuDescriptorWriteAccess,
            CD3DX12_GPU_DESCRIPTOR_HANDLE(),    // unused mean resource output
            Args::AODenoiseDepthSigma,
            Args::AODenoiseNormalSigma,
            false,
            Args::RTAODenoising_Variance_UseDepthWeights,
            Args::RTAODenoising_Variance_UseNormalWeights,
            Args::RTAOVarianceFilterKernelWidth);

        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        resourceStateTracker->TransitionResource(&varianceResource->resource.Get(), after));
    }

    // ToDo, should the smoothing be applied after each pass?
    // Smoothen the local variance which is prone to error due to undersampled input.
    {
        ScopedTimer _prof(L"VarianceSmoothing", commandList);
        resourceStateTracker->FlushResourceBarriers();
        m_gaussianSmoothingKernel.Execute(
            commandList,
            width,
            height,
            GpuKernels::GaussianFilter::Filter3x3,
            m_cbvSrvUavHeap->GetHeap(),
            varianceResource->gpuDescriptorReadAccess,
            smoothedVarianceResource->gpuDescriptorWriteAccess);
    }

    // Transition Variance resource to shader resource state.
    // Also prepare smoothed AO resource for the next pass and transition it to UAV.
    {
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES after = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        D3D12_RESOURCE_BARRIER barriers[] = {
            resourceStateTracker->TransitionResource(&smoothedVarianceResource->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
      // ToDo Remove      resourceStateTracker->TransitionResource(&outSmoothedValueResource->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
    }
#endif

#if RAYTRACING_MANUAL_KERNEL_STEP_SHIFTS
    UINT offsets[5] = {
        static_cast<UINT>(Args::RTAO_KernelStepShift0),
        static_cast<UINT>(Args::RTAO_KernelStepShift1),
        static_cast<UINT>(Args::RTAO_KernelStepShift2),
        static_cast<UINT>(Args::RTAO_KernelStepShift3),
        static_cast<UINT>(Args::RTAO_KernelStepShift4) };

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
        resourceStateTracker->FlushResourceBarriers();
        m_atrousWaveletTransformFilter.Execute(
            commandList,
            m_cbvSrvUavHeap->GetHeap(),
            static_cast<GpuKernels::AtrousWaveletTransformCrossBilateralFilter::FilterType>(static_cast<UINT>(Args::DenoisingMode)),
            inValueResource.gpuDescriptorReadAccess,
            inNormalDepthResource.gpuDescriptorReadAccess,
            inDepthResource.gpuDescriptorReadAccess,
            smoothedVarianceResource->gpuDescriptorReadAccess,
            inRayHitDistanceResource.gpuDescriptorReadAccess,
            inPartialDistanceDerivativesResource.gpuDescriptorReadAccess,
            m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorReadAccess,
            outSmoothedValueResource,
            Args::AODenoiseValueSigma,
            Args::AODenoiseDepthSigma,
            Args::AODenoiseNormalSigma,
            Args::RTAODenoising_WeightScale,
            static_cast<TextureResourceFormatRGB::Type>(static_cast<UINT>(Args::RTAO_TemporalSupersampling_NormalDepthResourceFormat)),
            offsets,
            Args::AtrousFilterPasses,
            GpuKernels::AtrousWaveletTransformCrossBilateralFilter::Mode::OutputFilteredValue,
            Args::ReverseFilterOrder,
            Args::UseSpatialVariance,
            Args::RTAODenoisingPerspectiveCorrectDepthInterpolation,
            Args::RTAODenoisingUseAdaptiveKernelSize,
            Args::RTAO_Denoising_AdaptiveKernelSize_MinHitDistanceScaleFactor,
            Args::RTAODenoisingFilterMinKernelWidth,
            static_cast<UINT>((Args::RTAODenoisingFilterMaxKernelWidthPercentage / 100) * m_raytracingWidth),
            Args::RTAODenoisingFilterVarianceSigmaScaleOnSmallKernels,
            RTAO::Args::QuarterResAO);
    }
#endif
}




void D3D12RaytracingAmbientOcclusion::DownsampleRaytracingOutput()
{
	auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();

    ScopedTimer _prof(L"DownsampleToBackbuffer", commandList);

    resourceStateTracker->TransitionResource(&m_raytracingOutputIntermediate, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    // ToDo pass the filter to the kernel instead of using 3 different instances
    resourceStateTracker->FlushResourceBarriers();
	switch (Args::AntialiasingMode)
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

	resourceStateTracker->TransitionResource(&m_raytracingOutputIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}


// Upsample quarter resources
void D3D12RaytracingAmbientOcclusion::UpsampleResourcesForRenderComposePass()
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
        inputLowResValueResource = &m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance];
        outputHiResValueResource = &m_AOResources[AOResource::RayHitDistance];
        break;
    }
    case CompositionType::AmbientOcclusionVariance:
    {
        passName = L"Upsample AO variance";
        inputLowResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_lowResVarianceResource[AOVarianceResource::Smoothed] : &m_lowResVarianceResource[AOVarianceResource::Raw];
        outputHiResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_varianceResource[AOVarianceResource::Smoothed] : &m_varianceResource[AOVarianceResource::Raw];
        break;
    }
    case CompositionType::AmbientOcclusionLocalVariance:
    {
        passName = L"Upsample AO local variance";
        filterType = GpuKernels::UpsampleBilateralFilter::Filter2x2RG;
        inputLowResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_lowResLocalMeanVarianceResource[AOVarianceResource::Smoothed] : &m_lowResLocalMeanVarianceResource[AOVarianceResource::Raw];
        outputHiResValueResource = Args::RTAODenoisingUseSmoothedVariance ? &m_localMeanVarianceResource[AOVarianceResource::Smoothed] : &m_localMeanVarianceResource[AOVarianceResource::Raw];
        break;
    }
    default:
        break;
    }

    if (inputLowResValueResource)
    {
        // ToDo move this within BilateralUpsample().
        GpuResource* GBufferLowResResources = m_pathtracer.GetGBufferResources(true);
        GpuResource* GBufferResources = m_pathtracer.GetGBufferResources();

        BilateralUpsample(
            m_GBufferWidth,
            m_GBufferHeight,
            filterType,
            inputLowResValueResource->gpuDescriptorReadAccess,
            GBufferLowResResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
            GBufferResources[GBufferResource::PartialDepthDerivatives].gpuDescriptorReadAccess,
            outputHiResValueResource,
            passName.c_str());
    }
}

// ToDo standardize naming AO vs AmbientOcclusion
void D3D12RaytracingAmbientOcclusion::BilateralUpsample(
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
    m_upsampleBilateralFilterKernel.Execute(
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

// Composite results from multiple passed into a final image.
void D3D12RaytracingAmbientOcclusion::RenderPass_ComposeRenderPassesCS(D3D12_GPU_DESCRIPTOR_HANDLE AOSRV)
{
	auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    ScopedTimer _prof(L"ComposeRenderPassesCS", commandList);
    
	// Update constant buffer.
	{
		m_csComposeRenderPassesCB->rtDimensions = XMUINT2(m_GBufferWidth, m_GBufferHeight);
        m_csComposeRenderPassesCB->enableAO = Args::AOEnabled;
        m_csComposeRenderPassesCB->compositionType = static_cast<CompositionType>(static_cast<UINT>(Args::CompositionMode));
        m_csComposeRenderPassesCB->defaultAmbientIntensity = Pathtracer::Args::DefaultAmbientIntensity;

        m_csComposeRenderPassesCB->variance_visualizeStdDeviation = Args::Compose_VarianceVisualizeStdDeviation;
        m_csComposeRenderPassesCB->variance_scale = Args::Compose_VarianceScale;
        m_csComposeRenderPassesCB->RTAO_MaxRayHitDistance = m_RTAO.GetMaxRayHitTime();



#if 0 // ToDo
        // ToDo use a unique CB for compose passes?
        m_csComposeRenderPassesCB->RTAO_UseAdaptiveSampling = Args::RTAOAdaptiveSampling;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMaxWeightSum = Args::RTAOAdaptiveSamplingMaxFilterWeight;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMinMaxSampling = Args::RTAOAdaptiveSamplingMinMaxSampling;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingScaleExponent = Args::RTAOAdaptiveSamplingScaleExponent;
        m_csComposeRenderPassesCB->RTAO_AdaptiveSamplingMinSamples = Args::RTAOAdaptiveSamplingMinSamples;
        m_csComposeRenderPassesCB->RTAO_MaxSPP = Args::AOSampleCountPerDimension * Args::AOSampleCountPerDimension;
#endif
        m_csComposeRenderPassesCB.CopyStagingToGpu(frameIndex);
	}

    // ToDo cleanup

	// Set pipeline state.
	{
		using namespace ComputeShader::RootSignature::ComposeRenderPassesCS;


        GpuResource* VarianceResource = Args::RTAODenoisingUseSmoothedVariance ? &m_varianceResource[AOVarianceResource::Smoothed] : &m_varianceResource[AOVarianceResource::Raw];
        GpuResource* LocalMeanVarianceResource = &m_localMeanVarianceResource[AOVarianceResource::Raw];
        GpuResource* RayHitDistance = RTAO::Args::QuarterResAO ? &m_AOResources[AOResource::RayHitDistance] : &m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::RayHitDistance];


		commandList->SetDescriptorHeaps(1, m_cbvSrvUavHeap->GetAddressOf());
		commandList->SetComputeRootSignature(m_computeRootSigs[CSType::ComposeRenderPassesCS].Get());
		commandList->SetPipelineState(m_computePSOs[CSType::ComposeRenderPassesCS].Get());

		// Bind outputs.
		commandList->SetComputeRootDescriptorTable(Slot::Output, m_raytracingOutputIntermediate.gpuDescriptorWriteAccess);
		
		// Bind inputs.
		commandList->SetComputeRootDescriptorTable(Slot::GBufferResources, Pathtracer::g_GBufferResources[0].gpuDescriptorReadAccess);
#if TWO_STAGE_AO_BLUR && !ATROUS_DENOISER
		commandList->SetComputeRootDescriptorTable(Slot::AO, m_AOResources[AOResource::Coefficient].gpuDescriptorReadAccess);
#else
        commandList->SetComputeRootDescriptorTable(Slot::AO, AOSRV);
#endif
		commandList->SetComputeRootDescriptorTable(Slot::Visibility, m_VisibilityResource.gpuDescriptorReadAccess);
		commandList->SetComputeRootShaderResourceView(Slot::MaterialBuffer, m_materialBuffer.GpuVirtualAddress());
		commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_csComposeRenderPassesCB.GpuVirtualAddress(frameIndex));
        commandList->SetComputeRootDescriptorTable(Slot::Variance, VarianceResource->gpuDescriptorReadAccess);
        commandList->SetComputeRootDescriptorTable(Slot::LocalMeanVariance, LocalMeanVarianceResource->gpuDescriptorReadAccess);

        commandList->SetComputeRootDescriptorTable(Slot::FilterWeightSum, m_AOResources[AOResource::FilterWeightSum].gpuDescriptorReadAccess);
        commandList->SetComputeRootDescriptorTable(Slot::AORayHitDistance, RayHitDistance->gpuDescriptorReadAccess);
        commandList->SetComputeRootDescriptorTable(Slot::Color, Pathtracer::g_GBufferResources[GBufferResource::Color].gpuDescriptorReadAccess);
        commandList->SetComputeRootDescriptorTable(Slot::AOSurfaceAlbedo, Pathtracer::g_GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorReadAccess);

        
        commandList->SetComputeRootDescriptorTable(Slot::FrameAge, m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorReadAccess);
	}

	// Dispatch.
	XMUINT2 groupSize(CeilDivide(m_GBufferWidth, DefaultComputeShaderParams::ThreadGroup::Width), CeilDivide(m_GBufferHeight, DefaultComputeShaderParams::ThreadGroup::Height));

    resourceStateTracker->FlushResourceBarriers();
	commandList->Dispatch(groupSize.x, groupSize.y, 1);
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
			0.001f* NumMPixelsPerSecond(GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_GBuffer), m_GBufferWidth, m_GBufferHeight)  << " GigaRay/s\n";
        // ToDo use profiler from MiniEngine
		float numAOGigaRays = 1e-6f * m_numCameraRayGeometryHits[ReduceSumCalculations::CameraRayHits] * (RTAO::Args::QuarterResAO ? 0.25f : 1) * m_sppAO / GpuTimeManager::instance().GetAverageMS(GpuTimers::Raytracing_AO);
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
        wLabel << L" " << L"AS update mode: " << Args::ASUpdateMode << L"\n";
        wLabel.precision(3);
        wLabel << L" " << L"AS memory footprint: " << static_cast<double>(m_ASmemoryFootprint)/(1024*1024) << L"MB\n";
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
                   L"RenderPass_TemporalSupersamplingReverseProjection",
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

	switch (Args::AntialiasingMode)
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

    if (RTAO::Args::QuarterResAO)
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
    m_atrousWaveletTransformFilter.CreateInputResourceSizeDependentResources(device, m_cbvSrvUavHeap.get(), m_raytracingWidth, m_raytracingHeight, m_RTAO.GetAOCoefficientFormat());
    
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

    m_cbvSrvUavHeap.reset();
    m_csHemisphereVisualizationCB.Release();

    m_pathtracer.ReleaseDeviceDependentResources();
    m_RTAO.ReleaseDeviceDependentResources();

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

// ToDo remove?
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
    resourceStateTracker->FlushResourceBarriers();
    commandList->Dispatch(rngWindowSize.x, rngWindowSize.y, 1);
#endif
}



void D3D12RaytracingAmbientOcclusion::CreateSamplesRNGVisualization()
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


// Render the scene.
void D3D12RaytracingAmbientOcclusion::OnRender()
{
    if (!m_deviceResources->IsWindowVisible())
    {
        return;
    }

#if 1
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

            if (!(Args::TAO_LazyRender && m_cameraChangedIndex <= 0))
            {

#if USE_GRASS_GEOMETRY
                GenerateGrassGeometry();
#endif
                UpdateAccelerationStructure();

                // Render.
                m_pathtracer.OnRender(g_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress());

                // AO. 
                if (Args::AOMode == Args::AOType::RTAO)
                {
                    ScopedTimer _prof(L"RTAO_Root", commandList);

                    GpuResource* GBufferResources = m_pathtracer.GetGBufferResources(RTAO::Args::QuarterResAO);

                    RenderPass_TemporalSupersamplingReverseProjection();
                    m_RTAO.OnRender(
                        g_accelerationStructure->GetTopLevelASResource()->GetGPUVirtualAddress(),
                        GBufferResources[GBufferResource::HitPosition].gpuDescriptorReadAccess,
                        GBufferResources[GBufferResource::SurfaceNormalDepth].gpuDescriptorReadAccess,
                        GBufferResources[GBufferResource::AOSurfaceAlbedo].gpuDescriptorReadAccess,
                        m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::FrameAge].gpuDescriptorReadAccess);

                    RenderPass_TemporalSupersamplingBlendWithCurrentFrame();

                    ApplyAtrousWaveletTransformFilter(true);

                    if (Args::RTAODenoisingLowTspp)
                    {
                        MultiPassBlur();
                    }
                    if (Args::RTAODenoisingUseMultiscale)
                    {
                        ApplyMultiScaleAtrousWaveletTransformFilter(false);
                    }
                    else
                    {

#if 0
                        if (Args::RTAO_TemporalSupersampling_CacheDenoisedOutput)
                        {

                            // Cache current frame's normal depth buffer.
                            GpuResource* TSSAOCoefficient = RTAO::Args::QuarterResAO ? m_lowResTSSAOCoefficient : m_TSSAOCoefficient;
                            GpuResource* AOResources = m_RTAO.AOResources();

                            ToDo
                            commandList->ResourceBarrier(1, &resourceStateTracker->InsertUAVBarrier(&AOResources[AOResource::Smoothed]));

                            resourceStateTracker->FlushResourceBarriers();
                            CopyTextureRegion(
                                commandList,
                                AOResources[AOResource::Smoothed],
                                TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex],
                                &CD3DX12_BOX(0, 0, m_raytracingWidth, m_raytracingHeight),
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

                            if (Args::RTAO_TemporalSupersampling_CacheSquaredMean)
                            {
                                resourceStateTracker->FlushResourceBarriers();
                                CopyTextureRegion(
                                    commandList,
                                    m_atrousWaveletTransformFilter.VarianceOutputResource(),
                                    m_temporalCache[m_temporalCacheCurrentFrameResourceIndex][TemporalSupersampling::CoefficientSquaredMean],
                                    &CD3DX12_BOX(0, 0, m_raytracingWidth, m_raytracingHeight),
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            }
                            commandList->ResourceBarrier(1, &resourceStateTracker->InsertUAVBarrier(&TSSAOCoefficient[m_temporalCacheCurrentFrameTSSAOCoefficientResourceIndex]));
                        }
#endif
                        //ApplyAtrousWaveletTransformFilter(false);
                    }

                    if (RTAO::Args::QuarterResAO)
                    {
                        UpsampleResourcesForRenderComposePass();
                    }
                    else // ToDo move this to ApplyAtrousWaveletTransformFilter?
                    {
                        // Transition AO Smoothed resource to SRV.
                        //resourceStateTracker->TransitionResource(&m_AOResources[AOResource::Smoothed], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
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
            }

            // ToDo ping-pong the resource instead of copy
            {
                GpuResource* GBufferResources = m_pathtracer.GetGBufferResources(RTAO::Args::QuarterResAO);
                CopyTextureRegion(
                    commandList,
                    GBufferResources[GBufferResource::SurfaceNormalDepth].GetResource(),
                    m_prevFrameGBufferNormalDepth.GetResource(),
                    &CD3DX12_BOX(0, 0, m_raytracingWidth, m_raytracingHeight),
                    GBufferResources[GBufferResource::SurfaceNormalDepth].m_UsageState,
                    m_prevFrameGBufferNormalDepth.m_UsageState);
            }

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

   // Args::TAO_LazyRender.Bang();
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
}