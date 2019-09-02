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

#ifndef RAYTRACINGHLSLCOMPAT_H
#define RAYTRACINGHLSLCOMPAT_H

/*
//ToDo
// ToDo switch to default column major in hlsl and do a transpose before passing matrices to HLSL.
- todo finetune clamping/remove ghosting (test gliding spaceship)
- Adaptive kernel size - overblur under roof edge
- Full res - adaptive kernel size visible horizontal lines
- test impact of depth hash key in RaySort
- fix adaptive kernel size
- improve the multi-blur - skip higher iter blur on higher frame age.
- Run ray gen only for active pixels on checkerboard. Run Ray sort only for active pixels and combine two groups? 128x128?
- initialize resources
- progressive samplin
- depth aware variance calculation
- lower temporal blur on motion
- Add bounce ID as edge stopping function
Optimization
- Skip
- Get RTAO perf close to 50% at 50% sampling.
- combine resources, lower bit format (ray hit distance)
- RaySorting - test if need inverted indices and strip them if not. Strip depth bounds calculation?

- demo video
--- increase bounce
-- comparison 100spp AO, 1spp, 1/2 spp
-- PBR lighting AO ON / Off
-- reflections



- Multi-scale denoiser
- consider alternating min/max pattern on downsampling
- 3x3 vs 5x5 blur at lower resolutions
- Atrous vs separable at lower resolutions
- mirror weights
- weigh samples with higher tspp more when up/down sampling
- test energy conservation on ~4/16 spp at tspp up to 32
- use split barriers vs UAV if there's a work in between? vs UAV barrier right before the read?


- set max bounce to 2/3 - support windows in reflections.
- match denoised  AO at fidelity closer to that of temporal variance sharpness image
- improve matching on Temporal. Dragon surface hits lots of likely unnecessary disocclusions on camero movement/zoom.
- fix the temporal variance shimmer on boundaries with skybox.
- map local variance to only valid AO values
- retain per window frame seed on static geometry

- TAO fails on dragons surface on small rotaions
- ToDo motion vector can be nan at some reflections and reprojected depth of bad value. [1638,647 x 1440p]
- very large ddxy at 525 199
- blur away disocclussions
- add dynamic objects testing clamping
- overblur on mouse camera movement
-  no modes, no AOon phong, denoised artifacts onb normalMaps
- Fireflies
- quad blur in Variance
- AO raypayload of 2B.
- split temporal pass to Reprojection and Clamping. 
    -Use reprojection to drive AO spp. 
    - Cache kernel weight sum, min hit distance, frame age, variance and reproject to drive ao sampling.
- Temporal:
   - Fine tune min std dev tolerance in clamping
   - Try lower mip level on disocclusion.
 - option to disable variance smoothing
- ToDo test AO perf w/o tHit - if meaningful look into heuristic limitting ray groups to that. Maybe those that don't get sorted.
- Double check that passes either check whether a pixel value is valid from previous pass/frame or the value gets ignored.
- Optimizations:
    - Encode ray hit distance to 8bit.
    - replace multiple loads with gathers.
    - tighten texture format sizes

- Glitches
    - RayHitDistance is wrong behind tires.
    - clean up PIX /GPU validation warnings on Debug
    - Debug break on FS on 1080 display resolution.
    - Tearing with VSync on at 4K full res.
    - White halo under tiers.
    - Upsampling artifacts on 1080(540p)
    - RTAO invisible wall inside kitchen on long rays

- Cleanup:
    - ToDo remove .f specifier from floating numbers in hlsl
    - ToDo clean up scoped timer names.
    - Add/revise comments
    - Add UAV barrier to SRV - UAV transitions. Remove post D3D barriers.
    - Move global defines in RaytracingSceneDefines.h locally for RTAO and Denoiser.
    - Add dtors/release . Wait on GPU?
    - Build with higher warning bar and cleanup

- Sample generic
    - Add device removal support


*/
// Workarounds - ToDo remove/document
#define REPRO_DEVICE_REMOVAL_ON_HARD_CODED_AO_COEF 0
#define REPRO_INVISIBLE_WALL 0

//**********************************************************************************************
//
// RaytracingHLSLCompat.h
//
// A header with shared definitions for C++ and HLSL source files. 
//
//**********************************************************************************************

#define FOVY 45.f
// ToDo remove
#define NEAR_PLANE 0.001f
#define FAR_PLANE 1000.0f   // ToDo pass form the app

#define MARK_PERFECT_MIRRORS_AS_NOT_OPAQUE 1

#define RAYTRACING_MANUAL_KERNEL_STEP_SHIFTS 1
#define AO_SPP_N 1
#define AO_SPP_N_MAX 1          // ToDo Uses no loop on 1 => saves 0.3 ms on TitanXp at 1080p

#define SIGNED_DDXY 1           // Preserve the sign of ddxy

#define MOVE_ONCE_ON_STRAFE 1


#define RTAO_MARK_CACHED_VALUES_NEGATIVE 1
#define RTAO_GAUSSIAN_BLUR_AFTER_Temporal 0
#if RTAO_GAUSSIAN_BLUR_AFTER_Temporal && RTAO_MARK_CACHED_VALUES_NEGATIVE
Incompatible macros
#endif
#define STOP_TRACING_AND_DENOISING_AFTER_FEW_FRAMES 0

// ToDo TAO is swimming in reflections
#define CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN 0
#define USE_UV_DERIVATIVES 0
#define HACK_CLAMP_DDXY_TO_BE_SMALL 1

//#define SAMPLER_FILTER D3D12_FILTER_MIN_MAG_MIP_LINEAR
#define SAMPLER_FILTER D3D12_FILTER_ANISOTROPIC  // TODo blurry at various angles

// ToDo
#define ENABLE_PROFILING 0
#define ENABLE_LAZY_RENDER 0
#define ENABLE_SSAA 0

#define DISTANCE_ON_MISS 65504  // ~FLT_MAX within 16 bit format // ToDo explain

#define PRINT_OUT_TC_MATRICES 0
#define PRINT_OUT_CAMERA_CONFIG 0
#define DEBUG_PRINT_OUT_SEED_VALUE 0

#ifdef HLSL
typedef uint NormalDepthTexFormat;
#else
#define COMPACT_NORMAL_DEPTH_DXGI_FORMAT DXGI_FORMAT_R32_UINT
#endif
#define GBUFFER_AO_COUNT_AO_HITS 0

// ToDO enable Vsync via cmdline/or UI
#define ENABLE_VSYNC 1
#if ENABLE_VSYNC
#define VSYNC_PRESENT_INTERVAL 1  
#endif

// ToDo Fix missing DirectXTK12.lib in Profile config - as the nuget doesnt provide profile
// ToDo remove PROFILE preprocesser macro from Release

#define ATROUS_DENOISER 1
#define ATROUS_DENOISER_MAX_PASSES 10
#define RENDER_RNG_SAMPLE_VISUALIZATION 0   // ToDo doesn't render for all AA settings
#define ATROUS_ONELEVEL_ONLY 0

#define APPLY_SRGB_CORRECTION 0

// ToDO this wasn't necessary before..
#define VBIB_AS_NON_PIXEL_SHADER_RESOURCE 0 // ToDo spec requires it but it works without it?

#define USE_GRASS_GEOMETRY 1
#define GRASS_NO_DEGENERATE_INSTANCES 1 // Degenerate instances cause long trace ray times

#define LOAD_PBRT_SCENE 1       // loads PBRT(1) or SquidRoom(0)
#ifdef _DEBUG
#define LOAD_ONLY_ONE_PBRT_MESH 1  // for LOAD_PBRT_SCENE == 1 only
#else
#define LOAD_ONLY_ONE_PBRT_MESH 0  // for LOAD_PBRT_SCENE == 1 only
#endif
#define FACE_CULLING !LOAD_PBRT_SCENE

#if LOAD_PBRT_SCENE
#define DISTANCE_FALLOFF 0.000000005
#define AO_RAY_T_MAX 22
#define SCENE_SCALE 300     
#define GENERATE_GRASS 1
#else
#define GENERATE_GRASS 0
#define DISTANCE_FALLOFF 0
#define AO_RAY_T_MAX 150
#define SCENE_SCALE 2000
#endif
#define INDEX_FORMAT_UINT 1



// ToDo separate per-vertex attributes from VB
// ToDo dedupe the ones matching default
// ToDo move
// ToDo encapsulate under CS namespace
namespace ReduceSumCS {
	namespace ThreadGroup {
		enum Enum { Width = 8, Height = 16, Size = Width * Height, NumElementsToLoadPerThread = 10 };	
	}
}

// ToDo perf vs 8x8?
namespace AtrousWaveletTransformFilterCS {
    namespace ThreadGroup {
        enum Enum { Width = 16, Height = 16, Size = Width * Height };
    }
}

// ToDo combine and reuse a default 8x8 ?
namespace DefaultComputeShaderParams {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8, Size = Width * Height };
    }
}


#ifdef HLSL
#include "util\HlslCompat.h"
#if INDEX_FORMAT_UINT
typedef UINT Index;
#endif
#else
using namespace DirectX;

#if INDEX_FORMAT_UINT
typedef UINT Index;
#else
typedef UINT16 Index;
#endif
#endif


// ToDo revise
// PERFORMANCE TIP: Set max recursion depth as low as needed
// as drivers may apply optimization strategies for low recursion depths.
#define MAX_RAY_RECURSION_DEPTH 5    // ~ primary rays + 2 x reflections + shadow rays from reflected geometry.  ToDo
// ToDo add recursion viz

// ToDo:
// Options:
// - shading - simple/complex
// - instanced/unique goemetry
// - deformed geometry
// - Dynamic options
// - Update/Build

// ToDo clean up
struct ProceduralPrimitiveAttributes
{
    XMFLOAT3 normal;
};

struct RayPayload
{
    XMFLOAT4 color;
    UINT   recursionDepth; // encode?
};

struct Ray
{
    XMFLOAT3 origin;
    XMFLOAT3 direction;
};

struct AmbientOcclusionGBuffer
{
    float tHit;
    XMFLOAT3 hitPosition;           // Position of the hit for which to calculate Ambient coefficient.
    UINT diffuseByte3;              // Diffuse reflectivity of the hit surface.
    XMFLOAT2 encodedNormal;         // Normal of the hit surface. // ToDo encode as 16bit

    // Members for Motion Vector calculation.
    XMFLOAT3 _virtualHitPosition;   // virtual hitPosition in the previous frame.
                                    // For non-reflected points this is a true world position of a hit.
                                    // For reflected points, this is a world position of a hit reflected across the reflected surface 
                                    //   ultimately giving the same screen space coords when projected and the depth corresponding to the ray depth.
    XMFLOAT2 _encodedNormal;        // normal in the previous frame
};


// ToDo rename To Pathtracer
struct GBufferRayPayload
{
    UINT rayRecursionDepth;
    XMFLOAT3 radiance;
	//XMUINT2 materialInfo;   // {materialID, 16b 2D texCoord}
    AmbientOcclusionGBuffer AOGBuffer;
#if USE_UV_DERIVATIVES
    Ray rx;    // Auxilary camera ray offset by one pixel in x dimension in screen space.
    Ray ry;    // Auxilary camera ray offset by one pixel in y dimension in screen space.
#endif
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
    float rxTHit;
    float ryTHit;
#endif
};

struct ShadowRayPayload
{
    float tHit;         // Hit time <0,..> on Hit. -1 on miss.
};

struct RNGConstantBuffer
{
    XMUINT2 uavOffset;     // offset where [0,0] thread should write to.
    XMUINT2 dispatchDimensions;  // for 2D dispatches
	UINT sampleSetBase;
    UINT numSamples;
    UINT numSampleSets;
    UINT numSamplesToShow; 
    // TODo: Why is padding to 16 needed? cb gets corrupted otherwise. Put a static_assert in ConstantBuffer
    XMUINT2 stratums;      // Stratum resolution
    XMUINT2 grid;      // Grid resolution
};

// ToDo remove obsolete params in cbs

struct AtrousWaveletTransformFilterConstantBuffer
{
    // ToDo pad?
    XMUINT2 textureDim;
    UINT kernelStepShift;
    UINT kernelWidth;

    float valueSigma;
    float depthSigma;
    float normalSigma;
    UINT useCalculatedVariance;
    
    UINT useApproximateVariance;
    BOOL outputFilteredValue;
    BOOL outputFilteredVariance;
    BOOL outputFilterWeightSum;

    BOOL perspectiveCorrectDepthInterpolation;
    BOOL useAdaptiveKernelSize;
    float minHitDistanceToKernelWidthScale;
    UINT minKernelWidth;

    UINT maxKernelWidth;
    float varianceSigmaScaleOnSmallKernels;
    bool usingBilateralDownsampledBuffers;
    UINT DepthNumMantissaBits;

    float minVarianceToDenoise;
    float weightScale;
    float staleNeighborWeightScale;
    float depthWeightCutoff;

    BOOL useProjectedDepthTest;
    BOOL forceDenoisePass;
    float kernelStepScale;
    float weightByFrameAge;
};

// ToDo remove obsolete params in cbs
struct CalculateVariance_BilateralFilterConstantBuffer
{
    XMUINT2 textureDim;
    float normalSigma;
    float depthSigma;

    BOOL outputMean;
    BOOL useDepthWeights;
    BOOL useNormalWeights;
    UINT kernelWidth;

    UINT kernelRadius;
    float padding[3];
};

struct CalculateMeanVarianceConstantBuffer
{
    XMUINT2 textureDim;
    UINT kernelWidth;
    UINT kernelRadius;

    BOOL doCheckerboardSampling;
    BOOL areEvenPixelsActive;
    UINT pixelStepY;
    float padding;
};

// ToDo standardzie capitalization
struct AdaptiveRayGenConstantBuffer
{
    XMUINT2 textureDim;
    XMUINT2 QuadDim;

    UINT MaxFrameAge;
    UINT MinFrameAgeForAdaptiveSampling;    // Frame age at which the adaptive sampling kicks in.
    UINT FrameID;       // Looping FrameID within <0, QuadDim.size - 1>
    UINT MaxRaysPerQuad;

    UINT seed;
    UINT numSamplesPerSet;
    UINT numSampleSets;
    UINT numPixelsPerDimPerSet;

    UINT MaxFrameAgeToGenerateRaysFor;
    BOOL doCheckerboardRayGeneration;
    BOOL checkerboardGenerateRaysForEvenPixels;
    float padding;
};

struct SortRaysConstantBuffer
{
    XMUINT2 dim;

    BOOL useOctahedralRayDirectionQuantization;

    // Depth for a bin within which to sort further based on direction.
    float binDepthSize;
};

#define RTAO_RAY_SORT_1DRAYTRACE 1
#define RTAO_RAY_SORT_ENUMERATE_ELEMENT_ID_IN_MORTON_CODE 0
#define RTAO_RAY_SORT_STORE_RAYS_IN_MORTON_ORDER_X_MAJOR 0
#define RTAO_RAY_SORT_MORTON_MIRROR 0
#define RTAO_RAY_SORT_Y_AXIS_MAJOR 0

#define RTAO_RAY_SORT_NO_SMEM 0
#if RTAO_RAY_SORT_NO_SMEM
namespace SortRays {
    namespace ThreadGroup {
        enum Enum { Width = 256, Height = 128, Size = Width * Height };
    }
    namespace RayGroup {
        enum Enum { Width = ThreadGroup::Width, Height = 2 * ThreadGroup::Height, Size = Width * Height };
    }
}
#else


namespace SortRays {
    namespace ThreadGroup {
        enum Enum { Width = 64, Height = 16, Size = Width * Height };
    }


    // ToDo comment ray group's heigh can only go up to 64 as the most significant bit is used to test if the cached value is valid.
    namespace RayGroup {
        enum Enum { NumElementPairsPerThread = 4, Width = ThreadGroup::Width, Height = NumElementPairsPerThread * 2 * ThreadGroup::Height, Size = Width * Height };
    }
#ifndef HLSL
    static_assert( RayGroup::Width < 128 
                && RayGroup::Height < 256
                && RayGroup::Size <= 8192, "Ray group dimensions are outside the supported limits set by the Counting Sort shader.");
#endif
}
#endif


// ToDo split CB?
// ToDo capitalize?
// ToDo padding? or force align.
// ToDo remove unused
// ToDo PIX shows empty rows (~as many as valid rows) in between entries in multi frame CB.
struct PathtracerConstantBuffer
{
    // ToDo rename to world to view matrix and drop (0,0,0) note.
    XMMATRIX projectionToWorldWithCameraEyeAtOrigin;	// projection to world matrix with Camera at (0,0,0).
    XMFLOAT3 cameraPosition;
    BOOL     useDiffuseFromMaterial;
    XMFLOAT3 lightPosition;     // ToDo use float3
    BOOL     useNormalMaps;
    XMFLOAT3 lightColor;
    float    defaultAmbientIntensity;

    XMMATRIX prevViewProj;    // ToDo standardzie proj vs projection
    XMMATRIX prevProjToWorldWithCameraEyeAtOrigin;	// projection to world matrix with Camera at (0,0,0).
    XMFLOAT3 prevCameraPosition;
    float    padding;

	float Znear;     // ToDo rename to zNear | remove
	float Zfar;
    UINT  maxRadianceRayRecursionDepth;
    UINT  maxShadowRayRecursionDepth;

};

// ToDo split CB?
// ToDo capitalize?
// ToDo cleanup padding?
// ToDo remove RTAO prefix
struct RTAOConstantBuffer
{
    UINT seed;
    UINT numSamplesPerSet;
    UINT numSampleSets;
    UINT numPixelsPerDimPerSet;

    // ToDo rename to AOray
    float RTAO_maxShadowRayHitTime;             // Max shadow ray hit time used for tMax in TraceRay.
    BOOL RTAO_approximateInterreflections;      // Approximate interreflections. 
    float RTAO_diffuseReflectanceScale;              // Diffuse reflectance from occluding surfaces. 
    float RTAO_MinimumAmbientIllumination;       // Ambient illumination coef when a ray is occluded.

    // toDo rename shadow to AO
    float RTAO_maxTheoreticalShadowRayHitTime;  // Max shadow ray hit time used in falloff computation accounting for
                                                // RTAO_ExponentialFalloffMinOcclusionCutoff and RTAO_maxShadowRayHitTime.    
    BOOL RTAO_UseSortedRays;
    XMUINT2 raytracingDim;

    BOOL RTAO_IsExponentialFalloffEnabled;               // Apply exponential falloff to AO coefficient based on ray hit distance.    
    float RTAO_exponentialFalloffDecayConstant;
    BOOL doCheckerboardSampling;
    BOOL areEvenPixelsActive;
};

 
// ToDo use namespace?
// Final render output composition modes.
enum CompositionType {
    PBRShading = 0,  // ToDo rename
    AmbientOcclusionOnly_Denoised,
    AmbientOcclusionOnly_TemporallySupersampled,
    AmbientOcclusionOnly_RawOneFrame,
    AmbientOcclusionAndDisocclusionMap, // ToDo quarter res support
    AmbientOcclusionVariance,
    AmbientOcclusionLocalVariance,  // ToDo rename spatial to local variance references
    RTAOHitDistance,    // ToDo standardize naming
    NormalsOnly,
    DepthOnly,
    Diffuse,
    DisocclusionMap,
    Count
};

namespace TextureResourceFormatRGB
{
    enum Type {
        R32G32B32A32_FLOAT = 0,
        R16G16B16A16_FLOAT,
        R11G11B10_FLOAT,
        Count
    };
#ifndef HLSL
    inline DXGI_FORMAT ToDXGIFormat(UINT type)
    {
        switch (type)
        {
        case R32G32B32A32_FLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case R16G16B16A16_FLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case R11G11B10_FLOAT: return DXGI_FORMAT_R11G11B10_FLOAT;
        }
        return DXGI_FORMAT_UNKNOWN;
    }
#endif
}

namespace TextureResourceFormatR
{
    enum Type {
        R32_FLOAT = 0,
        R16_FLOAT,
        R8_UNORM,
        Count
    };
#ifndef HLSL
    inline DXGI_FORMAT ToDXGIFormat(UINT type)
    {
        switch (type)
        {
        case R32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
        case R16_FLOAT: return DXGI_FORMAT_R16_FLOAT;
        case R8_UNORM: return DXGI_FORMAT_R8_UNORM;
        }
        return DXGI_FORMAT_UNKNOWN;
    }
#endif
}


namespace TextureResourceFormatRG
{
    enum Type {
        R32G32_FLOAT = 0,
        R16G16_FLOAT,
        R8G8_SNORM,
        Count
    };
#ifndef HLSL
    inline DXGI_FORMAT ToDXGIFormat(UINT type)
    {
        switch (type)
        {
        case R32G32_FLOAT: return DXGI_FORMAT_R32G32_FLOAT;
        case R16G16_FLOAT: return DXGI_FORMAT_R16G16_FLOAT;
        case R8G8_SNORM: return DXGI_FORMAT_R8G8_SNORM;
        }
        return DXGI_FORMAT_UNKNOWN;
    }
#endif
}

// ToDo compress
// ToDo explain padding
struct ComposeRenderPassesConstantBuffer
{
    CompositionType compositionType;
    UINT isAOEnabled;
    float RTAO_MaxRayHitDistance;   // ToDo standardize ray hit time vs distance
    float defaultAmbientIntensity;
    
    BOOL variance_visualizeStdDeviation;
    float variance_scale;
    float padding3[2];
};

// ToDo standardize Texture vs Tex, Dim ...
struct DownsampleFilterConstantBuffer
{
	XMUINT2 inputTextureDimensions;
	XMFLOAT2 invertedInputTextureDimensions;
};

struct TextureDimConstantBuffer
{
    XMUINT2 textureDim;
    XMFLOAT2 invTextureDim;
};


// ToDo capitalize CB members?
struct FilterConstantBuffer
{
    XMUINT2 textureDim;
    UINT step;
    float padding;
};


// ToDo rename be more specific
struct BilateralFilterConstantBuffer
{
    XMUINT2 textureDim;
    UINT step;
    BOOL readWriteUAV_and_skipPassthrough;

    float normalWeightExponent;
    float minNormalWeightStrength;
    float padding[2];
};

struct TemporalSupersampling_ReverseReprojectConstantBuffer
{
    // ToDo pix missinterprets the format
    XMUINT2 textureDim;
    XMFLOAT2 invTextureDim; // ToDo test what impact passing inv tex dim makes

    XMMATRIX projectionToWorldWithCameraEyeAtOrigin;
    XMMATRIX prevProjectionToWorldWithCameraEyeAtOrigin;

    // ToDo moving this 4Bs above XMFLOATs causes issues
    BOOL useDepthWeights;
    BOOL useNormalWeights;
    float depthSigma;
    float depthTolerance;

    BOOL useWorldSpaceDistance;
    UINT DepthNumMantissaBits;      // Number of Mantissa Bits in the floating format of the input depth resources format.
    BOOL usingBilateralDownsampledBuffers;
    BOOL perspectiveCorrectDepthInterpolation;

    float floatEpsilonDepthTolerance;
    float depthDistanceBasedDepthTolerance;
    UINT numRaysToTraceAfterTemporalAtMaxFrameAge;
    UINT maxFrameAge;       // ToDo rename maxFrameAge to tspp

    BOOL testFlag;
};

struct TemporalSupersampling_BlendWithCurrentFrameConstantBuffer
{
    // ToDo pix missinterprets the format
    XMUINT2 textureDim;
    XMFLOAT2 invTextureDim; // ToDo test what impact passing inv tex dim makes

    BOOL  forceUseMinSmoothingFactor;  // ToDo remove?
    BOOL clampCachedValues;
    float minSmoothingFactor;
    float stdDevGamma;

    UINT minFrameAgeToUseTemporalVariance;
    float minStdDevTolerance;
    float frameAgeAdjustmentDueClamping;
    float clampDifferenceToFrameAgeScale;

    UINT numFramesToDenoiseAfterLastTracedRay;
    UINT blurStrength_MaxFrameAge;
    float blurDecayStrength;
    float padding;

    BOOL doCheckerboardSampling;
    BOOL areEvenPixelsActive;
    float padding2[2];

};

struct CalculatePartialDerivativesConstantBuffer
{
    XMUINT2 textureDim;
    float padding[2];
};

struct DownAndUpsampleFilterConstantBuffer
{
    XMFLOAT2 invHiResTextureDim;
    XMFLOAT2 invLowResTextureDim;

    // ToDo remove
    BOOL useNormalWeights;
    BOOL useDepthWeights;
    BOOL useBilinearWeights;
    BOOL useDynamicDepthThreshold;
};

struct GenerateGrassStrawsConstantBuffer_AppParams
{
    XMUINT2 activePatchDim; // Dimensions of active grass straws.
    XMUINT2 maxPatchDim;    // Dimensions of the whole vertex buffer.

    XMFLOAT2 timeOffset;
    float grassHeight;
    float grassScale;
    
    XMFLOAT3 patchSize;
    float grassThickness;

    XMFLOAT3 windDirection;
    float windStrength;

    float positionJitterStrength;
    float bendStrengthAlongTangent;
    float padding[2];
};

// ToDo move?
#define N_GRASS_TRIANGLES 5
#define N_GRASS_VERTICES 7
#define MAX_GRASS_STRAWS_1D 100
struct GenerateGrassStrawsConstantBuffer
{
    XMFLOAT2 invActivePatchDim;
    float padding1; // ToDo doing float p[2]; instead adds extra padding - as per PIX.
    float padding2;
    GenerateGrassStrawsConstantBuffer_AppParams p;
};


namespace BxDFType {
    enum Type {
        BSDF_DIFFUSE        = 1 << 0,
        BSDF_SPECULAR       = 1 << 1,
        BSDF_REFLECTION     = 1 << 2,
        BSDF_TRANSMISSION   = 1 << 3,
        BSDF_ALL            = BSDF_DIFFUSE | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION
    };
}


// Attributes per primitive type.
struct PrimitiveConstantBuffer
{
	UINT     materialID;          
    UINT     isVertexAnimated; 
    UINT     padding[2];
};

namespace MaterialType {
    enum Type {
        Default,
        Matte,  // Lambertian scattering
        Mirror,   // Specular reflector that isn't modified by the Fersnel equations.
        AnalyticalCheckerboardTexture
    };
}

// ToDO use same naming as in PBR Material
struct PrimitiveMaterialBuffer
{
	XMFLOAT3 Kd;
	XMFLOAT3 Ks;
    XMFLOAT3 Kr;
    XMFLOAT3 Kt;
    XMFLOAT3 opacity;
    XMFLOAT3 eta;
    float roughness;
    // ToDo use a bitmask?
    UINT hasDiffuseTexture; // ToDO use BOOL?
    UINT hasNormalTexture;
    UINT hasPerVertexTangents;
    MaterialType::Type type;
    float padding;
};

// Attributes per primitive instance.
struct PrimitiveInstanceConstantBuffer
{
    // ToDo should this be padded?
    UINT instanceIndex;  
    UINT primitiveType; // Procedural primitive type
};

// Dynamic attributes per primitive instance.
struct PrimitiveInstancePerFrameBuffer
{
    XMMATRIX localSpaceToBottomLevelAS;   // Matrix from local primitive space to bottom-level object space.
    XMMATRIX bottomLevelASToLocalSpace;   // Matrix from bottom-level object space to local primitive space.
};

struct AlignedUnitSquareSample2D
{
    XMFLOAT2 value;
    XMUINT2 padding;  // Padding to 16B
};

struct AlignedHemisphereSample3D
{
    XMFLOAT3 value;
    UINT padding;  // Padding to 16B
};

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
};

struct VertexPositionNormalTexture
{
	XMFLOAT3 position;
	XMFLOAT3 normal;
	XMFLOAT2 uv;
};

// ToDo dedupe with Vertex in PBRT.
struct VertexPositionNormalTextureTangent
{
	XMFLOAT3 position;
	XMFLOAT3 normal;
	XMFLOAT2 textureCoordinate;
	XMFLOAT3 tangent;
};


// Ray types traced in this sample.
namespace RayType {
    enum Enum {
        GBuffer = 0,	 // ToDo update	// ~ Primary camera ray generating GBuffer data.
        Shadow,         // ~ Shadow/visibility rays, only testing for occlusion
        Count
    };
}

namespace RTAORayType {
    enum Enum {
        AO = 0,	
        Count
    };
}


namespace TraceRayParameters
{
    static const UINT InstanceMask = ~0;   // Everything is visible.
    namespace HitGroup {
        static const UINT Offset[RayType::Count] =
        {
            0, // GBuffer ray
            1, // Shadow ray
        };
		// ToDo For now all geometries reusing shader records
		static const UINT GeometryStride = RayType::Count;
    }
    namespace MissShader {
        static const UINT Offset[RayType::Count] =
        {
            0, // GBuffer ray
            1, // Shadow ray
        };
    }
}

namespace RTAOTraceRayParameters
{
    static const UINT InstanceMask = ~0;   // Everything is visible.
    namespace HitGroup {
        static const UINT Offset[RTAORayType::Count] =
        {
            0, // AO ray
        };
        // Since there is only one closest hit shader across shader records in RTAO, 
        // always access the first shader record of each BLAS instance shader record range.
        // Optimally, we should specify just a single shader record, but because
        // the TLAS is reused with other RTPSOs and per 2+ BLAS instance shader records,
        // we have to make sure indexing due InstanceContributionToHitGroupIndex > 0 lands on
        // a valid shader record.
        static const UINT GeometryStride = 0;
    }
    namespace MissShader {
        static const UINT Offset[RTAORayType::Count] =
        {
            0, // AO ray
        };
    }
}


static const XMFLOAT4 BackgroundColor = XMFLOAT4(0.79f, 0.88f, 0.98f, 1.0f);
// ToDo
static const float InShadowRadiance = 0.35f;

#endif // RAYTRACINGHLSLCOMPAT_H