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

#include "SSAO//GlobalSharedHlslCompat.h"

// Workarounds - ToDo remove/document
#define REPRO_BLOCKY_ARTIFACTS_NONUNIFORM_CB_REFERENCE_SSAO 0
#define REPRO_DEVICE_REMOVAL_ON_HARD_CODED_AO_COEF 0
#define REPRO_INVISIBLE_WALL 0

//**********************************************************************************************
//
// RaytracingHLSLCompat.h
//
// A header with shared definitions for C++ and HLSL source files. 
//
//**********************************************************************************************
#define ENABLE_RAYTRACING 1
#define RUNTIME_AS_UPDATES 1
#define USE_GPU_TRANSFORM 1

#define WORKAROUND_ATROUS_VARYING_OUTPUTS 1

#define RAYTRACING_MANUAL_KERNEL_STEP_SHIFTS 1
#define AO_SPP_N 1
#define USE_ENVIRONMENT_MAP 1
#define DEBUG_AS 0

#define MOVE_ONCE_ON_STRAFE 1
#define PBRT_APPLY_INITIAL_TRANSFORM_TO_VB_ATTRIBUTES 1

#define ALLOW_MIRRORS 1

#define TEST_EARLY_EXIT 0

// ToDo set max recursion
// Give opacity to mirrors and shade. Some mirrors are tesselated in the kitchen and its not clear from pure reflections.
// ToDo TAO is swimming in reflections
#if ALLOW_MIRRORS
// Use anyhit instead??
#define TURN_MIRRORS_SEETHROUGH 0
#endif

#define CAMERA_PRESERVE_UP_ORIENTATION 1

#define DISABLE_DENOISING 0
#define DOUBLE_ALL_FACES 0
#define ADD_INVERTED_FACE 0
#define CORRECT_NORMALS 0

#define CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN 0

//#define SAMPLER_FILTER D3D12_FILTER_MIN_MAG_MIP_LINEAR
#define SAMPLER_FILTER D3D12_FILTER_ANISOTROPIC  // TODo blurry at various angles

// ToDo
// SSAO
#define SSAO_NOISE_W 100
#define SSAO_MAX_SAMPLES 15
#define SSAO_MAX_OCCLUSION_RAYS (SSAO_MAX_SAMPLES * SSAO_MAX_SAMPLES)
// ~SSAO

#define DISTANCE_ON_MISS 65504  // ~FLT_MAX within 16 bit format // ToDo explain

#define PRINT_OUT_TC_MATRICES 0
#define DEBUG_CAMERA_POS 1
#define PRINT_OUT_CAMERA_CONFIG 1

#define USE_NORMALIZED_Z 0  // Whether to normalize z to [0, 1] within [near, far] plane range. // ToDo

// ToDo 16bit per component normals?
#define FLOAT_TEXTURE_AS_R8_UNORM_1BYTE_FORMAT 0    // ToDo
#define FLOAT_TEXTURE_AS_R16_FLOAT_2BYTE_FORMAT 1
#define OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL 0
#define GBUFFER_RAYLENGTH_ALONG_CENTER_CAMERA_EYE_RAY 0 // Incompatible with reflected/refracted rays
#define GBUFFER_AO_NORMAL_VISUALIZATION 0
#define GBUFFER_AO_COUNT_AO_HITS 0
#define AO_ANY_HIT_FULL_OCCLUSION 0
#define TWO_STAGE_AO_BLUR 1
#define AO_RANDOM_SEED_EVERY_FRAME 0
#define AO_HITPOSITION_BASED_SEED 1
#define AO_SAMPLES_SPREAD_ACCROSS_PIXELS 1
#define VARIANCE_APPROXIMATION 1

#define COMPRES_NORMALS 1
#define PACK_NORMAL_AND_DEPTH 1

#define ADAPTIVE_KERNEL_SIZE 1

#define AO_PROGRESSIVE_SAMPLING 0

#define ENABLE_VSYNC 0
#if ENABLE_VSYNC
#define VSYNC_PRESENT_INTERVAL 1  
#endif

// ToDo Fix missing DirectXTK12.lib in Profile config - as the nuget doesnt provide profile
// ToDo remove PROFILE preprocesser macro from Release

#define BLUR_AO 1
#define ATROUS_DENOISER 1
#define ATROUS_DENOISER_MAX_PASSES 10
#define RENDER_RNG_SAMPLE_VISUALIZATION 0   // ToDo doesn't render for all AA settings
#define ATROUS_ONELEVEL_ONLY 0

#define DEBUG_MULTI_BLAS_BUILD 0

#define CAMERA_JITTER 0
#define APPLY_SRGB_CORRECTION 0
#define AO_ONLY 0
// ToDO this wasn't necessary before..
#define VBIB_AS_NON_PIXEL_SHADER_RESOURCE 0 // ToDo spec requires it but it works without it?

#define USE_GRASS_GEOMETRY 1

#define ONLY_SQUID_SCENE_BLAS 1
#if ONLY_SQUID_SCENE_BLAS
#define LOAD_PBRT_SCENE 1       // loads PBRT(1) or SquidRoom(0)
#define LOAD_ONLY_ONE_PBRT_MESH 0   // for LOAD_PBRT_SCENE == 1 only
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
#define CAMERA_Y_SCALE 1
#define FLAT_FACE_NORMALS 0
#define INDEX_FORMAT_UINT 1
#define NUM_GEOMETRIES 1
#else
#define AO_RAY_T_MAX 0.06
#define FACE_CULLING 1
#define INDEX_FORMAT_UINT 0
#define FLAT_FACE_NORMALS 1
#define CAMERA_Y_SCALE 1.3f
#define NUM_GEOMETRIES 100000

#endif

#define AO_OVERDOSE_BEND_NORMALS_DOWN 0
#define TESSELATED_GEOMETRY_BOX 1
#define TESSELATED_GEOMETRY_TEAPOT 1
#define TESSELATED_GEOMETRY_BOX_TETRAHEDRON 1
#define TESSELATED_GEOMETRY_BOX_TETRAHEDRON_REMOVE_BOTTOM_TRIANGLE 1
#define TESSELATED_GEOMETRY_THIN 1
#define TESSELATED_GEOMETRY_TILES 0
#define TESSELATED_GEOMETRY_TILES_WIDTH 4
#define TESSELATED_GEOMETRY_ASPECT_RATIO_DIMENSIONS 1


// ToDo separate per-vertex attributes from VB

// ToDo move
namespace ReduceSumCS {
	namespace ThreadGroup {
		enum Enum { Width = 8, Height = 16, Size = Width * Height, NumElementsToLoadPerThread = 10 };	
	}
}

namespace AtrousWaveletTransformFilterCS {
    namespace ThreadGroup {
        enum Enum { Width = 16, Height = 16, Size = Width * Height };
    }
}

namespace CalculateVariance_Bilateral {
    namespace ThreadGroup {
        enum Enum { Width = 16, Height = 8, Size = Width * Height };
    }
}

namespace PerPixelMeanSquareError {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8, Size = Width * Height};
    }
}

namespace DownsampleBoxFilter2x2 {
	namespace ThreadGroup {
		enum Enum { Width = 8, Height = 8 };
	}
}
// ToDo cleanup
namespace DownsampleGaussianFilter {
	namespace ThreadGroup {
		enum Enum { Width = 8, Height = 8 };
	}
}

namespace DownsampleNormalDepthHitPositionGeometryHitBilateralFilter {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8 };
    }
}

namespace DownsampleValueNormalDepthBilateralFilter {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8 };
    }
}

// ToDo combine and reuse a default 8x8 ?
namespace DefaultComputeShaderParams {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8 };
    }
}

namespace UpsampleBilateralFilter {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8 };
    }
}

namespace MultiScale_UpsampleBilateralFilterAndCombine {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8 };
    }
}

namespace GaussianFilter {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8 };
    }
}

namespace RootMeanSquareError {
    namespace ThreadGroup {
        enum Enum { Width = 8, Height = 8 };
    }
}

namespace ComposeRenderPassesCS {
	namespace ThreadGroup {
		enum Enum { Width = 8, Height = 8, Size = Width * Height };
	}
}

namespace AoBlurCS {
    namespace ThreadGroup {
		enum Enum { Width = 8, Height = 8 };
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
#define MAX_RAY_RECURSION_DEPTH 4    // ~ primary rays + 2 x reflections + shadow rays from reflected geometry.  ToDo
// ToDo add recursion viz

// ToDo:
// Options:
// - shading - simple/complex
// - instanced/unique goemetry
// - deformed geometry
// - Dynamic options
// - Update/Build
#define ALBEDO_SHADING 0
#define NORMAL_SHADING 0
#define DEPTH_SHADING 0
#define SINGLE_COLOR_SHADING 0

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
    UINT hit; // ToDo compact - use BOOL?
    float tHit;
    XMFLOAT3 hitPosition;
    float obliqueness; // obliqueness of the hit surface ~ sin(incidentAngle)
    XMFLOAT3 diffuse;               // Diffuse reflectivity of the hit surface.
    XMFLOAT3 normal;
    XMFLOAT3 _virtualHitPosition;   // virtual hitPosition in the previous frame.
                                    // For non-reflected points this is a true world position of a hit.
                                    // For reflected points, this is a world position of a hit reflected across the reflected surface 
                                    //   ultimately giving the same screen space coords when projected and the depth corresponding to the ray depth.
    XMFLOAT3 _normal;               // normal in the previous frame
};

struct GBufferRayPayload
{
    // ToDo Having rayRecursionDepth causes DeviceRemoved on recursive trace ray. Check on alignments mismatch?
#if ALLOW_MIRRORS
    UINT rayRecursionDepth;
#endif

    // Contribution scaling factor of the traced ray. 
    // Used to avoid tracing rays with very small contributions.
    float bounceContribution;  

    XMFLOAT3 radiance;
	//XMUINT2 materialInfo;   // {materialID, 16b 2D texCoord}
    AmbientOcclusionGBuffer AOGBuffer;
    Ray rx;    // Auxilary camera ray offset by one pixel in x dimension in screen space.
    Ray ry;    // Auxilary camera ray offset by one pixel in y dimension in screen space.
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
    float rxTHit;
    float ryTHit;
#endif
};

struct ShadowRayPayload
{
    // ToDo use 1 byte value for true/false?
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
    UINT scatterOutput;

    float valueSigma;
    float depthSigma;
    float normalSigma;
    UINT useCalculatedVariance;
    
    UINT useApproximateVariance;
    BOOL outputFilteredValue;
    BOOL outputFilteredVariance;
    BOOL outputFilterWeigthSum;

    BOOL pespectiveCorrectDepthInterpolation;
    BOOL useAdaptiveKernelSize;
    float minHitDistanceToKernelWidthScale;
    UINT minKernelWidth;

    UINT maxKernelWidth;
    float varianceSigmaScaleOnSmallKernels;
    bool usingBilateralDownsampledBuffers;
    UINT padding;

    UINT kernelWidth;
    UINT padding2[3];
};

// ToDo remove obsolete params in cbs

struct CalculateVariance_BilateralFilterConstantBuffer
{
    XMUINT2 textureDim;
    float depthSigma;
    float normalSigma;
    float padding;

    BOOL outputMean;
    BOOL useDepthWeights;
    BOOL useNormalWeights;
    UINT kernelWidth;
};



// ToDo split CB?
// ToDo capitalize?
// ToDo padding?
struct SceneConstantBuffer
{
    // ToDo rename to world to view matrix and drop (0,0,0) note.
    XMMATRIX projectionToWorldWithCameraEyeAtOrigin;	// projection to world matrix with Camera at (0,0,0).
    XMMATRIX viewProjection;    // ToDo remove // world to projection matrix with Camera at (0,0,0).
    XMVECTOR cameraPosition;
	XMVECTOR lightPosition;
    XMFLOAT3 lightColor;
    float defaultAmbientIntensity;
    XMMATRIX prevViewProj;    // ToDo standardzie proj vs projection
    XMMATRIX prevProjToWorldWithCameraEyeAtOrigin;	// projection to world matrix with Camera at (0,0,0).
    XMVECTOR prevCameraPosition;

    // ToDo remove
    XMVECTOR cameraAt;
    XMVECTOR cameraUp;
    XMVECTOR cameraRight;

    float reflectance;
    float elapsedTime;                 // Elapsed application time.
	float Zmin;     // ToDo rename to zNear
	float Zmax;
    
    UINT seed;
    UINT numSamplesPerSet;
    UINT numSampleSets;
    UINT numSamplesToUse;       // ToDo rename to max samples
    
    UINT numPixelsPerDimPerSet;
    BOOL useShadowRayHitTime;           // ToDo Rename "use"
    XMFLOAT2 cameraJitter;

    float RTAO_maxShadowRayHitTime;             // Max shadow ray hit time used for tMax in TraceRay.
    BOOL RTAO_approximateInterreflections;      // Approximate interreflections. 
    float RTAO_diffuseReflectanceScale;              // Diffuse reflectance from occluding surfaces. 
    float RTAO_minimumAmbientIllumnination;       // Ambient illumination coef when a ray is occluded.

    // toDo rename shadow to AO?
    float RTAO_maxTheoreticalShadowRayHitTime;  // Max shadow ray hit time used in falloff computation accounting for
                                                // RTAO_ExponentialFalloffMinOcclusionCutoff and RTAO_maxShadowRayHitTime.
    UINT  maxRadianceRayRecursionDepth;
    UINT  maxShadowRayRecursionDepth;


    BOOL RTAO_IsExponentialFalloffEnabled;               // Apply exponential falloff to AO coefficient based on ray hit distance.    
    float RTAO_exponentialFalloffDecayConstant; 
    BOOL RTAO_UseAdaptiveSampling;
    float RTAO_AdaptiveSamplingMaxWeightSum;

    float RTAO_AdaptiveSamplingScaleExponent;   // ToDo weight exponent instead?
    BOOL RTAO_UseNormalMaps;
    BOOL RTAO_AdaptiveSamplingMinMaxSampling;
    UINT RTAO_AdaptiveSamplingMinSamples;

    float RTAO_TraceRayOffsetAlongNormal;
    float RTAO_TraceRayOffsetAlongRayDirection;
    float RTAO_minimumFrBounceCoefficient;  // Minimum bounce coefficient for reflection ray to consider executing a TraceRay for it.
    float RTAO_minimumFtBounceCoefficient;  // Minimum bounce coefficient for transmission ray to consider executing a TraceRay for it.
    BOOL useDiffuseFromMaterial;
    BOOL doShading;                         // Do shading during path tracing. If false, collects only information needed for AO pass.
};
 
// Final render output composition modes.
enum CompositionType {
    PhongLighting = 0,
    AmbientOcclusionOnly,
    AmbientOcclusionOnly_RawOneFrame,
    AmbientOcclusionHighResSamplingPixels,
    AmbientOcclusionAndDisocclusionMap, // ToDo quarter res support
    RTAOHitDistance,    // ToDo standardize naming
    NormalsOnly,
    DepthOnly,
    Diffuse,
    DisocclusionMap,
    Count
};

// ToDo compress
// ToDo explain padding
struct ComposeRenderPassesConstantBuffer
{
	XMUINT2 rtDimensions;
	XMFLOAT2 padding1;

	XMFLOAT3 cameraPosition;
    float padding2;

	XMFLOAT3 lightPosition;     // ToDo cb doesn't match if XMFLOAT starts at offset 1. Can this be caught?
    UINT enableAO;

	XMFLOAT3 lightAmbientColor;
    CompositionType compositionType;

	XMFLOAT3 lightDiffuseColor;		
    float RTAO_AdaptiveSamplingMaxWeightSum;

    BOOL RTAO_UseAdaptiveSampling;
    float RTAO_AdaptiveSamplingScaleExponent;   // ToDo weight exponent instead?
    BOOL RTAO_AdaptiveSamplingMinMaxSampling;
    UINT RTAO_AdaptiveSamplingMinSamples;

    UINT RTAO_MaxSPP;
    float RTAO_MaxRayHitDistance;   // ToDo standardize ray hit time vs distance
    float defaultAmbientIntensity;
    float padding3[2];
};

struct AoBlurConstantBuffer
{
	XMFLOAT2 kRcpBufferDim;
	float kDistanceTolerance;
};

struct DownsampleFilterConstantBuffer
{
	XMUINT2 inputTextureDimensions;
	XMFLOAT2 invertedInputTextureDimensions;
};

struct GaussianFilterConstantBuffer
{
    XMUINT2 textureDim;
    XMFLOAT2 invTextureDim;
};

struct RTAO_TemporalCache_ReverseReprojectConstantBuffer
{
    // ToDo can we get by with one matrix?
    XMMATRIX invProj;
    XMMATRIX invView;
    XMMATRIX reverseProjectionTransform;
    XMMATRIX invViewProjAndCameraTranslation;
    XMMATRIX prevInvViewProj;
    XMVECTOR cameraPosition;
    XMMATRIX projectionToWorldWithCameraEyeAtOrigin;
    XMVECTOR prevToCurrentFrameCameraTranslation;   // ToDo include this in one of the projection matrices?
    XMMATRIX prevProjectionToWorldWithCameraEyeAtOrigin;
    
    BOOL  forceUseMinSmoothingFactor;  // ToDo remove?
    float minSmoothingFactor;       
    float zNear; // ToDo take these from transform matrix directly?
    float zFar;

    // ToDo pix missinterprets the format
    XMUINT2 textureDim;
    XMFLOAT2 invTextureDim; // ToDo test what impact passing inv tex dim makes
    
    // ToDo moving this 4Bs above XMFLOATs causes issues
    float depthTolerance;
    BOOL useDepthWeights;
    BOOL useNormalWeights;
    BOOL clampCachedValues;

    float stdDevGamma;
    float floatEpsilonDepthTolerance;
    float depthDistanceBasedDepthTolerance;
    float depthSigma;

    BOOL useWorldSpaceDistance;
    float minStdDevTolerance;
    float frameAgeAdjustmentDueClamping;
    float padding;
};

struct CalculatePartialDerivativesConstantBuffer
{
    XMUINT2 textureDim;
    float padding[2];
};

struct DownAndUpsampleFilterConstantBuffer
{
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

/* TODO remove
 
 float3 position;
 float3 normal;
 float2 textureCoordinate;
 float3 tangent;

 */

namespace RayGenShaderType {
	enum Enum {
		GBuffer = 0,
		AOFullRes,
        AOQuarterRes,
		Visibility,
		Count
	};
}


// Ray types traced in this sample.
namespace RayType {
    enum Enum {
        Radiance = 0,   // ~ Primary, reflected camera/view rays calculating color for each hit.
        Shadow,         // ~ Shadow/visibility rays, only testing for occlusion
		GBuffer,		// ~ Primary camera ray generating GBuffer data.
        Count
    };
}

namespace TraceRayParameters
{
    static const UINT InstanceMask = ~0;   // Everything is visible.
    namespace HitGroup {
        static const UINT Offset[RayType::Count] =
        {
            0, // Radiance ray
            1, // Shadow ray
			2  // GBuffer ray
        };
		// ToDo For now all geometries reusing shader records
		static const UINT GeometryStride = RayType::Count;
    }
    namespace MissShader {
        static const UINT Offset[RayType::Count] =
        {
            0, // Radiance ray
            1, // Shadow ray
			2, // GBuffer ray
        };
    }
}

static const XMFLOAT4 BackgroundColor = XMFLOAT4(0.79f, 0.88f, 0.98f, 1.0f);
// ToDo
static const float InShadowRadiance = 0.35f;

#endif // RAYTRACINGHLSLCOMPAT_H