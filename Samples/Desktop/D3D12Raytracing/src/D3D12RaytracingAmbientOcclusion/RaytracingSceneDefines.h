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

#include "RayTracingHlslCompat.h"

// ToDo standardize use of CS suffix
namespace ComputeShader {
	namespace Type {
		enum Enum {
			HemisphereSampleSetVisualization = 0,
			ReduceSum,
			ComposeRenderPassesCS,
            AoBlurCS,
            AoBlurAndUpsampleCS,
			Count
		};
	}
	namespace RootSignature {
		namespace HemisphereSampleSetVisualization {
			namespace Slot {
				enum Enum {
					Output = 0,
					SampleBuffers,
					SceneConstant,
					Count
				};
			}
		}		
		namespace ReduceSum {
			namespace Slot {
				enum Enum {
					Output = 0,
					Input,
					Count
				};
			}
		}

		namespace ComposeRenderPassesCS {
			namespace Slot {
				enum Enum {
					Output = 0,
					GBufferResources,
					AO,
					Visibility,
					MaterialBuffer,
					ConstantBuffer,
                    FilterWeightSum,
                    AORayHitDistance,
                    FrameAge,   // ToDo use same name as in the shader
                    Color,
                    AODiffuse,
                    Count
				};
			}
		}
		
		namespace AoBlurCS {
			namespace Slot {
				enum Enum {
					Output = 0,
					Normal,
                    Distance,
                    InputAO,
					ConstantBuffer,
					Count
				};
			}
		}
		
	}
	namespace RS = RootSignature;
}
namespace CSType = ComputeShader::Type;
namespace CSRootSignature = ComputeShader::RootSignature;

namespace RaytracingType {
    enum Enum {
        Pathtracing = 0,
        AmbientOcclusion,
        Count
    };
}

namespace DownsampleFilter {
	enum Enum {
		None = 0,
		BoxFilter2x2,
		GaussianFilter9Tap,
		GaussianFilter25Tap,
		Count
	};
}

namespace GlobalRootSignature {
    namespace Slot {
        enum Enum {
            Output = 0,
			GBufferResources,
			GBufferResourcesIn,
			AOResourcesOut,	// ToDo cleanup, move to local root sigs 
			VisibilityResource,	// ToDo cleanup, move to local root sigs 
            AccelerationStructure,
            SceneConstant,
			MaterialBuffer,
            SampleBuffers,
            EnvironmentMap,
            FilterWeightSum,
            GBufferDepth,   // ToDo move to the above slot for GBufferResources ?
            GbufferNormalRGB,
            NormalDepthLowPrecision,
            AORayHitDistance,
            AOFrameAge,
            AORayDirectionOriginDepthHitSRV,
            AORayDirectionOriginDepthHitUAV,
            AOSourceToSortedRayIndex,
#if CALCULATE_PARTIAL_DEPTH_DERIVATIVES_IN_RAYGEN
            PartialDepthDerivatives,
#endif
            PrevFrameBottomLevelASIstanceTransforms,
            MotionVector,
            ReprojectedHitPosition,
            Color,
            AODiffuse,
            ShadowMapUAV,   // ToDo Standardize nmaming
            ShadowMapSRV,
            Count
        };
    }
}

namespace LocalRootSignature {
    namespace Type {
        enum Enum {
            Triangle = 0,
            AABB,
            Count
        };
    }
}

namespace LocalRootSignature {
    namespace Triangle {
        namespace Slot {
            enum Enum {
                ConstantBuffer = 0,
                IndexBuffer,
                VertexBuffer,
                PreviousFrameVertexBuffer,
                DiffuseTexture,
                NormalTexture,
                Count
            };
        }
        struct RootArguments {
            PrimitiveConstantBuffer cb;
            // ToDo add align specifier
            // Bind each resource via a descriptor.
            // This design was picked for simplicity, but one could optimize for shader record size by:
            //    1) Binding multiple descriptors via a range descriptor instead.
            //    2) Storing 4 Byte indices (instead of 8 Byte descriptors) to a global pool resources.
            D3D12_GPU_DESCRIPTOR_HANDLE indexBufferGPUHandle;
            D3D12_GPU_DESCRIPTOR_HANDLE vertexBufferGPUHandle;
            D3D12_GPU_DESCRIPTOR_HANDLE previousFrameVertexBufferGPUHandle;
            D3D12_GPU_DESCRIPTOR_HANDLE diffuseTextureGPUHandle;
            D3D12_GPU_DESCRIPTOR_HANDLE normalTextureGPUHandle;
        };
    }
}

namespace LocalRootSignature {
    inline UINT MaxRootArgumentsSize()
    {
        return sizeof(Triangle::RootArguments);
    }
}

// Todo rename
namespace ReduceSumCalculations {
	enum Enum {
		CameraRayHits = 0,
		AORayHits,
		Count
	};
}

namespace GeometryType {
    enum Enum {
        Plane = 0,
        Sphere,
		SquidRoom,
		PBRT,
        Count
    };
}

// ToDo update descriptions, prune redundant.
namespace GBufferResource {
	enum Enum {
		Hit = 0,		// Geometry hit or not.
		Material,		// Material of the object hit ~ {MaterialID, texCoord}.
		HitPosition,	// 3D position of hit.
		SurfaceNormal,	// 3D normal at a hit and dot(normal, rayDir) in W, ToDo rename to encoded or sth
        Distance,       // Length along ray of hit. // ToDo update (depth?)
        Depth,          // Non-linear depth of the hit. // ToDo remove
        SurfaceNormalRGB, // 3D normal at a hit. // ToDo deduplicate remove Surface prefix
        NormalDepthLowPrecision,
        PartialDepthDerivatives,
        MotionVector,
        ReprojectedHitPosition,
        Color,
        AODiffuse, 
		Count
	};
}

namespace AOResource {
	enum Enum {
		Coefficient = 0,
        Smoothed,
		HitCount,
        RayHitDistance,
        FilterWeightSum,
		Count
	};
}

namespace TemporalCache {
    enum Enum {
        AO = 0,
        NormalDepth,
        FrameAge,
        Count
    };
}


namespace Scene {
	namespace Type {
		enum Enum {
			SingleObject = 0,
			GeometricForest,
			SquidRoom,
			PBRT,	// Rename
			Count
		};
		extern const WCHAR* Names[Count];
	}


	struct Camera
	{
		struct CameraPosition
		{
			XMVECTOR eye, at, up;
		};

		struct CameraBoundaries
		{
			XMVECTOR min, max;
		};

		CameraPosition position;
		CameraBoundaries boundaries;
	};

	struct Params {
		Camera camera;
	};

	class Initialize
	{
	public:
		Initialize();
	};
	extern Params args[Scene::Type::Count];
}

namespace SceneEnums
{
	namespace VertexBuffer {
		enum Value { SceneGeometry = 0, Count };
	}
}




// Bottom-level acceleration structures (BottomLevelASType).
// This sample uses two BottomLevelASType, one for AABB and one for Triangle geometry.
// Mixing of geometry types within a BLAS is not supported.
namespace BottomLevelASType = GeometryType;