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
namespace ComputeShader {   // ToDo remove this?
	namespace Type {
		enum Enum {
			HemisphereSampleSetVisualization = 0,
			ReduceSum,
			CompositionCS,
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

		namespace CompositionCS {
			namespace Slot {
				enum Enum {
					Output = 0,
					GBufferResources,
					AO,
					MaterialBuffer,
					ConstantBuffer,
                    Variance,
                    LocalMeanVariance,
                    AORayHitDistance,
                    FrameAge,   // ToDo use same name as in the shader
                    Color,
                    AOSurfaceAlbedo,
                    Count
				};
			}
		}		
	}
	namespace RS = RootSignature;
}
namespace CSType = ComputeShader::Type;
namespace CSRootSignature = ComputeShader::RootSignature;

// ToDo move?
namespace RayGenShaderType {
    enum Enum {
        GBuffer = 0,
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
        // ToDo rename to AORay hit members?
		Material,		// Material of the object hit ~ {MaterialID, texCoord}.
		HitPosition,	// 3D position of hit.
		SurfaceNormalDepth,	// Encoded normal.
        Depth,          // Linear depth of the hit.
        PartialDepthDerivatives,
        MotionVector,
        ReprojectedNormalDepth,
        Color,
        AOSurfaceAlbedo,
		Count
	};
}

namespace AOResource {
	enum Enum {
		Coefficient = 0,
		HitCount,
        Smoothed,   // ToDo remove
        RayHitDistance,
		Count
	};
}

namespace AOVarianceResource {
    enum Enum {
        Raw = 0,
        Smoothed,
        Count
    };
}

namespace TemporalSupersampling {
    enum Enum {
        // ToDo rename to TRpp
        FrameAge = 0,
        RayHitDistance,
        CoefficientSquaredMean,
        Count
    };
}


namespace SampleScene {
	namespace Type {
		enum Enum {
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
	extern Params args[SampleScene::Type::Count];
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