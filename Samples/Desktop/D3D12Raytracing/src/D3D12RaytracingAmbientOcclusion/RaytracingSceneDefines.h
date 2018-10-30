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
					Count
				};
			}
		}
	}
	namespace RS = RootSignature;
}
namespace CSType = ComputeShader::Type;
namespace CSRootSignature = ComputeShader::RootSignature;



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
                MaterialID = 0,
                VertexBuffers,
                Count
            };
        }
        struct RootArguments {
            UINT materialID;
			XMUINT3 padding;		// ToDo explain why
            D3D12_GPU_DESCRIPTOR_HANDLE vertexBufferGPUHandle;
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

// ToDo Gputimers bug out changing enums around
namespace GpuTimers {
	enum Enum {
		Raytracing_AO = 0,
		Raytracing_Visibility,
		ComposeRenderPassesCS,
		Raytracing_GBuffer,
		UpdateBLAS,
		ReduceSum,
		UpdateTLAS,
		Count
	};
}

namespace UIParameters {
	enum Enum {
		RaytracingAPI = 0,
		BuildQuality,
		UpdateAlgorithm,
		TesselationQuality,
		NumberOfObjects,
		Count
	};
}

namespace GBufferResource {
	enum Enum {
		Hit = 0,		// Geometry hit or not.
		MaterialID,		// Material ID of the object hit.
		HitPosition,	// 3D position of hit.
		SurfaceNormal,	// 3D normal at a hit.
		Depth,			// Linear depth of the hit in [0,1].
		Count
	};
}

namespace AOResource {
	enum Enum {
		Coefficient = 0,
		HitCount,
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