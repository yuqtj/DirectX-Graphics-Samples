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

#include "RayTracingSceneDefines.h"


namespace Scene
{
	const WCHAR* Type::Names[] = { L"Single object", L" Geometric forest", L"Squid room" };
	Params args[Scene::Type::Count];

	// Initialize scene parameters
	Initialize initializeObject;
	Initialize::Initialize()
	{
		// Camera Position
		{
			auto& camera = args[Scene::Type::SingleObject].camera;
#if TESSELATED_GEOMETRY_BOX
#if NUM_GEOMETRIES_1000
			camera.position.eye = { 0, (TESSELATED_GEOMETRY_TILES*TESSELATED_GEOMETRY_TILES_WIDTH + CAMERA_Y_SCALE) * 0.43f, 0, 1 };
#elif NUM_GEOMETRIES_100000
			camera.position.eye = { 0,  (TESSELATED_GEOMETRY_TILES*TESSELATED_GEOMETRY_TILES_WIDTH + CAMERA_Y_SCALE)*3.65f, 0, 1 };
#else
			camera.position.eye = { 0,  (TESSELATED_GEOMETRY_TILES*TESSELATED_GEOMETRY_TILES_WIDTH + CAMERA_Y_SCALE)*11.3f, 0, 1 };
#endif
			camera.position.at = { 0, 0, 0, 1 };
			camera.position.up = { 0, 0, 1, 0 };
#elif 1
			camera.position.eye = { 0.509415329f, 2.28009248f, -4.69469929f, 1 };
			camera.position.at = { 1.17371607f, -6.80418682f, 2.02239656f, 1 };
			camera.position.up = { 0.109009691f, 0.596148014f, 0.795441568f, 0};
#else
			camera.position.eye = { 0, 6.3f * CAMERA_Y_SCALE , -10.0f, 1 };
			camera.position.at = { 0, 1, 0, 1 };
			XMVECTOR right = { 1.0f, 0.0f, 0.0f, 0.0f };
			XMVECTOR forward = XMVector4Normalize(camera.position.at - camera.position.eye);
			camera.position.up = XMVector3Normalize(XMVector3Cross(forward, right));
#endif
			camera.boundaries.min = -XMVectorSplatInfinity();
			camera.boundaries.max = XMVectorSplatInfinity();
		}
		{
			// ToDo
			auto& camera = args[Scene::Type::GeometricForest].camera;
			camera.position.eye = { 0, 80, 268.555980f, 1 };
			camera.position.at = { 0, 80, 0, 1 };
			camera.position.up = { 0, 1, 0, 0 };
			camera.boundaries.min = -XMVectorSplatInfinity();
			camera.boundaries.max = XMVectorSplatInfinity();
		}
		{

			auto& camera = args[Scene::Type::SquidRoom].camera;
			camera.position.eye = { 0, 80, 268.555980f, 1 };
			camera.position.at = { 0, 80, 0, 1 };
			camera.position.up = { 0, 1, 0, 0 };
#if LOAD_PBRT_SCENE
			camera.boundaries.min = -XMVectorSplatInfinity();
			camera.boundaries.max = XMVectorSplatInfinity();
#else
			camera.boundaries.min = { -430, 2.2f, -428, 1 };
			camera.boundaries.max = { 408, 358, 416, 1 };
#endif

#if DEBUG_CAMERA_POS
#if 1   // Isometric view of all objects and grass around
            camera.position.at = { -47.2393f, 27.2846f, -30.9385f, 1 };
            camera.position.up = { 0.499065f, 0.722306f, 0.478741f, 0 };
            camera.position.eye = { -47.8167f, 27.891f, -31.4858f, 1 };
#elif 0 // Top-down spaceship front rod - long ray distances
    camera.position.at = { -4.02726f, 1.08747f, -14.9725f, 1 };
    camera.position.up = { 0.0153231f, -0.232894f, -0.972375f, 0 };
    camera.position.eye = { -4.02189f, 2.08023f, -15.0949f, 1 };
#elif 0
            // SpaceShip
            camera.position.at = { -4.69957f, 2.73596f, -18.8503f, 1 };
            camera.position.up = { 0.333712f, 0.88733f, 0.31823f, 0 };
            camera.position.eye = { -5.38028f, 3.09481f, -19.4894f, 1 };
#elif 1 // Grass shot - short ray distances
            camera.position.at = { -4.14706f, 1.13742f, -18.0755f, 1 };
            camera.position.up = { -0.0265382f, 0.874558f, -0.484183f, 0 };
            camera.position.eye = { -4.08963f, 1.52135f, -17.1536f, 1 };
#elif 1// Car side behind
            camera.position.at = { -14.7492f, 0.735575f, 2.19938f, 1 };
            camera.position.up = { -0.254448f, 0.961197f, 0.106525f, 0 };
            camera.position.eye = { -13.8442f, 0.901969f, 1.80731f, 1 };
#elif 1
            camera.position.at = { -46.0809f, 8.99305f, -30.2576f, 1 };
            camera.position.up = { 0.20296f, 0.959185f, 0.196876f, 0 };
            camera.position.eye = { -46.7962f, 9.16654f, -30.935f, 1 };

#elif 1
            camera.position.eye = { -28.2961f, 1.7579f, 0.0533502f, 1 };
            camera.position.at = { -27.5242f, 1.49657f, 0.633411f, 1 };
            camera.position.up = { 0.291956f, 0.92989f, 0.223728f, 0 };
#elif 1
            camera.position.eye = { -31.7151f, 7.36261f, -15.3756f, 1 };
            camera.position.at = { -31.1283f, 7.10824f, -14.6065f, 1 };
            camera.position.up = { 0.216289f, 0.932514f, 0.289179f, 0 };
#else
            camera.position.eye = { -27.2654f, 8.0924f, -9.16976f, 1 };
            camera.position.at = { -26.4939f, 7.89591f, -8.56419f, 1 };
            camera.position.up = { 0.237911f, 0.952311f, 0.191048f, 0 };
#endif
#endif
		}
	}
}
