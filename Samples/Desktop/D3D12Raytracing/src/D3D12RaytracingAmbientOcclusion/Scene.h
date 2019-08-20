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

// ToDo move to cpp
#include "RaytracingSceneDefines.h"
#include "DirectXRaytracingHelper.h"
#include "RaytracingAccelerationStructure.h"
#include "CameraController.h"
#include "PerformanceTimers.h"
#include "Sampler.h"
#include "GpuKernels.h"
#include "EngineTuning.h"
#include "SceneParameters.h"
#include "StepTimer.h"


// ToDo remove namespce?
namespace Scene
{
    namespace Args
    {
    }

    class Scene
    {
    public:
        // Ctors.
        Scene();

        // Public methods.
        void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap, UINT maxInstanceContributionToHitGroupIndex);
        void OnUpdate();
        void OnRender(D3D12_GPU_VIRTUAL_ADDRESS accelerationStructure, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceHitPositionResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceNormalDepthResource, D3D12_GPU_DESCRIPTOR_HANDLE rayOriginSurfaceAlbedoResource, D3D12_GPU_DESCRIPTOR_HANDLE frameAgeResource);
        void ReleaseDeviceDependentResources();
        void ReleaseWindowSizeDependentResources() {}; // ToDo

        const GameCore::Camera& Camera() { return m_camera; }
        const GameCore::Camera& PrevFrameCamera() { return m_prevFrameCamera; }
        const StepTimer& Timer() { return m_timer; }
        const std::map<std::wstring, BottomLevelAccelerationStructureGeometry>& BottomLevelASGeometries() { return m_bottomLevelASGeometries; }
        const std::unique_ptr<RaytracingAccelerationStructureManager>& AccelerationStructure() { return m_accelerationStructure; }


        const GpuResource(&GrassPatchVB())[UIParameters::NumGrassGeometryLODs][2] { return m_grassPatchVB; }
        

    private:
        void CreateDeviceDependentResources(UINT maxInstanceContributionToHitGroupIndex);
        void CreateConstantBuffers();
        void CreateAuxilaryDeviceResources();
        void CreateTextureResources();
        void CreateResolutionDependentResources();

        void GenerateGrassGeometry();
        void LoadSquidRoom();
        void CreateIndexAndVertexBuffers(const GeometryDescriptor& desc, D3DGeometry* geometry);
        void LoadPBRTScene();
        void LoadSceneGeometry();
        void UpdateCameraMatrices();
        void UpdateGridGeometryTransforms();
        void InitializeScene();
        void UpdateAccelerationStructure();
        void InitializeGrassGeometry();
        void InitializeGeometry();
        void BuildPlaneGeometry();
        void BuildTesselatedGeometry();
        void InitializeAllBottomLevelAccelerationStructures();
        void InitializeAccelerationStructures();

        std::shared_ptr<DX::DeviceResources> m_deviceResources;
        std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;



        // Application state.
        StepTimer m_timer;
        bool m_animateCamera;
        bool m_animateLight;
        bool m_animateScene;
        bool m_isCameraFrozen;
        int m_cameraChangedIndex = 0;
        bool m_hasCameraChanged = true;
        GameCore::Camera m_camera;
        GameCore::Camera m_prevFrameCamera;
        float m_manualCameraRotationAngle = 0; // ToDo remove
        std::unique_ptr<GameCore::CameraController> m_cameraController;

        // Grass geometry.
        static const UINT NumGrassPatchesX = 30;
        static const UINT NumGrassPatchesZ = 30;
        static const UINT MaxBLAS = 10 + NumGrassPatchesX * NumGrassPatchesZ;   // ToDo enumerate all instances in the comment

        GpuKernels::GenerateGrassPatch      m_grassGeometryGenerator;
        UINT                                m_animatedCarInstanceIndex;
        UINT                                m_grassInstanceIndices[NumGrassPatchesX * NumGrassPatchesZ];
        UINT                                m_currentGrassPatchVBIndex = 0;
        // ToDo remove
        D3DBuffer                           m_nullVB;               // Null vertex Buffer - used for geometries that don't animate and don't need double buffering for motion vector calculation.
        UINT                                m_grassInstanceShaderRecordOffsets[2];
        UINT                                m_prevFrameLODs[NumGrassPatchesX * NumGrassPatchesZ];

        std::map<std::wstring, BottomLevelAccelerationStructureGeometry> m_bottomLevelASGeometries;
        std::unique_ptr<RaytracingAccelerationStructureManager> m_accelerationStructure;
        GpuResource m_grassPatchVB[UIParameters::NumGrassGeometryLODs][2];      // Two VBs: current and previous frame.

        // ToDo remove?
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

        D3DTexture m_environmentMap;

        StructuredBuffer<XMFLOAT3X4> m_prevFrameBottomLevelASInstanceTransforms;        // Bottom-Level AS Instance transforms used for previous frame. Used for Temporal Reprojection.
        UINT m_maxInstanceContributionToHitGroupIndex;

        const UINT MaxNumBottomLevelInstances = 10100;           // ToDo tighten this to only what needed or add support a copy of whats used from StructuredBuffers to GPU.



        // ToDo cleanup
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
    };
    StructuredBuffer<AlignedGeometryTransform3x4> m_geometryTransforms;
}