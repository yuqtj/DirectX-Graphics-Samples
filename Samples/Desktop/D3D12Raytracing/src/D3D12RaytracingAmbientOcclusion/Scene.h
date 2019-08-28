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
namespace Scene_Args
{
}

class Scene
{
public:
    // Ctors.
    Scene();

    // Public methods.
    void Setup(std::shared_ptr<DX::DeviceResources> deviceResources, std::shared_ptr<DX::DescriptorHeap> descriptorHeap);
    void OnUpdate();
    void OnRender();
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources() {}; // ToDo

    const GameCore::Camera& Camera() { return m_camera; }
    const GameCore::Camera& PrevFrameCamera() { return m_prevFrameCamera; }
    const StepTimer& Timer() { return m_timer; }
    const std::map<std::wstring, BottomLevelAccelerationStructureGeometry>& BottomLevelASGeometries() { return m_bottomLevelASGeometries; }
    const std::unique_ptr<RaytracingAccelerationStructureManager>& AccelerationStructure() { return m_accelerationStructure; }
 
    // Getters & setters.
    // ToDO direct access instead?
    GpuResource(&GrassPatchVB())[UIParameters::NumGrassGeometryLODs][2] { return m_grassPatchVB; }
    D3DTexture& EnvironmentMap() { return m_environmentMap; }
    StructuredBuffer<PrimitiveMaterialBuffer>& MaterialBuffer() { return m_materialBuffer; }
    StructuredBuffer<XMFLOAT3X4>& PrevFrameBottomLevelASInstanceTransforms() { return m_prevFrameBottomLevelASInstanceTransforms; }

    void RequestGeometryInitialization(bool bRequest) { m_isGeometryInitializationRequested = bRequest; }
    void RequestASInitialization(bool bRequest) { m_isASinitializationRequested = bRequest; }

    void ToggleAnimateLight() { m_animateLight = !m_animateLight; }
    void ToggleAnimateCamera() { m_animateCamera = !m_animateCamera; }
private:
    void CreateDeviceDependentResources();
    void CreateConstantBuffers();
    void CreateAuxilaryDeviceResources();
    void CreateTextureResources();
    void CreateResolutionDependentResources();

    void GenerateGrassGeometry();
    void LoadSquidRoom();
    void CreateIndexAndVertexBuffers(const GeometryDescriptor& desc, D3DGeometry* geometry);
    void LoadPBRTScene();
    void LoadSceneGeometry();
    void UpdateSSAOCameraMatrices();
    void InitializeScene();
    void UpdateAccelerationStructure();
    void InitializeGrassGeometry();
    void InitializeGeometry();
    void BuildPlaneGeometry();
    void InitializeAllBottomLevelAccelerationStructures();
    void InitializeAccelerationStructures();
    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    std::shared_ptr<DX::DescriptorHeap> m_cbvSrvUavHeap;

    // Application state.
    StepTimer m_timer;
    bool m_animateCamera = false;
    bool m_animateLight = false;
    bool m_animateScene = false;
    bool m_isCameraFrozen = false;
    int m_cameraChangedIndex = 0;
    bool m_hasCameraChanged = true;
    GameCore::Camera m_camera;
    GameCore::Camera m_prevFrameCamera;
    float m_manualCameraRotationAngle = 0; // ToDo remove
    std::unique_ptr<GameCore::CameraController> m_cameraController;

    // ToDo remove?
    bool m_isGeometryInitializationRequested = false;
    bool m_isASinitializationRequested = false;

    // Geometry.
    UINT m_numTriangles;
    UINT m_numInstancedTriangles;

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

    StructuredBuffer<XMFLOAT3X4> m_prevFrameBottomLevelASInstanceTransforms;        // Bottom-Level AS Instance transforms used for previous frame. Used for Temporal Reprojection.
    const UINT MaxNumBottomLevelInstances = 10100;           // ToDo tighten this to only what needed or add support a copy of whats used from StructuredBuffers to GPU.


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

    // Materials & textures.
    std::vector<PrimitiveMaterialBuffer> m_materials;	// ToDO dedupe mats - hash materials
    StructuredBuffer<PrimitiveMaterialBuffer> m_materialBuffer;
    StructuredBuffer<AlignedGeometryTransform3x4> m_geometryTransforms;
    D3DTexture m_environmentMap;
    D3DTexture m_nullTexture;

    XMVECTOR m_lightPosition;
    XMFLOAT3 m_lightColor;

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

    friend class Pathtracer;
    friend class Composition;
};