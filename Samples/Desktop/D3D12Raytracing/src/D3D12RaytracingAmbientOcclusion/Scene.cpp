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

#include "stdafx.h"
#include "Scene.h"
#include "GameInput.h"
#include "EngineTuning.h"
#include "EngineProfiling.h"
#include "GpuTimeManager.h"
#include "Scene.h"
#include "D3D12RaytracingAmbientOcclusion.h"
#include "SquidRoom.h"
#include "RaytracingSceneDefines.h"

// ToDo prune unused
using namespace std;
using namespace DX;
using namespace DirectX;
using namespace SceneEnums;
using namespace GameCore;

UIParameters g_UIparameters;    // ToDo move

namespace Scene_Args
{
    void OnSceneChange(void*)
    {
        Sample::instance().RequestSceneInitialization();
    }

    BoolVar EnableGeometryAndASBuildsAndUpdates(L"Render/Acceleration structure/Enable geometry & AS builds and updates", true);

    EnumVar SceneType(L"Scene", SampleScene::Type::SquidRoom, SampleScene::Type::Count, SampleScene::Type::Names, OnSceneChange, nullptr);
       
    NumVar CameraRotationDuration(L"Scene2/Camera rotation time", 48.f, 1.f, 120.f, 1.f);
    BoolVar AnimateGrass(L"Scene2/Animate grass", true);

    NumVar DebugVar(L"Render/Debug var", -20, -90, 90, 0.5f);
}

void Scene::Setup(shared_ptr<DeviceResources> deviceResources, shared_ptr<DX::DescriptorHeap> descriptorHeap)
{
    m_deviceResources = deviceResources;
    m_cbvSrvUavHeap = descriptorHeap;

    CreateDeviceDependentResources();
}

void Scene::Release()
{
}

void Scene::CreateDeviceDependentResources()
{
    auto device = m_deviceResources->GetD3DDevice();

    CreateAuxilaryDeviceResources();

    InitializeGeometry();
    InitializeAccelerationStructures();

    m_prevFrameBottomLevelASInstanceTransforms.Create(device, MaxNumBottomLevelInstances, Sample::FrameCount, L"GPU buffer: Bottom Level AS Instance transforms for previous frame");
}

// ToDo rename
void Scene::CreateAuxilaryDeviceResources()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandQueue = m_deviceResources->GetCommandQueue();

    // ToDo pass this from sample?
    ResourceUploadBatch resourceUpload(device);
    resourceUpload.Begin();

    m_grassGeometryGenerator.Initialize(device, L"Assets\\wind\\wind2.jpg", m_cbvSrvUavHeap.get(), &resourceUpload, Sample::FrameCount, UIParameters::NumGrassGeometryLODs);
   
    // Upload the resources to the GPU.
    auto finish = resourceUpload.End(commandQueue);

    // Wait for the upload thread to terminate
    finish.wait();
}

void Scene::OnRender()
{
#if USE_GRASS_GEOMETRY
    GenerateGrassGeometry();
#endif
    UpdateAccelerationStructure();
}

void Scene::OnUpdate()
{
    m_timer.Tick();
    float elapsedTime = static_cast<float>(m_timer.GetElapsedSeconds());

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
        //m_bClearTemporalSupersampling = true;
    }

    if (m_animateScene)
    {
        float animationDuration = 180.0f;
        float t = static_cast<float>(m_timer.GetTotalSeconds());
        float rotAngle1 = XMConvertToRadians(t * 360.0f / animationDuration);
        float rotAngle2 = XMConvertToRadians((t + 12) * 360.0f / animationDuration);
        float rotAngle3 = XMConvertToRadians((t + 24) * 360.0f / animationDuration);
        //m_accelerationStructure->GetBottomLevelASInstance(5).SetTransform(XMMatrixRotationAxis(XMVectorSet(0, 1, 0, 0), rotAngle1)
        //    * XMMatrixTranslationFromVector(XMVectorSet(-10, 4, -10, 0)));
        //m_accelerationStructure->GetBottomLevelASInstance(6).SetTransform(XMMatrixRotationAxis(XMVectorSet(0, 1, 0, 0), rotAngle2)
        //    * XMMatrixTranslationFromVector(XMVectorSet(-15, 4, -10, 0)));
        //m_accelerationStructure->GetBottomLevelASInstance(7).SetTransform(XMMatrixRotationAxis(XMVectorSet(0, 1, 0, 0), rotAngle3)
        //    * XMMatrixTranslationFromVector(XMVectorSet(-5, 4, -10, 0)));

        //m_accelerationStructure->GetBottomLevelASInstance(3).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(-5 + 10 * t, 0, 0, 0)));
        //m_accelerationStructure->GetBottomLevelASInstance(0).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(0, 10 * t, 0, 0)));
        //m_accelerationStructure->GetBottomLevelASInstance(1).SetTransform(XMMatrixRotationX(XMConvertToRadians((t-0.5f) * 20)));

        // Animated car.
        {
            float radius = 64;
            XMMATRIX mTranslationSceneCenter = XMMatrixTranslation(-7, 0, 7);
            XMMATRIX mTranslation = XMMatrixTranslation(0, 0, radius);

            float lapSeconds = 50;
            float angleToRotateBy = 360.0f * (-t) / lapSeconds;
            XMMATRIX mRotateSceneCenter = XMMatrixRotationY(XMConvertToRadians(Scene_Args::DebugVar));
            XMMATRIX mRotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
            float scale = 1;
            XMMATRIX mScale = XMMatrixScaling(scale, scale, scale);
            XMMATRIX mTransform = mScale * mRotateSceneCenter * mTranslation * mRotate * mTranslationSceneCenter;

            m_accelerationStructure->GetBottomLevelASInstance(m_animatedCarInstanceIndex).SetTransform(mTransform);
        }
    }

    // Rotate the camera around Y axis.
    if (m_animateCamera)
    {
        m_hasCameraChanged = true;
        // ToDo
        float secondsToRotateAround = Scene_Args::CameraRotationDuration;
        float angleToRotateBy = 360.0f * (elapsedTime / secondsToRotateAround);
        XMMATRIX axisCenter = XMMatrixTranslation(5.87519f, 0, 8.52134f);
        XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));

        XMVECTOR eye = m_camera.Eye();
        XMVECTOR at = m_camera.At();
        XMVECTOR up = m_camera.Up();
        at = XMVector3TransformCoord(at, rotate);
        eye = XMVector3TransformCoord(eye, rotate);
        up = XMVector3TransformNormal(up, rotate);
        m_camera.Set(eye, at, up);
    }

    // ToDo remove
    if (fabs(m_manualCameraRotationAngle) > 0)
    {
        m_hasCameraChanged = true;
        m_cameraChangedIndex = 2;
        m_camera.RotateAroundYAxis(XMConvertToRadians(m_manualCameraRotationAngle));
        m_manualCameraRotationAngle = 0;
    }


    // Rotate the second light around Y axis.
    if (m_animateLight)
    {
        float secondsToRotateAround = 8.0f;
        float angleToRotateBy = -360.0f * (elapsedTime / secondsToRotateAround);
        XMMATRIX rotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
        XMVECTOR prevLightPosition = m_lightPosition;

        m_lightPosition = XMVector3Transform(m_lightPosition, rotate);
    }

#if ENABLE_SSAO
    UpdateCameraMatrices();
#endif
}

void Scene::CreateResolutionDependentResources()
{
}

void Scene::CreateTextureResources()
{
}

void Scene::CreateIndexAndVertexBuffers(
    const GeometryDescriptor& desc,
    D3DGeometry* geometry)
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

    CreateGeometry(device, commandList, m_cbvSrvUavHeap.get(), desc, geometry);
}

// ToDo move
void Scene::LoadPBRTScene()
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
        {L"Quad", "Assets\\quad\\scene.pbrt"},
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
            auto& mesh = pbrtScene.m_Meshes[i];
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

            for (auto& parseIndex : mesh.m_IndexBuffer)
            {
                Index index = parseIndex;
                indexBuffer.push_back(index);
            }
            desc.ib.indices = indexBuffer.data();

            for (auto& parseVertex : mesh.m_VertexBuffer)
            {
                VertexPositionNormalTextureTangent vertex;

                // Apply the initial transform to VB attributes.
                XMStoreFloat3(&vertex.normal, XMVector3TransformNormal(parseVertex.Normal.GetXMVECTOR(), mesh.m_transform));
                XMStoreFloat3(&vertex.position, XMVector3TransformCoord(parseVertex.Position.GetXMVECTOR(), mesh.m_transform));

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

            if (
                cb.opacity.x > 0.99f && cb.opacity.y > 0.99f && cb.opacity.z > 0.99f
#if MARK_PERFECT_MIRRORS_AS_NOT_OPAQUE
                && !(cb.Kr.x > 0.99f && cb.Kr.y > 0.99f && cb.Kr.z > 0.99f))
#endif
            {
                geometryFlags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            }
            else
            {
                geometryFlags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
            }

            bottomLevelASGeometry.m_geometryInstances.push_back(GeometryInstance(geometry, materialID, diffuseTexture->gpuDescriptorHandle, normalTexture->gpuDescriptorHandle, geometryFlags, isVertexAnimated));
            numTriangles += desc.ib.count / 3;
        }
    }

    // Upload the resources to the GPU.
    auto finish = resourceUpload.End(commandQueue);

    // Wait for the upload thread to terminate
    finish.wait();
}

void Scene::InitializeScene()
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
            float stepScale = 1.0f)
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
        auto& camera = SampleScene::args[Scene_Args::SceneType].camera;
        m_camera.Set(camera.position.eye, camera.position.at, camera.position.up);
        m_cameraController = make_unique<CameraController>(m_camera);
        m_cameraController->SetBoundaries(camera.boundaries.min, camera.boundaries.max);
        // ToDo
        m_cameraController->EnableMomentum(false);
        m_prevFrameCamera = m_camera;
    }

    // Setup lights.
    {
        m_lightPosition = XMVectorSet(-20.0f, 60.0f, 20.0f, 0);
        m_lightColor = XMFLOAT3(0.6f, 0.6f, 0.6f);
    }
}

void Scene::BuildPlaneGeometry()
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
    
// ToDo move this out as a helper
void Scene::LoadSquidRoom()
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
        Sample::instance().GetAssetFullPath(SquidRoomAssets::DataFileName).c_str(),
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
       
void Scene::LoadSceneGeometry()
{
#if LOAD_PBRT_SCENE
    LoadPBRTScene();
#else
    LoadSquidRoom();
#endif

#if USE_GRASS_GEOMETRY
    InitializeGrassGeometry();
#endif
}

// Build geometry used in the sample.
void Scene::InitializeGrassGeometry()
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
void Scene::InitializeGeometry()
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

// Build acceleration structure needed for raytracing.
void Scene::InitializeAllBottomLevelAccelerationStructures()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto FrameCount = m_deviceResources->GetBackBufferCount();

    m_accelerationStructure = make_unique<RaytracingAccelerationStructureManager>(device, MaxNumBottomLevelInstances, FrameCount);

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;    // ToDo specify via Scene_Args
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
            compactAS = false;
        }
        m_accelerationStructure->AddBottomLevelAS(device, buildFlags, bottomLevelASGeometry, updateOnBuild, updateOnBuild, compactAS);
    }
}


// Build acceleration structure needed for raytracing.
void Scene::InitializeAccelerationStructures()
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
        L"House"
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


#if !LOAD_ONLY_ONE_PBRT_MESH
    float radius = 75;
    XMMATRIX mTranslationSceneCenter = XMMatrixTranslation(-7, 0, 7);
    XMMATRIX mTranslation = XMMatrixTranslation(0, -1.5, radius);
    XMMATRIX mScale = XMMatrixScaling(10, 20, 1);
    int NumMirrorQuads = 12;
    for (int i = 0; i < NumMirrorQuads; i++)
    {
        float angleToRotateBy = 360.0f * (2.f * i / (2.f * NumMirrorQuads));
        XMMATRIX mRotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
        XMMATRIX mTransform = mScale * mTranslation * mRotate * mTranslationSceneCenter;
        m_accelerationStructure->AddBottomLevelASInstance(L"Quad", UINT_MAX, mTransform);
    }

    for (int i = 0; i < NumMirrorQuads; i++)
    {
        float angleToRotateBy = 360.0f * ((2.f * i + 1) / (2.f * NumMirrorQuads));
        XMMATRIX mRotate = XMMatrixRotationY(XMConvertToRadians(angleToRotateBy));
        XMMATRIX mTransform = mScale * mTranslation * mRotate * mTranslationSceneCenter;
        m_accelerationStructure->AddBottomLevelASInstance(L"MirrorQuad", UINT_MAX, mTransform);
    }

    m_animatedCarInstanceIndex = m_accelerationStructure->AddBottomLevelASInstance(L"Car", UINT_MAX, XMMatrixIdentity());
#endif


    //m_accelerationStructure->GetBottomLevelASInstance(5).SetTransform(XMMatrixTranslationFromVector(XMVectorSet(-10, 4, -10, 0)));

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
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS buildFlags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;    // ToDo specify via Scene_Args
    bool allowUpdate = false;
    bool performUpdateOnBuild = false;
    m_accelerationStructure->InitializeTopLevelAS(device, buildFlags, allowUpdate, performUpdateOnBuild, L"Top-Level Acceleration Structure");
}



// ToDo move to class
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

void Scene::GenerateGrassGeometry()
{
#if !GENERATE_GRASS
    return;
#endif
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
    float totalTime = Scene_Args::AnimateGrass ? static_cast<float>(m_timer.GetTotalSeconds()) : 0;

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
            resourceStateTracker->TransitionResource(&grassPatchVB, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            resourceStateTracker->InsertUAVBarrier(&grassPatchVB);
        }

        resourceStateTracker->FlushResourceBarriers();
        m_grassGeometryGenerator.Run(commandList, params, m_cbvSrvUavHeap->GetHeap(), grassPatchVB.gpuDescriptorWriteAccess);

        // Transition the output vertex buffer to VB state and make sure the CS is done writing.        
        {
            resourceStateTracker->TransitionResource(&grassPatchVB, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resourceStateTracker->InsertUAVBarrier(&grassPatchVB);
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
                    float approxDistanceToCamera = max(0.f, XMVectorGetX(XMVector3Length((centerPosition - m_camera.Eye()))) - 0.5f * width);
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

void Scene::UpdateAccelerationStructure()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto resourceStateTracker = m_deviceResources->GetGpuResourceStateTracker();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

    if (Scene_Args::EnableGeometryAndASBuildsAndUpdates)
    {
        bool forceBuild = false;    // ToDo

        resourceStateTracker->FlushResourceBarriers();
        m_accelerationStructure->Build(commandList, m_cbvSrvUavHeap->GetHeap(), frameIndex, forceBuild);
    }

    // ToDo we could use single structured buffer and use previous frame copy?
    // Copy previous frame Bottom Level AS instance transforms to GPU. 
    m_prevFrameBottomLevelASInstanceTransforms.CopyStagingToGpu(frameIndex);

    // Update the CPU staging copy with the current frame transforms.
    const auto& bottomLevelASInstanceDescs = m_accelerationStructure->GetBottomLevelASInstancesBuffer();
    for (UINT i = 0; i < bottomLevelASInstanceDescs.NumElements(); i++)
    {
        m_prevFrameBottomLevelASInstanceTransforms[i] = *reinterpret_cast<const XMFLOAT3X4*>(bottomLevelASInstanceDescs[i].Transform);
    }
}