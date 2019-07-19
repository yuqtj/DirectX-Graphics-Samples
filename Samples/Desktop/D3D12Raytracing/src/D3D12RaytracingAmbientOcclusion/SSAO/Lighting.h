//--------------------------------------------------------------------------------------
// D3D12RaytracingAO.h
//
// Advanced Technology Group (ATG)
// Copyright (C) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "CommonStates.h"

#if SSAO_DISABLED_CODE
#include "Menus.h"
#include "Mesh.h"
#endif

class Lighting
{
public:
    virtual ~Lighting()
    {
        m_deviceResources.reset();
#if SSAO_DISABLED_CODE
        m_primitiveBatch.reset();
        m_basicEffect.reset();
#endif
    }

    virtual void Setup(
        std::shared_ptr<DX::DeviceResources> pDeviceResources)
    {
        m_deviceResources = pDeviceResources;

#if SSAO_DISABLED_CODE
        SetupSplitRendering();
#endif
    }

#if SSAO_DISABLED_CODE
    virtual void Run(ComPtr<ID3D12Resource> pSceneConstantResource) = 0;
#endif

#if SSAO_DISABLED_CODE
    virtual void SetMesh(std::shared_ptr<Mesh> pMesh) = 0;
#endif
    virtual void OnSizeChanged() { }

#if SSAO_DISABLED_CODE
    virtual void OnOptionUpdate(std::shared_ptr<Menus> pMenu) = 0;
#endif

    virtual void OnCameraChanged(XMMATRIX& world, XMMATRIX& view, XMMATRIX& projection)
    {
        UNREFERENCED_PARAMETER(world);
        UNREFERENCED_PARAMETER(view);
        UNREFERENCED_PARAMETER(projection);
    }

    virtual void ChangeScreenScale(float pScreenWidthScale)
    {
        m_deviceResources->WaitForGpu();

        m_screenWidthScale = pScreenWidthScale;
        OnSizeChanged();

        m_deviceResources->WaitForGpu();
    }

#if SSAO_DISABLED_CODE
    void SetupSplitRendering()
    {
        auto device = m_deviceResources->GetD3DDevice();

        // Setup split rendering.
        {
            m_primitiveBatch = std::make_unique<PrimitiveBatch<VertexPositionTexture>>(device);

            RenderTargetState rtState(m_deviceResources->GetBackBufferFormat(), DXGI_FORMAT_UNKNOWN);
            EffectPipelineStateDescription pd(
                &VertexPositionTexture::InputLayout,
                CommonStates::Opaque,
                CommonStates::DepthNone,
                CommonStates::CullNone,
                rtState
            );

            m_basicEffect = std::make_unique<BasicEffect>(device, EffectFlags::Texture, pd);

            m_basicEffect->SetProjection(
                XMMatrixOrthographicOffCenterRH(
                    0,
                    1.f,
                    1.f,
                    0,
                    0,
                    1)
            );
        }
    }
#endif

protected:
    std::shared_ptr<DX::DeviceResources> m_deviceResources;
    float m_screenWidthScale = 1.f;

#if SSAO_DISABLED_CODE
    // Split rendering.
    std::unique_ptr<DirectX::PrimitiveBatch<DirectX::VertexPositionTexture>> m_primitiveBatch;
    std::unique_ptr<DirectX::BasicEffect> m_basicEffect;
#endif
};