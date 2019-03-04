// ToDo fix up file headers 
#pragma once

// ToDo move to DXSampleHelper.h

#define NAME_D3D12_OBJECT(x) SetName((x).Get(), L#x)

// Round up size to the next 32 bit multiple.
inline constexpr UINT RoundUp32(const UINT size)
{
    return ((size - 1) / sizeof(UINT32) + 1);
}

// Align to a certain value.
inline constexpr UINT AlignArbitrary(const UINT size, const UINT alignment)
{
    return size - 1 - (size - 1) % alignment + alignment;
}

inline constexpr UINT GetNumGrps(const UINT size, const UINT numThreads)
{
    return (size + numThreads - 1) / numThreads;
}

// Allocate texture2D and upload data to the GPU.
inline void AllocateTexture2D(
    ID3D12Device* pDevice,
    void *pData,
    ID3D12Resource **ppResource,
    ID3D12Resource **ppUploadResource,
    ID3D12GraphicsCommandList* commandList,
    UINT width,
    UINT height,
    UINT stride,
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT,
    const wchar_t* resourceName = nullptr)
{
    auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto texDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, 1);
    ThrowIfFailed(pDevice->CreateCommittedResource(
        &defaultHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(ppResource)));

    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC buffer = CD3DX12_RESOURCE_DESC::Buffer(GetRequiredIntermediateSize(*ppResource, 0, 1));

    ThrowIfFailed(pDevice->CreateCommittedResource(
        &uploadHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &buffer,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(ppUploadResource)));

    if (resourceName)
    {
        (*ppResource)->SetName(resourceName);
    }

    // Upload the texture info to the GPU.
    D3D12_SUBRESOURCE_DATA resource;
    resource.pData = pData;
    resource.RowPitch = width * stride;
    resource.SlicePitch = resource.RowPitch * height;

    UpdateSubresources(commandList, *ppResource, *ppUploadResource, 0, 0, 1, &resource);

    // Change state to read.
    D3D12_RESOURCE_BARRIER barrier[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(*ppResource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ)
    };
    commandList->ResourceBarrier(_countof(barrier), barrier);
}

inline void AllocateTexture2DArr(
    ID3D12Device* pDevice,
    ID3D12Resource **ppResource,
    UINT width,
    UINT height,
    UINT arr,
    D3D12_CLEAR_VALUE* pClear = nullptr,
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT,
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
    const wchar_t* resourceName = nullptr)
{
    auto texDesc = CD3DX12_RESOURCE_DESC::Tex2D(format, width, height, (UINT16)arr, 1, 1, 0, flags);
    CD3DX12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    ThrowIfFailed(pDevice->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        state,
        pClear,
        IID_PPV_ARGS(ppResource)));

    if (resourceName)
    {
        (*ppResource)->SetName(resourceName);
    }
}


// Allocate Buffer.
inline void AllocateBuffer(
    ID3D12Device* pDevice,
    UINT64 bufferSize,
    ID3D12Resource **ppResource,
    D3D12_RESOURCE_STATES initialResourceState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    const wchar_t* resourceName = nullptr)
{
    auto defaultProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);
    ThrowIfFailed(pDevice->CreateCommittedResource(
        &defaultProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        initialResourceState,
        nullptr,
        IID_PPV_ARGS(ppResource)));

    if (resourceName)
    {
        (*ppResource)->SetName(resourceName);
    }
}

// Create CBV.
inline void CreateCBV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    UINT sizeInBytes,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle)
{
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = pResources->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = sizeInBytes;

    pDevice->CreateConstantBufferView(&cbvDesc, cpuDescriptorHandle);
}

// Create rtv.
inline void CreateRTV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    D3D12_RENDER_TARGET_VIEW_DESC* rtvDesc = nullptr)
{
    pDevice->CreateRenderTargetView(pResources, rtvDesc, cpuDescriptorHandle);
}

// Create dsv.
inline void CreateDSV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    D3D12_DEPTH_STENCIL_VIEW_DESC* dsvDesc = nullptr
)
{
    pDevice->CreateDepthStencilView(pResources, dsvDesc, cpuDescriptorHandle);
}

// Create uav.
inline void CreateUAV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    D3D12_UNORDERED_ACCESS_VIEW_DESC* uavDesc = nullptr,
    ID3D12Resource* pCounterResource = nullptr)
{
    pDevice->CreateUnorderedAccessView(
        pResources,
        pCounterResource,
        uavDesc,
        cpuDescriptorHandle
    );
}

// Create srv.
inline void CreateSRV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    D3D12_SHADER_RESOURCE_VIEW_DESC* srvDesc = nullptr)
{
    pDevice->CreateShaderResourceView(
        pResources,
        srvDesc,
        cpuDescriptorHandle
    );
}

// Create SRV for a buffer.
inline void CreateBufferSRV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    UINT firstElement,
    UINT numElements,
    UINT elementSize,
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN,
    D3D12_BUFFER_SRV_FLAGS flags = D3D12_BUFFER_SRV_FLAG_NONE)
{
    // SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = format;
    srvDesc.Buffer.NumElements = numElements;
    srvDesc.Buffer.Flags = flags;
    srvDesc.Buffer.StructureByteStride = elementSize;
    srvDesc.Buffer.FirstElement = firstElement;

    CreateSRV(pDevice, pResources, cpuDescriptorHandle, &srvDesc);
}

// Create SRV for a texture2D.
inline void CreateTexture2DSRV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT,
    UINT mipLevels = 1)
{
    // SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = format;
    srvDesc.Texture2D.MipLevels = mipLevels;

    CreateSRV(pDevice, pResources, cpuDescriptorHandle, &srvDesc);
};

inline void CreateTexture2DUAV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT,
    UINT mipSlice = 0)
{
    // UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uavDesc.Format = format;
    uavDesc.Texture2D.MipSlice = mipSlice;

    CreateUAV(pDevice, pResources, cpuDescriptorHandle, &uavDesc);
}

inline void CreateTexture2DArrSRV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    UINT arraySize,
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT,
    UINT mipLevels = 1)
{
    // SRV
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = format;
    srvDesc.Texture2DArray.ArraySize = arraySize;
    srvDesc.Texture2DArray.MipLevels = mipLevels;

    CreateSRV(pDevice, pResources, cpuDescriptorHandle, &srvDesc);
}

inline void CreateTexture2DArrUAV(
    ID3D12Device* pDevice,
    ID3D12Resource* pResources,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    UINT arraySize,
    DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT)
{
    // UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    uavDesc.Format = format;
    uavDesc.Texture2DArray.ArraySize = arraySize;

    CreateUAV(pDevice, pResources, cpuDescriptorHandle, &uavDesc);
}

// Create sampler for a texture2D.
inline void CreateTexture2DSampler(
    ID3D12Device* pDevice,
    D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle,
    D3D12_FILTER filter,
    D3D12_TEXTURE_ADDRESS_MODE addressU,
    D3D12_TEXTURE_ADDRESS_MODE addressV,
    D3D12_TEXTURE_ADDRESS_MODE addressW,
    float* borderColor = nullptr,
    D3D12_COMPARISON_FUNC compFunc = D3D12_COMPARISON_FUNC_ALWAYS,
    UINT maxAnisotropy = 1,
    float minLod = 0,
    float maxLod = D3D12_FLOAT32_MAX
)
{
    D3D12_SAMPLER_DESC samplerDescNoWrap = {};
    samplerDescNoWrap.Filter = filter;
    samplerDescNoWrap.AddressU = addressU;
    samplerDescNoWrap.AddressV = addressV;
    samplerDescNoWrap.AddressW = addressW;
    samplerDescNoWrap.MinLOD = minLod;
    samplerDescNoWrap.MaxLOD = maxLod;
    samplerDescNoWrap.MipLODBias = 0.0f;
    samplerDescNoWrap.MaxAnisotropy = maxAnisotropy;
    samplerDescNoWrap.ComparisonFunc = compFunc;

    if (borderColor != nullptr)
        memcpy(samplerDescNoWrap.BorderColor, borderColor, sizeof(samplerDescNoWrap.BorderColor));

    pDevice->CreateSampler(&samplerDescNoWrap, cpuDescriptorHandle);
}

// Create root signature.
inline void SerializeAndCreateRootSignature(
    ID3D12Device* device,
    D3D12_ROOT_SIGNATURE_DESC& desc,
    ComPtr<ID3D12RootSignature>& rootSig)
{
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;
    ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ThrowIfFailed(device->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(rootSig.GetAddressOf())));
}

union Param
{
    Param(float f) : f(f) {}
    Param(UINT u) : u(u) {}
    Param(INT i) : i(i) {}

    void operator= (float pf) { f = pf; }
    void operator= (UINT pu) { u = pu; }
    void operator= (INT pi) { i = pi; }

    union
    {
        float f;
        UINT u;
        INT i;
    };
};