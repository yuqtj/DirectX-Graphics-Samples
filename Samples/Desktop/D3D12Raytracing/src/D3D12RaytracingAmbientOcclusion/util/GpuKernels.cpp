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

// ToDo standardize naming
#include "../stdafx.h"
#include "PerformanceTimers.h"
#include "GpuKernels.h"
#include "CompiledShaders\ReduceSumUintCS.hlsl.h"
#include "CompiledShaders\ReduceSumFloatCS.hlsl.h"
#include "CompiledShaders\DownsampleBoxFilter2x2CS.hlsl.h"
#include "CompiledShaders\DownsampleGaussian9TapFilterCS.hlsl.h"
#include "CompiledShaders\DownsampleGaussian25TapFilterCS.hlsl.h"
#include "CompiledShaders\DownsampleBilateralFilter2x2CS.hlsl.h"
#include "CompiledShaders\UpsampleBilateralFilter2x2CS.hlsl.h"
#include "CompiledShaders\GaussianFilter3x3CS.hlsl.h"
#include "CompiledShaders\PerPixelMeanSquareErrorCS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Box3x3CS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian3x3CS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS.hlsl.h"
#include "CompiledShaders\CalculateVariance_Bilateral5x5CS.hlsl.h"
#include "CompiledShaders\CalculateVariance_Bilateral7x7CS.hlsl.h"

using namespace std;

namespace GpuKernels
{
	namespace RootSignature {
		namespace ReduceSum {
			namespace Slot {
				enum Enum {
					Output = 0,
					Input,
					Count
				};
			}
		}
	}

	void ReduceSum::Initialize(ID3D12Device* device, Type type)
	{
        m_resultType = type;

		// Create root signature.
		{
			using namespace RootSignature::ReduceSum;

			CD3DX12_DESCRIPTOR_RANGE ranges[2];
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
			ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

			CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
			rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
			rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[1]);

			CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
			SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: ReduceSum");
		}

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            
            switch (m_resultType)
            {
            case Uint:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pReduceSumUintCS), ARRAYSIZE(g_pReduceSumUintCS));
                break;
            case Float:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pReduceSumFloatCS), ARRAYSIZE(g_pReduceSumFloatCS));
                break;
            }

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: ReduceSum");
        }
	}

	void ReduceSum::CreateInputResourceSizeDependentResources(
		ID3D12Device* device,
		DescriptorHeap* descriptorHeap,
		UINT frameCount,
		UINT width,
		UINT height,
		UINT numInvocationsPerFrame)
	{
		// Create shader resources
		{
			width = CeilDivide(width, ReduceSumCS::ThreadGroup::NumElementsToLoadPerThread);
			
			// Number of reduce iterations to bring [width, height] down to [1, 1]
			UINT numIterations = max(
				CeilLogWithBase(width, ReduceSumCS::ThreadGroup::Width),
				CeilLogWithBase(height, ReduceSumCS::ThreadGroup::Height));

            DXGI_FORMAT format;
            switch (m_resultType)
            {
            case Uint: format = DXGI_FORMAT_R32_UINT; break;
            case Float: format = DXGI_FORMAT_R32_FLOAT; break;
            }

			m_csReduceSumOutputs.resize(numIterations);
			for (UINT i = 0; i < numIterations; i++)
			{
				width = max(1, CeilDivide(width, ReduceSumCS::ThreadGroup::Width));
				height = max(1, CeilDivide(height, ReduceSumCS::ThreadGroup::Height));

				m_csReduceSumOutputs[i].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
				CreateRenderTargetResource(device, format, width, height, descriptorHeap,
					&m_csReduceSumOutputs[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: ReduceSum intermediate output");
			}

			// ToDo should we allocate FrameCount + 1 in GPUTImeras we're depending on Present to stall?
            switch (m_resultType)
            {
            case Uint: m_resultSize = sizeof(UINT); break;
            case Float: m_resultSize = sizeof(float); break;
                break;
            }
			m_readbackResources.resize(numInvocationsPerFrame);
			for (UINT i = 0; i < m_readbackResources.size(); i++)
				for (auto& readbackResource : m_readbackResources)
				{
					wstringstream wResourceName;
					wResourceName << L"Readback buffer - ReduceSum output [" << i << L"]";
					AllocateReadBackBuffer(device, frameCount * m_resultSize, &m_readbackResources[i], D3D12_RESOURCE_STATE_COPY_DEST, wResourceName.str().c_str());
				}
		}
	}

	void ReduceSum::Execute(
		ID3D12GraphicsCommandList* commandList,
		ID3D12DescriptorHeap* descriptorHeap, 
		UINT frameIndex,
		UINT invocationIndex,   // per frame invocation index
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
		void* resultSum)
	{
		using namespace RootSignature::ReduceSum;
		
		// ToDo move out or rename
		PIXBeginEvent(commandList, 0, L"CalculateNumCameraRayHits");

		// Set pipeline state.
		{
			commandList->SetDescriptorHeaps(1, &descriptorHeap);
			commandList->SetComputeRootSignature(m_rootSignature.Get());
			commandList->SetPipelineState(m_pipelineStateObject.Get());
		}

		//
		// Iterative sum reduce [width, height] to [1,1]
		//
		SIZE_T readBackBaseOffset = frameIndex * m_resultSize;
		{
			// First iteration reads from input resource.		
			commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
			commandList->SetComputeRootDescriptorTable(Slot::Output, m_csReduceSumOutputs[0].gpuDescriptorWriteAccess);

			for (UINT i = 0; i < m_csReduceSumOutputs.size(); i++)
			{
				auto outputResourceDesc = m_csReduceSumOutputs[i].resource.Get()->GetDesc();

				// Each group writes out a single summed result across group threads.
				XMUINT2 groupSize(static_cast<UINT>(outputResourceDesc.Width), static_cast<UINT>(outputResourceDesc.Height));

				// Dispatch.
				commandList->Dispatch(groupSize.x, groupSize.y, 1);

				// Set the output resource as input in the next iteration. 
				if (i < m_csReduceSumOutputs.size() - 1)
				{
					commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_csReduceSumOutputs[i].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
					commandList->SetComputeRootDescriptorTable(Slot::Input, m_csReduceSumOutputs[i].gpuDescriptorReadAccess);
					commandList->SetComputeRootDescriptorTable(Slot::Output, m_csReduceSumOutputs[i + 1].gpuDescriptorWriteAccess);
				}
				else  // We're done, prepare the last output for copy to readback.
				{
					commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_csReduceSumOutputs.back().resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
				}
			}

			// Copy the sum result to the readback buffer.
            // ToDo should the readback take frameIndex into consideration in addition to invocationIndex if we dont wait on GPU below?
			auto destDesc = m_readbackResources[invocationIndex]->GetDesc();
			auto srcDesc = m_csReduceSumOutputs.back().resource.Get()->GetDesc();
			D3D12_PLACED_SUBRESOURCE_FOOTPRINT bufferFootprint = {};
			bufferFootprint.Offset = 0;
			bufferFootprint.Footprint.Width = static_cast<UINT>(destDesc.Width / m_resultSize);
			bufferFootprint.Footprint.Height = 1;
			bufferFootprint.Footprint.Depth = 1;
			bufferFootprint.Footprint.RowPitch = Align(static_cast<UINT>(destDesc.Width) * m_resultSize, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
			bufferFootprint.Footprint.Format = srcDesc.Format;
			CD3DX12_TEXTURE_COPY_LOCATION copyDest(m_readbackResources[invocationIndex].Get(), bufferFootprint);
			CD3DX12_TEXTURE_COPY_LOCATION copySrc(m_csReduceSumOutputs.back().resource.Get(), 0);
			commandList->CopyTextureRegion(&copyDest, frameIndex, 0, 0, &copySrc, nullptr);

			// Transition the intermediate output resources back.
			{
				std::vector<D3D12_RESOURCE_BARRIER> barriers;
				barriers.resize(m_csReduceSumOutputs.size());
				for (UINT i = 0; i < m_csReduceSumOutputs.size() - 1; i++)
				{
					barriers[i] = CD3DX12_RESOURCE_BARRIER::Transition(m_csReduceSumOutputs[i].resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
				}
				barriers[m_csReduceSumOutputs.size() - 1] = CD3DX12_RESOURCE_BARRIER::Transition(m_csReduceSumOutputs.back().resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

				commandList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
			}
		}

		// Performance optimization.
		// To avoid stalling CPU until GPU is done, grab the data from a finished frame FrameCount ago.
		// This is fine for the informational purposes of using the value for UI display only.
		UINT* mappedData = nullptr;
		CD3DX12_RANGE readRange(readBackBaseOffset, readBackBaseOffset + m_resultSize);
		ThrowIfFailed(m_readbackResources[invocationIndex]->Map(0, &readRange, reinterpret_cast<void**>(&mappedData)));
		memcpy(resultSum, mappedData, m_resultSize);
		m_readbackResources[invocationIndex]->Unmap(0, &CD3DX12_RANGE(0, 0));

		PIXEndEvent(commandList);
	}

	namespace RootSignature {
		namespace DownsampleBoxFilter2x2 {
			namespace Slot {
				enum Enum {
					Output = 0,
					Input,
					ConstantBuffer,
					Count
				};
			}
		}
	}

	void DownsampleBoxFilter2x2::Initialize(ID3D12Device* device)
	{
		// Create root signature.
		{
			using namespace RootSignature::DownsampleBoxFilter2x2;

			CD3DX12_DESCRIPTOR_RANGE ranges[2]; // Perfomance TIP: Order from most frequent to least frequent.
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
			ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

			CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
			rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
			rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[1]);
			rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

			CD3DX12_STATIC_SAMPLER_DESC staticSampler(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

			CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, 1, &staticSampler);
			SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: DownsampleBoxFilter2x2");
		}

		// Create compute pipeline state.
		{
			D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
			descComputePSO.pRootSignature = m_rootSignature.Get();
			descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleBoxFilter2x2CS), ARRAYSIZE(g_pDownsampleBoxFilter2x2CS));

			ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
			m_pipelineStateObject->SetName(L"Pipeline state object: DownsampleBoxFilter2x2");
		}
	}

	void DownsampleBoxFilter2x2::CreateInputResourceSizeDependentResources(
		ID3D12Device* device,
		UINT width,
		UINT height)
	{
		// Create shader resources
		{
			m_CB.Create(device, 1, L"Constant Buffer: DownsampleBoxFilter2x2");
			m_CB->inputTextureDimensions = XMUINT2(width, height);
			m_CB->invertedInputTextureDimensions = XMFLOAT2(1.f/width, 1.f/height);
			m_CB.CopyStagingToGpu();
		}
	}

	// Downsamples input resource.
	// width, height - dimensions of the input resource.
	void DownsampleBoxFilter2x2::Execute(
		ID3D12GraphicsCommandList* commandList,
		UINT width,
		UINT height,
		ID3D12DescriptorHeap* descriptorHeap,
		const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
		const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
	{
		using namespace RootSignature::DownsampleBoxFilter2x2;
		using namespace DownsampleBoxFilter2x2;

		PIXBeginEvent(commandList, 0, L"DownsampleBoxFilter2x2");

		// Set pipeline state.
		{
			commandList->SetDescriptorHeaps(1, &descriptorHeap);
			commandList->SetComputeRootSignature(m_rootSignature.Get());
			commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
			commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
			commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress());
			commandList->SetPipelineState(m_pipelineStateObject.Get());
		}

		// ToDo handle misaligned input
		XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));

		// Dispatch.
		commandList->Dispatch(groupSize.x, groupSize.y, 1);

		PIXEndEvent(commandList);
	}

	namespace RootSignature {
		namespace DownsampleGaussianFilter {
			namespace Slot {
				enum Enum {
					Output = 0,
					Input,
					ConstantBuffer,
					Count
				};
			}
		}
	}

	void DownsampleGaussianFilter::Initialize(ID3D12Device* device, Type type)
	{
		// Create root signature.
		{
			using namespace RootSignature::DownsampleGaussianFilter;

			CD3DX12_DESCRIPTOR_RANGE ranges[2]; // Perfomance TIP: Order from most frequent to least frequent.
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
			ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

			CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
			rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
			rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[1]);
			rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

			CD3DX12_STATIC_SAMPLER_DESC staticSampler(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

			CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, 1, &staticSampler);
			SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: DownsampleGaussianFilter");
		}

		// Create compute pipeline state.
		{
			D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
			descComputePSO.pRootSignature = m_rootSignature.Get();
			switch (type)
			{
			case Tap9:
				descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleGaussian9TapFilterCS), ARRAYSIZE(g_pDownsampleGaussian9TapFilterCS));
				break;
			case Tap25:
				descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleGaussian25TapFilterCS), ARRAYSIZE(g_pDownsampleGaussian25TapFilterCS));
				break;
			}

			ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
			m_pipelineStateObject->SetName(L"Pipeline state object: DownsampleGaussianFilter");
		}
	}

	void DownsampleGaussianFilter::CreateInputResourceSizeDependentResources(
		ID3D12Device* device,
		UINT width,
		UINT height)
	{
		// Create shader resources
		{
			m_CB.Create(device, 1, L"Constant Buffer: DownsampleGaussianFilter");
			m_CB->inputTextureDimensions = XMUINT2(width, height);
			m_CB->invertedInputTextureDimensions = XMFLOAT2(1.f / width, 1.f / height);
			m_CB.CopyStagingToGpu();
		}
	}

	// Downsamples input resource.
	// width, height - dimensions of the input resource.
	void DownsampleGaussianFilter::Execute(
		ID3D12GraphicsCommandList* commandList,
		UINT width,
		UINT height,
		ID3D12DescriptorHeap* descriptorHeap,
		const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
		const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
	{
		using namespace RootSignature::DownsampleGaussianFilter;
		using namespace DownsampleGaussianFilter;

		PIXBeginEvent(commandList, 0, L"DownsampleGaussianFilter");

		// Set pipeline state.
		{
			commandList->SetDescriptorHeaps(1, &descriptorHeap);
			commandList->SetComputeRootSignature(m_rootSignature.Get());
			commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
			commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
			commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress());
			commandList->SetPipelineState(m_pipelineStateObject.Get());
		}

		// ToDo handle misaligned input
		XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));

		// Dispatch.
		commandList->Dispatch(groupSize.x, groupSize.y, 1);

		PIXEndEvent(commandList);
	}


    namespace RootSignature {
        namespace DownsampleBilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    OutputNormal,
                    OutputPosition,
                    OutputGeometryHit,
                    Input,
                    InputNormal,
                    InputPosition,
                    InputGeometryHit,
                    Count
                };
            }
        }
    }

    void DownsampleBilateralFilter::Initialize(ID3D12Device* device, Type type)
    {
        // Create root signature.
        {
            using namespace RootSignature::DownsampleBilateralFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[8]; // Perfomance TIP: Order from most frequent to least frequent.
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // 1 input normal texture
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // 1 input position texture
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);  // 1 input geometry hit texture
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
            ranges[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // 1 output normal texture
            ranges[6].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);  // 1 output position texture
            ranges[7].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3);  // 1 output geometry hit texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::InputNormal].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::InputPosition].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::InputGeometryHit].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::OutputNormal].InitAsDescriptorTable(1, &ranges[5]);
            rootParameters[Slot::OutputPosition].InitAsDescriptorTable(1, &ranges[6]);
            rootParameters[Slot::OutputGeometryHit].InitAsDescriptorTable(1, &ranges[7]);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: DownsampleBilateralFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            switch (type)
            {
            case Filter2X2:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleBilateralFilter2x2CS), ARRAYSIZE(g_pDownsampleBilateralFilter2x2CS));
                break;
            }

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: DownsampleBilateralFilter");
        }
    }

    // Downsamples input resource.
    // width, height - dimensions of the input resource.
    void DownsampleBilateralFilter::Execute(
        ID3D12GraphicsCommandList* commandList,
        UINT width,
        UINT height,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputPositionResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputGeometryHitResourceHandle, 
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputPositionResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputGeometryHitResourceHandle)
    {
        using namespace RootSignature::DownsampleBilateralFilter;
        using namespace DownsampleBilateralFilter;

        PIXBeginEvent(commandList, 0, L"DownsampleBilateralFilter");

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputNormal, inputNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputPosition, inputPositionResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputGeometryHit, inputGeometryHitResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputNormal, outputNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputPosition, outputPositionResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputGeometryHit, outputGeometryHitResourceHandle);
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo handle misaligned input
        XMUINT2 groupSize(CeilDivide((width + 1) / 2 + 1, ThreadGroup::Width), CeilDivide((height + 1) / 2 + 1, ThreadGroup::Height));

        // Dispatch.
        commandList->Dispatch(groupSize.x, groupSize.y, 1);

        PIXEndEvent(commandList);
    }


    namespace RootSignature {
        namespace UpsampleBilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    InputLowResNormal,
                    InputHiResNormal,
                    Count
                };
            }
        }
    }

    // ToDo test downsample,upsample on odd resolution
    void UpsampleBilateralFilter::Initialize(ID3D12Device* device, Type type)
    {
        // Create root signature.
        {
            using namespace RootSignature::UpsampleBilateralFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[4]; // Perfomance TIP: Order from most frequent to least frequent.
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // 1 input normal low res texture
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // 1 input normal high res texture
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::InputLowResNormal].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::InputHiResNormal].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[3]);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: UpsampleBilateralFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            switch (type)
            {
            case Filter2X2:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pUpsampleBilateralFilter2x2CS), ARRAYSIZE(g_pUpsampleBilateralFilter2x2CS));
                break;
            }

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: UpsampleBilateralFilter");
        }
    }

    // Resamples input resource.
    // width, height - dimensions of the input resource.
    void UpsampleBilateralFilter::Execute(
        ID3D12GraphicsCommandList* commandList,
        UINT width,
        UINT height,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
    {
        using namespace RootSignature::UpsampleBilateralFilter;
        using namespace UpsampleBilateralFilter;

        PIXBeginEvent(commandList, 0, L"UpsampleBilateralFilter");

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputLowResNormal, inputLowResNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputHiResNormal, inputHiResNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo handle misaligned input
        // Start from -1,-1 pixel to account for high-res pixel border around low-res pixel border.
        XMUINT2 groupSize(CeilDivide(width + 1, ThreadGroup::Width), CeilDivide(height + 1, ThreadGroup::Height));

        // Dispatch.
        commandList->Dispatch(groupSize.x, groupSize.y, 1);

        PIXEndEvent(commandList);
    }

    namespace RootSignature {
        namespace GaussianFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    void GaussianFilter::Initialize(ID3D12Device* device)
    {
        // Create root signature.
        {
            using namespace RootSignature::GaussianFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[2]; // Perfomance TIP: Order from most frequent to least frequent.
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_STATIC_SAMPLER_DESC staticSampler(0, D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_MIRROR, D3D12_TEXTURE_ADDRESS_MODE_MIRROR);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, 1, &staticSampler);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: GaussianFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case Filter3X3:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pGaussianFilter3x3CS), ARRAYSIZE(g_pGaussianFilter3x3CS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: GaussianFilter");
            }
        }

        // Create shader resources
        {
            m_CB.Create(device, 1, L"Constant Buffer: GaussianFilter");
        }
    }

    // ToDo fix up input order to be same among kernels

    // Blurs input resource with a Gaussian filter.
    // width, height - dimensions of the input resource.
    void GaussianFilter::Execute(
        ID3D12GraphicsCommandList* commandList,
        UINT width,
        UINT height,
        FilterType type,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
    {
        using namespace RootSignature::GaussianFilter;
        using namespace GaussianFilter;

        PIXBeginEvent(commandList, 0, L"GaussianFilter");

        m_CB->textureDim = XMUINT2(width, height);
        m_CB->invTextureDim = XMFLOAT2(1.f / width, 1.f / height);
        m_CB.CopyStagingToGpu();

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress());
            commandList->SetPipelineState(m_pipelineStateObjects[type].Get());
        }

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);

        PIXEndEvent(commandList);
    }


    namespace RootSignature {
        namespace RootMeanSquareError {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    InputReference,
                    Count
                };
            }
        }
    }

    void RootMeanSquareError::Initialize(ID3D12Device* device)
    {
        // Create root signature.
        {
            using namespace RootSignature::RootMeanSquareError;

            CD3DX12_DESCRIPTOR_RANGE ranges[2];
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0);  // 2 input textures
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[1]);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: PerPixelMeanSquareError");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pPerPixelMeanSquareErrorCS), ARRAYSIZE(g_pPerPixelMeanSquareErrorCS));

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: PerPixelMeanSquareError");
        }

        m_reduceSumKernel.Initialize(device, ReduceSum::Float);
    }

    void RootMeanSquareError::CreateInputResourceSizeDependentResources(
        ID3D12Device* device,
        DescriptorHeap* descriptorHeap,
        UINT frameCount,
        UINT width,
        UINT height,
        UINT numInvocationsPerFrame)
    {
        // Create shader resources
        {
            m_perPixelMeanSquareError.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, width, height, descriptorHeap,
                &m_perPixelMeanSquareError, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: PerPixelMeanSquareError output");
        }

        m_reduceSumKernel.CreateInputResourceSizeDependentResources(device, descriptorHeap, frameCount, width, height, numInvocationsPerFrame);
    }

    // Calculates root mean square error
    //  1) Executes a CS calculating per-pixel mean square error
    //  2) Executes a Reduce Sum CS aggregating mean square error
    //  3) Takes a root square on the CPU of the readback result
    // width, height - dimensions of the input resource.
    void RootMeanSquareError::Execute(
        ID3D12GraphicsCommandList* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        UINT frameIndex,
        UINT invocationIndex,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
        float* rootMeanSquareError)
    {
        using namespace RootSignature::RootMeanSquareError;
        using namespace RootMeanSquareError;

        PIXBeginEvent(commandList, 0, L"RootMeanSquareError");

        // Calcualte per-pixel mean square error.
        auto desc = m_perPixelMeanSquareError.resource->GetDesc();
        {
            // Set pipeline state.
            {
                commandList->SetDescriptorHeaps(1, &descriptorHeap);
                commandList->SetComputeRootSignature(m_rootSignature.Get());
                commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
                commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
                commandList->SetPipelineState(m_pipelineStateObject.Get());
            }
            XMUINT2 groupSize(CeilDivide(static_cast<UINT>(desc.Width), ThreadGroup::Width), CeilDivide(static_cast<UINT>(desc.Height), ThreadGroup::Height));

            // Dispatch.
            commandList->Dispatch(groupSize.x, groupSize.y, 1);
        }

        // Sum the mean square error across all pixels.
        float sumPerPixelMeanSquareError;
        {
            // ToDo index the result to a frame its from or wait on the GPU to finish
            m_reduceSumKernel.Execute(commandList, descriptorHeap, frameIndex, invocationIndex, m_perPixelMeanSquareError.gpuDescriptorReadAccess, &sumPerPixelMeanSquareError);
        }

        // Calculate root mean square error.
        *rootMeanSquareError = sqrtf(sumPerPixelMeanSquareError / (desc.Width * desc.Height));

        PIXEndEvent(commandList);
    }

    // ToDo prune
    namespace RootSignature {
        namespace AtrousWaveletTransformCrossBilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    VarianceOutput,
                    Input,
                    Normals,
                    NormalsOct,
                    Depths,
                    Variance,
                    SmoothedVariance,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    // ToDo move type to execute
    void AtrousWaveletTransformCrossBilateralFilter::Initialize(ID3D12Device* device, UINT maxFilterPasses)
    {
        // Create root signature.
        {
            using namespace RootSignature::AtrousWaveletTransformCrossBilateralFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[8];
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input values
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // input normals
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // input depths
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);  // input octahedron compressed normals
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4);  // input variance
            ranges[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // input smoothed variance
            ranges[6].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output filtered values
            ranges[7].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // output filtered variance

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::Normals].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::Depths].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::NormalsOct].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::Variance].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::SmoothedVariance].InitAsDescriptorTable(1, &ranges[5]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[6]);
            rootParameters[Slot::VarianceOutput].InitAsDescriptorTable(1, &ranges[7]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: AtrousWaveletTransformCrossBilateralFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case EdgeStoppingGaussian5x5:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS), ARRAYSIZE(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS));
                    break;
                case EdgeStoppingGaussian3x3:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian3x3CS), ARRAYSIZE(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian3x3CS));
                    break;
                case EdgeStoppingBox3x3:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Box3x3CS), ARRAYSIZE(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Box3x3CS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: AtrousWaveletTransformCrossBilateralFilter");
            }
        }

        // Create shader resources.
        {
            m_CB.Create(device, maxFilterPasses, L"Constant Buffer: AtrousWaveletTransformCrossBilateralFilter");
        }
    }

    void AtrousWaveletTransformCrossBilateralFilter::CreateInputResourceSizeDependentResources(
        ID3D12Device* device,
        DescriptorHeap* descriptorHeap,
        UINT width,
        UINT height)
    {
        // Create shader resources
        {
            m_intermediateValueOutput.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, width, height, descriptorHeap,
                &m_intermediateValueOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter intermediate value output");

            m_intermediateVarianceOutputs[0].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, width, height, descriptorHeap,
                &m_intermediateVarianceOutputs[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter intermediate variance output 0");

            m_intermediateVarianceOutputs[1].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, width, height, descriptorHeap,
                &m_intermediateVarianceOutputs[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter intermediate variance output 1");
        }

        // Update the Constant Buffers.
        m_CB->textureDim = XMINT2(width, height);
        for (UINT i = 0; i < m_CB.NumInstances(); i++)
        {
            m_CB->kernelStepShift = i;
            m_CB.CopyStagingToGpu(i);
        }
    }

    // ToDo add option to allow input, output being the same
    // Expects, and returns, outputResource in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state.
    void AtrousWaveletTransformCrossBilateralFilter::Execute(
        ID3D12GraphicsCommandList* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        FilterType filterType,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsOctResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputVarianceResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputSmoothedVarianceResourceHandle,
        RWGpuResource* outputResource,
        float valueSigma,
        float depthSigma,
        float normalSigma,
        UINT numFilterPasses,
        bool reverseFilterPassOrder,
        bool useCalculatedVariance)
    {
        using namespace RootSignature::AtrousWaveletTransformCrossBilateralFilter;
        using namespace AtrousWaveletTransformFilterCS;

        PIXBeginEvent(commandList, 0, L"AtrousWaveletTransformCrossBilateralFilter");

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObjects[filterType].Get());
            commandList->SetComputeRootDescriptorTable(Slot::Normals, inputNormalsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Depths, inputDepthsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::NormalsOct, inputNormalsOctResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::SmoothedVariance, inputSmoothedVarianceResourceHandle);
        }

        // Update the Constant Buffers.
        for (UINT i = 0; i < m_CB.NumInstances(); i++)
        {
            // Ref: Dammertz2010
            // Tighten value range smoothing for higher passes.

            // Ref: Dundr2018
            // Reverse wavelet order to blur out ringing at larger offsets.
            int _i;
            if (reverseFilterPassOrder)
                _i = i == 0 ? 0 : numFilterPasses - 1 - i;
            else
                _i = i;
            if (useCalculatedVariance)
            {
                m_CB->valueSigma = valueSigma;
            }
            else
            {
                m_CB->valueSigma = _i > 0 ? valueSigma * powf(2.f, -float(i)) : 1;
            }
            m_CB->depthSigma = depthSigma;
            m_CB->normalSigma = normalSigma;
            m_CB->kernelStepShift = _i;
            m_CB->scatterOutput = _i == numFilterPasses - 1;
            m_CB->useCalculatedVariance = useCalculatedVariance;
            m_CB.CopyStagingToGpu(i);
        }

        //
        // Iterative filter
        //
        {
            auto resourceDesc = outputResource->resource.Get()->GetDesc();
            XMUINT2 resourceDim(static_cast<UINT>(resourceDesc.Width), static_cast<UINT>(resourceDesc.Height));
            XMUINT2 groupSize(CeilDivide(resourceDim.x, ThreadGroup::Width), CeilDivide(resourceDim.y, ThreadGroup::Height));

#if ATROUS_ONELEVEL_ONLY
            UINT i = numFilterPasses - 1;
            numFilterPasses = 1;

            RWGpuResource* outValueResources[2] = { outputResource, &m_intermediateValueOutput };

            commandList->SetComputeRootDescriptorTable(Slot::Input, inputValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outValueResources[0]->gpuDescriptorWriteAccess);
#else
            // Order the resources such that the final pass writes to outputResource.
            RWGpuResource* outValueResources[2] =
            {
                numFilterPasses % 2 == 1 ? outputResource : &m_intermediateValueOutput,
                numFilterPasses % 2 == 1 ? &m_intermediateValueOutput : outputResource,
            };


            // First iteration reads from input resource.		
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outValueResources[0]->gpuDescriptorWriteAccess);
            commandList->SetComputeRootDescriptorTable(Slot::VarianceOutput, m_intermediateVarianceOutputs[0].gpuDescriptorWriteAccess);

            for (UINT i = 0; i < numFilterPasses; i++)
#endif
            {
                commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(i));
                                
                // Dispatch.
                commandList->Dispatch(groupSize.x, groupSize.y, 1);

                // Transition and bind resources for the next pass.
                // Flip input/output. 
                if (i < numFilterPasses - 1)
                {
                    // Ping-pong input/output resources across passes.
                    UINT inputID = i % 2;
                    UINT outputID = (i + 1) % 2;

                    D3D12_RESOURCE_BARRIER barriers[4] = {
                        CD3DX12_RESOURCE_BARRIER::Transition(m_intermediateVarianceOutputs[inputID].resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                        CD3DX12_RESOURCE_BARRIER::Transition(outValueResources[inputID]->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),

                        CD3DX12_RESOURCE_BARRIER::Transition(m_intermediateVarianceOutputs[outputID].resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                        CD3DX12_RESOURCE_BARRIER::Transition(outValueResources[outputID]->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                    };
                    // Only flip outputs from pass 0 as the caller passed in resources were used as input.
                    commandList->ResourceBarrier(i == 0 ? 2 : ARRAYSIZE(barriers), barriers);

                    // Flip input, output resources.
                    commandList->SetComputeRootDescriptorTable(Slot::Input, outValueResources[inputID]->gpuDescriptorReadAccess);
                    commandList->SetComputeRootDescriptorTable(Slot::Output, outValueResources[outputID]->gpuDescriptorWriteAccess);
                    commandList->SetComputeRootDescriptorTable(Slot::SmoothedVariance, m_intermediateVarianceOutputs[inputID].gpuDescriptorReadAccess);
                    commandList->SetComputeRootDescriptorTable(Slot::VarianceOutput, m_intermediateVarianceOutputs[outputID].gpuDescriptorWriteAccess);
                }
            }
            
            // Transition the intermediate output resource back.
            if (numFilterPasses > 1)
            {
                bool isVar0ResourceInUAVState = ((numFilterPasses - 1) % 2) == 0;   
                D3D12_RESOURCE_BARRIER barriers[] = {
                    CD3DX12_RESOURCE_BARRIER::Transition(m_intermediateValueOutput.resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    CD3DX12_RESOURCE_BARRIER::Transition(m_intermediateVarianceOutputs[isVar0ResourceInUAVState ? 1 : 0].resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                };
                // Transition variance resources back only if they're not in their default state.
                commandList->ResourceBarrier(ARRAYSIZE(barriers), barriers);
            }
        }

        PIXEndEvent(commandList);
    }



    namespace RootSignature {
        namespace CalculateVariance {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    Normals,
                    NormalsOct,
                    Depths,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    // ToDo move type to execute
    void CalculateVariance::Initialize(ID3D12Device* device)
    {
        // Create root signature.
        {
            using namespace RootSignature::CalculateVariance;

            CD3DX12_DESCRIPTOR_RANGE ranges[5];
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input values
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // input normals
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // input depths
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);  // input octahedron compressed normals
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output filtered values

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::Normals].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::Depths].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::NormalsOct].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: CalculateVariance");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case Bilateral5x5:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateVariance_Bilateral5x5CS), ARRAYSIZE(g_pCalculateVariance_Bilateral5x5CS));
                    break;
                case Bilateral7x7:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateVariance_Bilateral7x7CS), ARRAYSIZE(g_pCalculateVariance_Bilateral7x7CS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: CalculateVariance");
            }
        }

        // Create shader resources.
        {
            m_CB.Create(device, 1, L"Constant Buffer: CalculateVariance");
        }
    }

    // ToDo add option to allow input, output being the same
    // Expects, and returns, outputResource in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state.
    void CalculateVariance::Execute(
        ID3D12GraphicsCommandList* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        FilterType filterType,
        UINT width,
        UINT height,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsOctResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
        float depthSigma,
        float normalSigma,
        bool useApproximateVariance)
    {
        using namespace RootSignature::CalculateVariance;
        using namespace CalculateVariance_Bilateral;

        // ToDo move out or rename
        PIXBeginEvent(commandList, 0, L"CalculateVariance_Bilateral5x5");

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObjects[filterType].Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Normals, inputNormalsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Depths, inputDepthsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::NormalsOct, inputNormalsOctResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
        }

        // Update the Constant Buffer.
        m_CB->depthSigma = depthSigma;
        m_CB->normalSigma = normalSigma;
        m_CB->textureDim = XMINT2(width, height);
        m_CB->useApproximateVariance = useApproximateVariance;
        m_CB.CopyStagingToGpu();
        commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress());


        // Dispatch.
        {
            XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
            commandList->Dispatch(groupSize.x, groupSize.y, 1);
        }

        PIXEndEvent(commandList);
    }
}