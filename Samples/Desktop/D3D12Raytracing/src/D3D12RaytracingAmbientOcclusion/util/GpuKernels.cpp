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

#include "../stdafx.h"
#include "PerformanceTimers.h"
#include "GpuKernels.h"
#include "CompiledShaders\ReduceSumUintCS.hlsl.h"
#include "CompiledShaders\ReduceSumFloatCS.hlsl.h"
#include "CompiledShaders\DownsampleBoxFilter2x2CS.hlsl.h"
#include "CompiledShaders\DownsampleGaussian9TapFilterCS.hlsl.h"
#include "CompiledShaders\DownsampleGaussian25TapFilterCS.hlsl.h"
#include "CompiledShaders\PerPixelMeanSquareErrorCS.hlsl.h"
#include "CompiledShaders\AtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian3x3CS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS.hlsl.h"

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
        using namespace RootSignature::DownsampleBoxFilter2x2;
        using namespace DownsampleBoxFilter2x2;

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

    namespace RootSignature {
        namespace AtrousWaveletTransformCrossBilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    Normals,
                    Depths,
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

            CD3DX12_DESCRIPTOR_RANGE ranges[4];
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input values
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // input normals
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // input depths
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output filtered values

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::Normals].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::Depths].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: ReduceSum");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case Gaussian5x5:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS), ARRAYSIZE(g_pAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS));
                    break;
                case EdgeStoppingGaussian5x5:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS), ARRAYSIZE(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS));
                    break;
                case EdgeStoppingGaussian3x3:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian3x3CS), ARRAYSIZE(g_pEdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian3x3CS));
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
            m_intermediateOutput.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, DXGI_FORMAT_R32_FLOAT, width, height, descriptorHeap,
                &m_intermediateOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter intermediate output");
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
        FilterType type,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
        RWGpuResource* outputResource,
        UINT numFilterPasses)
    {
        using namespace RootSignature::AtrousWaveletTransformCrossBilateralFilter;
        
        // ToDo move out or rename
        PIXBeginEvent(commandList, 0, L"AtrousWaveletTransformCrossBilateralFilter");

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObjects[type].Get());
            commandList->SetComputeRootDescriptorTable(Slot::Normals, inputNormalsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Depths, inputDepthsResourceHandle);
        }

        //
        // Iterative filter
        //
        {
            auto resourceDesc = outputResource->resource.Get()->GetDesc();
            XMUINT2 resourceDim(static_cast<UINT>(resourceDesc.Width), static_cast<UINT>(resourceDesc.Height));
            XMUINT2 groupSize(
                CeilDivide(resourceDim.x, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width),
                CeilDivide(resourceDim.y, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height));

            // Order the resources such that the final pass writes to outputResource.
            RWGpuResource* outResources[2] =
            {
                numFilterPasses % 2 == 1 ? outputResource : &m_intermediateOutput,
                numFilterPasses % 2 == 1 ? &m_intermediateOutput : outputResource,
            };

            // First iteration reads from input resource.		
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outResources[0]->gpuDescriptorWriteAccess);

            for (UINT i = 0; i < numFilterPasses; i++)
            {
                commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(i));
                                
                // Dispatch.
                commandList->Dispatch(groupSize.x, groupSize.y, 1);

                // Transition and bind resources for the next pass.
                // Flip input/output. 
                if (i < numFilterPasses - 1)
                {
                    // Ping-pong resources across passes.
                    UINT inputID = i % 2;
                    UINT outputID = (i + 1) % 2;

                    D3D12_RESOURCE_BARRIER barriers[2] = {
                        CD3DX12_RESOURCE_BARRIER::Transition(outResources[inputID]->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                        CD3DX12_RESOURCE_BARRIER::Transition(outResources[outputID]->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    };
                    // First iteration reads from input resource => only flip current pass output/next pass input.
                    commandList->ResourceBarrier(i == 0 ? 1 : ARRAYSIZE(barriers), barriers);

                    commandList->SetComputeRootDescriptorTable(Slot::Input, outResources[inputID]->gpuDescriptorReadAccess);
                    commandList->SetComputeRootDescriptorTable(Slot::Output, outResources[outputID]->gpuDescriptorWriteAccess);
                }
            }
            
            // Transition the intermediate output resource back.
            if (numFilterPasses > 1)
            {
                commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_intermediateOutput.resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            }
        }

        PIXEndEvent(commandList);
    }

}