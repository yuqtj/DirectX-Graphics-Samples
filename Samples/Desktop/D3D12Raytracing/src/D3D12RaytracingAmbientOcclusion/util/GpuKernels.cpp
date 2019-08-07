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

// ToDo shaders are in root while this is in Util...
// ToDo standardize naming
#include "../stdafx.h"
#include "EngineProfiling.h"
#include "GpuKernels.h"
#include "CompiledShaders\ReduceSumUintCS.hlsl.h"
#include "CompiledShaders\ReduceSumFloatCS.hlsl.h"
#include "CompiledShaders\DownsampleBoxFilter2x2CS.hlsl.h"
#include "CompiledShaders\DownsampleGaussian9TapFilterCS.hlsl.h"
#include "CompiledShaders\DownsampleGaussian25TapFilterCS.hlsl.h"
#include "CompiledShaders\DownsampleNormalDepthHitPositionGeometryHitBilateralFilter2x2CS.hlsl.h"
#include "CompiledShaders\DownsampleValueDepthNormal_DepthWeightedBilateralFilter2x2CS.hlsl.h"
#include "CompiledShaders\DownsampleValueDepthNormal_PointSamplingBilateralFilter2x2CS.hlsl.h"
#include "CompiledShaders\DownsampleValueDepthNormal_DepthNormalWeightedBilateralFilter2x2CS.hlsl.h"
#include "CompiledShaders\UpsampleBilateralFilter2x2FloatCS.hlsl.h"
#include "CompiledShaders\UpsampleBilateralFilter2x2Float2CS.hlsl.h"
#include "CompiledShaders\MultiScale_UpsampleBilateralAndCombine2x2CS.hlsl.h"
#include "CompiledShaders\GaussianFilter3x3CS.hlsl.h"
#include "CompiledShaders\GaussianFilterRG3x3CS.hlsl.h"
#include "CompiledShaders\PerPixelMeanSquareErrorCS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Box3x3CS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian3x3CS.hlsl.h"
#include "CompiledShaders\EdgeStoppingAtrousWaveletTransfromCrossBilateralFilter_Gaussian5x5CS.hlsl.h"
#include "CompiledShaders\CalculateVariance_BilateralFilterCS.hlsl.h"
#include "CompiledShaders\CalculateVariance_SeparableFilterCS.hlsl.h"
//#include "CompiledShaders\CalculateVariance_SeparableBilateralFilterCS.hlsl.h"
#include "CompiledShaders\CalculateMeanVariance_SeparableFilterCS.hlsl.h"
#include "CompiledShaders\CalculateMeanVariance_SeparableFilterCS_AnyToAnyWaveReadLaneAt.hlsl.h"
#include "CompiledShaders\CalculatePartialDerivativesViaCentralDifferencesCS.hlsl.h"
#include "CompiledShaders\TemporalSupersampling_BlendWithCurrentFrameCS.hlsl.h"
#include "CompiledShaders\TemporalSupersampling_ReverseReprojectCS.hlsl.h"
#include "CompiledShaders\WriteValueToTextureCS.hlsl.h"
#include "CompiledShaders\GenerateGrassStrawsCS.hlsl.h"
#include "CompiledShaders\CountingSort_SortRays_128x64rayGroupCS.hlsl.h"
#include "CompiledShaders\AdaptiveRayGenCS.hlsl.h"
#include "CompiledShaders\FillInMissingValuesFilter_SeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt.hlsl.h"
#include "CompiledShaders\FillInMissingValuesFilter_DepthAwareSeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt.hlsl.h"
#include "CompiledShaders\DepthAwareSeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt.hlsl.h"

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

	void ReduceSum::Initialize(ID3D12Device5* device, Type type)
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
		ID3D12Device5* device,
		DX::DescriptorHeap* descriptorHeap,
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
				width = max(1u, CeilDivide(width, ReduceSumCS::ThreadGroup::Width));
				height = max(1u, CeilDivide(height, ReduceSumCS::ThreadGroup::Height));

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
		ID3D12GraphicsCommandList4* commandList,
		ID3D12DescriptorHeap* descriptorHeap, 
		UINT frameIndex,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
		void* resultSum,
        UINT invocationIndex)   // per frame invocation index)
	{
		using namespace RootSignature::ReduceSum;
		
		// ToDo move out or rename
        ScopedTimer _prof(L"CalculateNumCameraRayHits", commandList);

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

            // ToDo move copy to CPU out to separate kernel

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
		// This is fine for informational purposes such as using the value for UI display.
		UINT* mappedData = nullptr;
		CD3DX12_RANGE readRange(readBackBaseOffset, readBackBaseOffset + m_resultSize);
		ThrowIfFailed(m_readbackResources[invocationIndex]->Map(0, &readRange, reinterpret_cast<void**>(&mappedData)));
		memcpy(resultSum, mappedData, m_resultSize);
		m_readbackResources[invocationIndex]->Unmap(0, &CD3DX12_RANGE(0, 0));
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

	void DownsampleBoxFilter2x2::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
	{
		// Create root signature.
		{
			using namespace RootSignature::DownsampleBoxFilter2x2;

			CD3DX12_DESCRIPTOR_RANGE ranges[2]; 
			ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
			ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

			CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
			rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
			rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[1]);
			rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            // ToDo use D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT
            // ToDo remove unused samplers
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

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: DownsampleBoxFilter2x2");
        }
	}

	// Downsamples input resource.
	// width, height - dimensions of the input resource.
	void DownsampleBoxFilter2x2::Execute(
		ID3D12GraphicsCommandList4* commandList,
		UINT width,
		UINT height,
		ID3D12DescriptorHeap* descriptorHeap,
		const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
		const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
	{
		using namespace RootSignature::DownsampleBoxFilter2x2;
		using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"DownsampleBoxFilter2x2", commandList);

        m_CB->inputTextureDimensions = XMUINT2(width, height);
        m_CB->invertedInputTextureDimensions = XMFLOAT2(1.f / width, 1.f / height);
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

		// Set pipeline state.
		{
			commandList->SetDescriptorHeaps(1, &descriptorHeap);
			commandList->SetComputeRootSignature(m_rootSignature.Get());
			commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
			commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
			commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
			commandList->SetPipelineState(m_pipelineStateObject.Get());
		}

		// ToDo handle misaligned input
		XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));

		// Dispatch.
		commandList->Dispatch(groupSize.x, groupSize.y, 1);
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

	void DownsampleGaussianFilter::Initialize(ID3D12Device5* device, Type type, UINT frameCount, UINT numCallsPerFrame)
	{
		// Create root signature.
		{
			using namespace RootSignature::DownsampleGaussianFilter;

			CD3DX12_DESCRIPTOR_RANGE ranges[2]; 
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

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: DownsampleGaussianFilter");
        }
	}

	// Downsamples input resource.
	// width, height - dimensions of the input resource.
	void DownsampleGaussianFilter::Execute(
		ID3D12GraphicsCommandList4* commandList,
		UINT width,
		UINT height,
		ID3D12DescriptorHeap* descriptorHeap,
		const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
		const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
	{
		using namespace RootSignature::DownsampleGaussianFilter;
		using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"DownsampleGaussianFilter", commandList);

        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();

        m_CB->inputTextureDimensions = XMUINT2(width, height);
        m_CB->invertedInputTextureDimensions = XMFLOAT2(1.f / width, 1.f / height);
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

		// Set pipeline state.
		{
			commandList->SetDescriptorHeaps(1, &descriptorHeap);
			commandList->SetComputeRootSignature(m_rootSignature.Get());
			commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
			commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
			commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
			commandList->SetPipelineState(m_pipelineStateObject.Get());
		}

		// ToDo handle misaligned input
		XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));

		// Dispatch.
		commandList->Dispatch(groupSize.x, groupSize.y, 1);
	}

    // ToDo dedupe
    namespace RootSignature {
        namespace DownsampleNormalDepthHitPositionGeometryHitBilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    OutputNormal,
                    OutputPosition,
                    OutputGeometryHit,
                    OutputPartialDistanceDerivative,
                    OutputMotionVector,
                    OutputPrevFrameHitPosition,
                    OutputDepth,
#if EXACT_DDXY_ON_QUARTER_RES_USING_DOWNSAMPLED_PIXEL_OFFSETS
#endif
                    Input,
                    InputNormal,
                    InputPosition,
                    InputGeometryHit,
                    InputPartialDistanceDerivative,
                    InputMotionVector,
                    InputPrevFrameHitPosition,
                    InputDepth,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    // ToDo move the Type parameter to Execute?
    void DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::Initialize(ID3D12Device5* device, Type type, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter;

            // ToDo review access frequency or remove performance tip
            CD3DX12_DESCRIPTOR_RANGE ranges[17]; 
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // 1 input normal texture
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // 1 input position texture
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);  // 1 input geometry hit texture
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4);  // 1 input partial distance derivative
            ranges[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
            ranges[6].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // 1 output normal texture
            ranges[7].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);  // 1 output position texture
            ranges[8].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3);  // 1 output geometry hit texture
            ranges[9].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 4);  // 1 output partial distance derivative
            ranges[10].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // 1 input depth
            ranges[11].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 5);  // 1 output depth
            ranges[12].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6);  // 1 input motion vector
            ranges[13].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 6);  // 1 output motion vector
            ranges[14].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7);  // 1 input previous frame hit position
            ranges[15].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 7);  // 1 output previous frame hit position
        
            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::InputNormal].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::InputPosition].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::InputGeometryHit].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::InputPartialDistanceDerivative].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[5]);
            rootParameters[Slot::OutputNormal].InitAsDescriptorTable(1, &ranges[6]);
            rootParameters[Slot::OutputPosition].InitAsDescriptorTable(1, &ranges[7]);
            rootParameters[Slot::OutputGeometryHit].InitAsDescriptorTable(1, &ranges[8]);
            rootParameters[Slot::OutputPartialDistanceDerivative].InitAsDescriptorTable(1, &ranges[9]);
            rootParameters[Slot::InputDepth].InitAsDescriptorTable(1, &ranges[10]);
            rootParameters[Slot::OutputDepth].InitAsDescriptorTable(1, &ranges[11]);
            rootParameters[Slot::InputMotionVector].InitAsDescriptorTable(1, &ranges[12]);
            rootParameters[Slot::OutputMotionVector].InitAsDescriptorTable(1, &ranges[13]);
            rootParameters[Slot::InputPrevFrameHitPosition].InitAsDescriptorTable(1, &ranges[14]);
            rootParameters[Slot::OutputPrevFrameHitPosition].InitAsDescriptorTable(1, &ranges[15]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_STATIC_SAMPLER_DESC staticSamplers[] = {
                CD3DX12_STATIC_SAMPLER_DESC(0, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP) };

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, ARRAYSIZE(staticSamplers), staticSamplers);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: DownsampleNormalDepthHitPositionGeometryHitBilateralFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            switch (type)
            {
            case FilterDepthAware2x2:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleNormalDepthHitPositionGeometryHitBilateralFilter2x2CS), ARRAYSIZE(g_pDownsampleNormalDepthHitPositionGeometryHitBilateralFilter2x2CS));
                break;
            }

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: DownsampleNormalDepthHitPositionGeometryHitBilateralFilter");
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: DownsampleNormalDepthHitPositionGeometryHitBilateralFilter");
        }
    }

    // Downsamples input resource.
    // width, height - dimensions of the input resource.
    void DownsampleNormalDepthHitPositionGeometryHitBilateralFilter::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputPositionResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputGeometryHitResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputPartialDistanceDerivativesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputMotionVectorResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputPrevFrameHitPositionResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputPositionResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputGeometryHitResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputPartialDistanceDerivativesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputMotionVectorResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputPrevFrameHitPositionResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputDepthResourceHandle)
    {
        using namespace RootSignature::DownsampleNormalDepthHitPositionGeometryHitBilateralFilter;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"DownsampleNormalDepthHitPositionGeometryHitBilateralFilter", commandList);

        m_CB->textureDim = XMUINT2(width, height);
        m_CB->invTextureDim = XMFLOAT2(1.f / width, 1.f / height);
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputNormal, inputNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputPosition, inputPositionResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputGeometryHit, inputGeometryHitResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputPartialDistanceDerivative, inputPartialDistanceDerivativesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputMotionVector, inputMotionVectorResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputPrevFrameHitPosition, inputPrevFrameHitPositionResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputDepth, inputDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputNormal, outputNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputPosition, outputPositionResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputGeometryHit, outputGeometryHitResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputPartialDistanceDerivative, outputPartialDistanceDerivativesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputMotionVector, outputMotionVectorResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputPrevFrameHitPosition, outputPrevFrameHitPositionResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputDepth, outputDepthResourceHandle);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo handle misaligned input
        XMUINT2 groupSize(CeilDivide((width + 1) / 2 + 1, ThreadGroup::Width), CeilDivide((height + 1) / 2 + 1, ThreadGroup::Height));

        // Dispatch.
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }



    namespace RootSignature {
        namespace DownsampleValueNormalDepthBilateralFilter {
            namespace Slot {
                enum Enum {
                    OutputValue = 0,
                    OutputNormalDepth,
                    OutputPartialDistanceDerivative,
                    InputValue,
                    InputNormalDepth,
                    InputPartialDistanceDerivative,
                    Count
                };
            }
        }
    }

    void DownsampleValueNormalDepthBilateralFilter::Initialize(ID3D12Device5* device, Type type)
    {
        // Create root signature.
        {
            using namespace RootSignature::DownsampleValueNormalDepthBilateralFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[6]; 
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input value texture
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // 1 input normal and depth texture
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // 1 input partial distance derivative texture
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output value texture
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // 1 output normal and depth texture
            ranges[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);  // 1 output partial distance derivative texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::InputValue].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::InputNormalDepth].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::InputPartialDistanceDerivative].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::OutputValue].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::OutputNormalDepth].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::OutputPartialDistanceDerivative].InitAsDescriptorTable(1, &ranges[5]);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: DownsampleValueNormalDepthBilateralFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            switch (type)
            {
            case FilterPointSampling2x2:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleValueDepthNormal_PointSamplingBilateralFilter2x2CS), ARRAYSIZE(g_pDownsampleValueDepthNormal_PointSamplingBilateralFilter2x2CS));
                break;
            case FilterDepthWeighted2x2:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleValueDepthNormal_DepthWeightedBilateralFilter2x2CS), ARRAYSIZE(g_pDownsampleValueDepthNormal_DepthWeightedBilateralFilter2x2CS));
                break;
            case FilterDepthNormalWeighted2x2:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDownsampleValueDepthNormal_DepthNormalWeightedBilateralFilter2x2CS), ARRAYSIZE(g_pDownsampleValueDepthNormal_DepthNormalWeightedBilateralFilter2x2CS));
                break;
            }

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: DownsampleValueNormalDepthBilateralFilter");
        }
    }

    // Downsamples input resource.
    // width, height - dimensions of the input resource.
    void DownsampleValueNormalDepthBilateralFilter::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputPartialDistanceDerivativesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputPartialDistanceDerivativesResourceHandle)
    {
        using namespace RootSignature::DownsampleValueNormalDepthBilateralFilter;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"DownsampleValueNormalDepthBilateralFilter", commandList);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputValue, inputValueResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputNormalDepth, inputNormalDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputPartialDistanceDerivative, inputPartialDistanceDerivativesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputValue, outputValueResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputNormalDepth, outputNormalDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputPartialDistanceDerivative, outputPartialDistanceDerivativesResourceHandle);
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo handle misaligned input
        XMUINT2 groupSize(CeilDivide((width + 1) / 2 + 1, ThreadGroup::Width), CeilDivide((height + 1) / 2 + 1, ThreadGroup::Height));

        // Dispatch.
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }

    namespace RootSignature {
        namespace UpsampleBilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    InputLowResNormal,
                    InputHiResNormal,
                    InputHiResPartialDistanceDerivative,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    // ToDo test downsample,upsample on odd resolution
    void UpsampleBilateralFilter::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::UpsampleBilateralFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[5]; 
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // 1 input normal low res texture
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // 1 input normal high res texture
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);  // 1 input partial distance derivative texture
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::InputLowResNormal].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::InputHiResNormal].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::InputHiResPartialDistanceDerivative].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_STATIC_SAMPLER_DESC staticSamplers[] = {
                CD3DX12_STATIC_SAMPLER_DESC(0, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP) };

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, ARRAYSIZE(staticSamplers), staticSamplers);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: UpsampleBilateralFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case Filter2x2R:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pUpsampleBilateralFilter2x2FloatCS), ARRAYSIZE(g_pUpsampleBilateralFilter2x2FloatCS));
                    break;
                case Filter2x2RG:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pUpsampleBilateralFilter2x2Float2CS), ARRAYSIZE(g_pUpsampleBilateralFilter2x2Float2CS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: UpsampleBilateralFilter");
            }
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: GaussianFilter");
        }
    }

    // Resamples input resource.
    // width, height - dimensions of the output resource.
    // ToDo should the input width/height be of output or input?
    void UpsampleBilateralFilter::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width, // Todo remove and deduce from outputResourceInstead?
        UINT height,
        FilterType type,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDistanceDerivativeResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
        bool useBilinearWeights,
        bool useDepthWeights,
        bool useNormalWeights,
        bool useDynamicDepthThreshold)
    {
        using namespace RootSignature::UpsampleBilateralFilter;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"UpsampleBilateralFilter", commandList);

        // Each shader execution processes 2x2 hiRes pixels
        XMUINT2 lowResDim = XMUINT2(CeilDivide(width, 2), CeilDivide(height, 2));

        m_CB->useBilinearWeights = useBilinearWeights;
        m_CB->useDepthWeights = useDepthWeights;
        m_CB->useNormalWeights = useNormalWeights;
        m_CB->useDynamicDepthThreshold = useDynamicDepthThreshold;
        m_CB->invHiResTextureDim = XMFLOAT2(1.f / width, 1.f / height);
        m_CB->invLowResTextureDim = XMFLOAT2(1.f / lowResDim.x, 1.f / lowResDim.y);
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);



        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputLowResNormal, inputLowResNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputHiResNormal, inputHiResNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputHiResPartialDistanceDerivative, inputHiResPartialDistanceDerivativeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObjects[type].Get());
        }

        // ToDo handle misaligned input
        // Start from -1,-1 pixel to account for high-res pixel border around low-res pixel border.
        XMUINT2 groupSize(CeilDivide(lowResDim.x + 1, ThreadGroup::Width), CeilDivide(lowResDim.y + 1, ThreadGroup::Height));

        // Dispatch.
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }



    namespace RootSignature {
        namespace MultiScale_UpsampleBilateralFilterAndCombine {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    InputLowResValue1,
                    InputLowResValue2,
                    InputLowResNormal,
                    InputHiResValue,
                    InputHiResNormal,
                    InputHiResPartialDistanceDerivative,
                    Count
                };
            }
        }
    }

    // ToDo test downsample,upsample on odd resolution
    void MultiScale_UpsampleBilateralFilterAndCombine::Initialize(ID3D12Device5* device, Type type)
    {
        // Create root signature.
        {
            using namespace RootSignature::MultiScale_UpsampleBilateralFilterAndCombine;

            CD3DX12_DESCRIPTOR_RANGE ranges[7]; 
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input low res value 1
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // 1 input low res value 2
            ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // 1 input low res normal
            ranges[3].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);  // 1 input hi res value
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4);  // 1 input hi res normal
            ranges[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // 1 input hi res partial distance derivatives
            ranges[6].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::InputLowResValue1].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::InputLowResValue2].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::InputLowResNormal].InitAsDescriptorTable(1, &ranges[2]);
            rootParameters[Slot::InputHiResValue].InitAsDescriptorTable(1, &ranges[3]);
            rootParameters[Slot::InputHiResNormal].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::InputHiResPartialDistanceDerivative].InitAsDescriptorTable(1, &ranges[5]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[6]);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: MultiScale_UpsampleBilateralFilterAndCombine");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            switch (type)
            {
            case Filter2x2:
                descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pMultiScale_UpsampleBilateralAndCombine2x2CS), ARRAYSIZE(g_pMultiScale_UpsampleBilateralAndCombine2x2CS));
                break;
            }

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: MultiScale_UpsampleBilateralFilterAndCombine");
        }
    }

    // Resamples input resource.
    // width, height - dimensions of the input resource.
    void MultiScale_UpsampleBilateralFilterAndCombine::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValue1ResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValue2ResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDistanceDerivativeResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
    {
        using namespace RootSignature::MultiScale_UpsampleBilateralFilterAndCombine;
        using namespace DefaultComputeShaderParams;

        // Each shader execution processes 2x2 hiRes pixels
        width = CeilDivide(width, 2);
        height = CeilDivide(height, 2);

        ScopedTimer _prof(L"MultiScale_UpsampleBilateralFilterAndCombine", commandList);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputLowResValue1, inputLowResValue1ResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputLowResValue2, inputLowResValue2ResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputLowResNormal, inputLowResNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputHiResValue, inputHiResValueResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputHiResNormal, inputHiResNormalResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputHiResPartialDistanceDerivative, inputHiResPartialDistanceDerivativeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo handle misaligned input
        // Start from -1,-1 pixel to account for high-res pixel border around low-res pixel border.
        XMUINT2 groupSize(CeilDivide(width + 1, ThreadGroup::Width), CeilDivide(height + 1, ThreadGroup::Height));

        // Dispatch.
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
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

    void GaussianFilter::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::GaussianFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[2]; 
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
                case Filter3x3:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pGaussianFilter3x3CS), ARRAYSIZE(g_pGaussianFilter3x3CS));
                    break;
                case Filter3x3RG:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pGaussianFilterRG3x3CS), ARRAYSIZE(g_pGaussianFilterRG3x3CS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: GaussianFilter");
            }
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: GaussianFilter");
        }
    }

    // ToDo fix up input order to be same among kernels

    // Blurs input resource with a Gaussian filter.
    // width, height - dimensions of the input resource.
    void GaussianFilter::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        FilterType type,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
    {
        using namespace RootSignature::GaussianFilter;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"GaussianFilter", commandList);

        m_CB->textureDim = XMUINT2(width, height);
        m_CB->invTextureDim = XMFLOAT2(1.f / width, 1.f / height);
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObjects[type].Get());
        }

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }


    namespace RootSignature {
        namespace FillInMissingValuesFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    Depth,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    void FillInMissingValuesFilter::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::FillInMissingValuesFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count];
            ranges[Slot::Input].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
            ranges[Slot::Depth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
            ranges[Slot::Output].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[Slot::Input]);
            rootParameters[Slot::Depth].InitAsDescriptorTable(1, &ranges[Slot::Depth]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[Slot::Output]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: FillInMissingValuesFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case GaussianFilter7x7:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pFillInMissingValuesFilter_SeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt), ARRAYSIZE(g_pFillInMissingValuesFilter_SeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt));
                    break;

                case DepthAware_GaussianFilter7x7:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pFillInMissingValuesFilter_DepthAwareSeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt), ARRAYSIZE(g_pFillInMissingValuesFilter_DepthAwareSeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt));
                    break;
                }
                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: FillInMissingValuesFilter");
            }
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: FillInMissingValuesFilter");
        }
    }

    // ToDo fix up input order to be same among kernels

    // Blurs input resource with a Gaussian filter.
    // width, height - dimensions of the input resource.
    void FillInMissingValuesFilter::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        FilterType type,
        UINT filterStep,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
    {
        using namespace RootSignature::FillInMissingValuesFilter;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"FillInMissingValuesFilter", commandList);

        m_CB->textureDim = XMUINT2(width, height);
        m_CB->step = filterStep;
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Depth, inputDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObjects[type].Get());
        }

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }


    namespace RootSignature {
        namespace BilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Input,
                    Depth,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    void BilateralFilter::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::BilateralFilter;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count];
            ranges[Slot::Input].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
            ranges[Slot::Depth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
            ranges[Slot::Output].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[Slot::Input]);
            rootParameters[Slot::Depth].InitAsDescriptorTable(1, &ranges[Slot::Depth]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[Slot::Output]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: BilateralFilter");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case DepthAware_GaussianFilter5x5:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pDepthAwareSeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt), ARRAYSIZE(g_pDepthAwareSeparableGaussianFilterCS_AnyToAnyWaveReadLaneAt));
                    break;
                }
                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: BilateralFilter");
            }
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: BilateralFilter");
        }
    }

    // ToDo fix up input order to be same among kernels

    // Blurs input resource with a Gaussian filter.
    // width, height - dimensions of the input resource.
    void BilateralFilter::Execute(
        ID3D12GraphicsCommandList4* commandList,
        FilterType type,
        UINT filterStep,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthResourceHandle,
        RWGpuResource* outputResource)
    {
        using namespace RootSignature::BilateralFilter;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"BilateralFilter", commandList);

        auto resourceDesc = outputResource->resource.Get()->GetDesc();
        XMUINT2 resourceDim(static_cast<UINT>(resourceDesc.Width), static_cast<UINT>(resourceDesc.Height));

        m_CB->textureDim = resourceDim;
        m_CB->step = filterStep;
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Depth, inputDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResource->gpuDescriptorWriteAccess);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObjects[type].Get());
        }

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(resourceDim.x, ThreadGroup::Width), CeilDivide(resourceDim.y, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
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

    void RootMeanSquareError::Initialize(ID3D12Device5* device)
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
        ID3D12Device5* device,
        DX::DescriptorHeap* descriptorHeap,
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
        ID3D12GraphicsCommandList4* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        UINT frameIndex,
        UINT invocationIndex,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
        float* rootMeanSquareError)
    {
        using namespace RootSignature::RootMeanSquareError;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"RootMeanSquareError", commandList);

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
            m_reduceSumKernel.Execute(commandList, descriptorHeap, frameIndex, m_perPixelMeanSquareError.gpuDescriptorReadAccess, &sumPerPixelMeanSquareError, invocationIndex);
        }

        // Calculate root mean square error.
        *rootMeanSquareError = sqrtf(sumPerPixelMeanSquareError / (desc.Width * desc.Height));
    }

    // ToDo prune
    namespace RootSignature {
        namespace AtrousWaveletTransformCrossBilateralFilter {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    VarianceOutput,
#if !WORKAROUND_ATROUS_VARYING_OUTPUTS
                    FilterWeightSumOutput,
#endif
                    // ToDo standardize naming in RootSigs
                    Input,
                    Normals,
                    Variance,
                    SmoothedVariance,
                    RayHitDistance,
                    PartialDistanceDerivatives,
                    FrameAge,
                    ConstantBuffer,
                    // ToDo remove
                    Debug1,
                    Debug2,
                    Count
                };
            }
        }
    }

    // ToDo move type to execute
    void AtrousWaveletTransformCrossBilateralFilter::Initialize(ID3D12Device5* device, UINT frameCount, UINT maxFilterPasses, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::AtrousWaveletTransformCrossBilateralFilter;

            // ToDo reorganize slots and descriptors
            CD3DX12_DESCRIPTOR_RANGE ranges[14];
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input values
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // input normals
            ranges[4].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4);  // input variance
            ranges[5].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);  // input smoothed variance
            ranges[6].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output filtered values
            ranges[7].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // output filtered variance
            ranges[8].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);  // output filter weight sum
            ranges[9].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6);  // input hit distance
            ranges[10].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7);  // input hit distance
            ranges[11].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);
            ranges[12].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3);
            ranges[13].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 4);


            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::Normals].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::Variance].InitAsDescriptorTable(1, &ranges[4]);
            rootParameters[Slot::SmoothedVariance].InitAsDescriptorTable(1, &ranges[5]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[6]);
            rootParameters[Slot::VarianceOutput].InitAsDescriptorTable(1, &ranges[7]);
#if !WORKAROUND_ATROUS_VARYING_OUTPUTS
            rootParameters[Slot::FilterWeightSumOutput].InitAsDescriptorTable(1, &ranges[8]);
#endif
            rootParameters[Slot::RayHitDistance].InitAsDescriptorTable(1, &ranges[9]);
            rootParameters[Slot::PartialDistanceDerivatives].InitAsDescriptorTable(1, &ranges[10]);
            rootParameters[Slot::FrameAge].InitAsDescriptorTable(1, &ranges[11]);
            rootParameters[Slot::Debug1].InitAsDescriptorTable(1, &ranges[12]);
            rootParameters[Slot::Debug2].InitAsDescriptorTable(1, &ranges[13]);
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
            m_maxFilterPasses = maxFilterPasses;
            UINT numInstancesPerFrame = maxFilterPasses * numCallsPerFrame;
            m_CB.Create(device, frameCount * numInstancesPerFrame, L"Constant Buffer: AtrousWaveletTransformCrossBilateralFilter");
            m_CBfilterWeight.Create(device, frameCount * numInstancesPerFrame, L"Constant Buffer: AtrousWaveletTransformCrossBilateralFilter FilterWeightSum");
        }
    }

    void AtrousWaveletTransformCrossBilateralFilter::CreateInputResourceSizeDependentResources(
        ID3D12Device5* device,
        DX::DescriptorHeap* descriptorHeap,
        UINT width,
        UINT height,
        DXGI_FORMAT format)
    {
        // Create shader resources
        {
            m_intermediateValueOutput.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            // ToDo pass in the format
            CreateRenderTargetResource(device, format, width, height, descriptorHeap,
                &m_intermediateValueOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter intermediate value output");

            m_intermediateVarianceOutputs[0].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, format, width, height, descriptorHeap,
                &m_intermediateVarianceOutputs[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter intermediate variance output 0");

            m_intermediateVarianceOutputs[1].rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, format, width, height, descriptorHeap,
                &m_intermediateVarianceOutputs[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter intermediate variance output 1");

            // ToDo remove
            m_filterWeightOutput.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, format, width, height, descriptorHeap,
                &m_filterWeightOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, L"UAV texture: AtrousWaveletTransformCrossBilateralFilter filter weight sum output");
        }
    }

    // ToDo add option to allow input, output being the same
    // Expects, and returns outputResource in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state.
    // Expects, and returns outputIntermediateResource in D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE state.
    void AtrousWaveletTransformCrossBilateralFilter::Execute(
        ID3D12GraphicsCommandList4* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        FilterType filterType,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
        // ToDo document this assumes variance in the X component of the passed in resource.
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputVarianceResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputHitDistanceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputPartialDistanceDerivativesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputFrameAgeResourceHandle,
        RWGpuResource* outputResource,
        RWGpuResource* outputIntermediateResource,
        RWGpuResource* outputDebug1Resource,
        RWGpuResource* outputDebug2Resource,
        float valueSigma,
        float depthSigma,
        float normalSigma,
        float weightScale,
#if !NORMAL_DEPTH_R8G8B16_ENCODING
        TextureResourceFormatRGB::Type normalDepthResourceFormat,
#endif
        UINT kernelStepShifts[5],
        UINT passNumberToOutputToIntermediateResource,
        UINT numFilterPasses,
        Mode filterMode,
        bool reverseFilterPassOrder,
        bool useCalculatedVariance,
        bool perspectiveCorrectDepthInterpolation,
        bool useAdaptiveKernelSize,
        float minHitDistanceToKernelWidthScale,
        UINT minKernelWidth,
        UINT maxKernelWidth,
        float varianceSigmaScaleOnSmallKernels,
        bool usingBilateralDownsampledBuffers,
        float minVarianceToDenoise,
        float staleNeighborWeightScale,
        float depthWeightCutoff,
        bool useProjectedDepthTest,
        bool forceDenoisePass)
    {

        // ToDo: cleanup use of variance
        // SmoothedVariance is used for edge stopping
        // Variance is used for intermediate variance calculations for the 

        using namespace RootSignature::AtrousWaveletTransformCrossBilateralFilter;
        using namespace AtrousWaveletTransformFilterCS;

        ScopedTimer _prof(L"AtrousWaveletTransformCrossBilateralFilter", commandList);

        m_CBinstanceID = ((m_CBinstanceID + 1) * m_maxFilterPasses) % m_CB.NumInstances();

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObjects[filterType].Get());
            commandList->SetComputeRootDescriptorTable(Slot::Normals, inputNormalsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Variance, inputVarianceResourceHandle);
            // ToDo Smoothen input variance or remove the dupe
            commandList->SetComputeRootDescriptorTable(Slot::SmoothedVariance, inputVarianceResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::RayHitDistance, inputHitDistanceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::PartialDistanceDerivatives, inputPartialDistanceDerivativesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::FrameAge, inputFrameAgeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Debug1, outputDebug1Resource->gpuDescriptorWriteAccess);
            commandList->SetComputeRootDescriptorTable(Slot::Debug2, outputDebug2Resource->gpuDescriptorWriteAccess);
        }

        // ToDo use input resource dims.
        auto resourceDesc = outputResource->resource.Get()->GetDesc();
        XMUINT2 resourceDim(static_cast<UINT>(resourceDesc.Width), static_cast<UINT>(resourceDesc.Height));

        // ToDo split these into separate GpuKernels?
        auto& CB = filterMode == OutputFilteredValue ? m_CB : m_CBfilterWeight;
        // Update the Constant Buffers.
        for (UINT i = 0; i < numFilterPasses; i++)
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
                CB->valueSigma = valueSigma;
            }
            else
            {
                CB->valueSigma = _i > 0 ? valueSigma * powf(2.f, -float(i)) : 1;
            }
            CB->depthSigma = depthSigma;
            CB->normalSigma = normalSigma;
            CB->weightScale = weightScale;
#if RAYTRACING_MANUAL_KERNEL_STEP_SHIFTS
            CB->kernelStepShift = kernelStepShifts[_i];
#else
            CB->kernelStepShift = _i;
#endif
            // Move vars not changing inside loop outside of it.
            CB->useCalculatedVariance = filterMode == OutputFilteredValue && useCalculatedVariance;
            CB->outputFilteredVariance = filterMode == OutputFilteredValue && useCalculatedVariance;
            CB->outputFilteredValue = filterMode == OutputFilteredValue;
            CB->outputFilterWeightSum = filterMode == OutputPerPixelFilterWeightSum;
            CB->perspectiveCorrectDepthInterpolation = perspectiveCorrectDepthInterpolation;
            CB->useAdaptiveKernelSize = useAdaptiveKernelSize;
            CB->minHitDistanceToKernelWidthScale = minHitDistanceToKernelWidthScale;
            CB->minKernelWidth = minKernelWidth;
            CB->maxKernelWidth = maxKernelWidth;
            CB->varianceSigmaScaleOnSmallKernels = varianceSigmaScaleOnSmallKernels;
            CB->usingBilateralDownsampledBuffers = usingBilateralDownsampledBuffers;
            CB->textureDim = resourceDim;
            CB->minVarianceToDenoise = minVarianceToDenoise;
            CB->staleNeighborWeightScale = _i == 0 ? staleNeighborWeightScale : 1;  // ToDo revise
            CB->depthWeightCutoff = depthWeightCutoff;
            CB->useProjectedDepthTest = useProjectedDepthTest;
            CB->forceDenoisePass = forceDenoisePass;


#if NORMAL_DEPTH_R8G8B16_ENCODING
            m_CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(16);
#else
            switch (normalDepthResourceFormat)
            {
            case TextureResourceFormatRGB::R32G32B32A32_FLOAT: CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(32); break;
            case TextureResourceFormatRGB::R16G16B16A16_FLOAT: CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(16); break;
            case TextureResourceFormatRGB::R11G11B10_FLOAT: CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(10); break;
            default: ThrowIfFalse(false, L"Invalid resource format specified.");
            }
#endif
            
            CB.CopyStagingToGpu(m_CBinstanceID + i);
        }

        //
        // Iterative filter
        //
        {
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
#if WORKAROUND_ATROUS_VARYING_OUTPUTS
            commandList->SetComputeRootDescriptorTable(Slot::Output, outValueResources[0]->gpuDescriptorWriteAccess);
#else
            ToDo
            UINT outputSlot = (filterMode == Mode::OutputFilteredValue) ? Slot::Output : Slot::FilterWeightSumOutput; // ToDo Cleanup
            commandList->SetComputeRootDescriptorTable(outputSlot, outValueResources[0]->gpuDescriptorWriteAccess);
#endif
            commandList->SetComputeRootDescriptorTable(Slot::VarianceOutput, m_intermediateVarianceOutputs[0].gpuDescriptorWriteAccess);

            for (UINT i = 0; i < numFilterPasses; i++)
#endif
            {
                commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, CB.GpuVirtualAddress(m_CBinstanceID + i));
                                
                // Dispatch.
                commandList->Dispatch(groupSize.x, groupSize.y, 1);

                // ToDo remove the copy, write directly to the resource instead.
                if (outputIntermediateResource && i == passNumberToOutputToIntermediateResource)
                {
                    commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(outValueResources[i % 2]->resource.Get()));

                    CopyTextureRegion(
                        commandList,
                        outValueResources[i % 2]->resource.Get(),
                        outputIntermediateResource->resource.Get(),
                        &CD3DX12_BOX(0, 0, resourceDim.x, resourceDim.y),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }

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
                    commandList->SetComputeRootDescriptorTable(Slot::Variance, m_intermediateVarianceOutputs[inputID].gpuDescriptorReadAccess);
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
    }



    // ToDo reverse CalculatePartialDerivatives::RootSig  and then use as RootSig::Slot:: below?
    namespace RootSignature {
        namespace CalculatePartialDerivatives {
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

    // ToDo move type to execute
    void CalculatePartialDerivatives::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::CalculatePartialDerivatives;

            CD3DX12_DESCRIPTOR_RANGE ranges[2];
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input values
            ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output filtered values

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[0]);
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[1]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: CalculatePartialDerivatives");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculatePartialDerivativesViaCentralDifferencesCS), ARRAYSIZE(g_pCalculatePartialDerivativesViaCentralDifferencesCS));
            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: CalculatePartialDerivatives");
        }

        // Create shader resources.
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: CalculatePartialDerivatives");
        }
    }

    // ToDo add option to allow input, output being the same
    // Expects, and returns, outputResource in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state.
    void CalculatePartialDerivatives::Execute(
        ID3D12GraphicsCommandList4* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        UINT width,
        UINT height,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle)
    {
        using namespace RootSignature::CalculatePartialDerivatives;
        using namespace DefaultComputeShaderParams;

        // ToDo move out or rename - there are scoped timers in the caller.
        ScopedTimer _prof(L"CalculatePartialDerivatives", commandList);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObject.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Output, outputResourceHandle);
        }

        // Update the Constant Buffer.
        {
            m_CB->textureDim = XMUINT2(width, height);
            m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
            m_CB.CopyStagingToGpu(m_CBinstanceID);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
        }

        // Dispatch.
        {
            XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
            commandList->Dispatch(groupSize.x, groupSize.y, 1);
        }
    }


    namespace RootSignature {
        namespace CalculateVariance {
            namespace Slot {
                enum Enum {
                    OutputVariance = 0,
                    OutputMean,
                    Input,
                    Depth,
                    Normal,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    // ToDo move type to execute
    void CalculateVariance::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::CalculateVariance;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count];
            ranges[Slot::Input].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input values
            ranges[Slot::Depth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // input depth
            ranges[Slot::Normal].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);  // input normal
            ranges[Slot::OutputVariance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output variance 
            ranges[Slot::OutputMean].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // output mean 

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[Slot::Input]);
            rootParameters[Slot::Normal].InitAsDescriptorTable(1, &ranges[Slot::Normal]);
            rootParameters[Slot::Depth].InitAsDescriptorTable(1, &ranges[Slot::Depth]);
            rootParameters[Slot::OutputVariance].InitAsDescriptorTable(1, &ranges[Slot::OutputVariance]);
            rootParameters[Slot::OutputMean].InitAsDescriptorTable(1, &ranges[Slot::OutputMean]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: CalculateVariance");
        }
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();


        }
        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case SquareBilateral:
                case SeparableBilateral:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateVariance_BilateralFilterCS), ARRAYSIZE(g_pCalculateVariance_BilateralFilterCS));
                    break;
                    // ToDo
                //case SeparableBilateral:
                 //   descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateVariance_SeparableBilateralFilterCS), ARRAYSIZE(g_pCalculateVariance_SeparableBilateralFilterCS));
                 //   break;
                case Separable:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateVariance_SeparableFilterCS), ARRAYSIZE(g_pCalculateVariance_SeparableFilterCS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: CalculateVariance");
            }
        }

        // Create shader resources.
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: CalculateVariance");
        }
    }

    // ToDo add option to allow input, output being the same
    // Expects, and returns, outputResource in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state.
    void CalculateVariance::Execute(
        ID3D12GraphicsCommandList4* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        UINT width,
        UINT height,
        FilterType filterType,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,  // ToDo standardize Normal vs Normals
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputVarianceResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputMeanResourceHandle,
        float depthSigma,
        float normalSigma,
        bool outputMean,
        bool useDepthWeights,
        bool useNormalWeights,
        UINT kernelWidth)
    {
        using namespace RootSignature::CalculateVariance;
        using namespace DefaultComputeShaderParams;

        // ToDo replace asserts with runtime fails?
        assert((kernelWidth & 1) == 1 && L"KernelWidth must be an odd number so that width == radius + 1 + radius");

        // ToDo move out or rename
        // ToDo add spaces to names?
        ScopedTimer _prof(L"CalculateVariance_Bilateral", commandList); // ToDo update name

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObjects[filterType].Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Normal, inputNormalsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::Depth, inputDepthsResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputVariance, outputVarianceResourceHandle);
            if (outputMean)
            {
                commandList->SetComputeRootDescriptorTable(Slot::OutputMean, outputMeanResourceHandle);
            }
        }

        // Update the Constant Buffer.
        m_CB->textureDim = XMUINT2(width, height);
        m_CB->depthSigma = depthSigma;
        m_CB->normalSigma = normalSigma;
        m_CB->outputMean = outputMean;
        m_CB->useDepthWeights = useDepthWeights;
        m_CB->useNormalWeights = useNormalWeights;
        m_CB->kernelWidth = kernelWidth;
        m_CB->kernelRadius = kernelWidth >> 1;
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);
        commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));


        // Dispatch.
        {
            XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
            commandList->Dispatch(groupSize.x, groupSize.y, 1);
        }
    }


    namespace RootSignature {
        namespace CalculateMeanVariance {
            namespace Slot {
                enum Enum {
                    OutputMeanVariance = 0,
                    Input,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    // ToDo move type to execute
    void CalculateMeanVariance::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::CalculateMeanVariance;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count];
            ranges[Slot::Input].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input values
            ranges[Slot::OutputMeanVariance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output mean and variance 

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[Slot::Input]);
            rootParameters[Slot::OutputMeanVariance].InitAsDescriptorTable(1, &ranges[Slot::OutputMeanVariance]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: CalculateMeanVariance");
        }
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();


        }
#if 0
        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {}; 
            descComputePSO.pRootSignature = m_rootSignature.Get(); 
            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateMeanVariance_SeparableFilterCS), ARRAYSIZE(g_pCalculateMeanVariance_SeparableFilterCS));
            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: CalculateMeanVariance");
        }
#else
        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case Separable_AnyToAnyWaveReadLaneAt:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateMeanVariance_SeparableFilterCS_AnyToAnyWaveReadLaneAt), ARRAYSIZE(g_pCalculateMeanVariance_SeparableFilterCS_AnyToAnyWaveReadLaneAt));
                    break;
                case Separable:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCalculateMeanVariance_SeparableFilterCS), ARRAYSIZE(g_pCalculateMeanVariance_SeparableFilterCS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: CalculateMeanVariance");
            }
        }
#endif
        // Create shader resources.
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: CalculateMeanVariance");
        }
    }

    // ToDo add option to allow input, output being the same
    // Expects, and returns, outputResource in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state.
    void CalculateMeanVariance::Execute(
        ID3D12GraphicsCommandList4* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        UINT width,
        UINT height,
        FilterType filterType,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputMeanVarianceResourceHandle,
        UINT kernelWidth)
    {
        using namespace RootSignature::CalculateMeanVariance;
        using namespace DefaultComputeShaderParams;

        // ToDo replace asserts with runtime fails?
        // ToDo pass kernel radius instead
        assert((kernelWidth & 1) == 1 && L"KernelWidth must be an odd number so that width == radius + 1 + radius");

        // ToDo move out or rename
        // ToDo add spaces to names?
        ScopedTimer _prof(L"CalculateMeanVariance", commandList); // ToDo update name

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObjects[filterType].Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputMeanVariance, outputMeanVarianceResourceHandle);
        }

        // Update the Constant Buffer.
        m_CB->textureDim = XMUINT2(width, height);
        m_CB->kernelWidth = kernelWidth;
        m_CB->kernelRadius = kernelWidth >> 1;
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);
        commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));


        // Dispatch.
        {
            XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
            commandList->Dispatch(groupSize.x, groupSize.y, 1);
        }
    }


    namespace RootSignature {
        namespace RTAO_TemporalSupersampling_ReverseReproject {
            namespace Slot {
                enum Enum {
                    OutputCacheFrameAge = 0,
                    OutputReprojectedCacheValues,
                    InputCurrentFrameNormalDepth,
                    InputCurrentFrameLinearDepthDerivative,
                    InputReprojectedNormalDepth,
                    InputTextureSpaceMotionVector,      // Texture space motion vector from the previous to the current frame.
                    InputCachedValue,
                    InputCachedNormalDepth,
                    InputCachedFrameAge,
                    InputCachedSquaredMeanValue,
                    InputCachedRayHitDistance,
                    OutputDebug1,
                    OutputDebug2,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    void RTAO_TemporalSupersampling_ReverseReproject::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::RTAO_TemporalSupersampling_ReverseReproject;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count];
            ranges[Slot::InputCurrentFrameNormalDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
            ranges[Slot::InputCurrentFrameLinearDepthDerivative].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
            ranges[Slot::InputReprojectedNormalDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);
            ranges[Slot::InputTextureSpaceMotionVector].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);
            ranges[Slot::InputCachedNormalDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4);
            ranges[Slot::InputCachedValue].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5);
            ranges[Slot::InputCachedFrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 6);
            ranges[Slot::InputCachedSquaredMeanValue].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 7);
            ranges[Slot::InputCachedRayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 8);

            ranges[Slot::OutputCacheFrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
            ranges[Slot::OutputReprojectedCacheValues].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
            
            ranges[Slot::OutputDebug1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 10);
            ranges[Slot::OutputDebug2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 11);

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::InputCurrentFrameNormalDepth].InitAsDescriptorTable(1, &ranges[Slot::InputCurrentFrameNormalDepth]);
            rootParameters[Slot::InputCurrentFrameLinearDepthDerivative].InitAsDescriptorTable(1, &ranges[Slot::InputCurrentFrameLinearDepthDerivative]);
            rootParameters[Slot::InputReprojectedNormalDepth].InitAsDescriptorTable(1, &ranges[Slot::InputReprojectedNormalDepth]);
            rootParameters[Slot::InputTextureSpaceMotionVector].InitAsDescriptorTable(1, &ranges[Slot::InputTextureSpaceMotionVector]);
            rootParameters[Slot::InputCachedValue].InitAsDescriptorTable(1, &ranges[Slot::InputCachedValue]);
            rootParameters[Slot::InputCachedNormalDepth].InitAsDescriptorTable(1, &ranges[Slot::InputCachedNormalDepth]);
            rootParameters[Slot::InputCachedFrameAge].InitAsDescriptorTable(1, &ranges[Slot::InputCachedFrameAge]);
            rootParameters[Slot::InputCachedSquaredMeanValue].InitAsDescriptorTable(1, &ranges[Slot::InputCachedSquaredMeanValue]);
            rootParameters[Slot::InputCachedRayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::InputCachedRayHitDistance]);
            rootParameters[Slot::OutputCacheFrameAge].InitAsDescriptorTable(1, &ranges[Slot::OutputCacheFrameAge]);
            rootParameters[Slot::OutputReprojectedCacheValues].InitAsDescriptorTable(1, &ranges[Slot::OutputReprojectedCacheValues]);
            rootParameters[Slot::OutputDebug1].InitAsDescriptorTable(1, &ranges[Slot::OutputDebug1]);
            rootParameters[Slot::OutputDebug2].InitAsDescriptorTable(1, &ranges[Slot::OutputDebug2]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_STATIC_SAMPLER_DESC staticSamplers[] = {
                CD3DX12_STATIC_SAMPLER_DESC(0, D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP) };

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, ARRAYSIZE(staticSamplers), staticSamplers);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: RTAO_TemporalSupersampling_ReverseReproject");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pTemporalSupersampling_ReverseReprojectCS), ARRAYSIZE(g_pTemporalSupersampling_ReverseReprojectCS));

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: RTAO_TemporalSupersampling_ReverseReproject");
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: RTAO_TemporalSupersampling_ReverseReproject");
        }
    }

    // ToDo desc
    void RTAO_TemporalSupersampling_ReverseReproject::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameLinearDepthDerivativeResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputReprojectedNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputTextureSpaceMotionVectorResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCachedValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCachedNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCachedFrameAgeResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCachedSquaredMeanValue,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCachedRayHitDistanceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputReprojectedCacheFrameAgeResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputReprojectedCacheValuesResourceHandle,
        float minSmoothingFactor,
        float depthTolerance,
        bool useDepthWeights,
        bool useNormalWeights,
        float floatEpsilonDepthTolerance,
        float depthDistanceBasedDepthTolerance,
        float depthSigma,
        bool useWorldSpaceDistance,
        bool usingBilateralDownsampledBuffers,
        bool perspectiveCorrectDepthInterpolation,
#if !NORMAL_DEPTH_R8G8B16_ENCODING
        TextureResourceFormatRGB::Type normalDepthResourceFormat,
#endif
        RWGpuResource debugResources[2],
        const XMMATRIX& projectionToWorldWithCameraEyeAtOrigin,
        const XMMATRIX& prevProjectionToWorldWithCameraEyeAtOrigin,
        UINT maxFrameAge,
        UINT numRaysToTraceSinceTSSMovement)
    {
        using namespace RootSignature::RTAO_TemporalSupersampling_ReverseReproject;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"RTAO_TemporalSupersampling_ReverseReproject", commandList);

        m_CB->useDepthWeights = useDepthWeights;
        m_CB->useNormalWeights = useNormalWeights;
        m_CB->depthTolerance = depthTolerance;
        m_CB->textureDim = XMUINT2(width, height);
        m_CB->invTextureDim = XMFLOAT2(1.f / width, 1.f / height);
        m_CB->floatEpsilonDepthTolerance = floatEpsilonDepthTolerance;
        m_CB->depthDistanceBasedDepthTolerance = depthDistanceBasedDepthTolerance;
        m_CB->depthSigma = depthSigma;
        m_CB->projectionToWorldWithCameraEyeAtOrigin = XMMatrixTranspose(projectionToWorldWithCameraEyeAtOrigin);
        m_CB->prevProjectionToWorldWithCameraEyeAtOrigin = XMMatrixTranspose(prevProjectionToWorldWithCameraEyeAtOrigin);
        m_CB->useWorldSpaceDistance = useWorldSpaceDistance;
        m_CB->usingBilateralDownsampledBuffers = usingBilateralDownsampledBuffers;
        m_CB->perspectiveCorrectDepthInterpolation = perspectiveCorrectDepthInterpolation;
        m_CB->numRaysToTraceAfterTSSAtMaxFrameAge = numRaysToTraceSinceTSSMovement;
        m_CB->maxFrameAge = maxFrameAge;

#if NORMAL_DEPTH_R8G8B16_ENCODING
        m_CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(16);
#else
        switch (normalDepthResourceFormat)
        {
        case TextureResourceFormatRGB::R32G32B32A32_FLOAT: m_CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(32); break;
        case TextureResourceFormatRGB::R16G16B16A16_FLOAT: m_CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(16); break;
        case TextureResourceFormatRGB::R11G11B10_FLOAT: m_CB->DepthNumMantissaBits = NumMantissaBitsInFloatFormat(10); break;
        default: ThrowIfFalse(false, L"Invalid resource format specified.");
        }
#endif
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputCurrentFrameNormalDepth, inputCurrentFrameNormalDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputCurrentFrameLinearDepthDerivative, inputCurrentFrameLinearDepthDerivativeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputReprojectedNormalDepth, inputReprojectedNormalDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputTextureSpaceMotionVector, inputTextureSpaceMotionVectorResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputCachedValue, inputCachedValueResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputCachedNormalDepth, inputCachedNormalDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputCachedFrameAge, inputCachedFrameAgeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputCachedSquaredMeanValue, inputCachedSquaredMeanValue);
            commandList->SetComputeRootDescriptorTable(Slot::InputCachedRayHitDistance, inputCachedRayHitDistanceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputCacheFrameAge, outputReprojectedCacheFrameAgeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputReprojectedCacheValues, outputReprojectedCacheValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputDebug1, debugResources[0].gpuDescriptorWriteAccess);
            commandList->SetComputeRootDescriptorTable(Slot::OutputDebug2, debugResources[1].gpuDescriptorWriteAccess);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo - streak artifacts on dragons nose on reprojection

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }


    namespace RootSignature {
        namespace RTAO_TemporalSupersampling_BlendWithCurrentFrame {
            namespace Slot {
                enum Enum {
                    InputOutputValue = 0,
                    InputOutputFrameAge,
                    InputOutputSquaredMeanValue,
                    InputOutputRayHitDistance,
                    OutputVariance,
                    InputCurrentFrameValue,
                    InputCurrentFrameLocalMeanVariance,
                    InputCurrentFrameRayHitDistance,
                    InputReprojectedCacheValues,
                    OutputDebug1,
                    OutputDebug2,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    void RTAO_TemporalSupersampling_BlendWithCurrentFrame::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::RTAO_TemporalSupersampling_BlendWithCurrentFrame;

            // ToDo remove comments from here and move descriptions to enum definition.
            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count];
            ranges[Slot::InputCurrentFrameValue].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
            ranges[Slot::InputCurrentFrameLocalMeanVariance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
            ranges[Slot::InputCurrentFrameRayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);
            ranges[Slot::InputReprojectedCacheValues].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3);
            ranges[Slot::InputOutputValue].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
            ranges[Slot::InputOutputFrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
            ranges[Slot::InputOutputSquaredMeanValue].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);
            ranges[Slot::InputOutputRayHitDistance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 3);
            ranges[Slot::OutputVariance].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 4);
            ranges[Slot::OutputDebug1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 10);
            ranges[Slot::OutputDebug2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 11);

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::InputCurrentFrameValue].InitAsDescriptorTable(1, &ranges[Slot::InputCurrentFrameValue]);
            rootParameters[Slot::InputCurrentFrameLocalMeanVariance].InitAsDescriptorTable(1, &ranges[Slot::InputCurrentFrameLocalMeanVariance]);
            rootParameters[Slot::InputCurrentFrameRayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::InputCurrentFrameRayHitDistance]);
            rootParameters[Slot::InputReprojectedCacheValues].InitAsDescriptorTable(1, &ranges[Slot::InputReprojectedCacheValues]);
            rootParameters[Slot::InputOutputValue].InitAsDescriptorTable(1, &ranges[Slot::InputOutputValue]);
            rootParameters[Slot::InputOutputFrameAge].InitAsDescriptorTable(1, &ranges[Slot::InputOutputFrameAge]);
            rootParameters[Slot::InputOutputSquaredMeanValue].InitAsDescriptorTable(1, &ranges[Slot::InputOutputSquaredMeanValue]);
            rootParameters[Slot::InputOutputRayHitDistance].InitAsDescriptorTable(1, &ranges[Slot::InputOutputRayHitDistance]);
            rootParameters[Slot::OutputVariance].InitAsDescriptorTable(1, &ranges[Slot::OutputVariance]);
            rootParameters[Slot::OutputDebug1].InitAsDescriptorTable(1, &ranges[Slot::OutputDebug1]);
            rootParameters[Slot::OutputDebug2].InitAsDescriptorTable(1, &ranges[Slot::OutputDebug2]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: RTAO_TemporalSupersampling_BlendWithCurrentFrame");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pTemporalSupersampling_BlendWithCurrentFrameCS), ARRAYSIZE(g_pTemporalSupersampling_BlendWithCurrentFrameCS));

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: RTAO_TemporalSupersampling_BlendWithCurrentFrame");
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: RTAO_TemporalSupersampling_BlendWithCurrentFrame");
        }
    }

    // ToDo desc
    void RTAO_TemporalSupersampling_BlendWithCurrentFrame::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameLocalMeanVarianceResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameRayHitDistanceResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputOutputValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputOutputFrameAgeResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputOutputSquaredMeanValueResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputOutputRayHitDistanceResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputReprojectedCacheValuesResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputVarianceResourceHandle,
        float minSmoothingFactor,
        bool forceUseMinSmoothingFactor,
        bool clampCachedValues,
        float clampStdDevGamma,
        float clampMinStdDevTolerance,
        UINT minFrameAgeToUseTemporalVariance,
        float clampDifferenceToFrameAgeScale,
        RWGpuResource debugResources[2],
        UINT numFramesToDenoiseAfterLastTracedRay)
    {
        using namespace RootSignature::RTAO_TemporalSupersampling_BlendWithCurrentFrame;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"RTAO_TemporalSupersampling_BlendWithCurrentFrame", commandList);

        m_CB->minSmoothingFactor = minSmoothingFactor;
        m_CB->forceUseMinSmoothingFactor = forceUseMinSmoothingFactor;
        m_CB->textureDim = XMUINT2(width, height);
        m_CB->invTextureDim = XMFLOAT2(1.f / width, 1.f / height);
        m_CB->clampCachedValues = clampCachedValues;
        m_CB->stdDevGamma = clampStdDevGamma;
        m_CB->minStdDevTolerance = clampMinStdDevTolerance;
        m_CB->minFrameAgeToUseTemporalVariance = minFrameAgeToUseTemporalVariance;
        m_CB->clampDifferenceToFrameAgeScale = clampDifferenceToFrameAgeScale;
        m_CB->numFramesToDenoiseAfterLastTracedRay = numFramesToDenoiseAfterLastTracedRay;

        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputCurrentFrameValue, inputCurrentFrameValueResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputCurrentFrameLocalMeanVariance, inputCurrentFrameLocalMeanVarianceResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputCurrentFrameRayHitDistance, inputCurrentFrameRayHitDistanceResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputOutputValue, inputOutputValueResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputOutputFrameAge, inputOutputFrameAgeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputOutputSquaredMeanValue, inputOutputSquaredMeanValueResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputOutputRayHitDistance, inputOutputRayHitDistanceResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputReprojectedCacheValues, inputReprojectedCacheValuesResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputVariance, outputVarianceResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputDebug1, debugResources[0].gpuDescriptorWriteAccess);
            commandList->SetComputeRootDescriptorTable(Slot::OutputDebug2, debugResources[1].gpuDescriptorWriteAccess);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo - streak artifacts on dragons nose on reprojection

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }


    // ToDo prune
    namespace RootSignature {
        namespace WriteValueToTexture {
            namespace Slot {
                enum Enum {
                    Output = 0,
                    Count
                };
            }
        }
    }

    // ToDo move type to execute
    void WriteValueToTexture::Initialize(ID3D12Device5* device, DX::DescriptorHeap* descriptorHeap)
    {
        // Create root signature.
        {
            using namespace RootSignature::WriteValueToTexture;

            // ToDo reorganize slots and descriptors
            CD3DX12_DESCRIPTOR_RANGE ranges[1];
            ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Output].InitAsDescriptorTable(1, &ranges[0]);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: WriteValueToTexture");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pWriteValueToTextureCS), ARRAYSIZE(g_pWriteValueToTextureCS));


            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: WriteValueToTexture");
        }

        // Create shader resources
        {
            m_output.rwFlags = ResourceRWFlags::AllowWrite | ResourceRWFlags::AllowRead;
            CreateRenderTargetResource(device, DXGI_FORMAT_R8_UINT, m_width, m_height, descriptorHeap,
                &m_output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, L"UAV texture: WriteValueToTexture intermediate value output");

        }
    }

    void WriteValueToTexture::Execute(
        ID3D12GraphicsCommandList4* commandList,
        ID3D12DescriptorHeap* descriptorHeap,
        D3D12_GPU_DESCRIPTOR_HANDLE* outputResourceHandle)
    {
        using namespace RootSignature::WriteValueToTexture;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"WriteValueToTexture", commandList);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Output, m_output.gpuDescriptorWriteAccess);
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // ToDo use non_pixel_shader_Resource everywherE?
        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_output.resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(m_width, ThreadGroup::Width), CeilDivide(m_height, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);

        commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_output.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));

        *outputResourceHandle = m_output.gpuDescriptorReadAccess;
    }



    namespace RootSignature {
        namespace GenerateGrassPatch {
            namespace Slot {
                enum Enum {
                    OutputVB = 0,
                    InputWindMap,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    // ToDo move resource upload to a separate call?
    void GenerateGrassPatch::Initialize(
        ID3D12Device5* device, 
        const wchar_t* windTexturePath,
        DX::DescriptorHeap* descriptorHeap,
        ResourceUploadBatch* resourceUpload, 
        UINT frameCount, 
        UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::GenerateGrassPatch;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count];
            ranges[Slot::InputWindMap].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // input wind texture
            ranges[Slot::OutputVB].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // output vertex buffer 

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::InputWindMap].InitAsDescriptorTable(1, &ranges[Slot::InputWindMap]);
            rootParameters[Slot::OutputVB].InitAsDescriptorTable(1, &ranges[Slot::OutputVB]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_STATIC_SAMPLER_DESC staticWrapLinearSampler(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters, 1, &staticWrapLinearSampler);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: GenerateGrassPatch");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pGenerateGrassStrawsCS), ARRAYSIZE(g_pGenerateGrassStrawsCS));

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: GenerateGrassPatch");
        }

        // Create shader resources.
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: GenerateGrassPatch");
        }

        // Load the wind texture.
        {
            LoadWICTexture(device, resourceUpload, windTexturePath, descriptorHeap, &m_windTexture.resource, &m_windTexture.heapIndex, &m_windTexture.cpuDescriptorHandle, &m_windTexture.gpuDescriptorHandle, false);
        };

    }

    // ToDo add option to allow input, output being the same
    // Expects, and returns, outputResource in D3D12_RESOURCE_STATE_UNORDERED_ACCESS state.
    void GenerateGrassPatch::Execute(
        ID3D12GraphicsCommandList4* commandList,
        const GenerateGrassStrawsConstantBuffer_AppParams& appParams,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputVertexBufferResourceHandle)
    {
        using namespace RootSignature::GenerateGrassPatch;
        using namespace DefaultComputeShaderParams;
        
        // ToDo move out or rename
        // ToDo add spaces to names?
        ScopedTimer _prof(L"Generate Grass Patch", commandList);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetPipelineState(m_pipelineStateObject.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputWindMap, m_windTexture.gpuDescriptorHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputVB, outputVertexBufferResourceHandle);
        }

        // Update the Constant Buffer.
        m_CB->p = appParams;
        m_CB->invActivePatchDim = XMFLOAT2(1.f / appParams.activePatchDim.x, 1.f / appParams.activePatchDim.y);
        m_CB->p = appParams;
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);
        commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));


        // Dispatch.
        {
            XMUINT2 dim = appParams.maxPatchDim;
            XMUINT2 groupSize(CeilDivide(dim.x, ThreadGroup::Width), CeilDivide(dim.y, ThreadGroup::Height));
            commandList->Dispatch(groupSize.x, groupSize.y, 1);
        }
    }


    namespace RootSignature {
        namespace SortRays {
            namespace Slot {
                enum Enum {
                    OutputSortedToSourceRayIndexOffset = 0,
                    OutputSourceToSortedRayIndexOffset,
                    Input,
                    OutputDebug,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    void SortRays::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::SortRays;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count]; 
            ranges[Slot::Input].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);  // 1 input texture
            ranges[Slot::OutputSortedToSourceRayIndexOffset].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
            ranges[Slot::OutputSourceToSortedRayIndexOffset].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);  // 1 output texture
            ranges[Slot::OutputDebug].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2);  // 1 output texture

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::Input].InitAsDescriptorTable(1, &ranges[Slot::Input]);
            rootParameters[Slot::OutputSortedToSourceRayIndexOffset].InitAsDescriptorTable(1, &ranges[Slot::OutputSortedToSourceRayIndexOffset]);
            rootParameters[Slot::OutputSourceToSortedRayIndexOffset].InitAsDescriptorTable(1, &ranges[Slot::OutputSourceToSortedRayIndexOffset]);
            rootParameters[Slot::OutputDebug].InitAsDescriptorTable(1, &ranges[Slot::OutputDebug]);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: Sort Rays");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();

            for (UINT i = 0; i < FilterType::Count; i++)
            {
                switch (i)
                {
                case CountingSort:
                    descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pCountingSort_SortRays_128x64rayGroupCS), ARRAYSIZE(g_pCountingSort_SortRays_128x64rayGroupCS));
                    break;
                }

                ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObjects[i])));
                m_pipelineStateObjects[i]->SetName(L"Pipeline state object: Sort Rays");
            }
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: Sort Rays");
        }
    }

    // Blurs input resource with a Gaussian filter.
    // width, height - dimensions of the input resource.
    void SortRays::Execute(
        ID3D12GraphicsCommandList4* commandList,
        float binDepthSize,
        UINT width,
        UINT height,
        FilterType type,
        bool useOctahedralRayDirectionQuantization,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputRayDirectionOriginDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputSortedToSourceRayIndexOffsetResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputSourceToSortedRayIndexOffsetResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputDebugResourceHandle)
    {
        using namespace RootSignature::SortRays;
        using namespace SortRays;

        ScopedTimer _prof(L"Sort Rays", commandList);

        m_CB->dim = XMUINT2(width, height);
        m_CB->useOctahedralRayDirectionQuantization = useOctahedralRayDirectionQuantization;
        m_CB->binDepthSize = binDepthSize;
        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::Input, inputRayDirectionOriginDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputSourceToSortedRayIndexOffset, outputSourceToSortedRayIndexOffsetResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputSortedToSourceRayIndexOffset, outputSortedToSourceRayIndexOffsetResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputDebug, outputDebugResourceHandle);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObjects[type].Get());
        }

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(width, RayGroup::Width), CeilDivide(height, RayGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }


    namespace RootSignature {
        namespace AdaptiveRayGenerator {
            namespace Slot {
                enum Enum {
                    OutputRayDirectionOriginDepth = 0,
                    InputRayOriginSurfaceNormalDepth,
                    InputRayOriginPosition,
                    InputFrameAge,
                    InputAlignedHemisphereSamples,
                    ConstantBuffer,
                    Count
                };
            }
        }
    }

    void AdaptiveRayGenerator::Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame)
    {
        // Create root signature.
        {
            using namespace RootSignature::AdaptiveRayGenerator;

            CD3DX12_DESCRIPTOR_RANGE ranges[Slot::Count]; 
            ranges[Slot::InputRayOriginSurfaceNormalDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
            ranges[Slot::InputRayOriginPosition].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
            ranges[Slot::InputFrameAge].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);
            ranges[Slot::OutputRayDirectionOriginDepth].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);

            CD3DX12_ROOT_PARAMETER rootParameters[Slot::Count];
            rootParameters[Slot::InputRayOriginSurfaceNormalDepth].InitAsDescriptorTable(1, &ranges[Slot::InputRayOriginSurfaceNormalDepth]);
            rootParameters[Slot::InputRayOriginPosition].InitAsDescriptorTable(1, &ranges[Slot::InputRayOriginPosition]);
            rootParameters[Slot::InputFrameAge].InitAsDescriptorTable(1, &ranges[Slot::InputFrameAge]);
            rootParameters[Slot::OutputRayDirectionOriginDepth].InitAsDescriptorTable(1, &ranges[Slot::OutputRayDirectionOriginDepth]);
           rootParameters[Slot::InputAlignedHemisphereSamples].InitAsShaderResourceView(3);
            rootParameters[Slot::ConstantBuffer].InitAsConstantBufferView(0);

            CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
            SerializeAndCreateRootSignature(device, rootSignatureDesc, &m_rootSignature, L"Compute root signature: Adaptive Ray Generator Rays");
        }

        // Create compute pipeline state.
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC descComputePSO = {};
            descComputePSO.pRootSignature = m_rootSignature.Get();
            descComputePSO.CS = CD3DX12_SHADER_BYTECODE(static_cast<const void*>(g_pAdaptiveRayGenCS), ARRAYSIZE(g_pAdaptiveRayGenCS));

            ThrowIfFailed(device->CreateComputePipelineState(&descComputePSO, IID_PPV_ARGS(&m_pipelineStateObject)));
            m_pipelineStateObject->SetName(L"Pipeline state object: Adaptive Ray Generator");
        }

        // Create shader resources
        {
            m_CB.Create(device, frameCount * numCallsPerFrame, L"Constant Buffer: Adaptive Ray Generator");
        }
    }

    // width, height - dimensions of the input resource.
    void AdaptiveRayGenerator::Execute(
        ID3D12GraphicsCommandList4* commandList,
        UINT width,
        UINT height,
        AdaptiveQuadSizeType adaptiveQuadSizetype,
        UINT maxFrameAge,
        UINT minFrameAgeForAdaptiveSampling,
        UINT maxFrameAgeToGenerateRaysFor,
        UINT maxRaysPerQuad,
        UINT seed,
        UINT numSamplesPerSet,
        UINT numSampleSets,
        UINT numPixelsPerDimPerSet,
        ID3D12DescriptorHeap* descriptorHeap,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputRayOriginSurfaceNormalDepthResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputRayOriginPositionResourceHandle,
        const D3D12_GPU_DESCRIPTOR_HANDLE& inputFrameAgeResourceHandle,
        const D3D12_GPU_VIRTUAL_ADDRESS& inputAlignedHemisphereSamplesBufferAddress,
        const D3D12_GPU_DESCRIPTOR_HANDLE& outputRayDirectionOriginDepthResourceHandle)
    {
        using namespace RootSignature::AdaptiveRayGenerator;
        using namespace DefaultComputeShaderParams;

        ScopedTimer _prof(L"Adaptive Ray Gen", commandList);

        m_CB->textureDim = XMUINT2(width, height);
        switch (adaptiveQuadSizetype)
        {
        case Quad1x1: m_CB->QuadDim = XMUINT2(1, 1); break;
        case Quad2x2: m_CB->QuadDim = XMUINT2(2, 2); break;
        case Quad4x4: m_CB->QuadDim = XMUINT2(4, 4); break;
        }
        m_CB->MaxFrameAge = maxFrameAge;
        m_CB->MinFrameAgeForAdaptiveSampling = minFrameAgeForAdaptiveSampling;
        m_CB->MaxRaysPerQuad = maxRaysPerQuad;
        m_CB->seed = seed;
        m_CB->numSamplesPerSet = numSamplesPerSet;
        m_CB->numPixelsPerDimPerSet = numPixelsPerDimPerSet;
        m_CB->numSampleSets = numSampleSets;
        m_CB->MaxFrameAgeToGenerateRaysFor = maxFrameAgeToGenerateRaysFor;
        static UINT frameID = 0;
        frameID = (frameID + 1) % (m_CB->QuadDim.x * m_CB->QuadDim.y);
        m_CB->FrameID = frameID;

        m_CBinstanceID = (m_CBinstanceID + 1) % m_CB.NumInstances();
        m_CB.CopyStagingToGpu(m_CBinstanceID);

        // Set pipeline state.
        {
            commandList->SetDescriptorHeaps(1, &descriptorHeap);
            commandList->SetComputeRootSignature(m_rootSignature.Get());
            commandList->SetComputeRootDescriptorTable(Slot::InputRayOriginSurfaceNormalDepth, inputRayOriginSurfaceNormalDepthResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputRayOriginPosition, inputRayOriginPositionResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::InputFrameAge, inputFrameAgeResourceHandle);
            commandList->SetComputeRootDescriptorTable(Slot::OutputRayDirectionOriginDepth, outputRayDirectionOriginDepthResourceHandle);
            commandList->SetComputeRootShaderResourceView(Slot::InputAlignedHemisphereSamples, inputAlignedHemisphereSamplesBufferAddress);
            commandList->SetComputeRootConstantBufferView(Slot::ConstantBuffer, m_CB.GpuVirtualAddress(m_CBinstanceID));
            commandList->SetPipelineState(m_pipelineStateObject.Get());
        }

        // Dispatch.
        XMUINT2 groupSize(CeilDivide(width, ThreadGroup::Width), CeilDivide(height, ThreadGroup::Height));
        commandList->Dispatch(groupSize.x, groupSize.y, 1);
    }
}