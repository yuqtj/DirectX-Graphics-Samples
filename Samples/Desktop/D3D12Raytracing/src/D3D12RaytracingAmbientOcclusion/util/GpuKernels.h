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

//
// Helpers for doing CPU & GPU performance timing and statitics
//

#pragma once

namespace GpuKernels
{
	class ReduceSum
	{
	public:
        enum Type {
            Uint = 0,
            Float
        };

		void Release()
		{
			assert(0 && L"ToDo");
		}

		void Initialize(ID3D12Device* device, Type type);
		void CreateInputResourceSizeDependentResources(
			ID3D12Device* device,
			DescriptorHeap* descriptorHeap,
			UINT frameCount,
			UINT width,
			UINT height,
			UINT numInvocationsPerFrame);
		void Execute(
			ID3D12GraphicsCommandList* commandList,
			ID3D12DescriptorHeap* descriptorHeap,
			UINT frameIndex,
			UINT invocationIndex,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
			void* resultSum);

	private:
        Type                                m_resultType;
        UINT                                m_resultSize;
		ComPtr<ID3D12RootSignature>         m_rootSignature;
		ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
		std::vector<RWGpuResource>			m_csReduceSumOutputs;
		std::vector<ComPtr<ID3D12Resource>>	m_readbackResources;
	};

	class DownsampleBoxFilter2x2
	{
	public:
		void Release()
		{
			assert(0 && L"ToDo");
		}

		void Initialize(ID3D12Device* device);
		void CreateInputResourceSizeDependentResources(
			ID3D12Device* device,
			UINT width,
			UINT height);
		void Execute(
			ID3D12GraphicsCommandList* commandList,
			UINT width,
			UINT height,
			ID3D12DescriptorHeap* descriptorHeap,
			const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
			const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle);

	private:
		typedef UINT ResultType;
		ComPtr<ID3D12RootSignature>         m_rootSignature;
		ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
		std::vector<RWGpuResource>			m_csReduceSumOutputs;
		std::vector<ComPtr<ID3D12Resource>>	m_readbackResources;
		ConstantBuffer<DownsampleFilterConstantBuffer> m_CB;
	};


	class DownsampleGaussianFilter
	{
	public:
		enum Type {
			Tap9 = 0,
			Tap25
		};

		void Release()
		{
			assert(0 && L"ToDo");
		}

		void Initialize(ID3D12Device* device, Type type);
		void CreateInputResourceSizeDependentResources(
			ID3D12Device* device,
			UINT width,
			UINT height);
		void Execute(
			ID3D12GraphicsCommandList* commandList,
			UINT width,
			UINT height,
			ID3D12DescriptorHeap* descriptorHeap,
			const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
			const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle);

	private:
		typedef UINT ResultType;
		ComPtr<ID3D12RootSignature>         m_rootSignature;
		ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
		std::vector<ComPtr<ID3D12Resource>>	m_readbackResources;
		ConstantBuffer<DownsampleFilterConstantBuffer> m_CB;
	};


    class RootMeanSquareError
    {
    public:
        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device);
        void CreateInputResourceSizeDependentResources(
            ID3D12Device* device,
            DescriptorHeap* descriptorHeap,
            UINT frameCount,
            UINT width,
            UINT height,
            UINT numInvocationsPerFrame);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            ID3D12DescriptorHeap* descriptorHeap,
            UINT frameIndex,
            UINT invocationIndex,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
            float* rootMeanSquareError);

    private:
        typedef UINT ResultType;
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
        RWGpuResource			            m_perPixelMeanSquareError;
        ReduceSum                           m_reduceSumKernel;
    };

    class AtrousWaveletTransformCrossBilateralFilter
    {
    public:
        enum FilterType {
            Gaussian5x5 = 0,
            EdgeStoppingGaussian5x5,
            EdgeStoppingGaussian3x3,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, UINT maxFilterPasses);
        void CreateInputResourceSizeDependentResources(
            ID3D12Device* device,
            DescriptorHeap* descriptorHeap,
            UINT width,
            UINT height);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            ID3D12DescriptorHeap* descriptorHeap, 
            FilterType type,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
            RWGpuResource* outputResourceHandle,
            UINT numFilterPasses = 5);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        RWGpuResource			            m_intermediateOutput;
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CB;
    };
}