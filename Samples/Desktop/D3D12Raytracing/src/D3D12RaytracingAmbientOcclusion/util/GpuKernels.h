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
			DX::DescriptorHeap* descriptorHeap,
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
		ComPtr<ID3D12RootSignature>         m_rootSignature;
		ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
		std::vector<RWGpuResource>			m_csReduceSumOutputs;
		std::vector<ComPtr<ID3D12Resource>>	m_readbackResources;
		ConstantBuffer<DownsampleFilterConstantBuffer> m_CB;
	};


	class DownsampleGaussianFilter
	{
	public:
        // ToDo images moves switching between.
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
		ComPtr<ID3D12RootSignature>         m_rootSignature;
		ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
		ConstantBuffer<DownsampleFilterConstantBuffer> m_CB;
	};

    // ToDo rename to GBuffer downsample
    class DownsampleNormalDepthHitPositionGeometryHitBilateralFilter
    {
    public:
        enum Type {
            FilterDepthAware2x2 = 0       // ToDo rename to PointSampled
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, Type type);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            UINT width,
            UINT height,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputPositionResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputGeometryHitResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputPartialDistanceDerivativesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputPositionResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputGeometryHitResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputPartialDistanceDerivativesResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
    };

    class DownsampleValueNormalDepthBilateralFilter
    {
    public:
        enum Type {
            FilterPointSampling2x2 = 0,
            FilterDepthWeighted2x2,
            FilterDepthNormalWeighted2x2,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, Type type);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            UINT width,
            UINT height,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputPartialDistanceDerivativesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputPartialDistanceDerivativesResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
    };

    class UpsampleBilateralFilter
    {
    public:
        enum Type {
            Filter2x2 = 0,
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, Type type, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            UINT width,
            UINT height,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
            UINT perFrameInstanceId = 0,
            bool useBilinearWeights = true,
            bool useDepthWeights = true,
            bool useNormalWeights = true,
            bool useDynamicDepthThreshold = true);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
        ConstantBuffer<DownAndUpsampleFilterConstantBuffer> m_CB;
    };

    // ToDo rename
    class MultiScale_UpsampleBilateralFilterAndCombine
    {
    public:
        enum Type {
            Filter2x2 = 0,
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, Type type);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            UINT width,
            UINT height,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValue1ResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResValue2ResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
    };

    class GaussianFilter
    {
    public:
        enum FilterType {
            Filter3X3 = 0,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            UINT width,
            UINT height,
            FilterType type,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
            UINT perFrameInstanceId = 0);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        ConstantBuffer<GaussianFilterConstantBuffer> m_CB;
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
            DX::DescriptorHeap* descriptorHeap,
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
        enum Mode {
            OutputFilteredValue,
            OutputPerPixelFilterWeightSum
        };

        enum FilterType {
            EdgeStoppingBox3x3 = 0,
            EdgeStoppingGaussian3x3,
            EdgeStoppingGaussian5x5,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, UINT maxFilterPasses, UINT numCallsPerFrame = 1);
        void CreateInputResourceSizeDependentResources(
            ID3D12Device* device,
            DX::DescriptorHeap* descriptorHeap,
            UINT width,
            UINT height);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            ID3D12DescriptorHeap* descriptorHeap, 
            FilterType type,
            // ToDo use helper structs to pass the data in
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputVarianceResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputSmoothedVarianceResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHitDistanceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputPartialDistanceDerivativesResourceHandle,   // ToDo standardize depth vs distance
            RWGpuResource* outputResourceHandle,
            float valueSigma,
            float depthSigma,
            float normalSigma,
            UINT kernelStepShifts[5],
            UINT numFilterPasses = 5,
            Mode filterMode = OutputFilteredValue,
            bool reverseFilterPassOrder = false,
            bool useCalculatedVariance = true,
            UINT perFrameInstanceId = 0);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        RWGpuResource			            m_intermediateValueOutput;
        RWGpuResource			            m_intermediateVarianceOutputs[2];
        RWGpuResource			            m_filterWeightOutput;
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CB;
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CBfilterWeigth;
        UINT                                m_maxFilterPasses = 0;
    };


    // ToDo use template / inheritance
    class CalculatePartialDerivatives
    {
    public:
        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            ID3D12DescriptorHeap* descriptorHeap,
            UINT width,
            UINT height,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
            UINT perFrameInstanceId = 0);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CB;
    };

    class CalculateVariance
    {
    public:
        enum FilterType {
            Bilateral5x5 = 0,
            Bilateral7x7,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            ID3D12DescriptorHeap* descriptorHeap,
            FilterType type,
            UINT width,
            UINT height,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
            float depthSigma,
            float normalSigma,
            bool useApproximateVariance = true,
            UINT perFrameInstanceId = 0);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CB;
    };

    // ToDo bundle  RTAO ones together?
    class CalculateVariance2
    {
    public:
        enum FilterType {
            Bilateral5x5 = 0,
            Bilateral7x7,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device* device);
        void Execute(
            ID3D12GraphicsCommandList* commandList,
            ID3D12DescriptorHeap* descriptorHeap,
            FilterType type,
            UINT width,
            UINT height,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
            float depthSigma,
            float normalSigma,
            bool useApproximateVariance = true);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CB;
    };
}

