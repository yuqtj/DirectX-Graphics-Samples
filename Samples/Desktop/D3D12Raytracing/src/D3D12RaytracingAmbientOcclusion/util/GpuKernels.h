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

		void Initialize(ID3D12Device5* device, Type type);
		void CreateInputResourceSizeDependentResources(
			ID3D12Device5* device,
			DX::DescriptorHeap* descriptorHeap,
			UINT frameCount,
			UINT width,
			UINT height,
			UINT numInvocationsPerFrame = 1);
		void Execute(
			ID3D12GraphicsCommandList4* commandList,
			ID3D12DescriptorHeap* descriptorHeap,
			UINT frameIndex,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
			void* resultSum,
            UINT invocationIndex = 0 );

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

		void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
		void Execute(
			ID3D12GraphicsCommandList4* commandList,
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
        UINT                                m_CBinstanceID = 0;
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

		void Initialize(ID3D12Device5* device, Type type, UINT frameCount, UINT numCallsPerFrame = 1);
		void Execute(
			ID3D12GraphicsCommandList4* commandList,
			UINT width,
			UINT height,
			ID3D12DescriptorHeap* descriptorHeap,
			const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
			const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle);

	private:
		ComPtr<ID3D12RootSignature>         m_rootSignature;
		ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
		ConstantBuffer<DownsampleFilterConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
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

        void Initialize(ID3D12Device5* device, Type type);
        void Execute(
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
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputNormalLowPrecisionResourceHandle,            
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputPositionResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputGeometryHitResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputPartialDistanceDerivativesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputMotionVectorResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputPrevFrameHitPositionResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputDepthResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
    };

    class DownsampleValueNormalDepthBilateralFilter
    {
    public:
        enum Type { // ToDo remove?
            FilterPointSampling2x2 = 0,
            FilterDepthWeighted2x2,
            FilterDepthNormalWeighted2x2,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, Type type);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
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

        void Initialize(ID3D12Device5* device, Type type, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            UINT width,
            UINT height,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputLowResNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResNormalResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHiResPartialDistanceDerivativeResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle,
            bool useBilinearWeights = true,
            bool useDepthWeights = true,
            bool useNormalWeights = true,
            bool useDynamicDepthThreshold = true);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
        ConstantBuffer<DownAndUpsampleFilterConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
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

        void Initialize(ID3D12Device5* device, Type type);
        void Execute(
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
            FilterRG3X3,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            UINT width,
            UINT height,
            FilterType type,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];

        ConstantBuffer<GaussianFilterConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
    };


    class RootMeanSquareError
    {
    public:
        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device);
        void CreateInputResourceSizeDependentResources(
            ID3D12Device5* device,
            DX::DescriptorHeap* descriptorHeap,
            UINT frameCount,
            UINT width,
            UINT height,
            UINT numInvocationsPerFrame);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
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

        void Initialize(ID3D12Device5* device, UINT maxFilterPasses, UINT frameCount, UINT numCallsPerFrame = 1);
        void CreateInputResourceSizeDependentResources(
            ID3D12Device5* device,
            DX::DescriptorHeap* descriptorHeap, // ToDo pass the same heap type in all inputs?
            UINT width,
            UINT height,
            DXGI_FORMAT format);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            ID3D12DescriptorHeap* descriptorHeap, 
            FilterType type,
            // ToDo use helper structs to pass the data in
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputVarianceResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputHitDistanceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputPartialDistanceDerivativesResourceHandle,   // ToDo standardize depth vs distance
            RWGpuResource* outputResourceHandle,
            float valueSigma,
            float depthSigma,
            float normalSigma,
            float weightScale,
            TextureResourceFormatRGB::Type normalDepthResourceFormat,
            UINT kernelStepShifts[5],
            UINT numFilterPasses = 5,
            Mode filterMode = OutputFilteredValue,
            bool reverseFilterPassOrder = false,
            bool useCalculatedVariance = true,
            bool perspectiveCorrectDepthInterpolation = false,
            bool useAdaptiveKernelSize = false, // ToDo revise defaults
            float minHitDistanceToKernelWidthScale = 1.f,
            UINT minKernelWidth = 5,
            UINT maxKernelWidth = 101,
            float varianceSigmaScaleOnSmallKernels = 2.f,
            bool usingBilateralDownsampledBuffers = false,
            float minVarianceToDenoise = 0);

        RWGpuResource& VarianceOutputResource() { return m_intermediateVarianceOutputs[0]; }

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        RWGpuResource			            m_intermediateValueOutput;
        RWGpuResource			            m_intermediateVarianceOutputs[2];
        RWGpuResource			            m_filterWeightOutput;
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CB;
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CBfilterWeight;
        UINT                                m_CBinstanceID = 0;
        UINT                                m_maxFilterPasses = 0;
    };


    // ToDo remove
    class WriteValueToTexture
    {
    public:

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, DX::DescriptorHeap* descriptorHeap);
        void Execute(ID3D12GraphicsCommandList4* commandList, ID3D12DescriptorHeap* descriptorHeap, D3D12_GPU_DESCRIPTOR_HANDLE* outputResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
        RWGpuResource			            m_output;
        UINT m_width = 3840;
        UINT m_height = 2160;
    };


    // ToDo use template / inheritance
    class CalculatePartialDerivatives
    {
    public:
        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            ID3D12DescriptorHeap* descriptorHeap,
            UINT width,
            UINT height,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
        ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
    };

    class CalculateVariance
    {
    public:
        enum FilterType {
            SquareBilateral = 0,
            SeparableBilateral,
            Separable,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            ID3D12DescriptorHeap* descriptorHeap,
            UINT width,
            UINT height,
            FilterType filterType,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputNormalsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputDepthsResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputVarianceResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputMeanResourceHandle,
            float depthSigma,
            float normalSigma,
            bool outputMean,
            bool useDepthWeights,
            bool useNormalWeights,
            UINT kernelWidth);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        ConstantBuffer<CalculateVariance_BilateralFilterConstantBuffer> m_CB;    // ToDo use a CB specific to CalculateVariance?
        UINT                                m_CBinstanceID = 0;
    };

    // ToDo rename to VarianceMean to match the result layout
    class CalculateMeanVariance
    {
    public:
        enum FilterType {
            // ToDo is this supported on all HW?
            Separable_AnyToAnyWaveReadLaneAt = 0,
            Separable,
            Count
        };


        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            ID3D12DescriptorHeap* descriptorHeap,
            UINT width,
            UINT height,
            FilterType filterType,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputValuesResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputMeanVarianceResourceHandle,
            UINT kernelWidth);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];
        ConstantBuffer<CalculateMeanVarianceConstantBuffer> m_CB;    // ToDo use a CB specific to CalculateVariance?
        UINT                                m_CBinstanceID = 0;
    };

       
    class RTAO_TemporalCache_ReverseReproject
    {
    public:
        void Release()
        {
            assert(0 && L"ToDo");
        }

        // ToDo set default parameters
        void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            UINT width,
            UINT height,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameSpatialMeanVarianceResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputCurrentFrameLinearDepthDerivativeResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputTemporalCacheValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputTemporalCacheNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputTemporalCacheFrameAgeResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputTemporalCacheCoefficientSquaredMeanResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputTextureSpaceMotionVectorResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputReprojectedNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputTemporalCacheValueResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputTemporalCacheFrameAgeResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputTemporalCacheCoefficientSquaredMeanResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputVarianceResourceHandle,
            float minSmoothingFactor,
            const XMMATRIX& invView,
            const XMMATRIX& invProj,
            const XMMATRIX& invViewProjAndCameraTranslation,
            const XMMATRIX& reverseProjectionTransform,
            const XMMATRIX& prevInvViewProj,
            float zMin,
            float zFar,
            float depthTolerance,
            bool useDepthWeights,
            bool useNormalWeights,
            bool forceUseMinSmoothingFactor,
            bool clampCachedValues,
            float clampStdDevGamma,
            float clampMinStdDevTolerance,
            float floatEpsilonDepthTolerance,
            float depthDistanceBasedDepthTolerance,
            float depthSigma,
            bool useWorldSpaceDistance,
            UINT minFrameAgeToUseTemporalVariance,
            bool usingBilateralDownsampledBuffers,
            bool perspectiveCorrectDepthInterpolation,
            float clampDifferenceToFrameAgeScale,
            TextureResourceFormatRGB::Type normalDepthResourceFormat,
            RWGpuResource debugResources[2],
            const XMVECTOR& currentFrameCameraPosition,
            const XMMATRIX& projectionToWorldWithCameraEyeAtOrigin,
            const XMVECTOR& prevToCurrentFrameCameraTranslation,
            const XMMATRIX& prevProjectionToWorldWithCameraEyeAtOrigin);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;
        ConstantBuffer<RTAO_TemporalCache_ReverseReprojectConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
    };

    class GenerateGrassPatch
    {
    public:
        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, const wchar_t* windTexturePath, DX::DescriptorHeap* descriptorHeap, ResourceUploadBatch* resourceUpload, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            const GenerateGrassStrawsConstantBuffer_AppParams& appParams,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputVertexBufferResourceHandle);

        UINT GetVertexBufferSize(UINT grassStrawsX, UINT grassStrawsY)
        {
            return grassStrawsX * grassStrawsY * N_GRASS_VERTICES * sizeof(VertexPositionNormalTextureTangent);
        }

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;

        ConstantBuffer<GenerateGrassStrawsConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
        D3DTexture                          m_windTexture;
    };

    class SortRays
    {
    public:
        // ToDo remove
        enum FilterType {
            CountingSort = 0,
            Count
        };

        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            float binDepthSize,
            UINT width,
            UINT height,
            FilterType type,
            bool useOctahedralDirectionQuantization,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputRayDirectionOriginDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputSortedToSourceRayIndexOffsetResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputSourceToSortedRayIndexOffsetResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputDebugResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObjects[FilterType::Count];

        ConstantBuffer<SortRaysConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
    };

    // ToDo rename to VariableRateRay...
    /// ToDo add header desc to each kernel.
    class AdaptiveRayGenerator
    {
    public:
        enum AdaptiveQuadSizeType {
            Quad2x2 = 0,
            Quad4x4,
            Count
        };


        void Release()
        {
            assert(0 && L"ToDo");
        }

        void Initialize(ID3D12Device5* device, UINT frameCount, UINT numCallsPerFrame = 1);
        void Execute(
            ID3D12GraphicsCommandList4* commandList,
            UINT width,
            UINT height,
            AdaptiveQuadSizeType adaptiveQuadSizetype,
            UINT maxFrameAge,
            UINT minFrameAgeForAdaptiveSampling,
            UINT seed,
            UINT numSamplesPerSet,
            UINT numSampleSets,
            UINT numPixelsPerDimPerSet,
            ID3D12DescriptorHeap* descriptorHeap,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputRayOriginSurfaceNormalDepthResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputRayOriginPositionResourceHandle,
            const D3D12_GPU_DESCRIPTOR_HANDLE& inputFrameAgeResourceHandle,
            const D3D12_GPU_VIRTUAL_ADDRESS& inputAlignedHemisphereSamplesBufferAddress,
            const D3D12_GPU_DESCRIPTOR_HANDLE& outputRayDirectionOriginDepthResourceHandle);

    private:
        ComPtr<ID3D12RootSignature>         m_rootSignature;
        ComPtr<ID3D12PipelineState>         m_pipelineStateObject;

        ConstantBuffer<AdaptiveRayGenConstantBuffer> m_CB;
        UINT                                m_CBinstanceID = 0;
    };

}

