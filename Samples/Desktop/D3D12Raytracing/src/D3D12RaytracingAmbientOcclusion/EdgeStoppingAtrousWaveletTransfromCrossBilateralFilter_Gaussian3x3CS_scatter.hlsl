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
#if 1
#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValues : register(t0);
Texture2D<float4> g_inNormal : register(t1);
Texture2D<float> g_inDepth : register(t2);
Texture2D<uint> g_inNormalOct : register(t3);

RWTexture2D<float> g_outFilteredValues : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

#define KERNEL_WIDTH 3
// Account for kernel neighbors outside the thread group
#define SMEM_WIDTH  (AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width + 2)
#define SMEM_HEIGHT (AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height + 2)
groupshared float VCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float DCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheX[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheY[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheZ[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float OCache[SMEM_HEIGHT][SMEM_WIDTH];

void LoadToSharedMemory(int2 pixel, int2 smemIndex)
{
    VCache[smemIndex.y][smemIndex.x] = g_inValues[pixel];
    DCache[smemIndex.y][smemIndex.x] = g_inDepth[pixel];
#if COMPRES_NORMALS
    uint id = g_inNormalOct[pixel];
    float3 normal = i_octahedral_32(id, 1u);
    OCache[smemIndex.y][smemIndex.x] = 1.f;
#else
    float4 normal4 = g_inNormal[pixel];
    float3 normal = normal4.xyz;
    OCache[smemIndex.y][smemIndex.x] = normal4.w;
#endif
    NCacheX[smemIndex.y][smemIndex.x] = normal.x;
    NCacheY[smemIndex.y][smemIndex.x] = normal.y;
    NCacheZ[smemIndex.y][smemIndex.x] = normal.z;
}

// Returns kernel center pixel position for the current thread.
int2 PrefetchData(uint2 Gid, uint2 GTid)
{
    const int2 Gdim = int2(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height);

    // Cache thread ID: account for kernel neigbor in the cache.
    int2 CTid = GTid + int2(1, 1);

    // Interleave kernel step width groups.
    // - Each pass executes kernel at 1 << cb.kernelStepShift pixel offsets. Therefore load only those pixels per group.
    // - Interleave neighboring groups given the offsets, such that G0[0], G1[0], G2[0],... G"step-1"[0], G0[1], ... etc
    int2 _Gid = Gid >> cb.kernelStepShift;
    int2 _SubGid = Gid - (_Gid << cb.kernelStepShift);

    // ToDo replace * GDim with a shift.
    int2 pixel = ((_Gid * Gdim + GTid) << cb.kernelStepShift) + _SubGid;
    
    if (pixel.x >= cb.textureDim.x && pixel.y >= cb.textureDim.y)
    {
        return pixel;
    }
    LoadToSharedMemory(pixel, CTid);

    //
    // Load remaining offcenter kernel values - group neighbors
    //

    // Load Top and Bottom rows
    if (GTid.y == 0 || GTid.y == Gdim.y - 1)
    {
        int offset = GTid.y == 0 ? -1 : 1;
        LoadToSharedMemory(pixel + int2(0, offset << cb.kernelStepShift), CTid + int2(0, offset));
    }

    // Load Left and Right columns
    if (GTid.x == 0 || GTid.x == Gdim.x - 1)
    {
        int offset = GTid.x == 0 ? -1 : 1;
        LoadToSharedMemory(pixel + int2(offset << cb.kernelStepShift, 0), CTid + int2(offset, 0));
    }

    // Load the corner neighbors
    if ((GTid.x == 0 && GTid.y == 0) ||
        (GTid.x == 0 && GTid.y == Gdim.y - 1) ||
        (GTid.x == Gdim.x - 1 && GTid.y == 0) ||
        (GTid.x == Gdim.x - 1 && GTid.y == Gdim.y - 1))
    {
        int2 offset = int2(GTid.x == 0 ? -1 : 1, GTid.y == 0 ? -1 : 1);
        LoadToSharedMemory(pixel + (offset << cb.kernelStepShift), CTid + offset);
    }

    return pixel;
}


void AddFilterContribution(inout float weightedValueSum, inout float weightSum, in float value, in float depth, in float3 normal, float obliqueness, in float w_h, in int2 pixelPos, in int2 srcPixelPos, in int2 offset)
{
    const float valueSigma = cb.valueSigma;
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    // Calculate neighbor's position
    pixelPos += offset;
    srcPixelPos += (offset << cb.kernelStepShift);

    //if (srcPixelPos.x >= 0 && srcPixelPos.y >= 0 && srcPixelPos.x < cb.textureDim.x && srcPixelPos.y < cb.textureDim.y)
    {
        float iValue = g_inValues[pixelPos];
        float w_x = valueSigma > 0.01f ? cb.kernelStepShift > 0 ? exp(-abs(value - iValue) / (valueSigma * valueSigma)) : 1.f : 1.f;

#if 0
        float  iDepth = g_inDepth[srcPixelPos];
        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;

        float3 iNormal = g_inNormal[srcPixelPos].xyz;
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;

        float w = w_h * w_x * w_n * w_d;
#else
        float w = w_h * w_x;
#endif

        weightedValueSum += w * iValue;
        weightSum += w;
    }
}

// ToDo test: interleave loading auxilary buffers into smem with computation
// ToDo test: store normal3 in single smem

// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint2 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= cb.textureDim.x || DTid.y >= cb.textureDim.y)
    {
        return;
    }
    
    const uint N = 3;
    const float kernel1D[N] = { 0.27901, 0.44198, 0.27901 };
    const float kernel[N][N] =
    {
        { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2] },
        { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2] },
        { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2] },
    };
#if COMPRES_NORMALS
    uint normalOct = g_inNormalOct[DTid];
    float3 normal = i_octahedral_32(normalOct, 16u);
    float obliqueness = 1.f;
#elif 1
    float4 normal4 = g_inNormal[DTid];
    float3 normal = normal4.xyz;
    float obliqueness = max(0.0001f, pow(normal4.w, 10));
#else
    float3 iNormal = float3(1, 1, 1);
    float obliqueness = 1.f;
#endif 

    float  depth = g_inDepth[DTid];
    float  value = g_inValues[DTid];

    float weightedValueSum = value * kernel[1][1];
    float weightSum = kernel[1][1];

    // Recalculate original pixel position for bounds checking and writing to output.
    uint2 _DTid = DTid << cb.kernelStepShift;
    uint2 srcPixelPos = (_DTid % cb.textureDim) + _DTid / cb.textureDim;

    // Convolve
    int2 pixel = DTid;
#if 1
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[0][0], pixel, srcPixelPos, int2(-1, -1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[0][1], pixel, srcPixelPos, int2(0, -1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[0][2], pixel, srcPixelPos, int2(1, -1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[1][0], pixel, srcPixelPos, int2(-1, 0));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[1][2], pixel, srcPixelPos, int2(1, 0));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[2][0], pixel, srcPixelPos, int2(-1, 1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[2][1], pixel, srcPixelPos, int2(0, 1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[2][2], pixel, srcPixelPos, int2(1, 1));
#endif

    int2 outPixel;
    if (!cb.scatterOutput)
    {
        outPixel.x = (pixel.x >> 1) + (pixel.x % 2 == 0 ? 0 : cb.textureDim.x >> 1);
        outPixel.y = (pixel.y >> 1) + (pixel.y % 2 == 0 ? 0 : cb.textureDim.y >> 1);
    }
    else
    {
        outPixel = srcPixelPos;
    }

    g_outFilteredValues[outPixel] = weightedValueSum / weightSum;
}
#else

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

#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValues : register(t0);
Texture2D<float4> g_inNormal : register(t1);
Texture2D<float> g_inDepth : register(t2);
Texture2D<uint> g_inNormalOct : register(t3);

RWTexture2D<float> g_outFilteredValues : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

#define KERNEL_WIDTH 3
// Account for kernel neighbors outside the thread group
#define SMEM_WIDTH  (AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width + 2)
#define SMEM_HEIGHT (AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height + 2)
groupshared float VCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float DCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheX[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheY[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheZ[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float OCache[SMEM_HEIGHT][SMEM_WIDTH];

void LoadToSharedMemory(int2 pixel, int2 smemIndex)
{
    VCache[smemIndex.y][smemIndex.x] = g_inValues[pixel];
    DCache[smemIndex.y][smemIndex.x] = g_inDepth[pixel];
#if COMPRES_NORMALS
    uint id = g_inNormalOct[pixel];
    float3 normal = i_octahedral_32(id, 1u);
    OCache[smemIndex.y][smemIndex.x] = 1.f;
#else
    float4 normal4 = g_inNormal[pixel];
    float3 normal = normal4.xyz;
    OCache[smemIndex.y][smemIndex.x] = normal4.w;
#endif
    NCacheX[smemIndex.y][smemIndex.x] = normal.x;
    NCacheY[smemIndex.y][smemIndex.x] = normal.y;
    NCacheZ[smemIndex.y][smemIndex.x] = normal.z;
}

// Returns kernel center pixel position for the current thread.
void PrefetchData(int2 pixel, int2 CTid, uint2 GTid)
{
    const int2 Gdim = int2(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height);

    LoadToSharedMemory(pixel, CTid);

    //
    // Load remaining offcenter kernel values - group neighbors
    //

    // Load Top and Bottom rows
    if (GTid.y == 0 || GTid.y == Gdim.y - 1)
    {
        int offset = GTid.y == 0 ? -1 : 1;
        LoadToSharedMemory(pixel + int2(0, offset << cb.kernelStepShift), CTid + int2(0, offset));
    }

    // Load Left and Right columns
    if (GTid.x == 0 || GTid.x == Gdim.x - 1)
    {
        int offset = GTid.x == 0 ? -1 : 1;
        LoadToSharedMemory(pixel + int2(offset << cb.kernelStepShift, 0), CTid + int2(offset, 0));
    }

    // Load the corner neighbors
    if ((GTid.x == 0 && GTid.y == 0) ||
        (GTid.x == 0 && GTid.y == Gdim.y - 1) ||
        (GTid.x == Gdim.x - 1 && GTid.y == 0) ||
        (GTid.x == Gdim.x - 1 && GTid.y == Gdim.y - 1))
    {
        int2 offset = int2(GTid.x == 0 ? -1 : 1, GTid.y == 0 ? -1 : 1);
        LoadToSharedMemory(pixel + (offset << cb.kernelStepShift), CTid + offset);
    }
}


void AddFilterContribution(inout float weightedValueSum, inout float weightSum, in float value, in float depth, in float3 normal, float obliqueness, in float w_h, in int2 pixel, in int2 CTid, in int2 offset)
{
    const float valueSigma = cb.valueSigma;
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    CTid += offset;
    pixel += offset << cb.kernelStepShift;
    if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < cb.textureDim.x && pixel.y < cb.textureDim.y)
    {
        float iValue = VCache[CTid.y][CTid.x];
        float3 iNormal = float3(NCacheX[CTid.y][CTid.x], NCacheY[CTid.y][CTid.x], NCacheZ[CTid.y][CTid.x]);
        float  iDepth = DCache[CTid.y][CTid.x];

        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
        float w_x = valueSigma > 0.01f ? cb.kernelStepShift > 0 ? exp(-abs(value - iValue) / (valueSigma * valueSigma)) : 1.f : 1.f;

        // Ref: SVGF
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;

        float w = w_h * w_x * w_n * w_d;

        weightedValueSum += w * iValue;
        weightSum += w;
    }
}

// ToDo test: interleave loading auxilary buffers into smem with computation
// ToDo test: store normal3 in single smem

// Atrous Wavelet Transform Cross Bilateral Filter
// Ref: Dammertz 2010, Edge-Avoiding A-Trous Wavelet Transform for Fast Global Illumination Filtering
[numthreads(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID, uint2 DTid : SV_DispatchThreadID)
{
    const int2 Gdim = int2(AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Width, AtrousWaveletTransformFilter_Gaussian5x5CS::ThreadGroup::Height);
  
    // Cache thread ID: account for kernel neigbor in the cache.
    int2 CTid = GTid + int2(1, 1);

    // Interleave kernel step width groups.
    // - Each pass executes kernel at 1 << cb.kernelStepShift pixel offsets. Therefore load only those pixels per group.
    // - Interleave neighboring groups given the offsets, such that G0[0], G1[0], G2[0],... G"step-1"[0], G0[1], ... etc
    int2 _Gid = Gid >> cb.kernelStepShift;
    int2 _SubGid = Gid - (_Gid << cb.kernelStepShift);

    // ToDo replace * GDim with a shift.
    int2 pixel = ((_Gid * Gdim + GTid) << cb.kernelStepShift) + _SubGid;

    if (pixel.x >= cb.textureDim.x && pixel.y >= cb.textureDim.y)
    {
        return;
    }
    
    const uint N = KERNEL_WIDTH;
    const float kernel1D[N] = { 0.27901, 0.44198, 0.27901 };
    const float kernel[N][N] =
    {
        { kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2] },
        { kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2] },
        { kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2] },
    };
    
    float3 normal = float3(NCacheX[CTid.y][CTid.x], NCacheY[CTid.y][CTid.x], NCacheZ[CTid.y][CTid.x]);
    float  depth = DCache[CTid.y][CTid.x];
    float  value = VCache[CTid.y][CTid.x];
    float  obliqueness = max(0.0001f, pow(OCache[CTid.y][CTid.x], 10));

    float weightedValueSum = value * kernel[1][1];
    float weightSum = kernel[1][1];


    // Prefetch all the data needed into group shared memory.
    PrefetchData(pixel, CTid, GTid);
    GroupMemoryBarrierWithGroupSync();

#if 1
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[0][0], pixel, CTid, int2(-1, -1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[0][1], pixel, CTid, int2(0, -1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[0][2], pixel, CTid, int2(1, -1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[1][0], pixel, CTid, int2(-1, 0));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[1][2], pixel, CTid, int2(1, 0));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[2][0], pixel, CTid, int2(-1, 1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[2][1], pixel, CTid, int2(0, 1));
    AddFilterContribution(weightedValueSum, weightSum, value, depth, normal, obliqueness, kernel[2][2], pixel, CTid, int2(1, 1));
#endif

#if 0
    g_outFilteredValues[pixel] = weightSum > 0.0001f ? weightedValueSum / weightSum : 0.f;
#else
    g_outFilteredValues[pixel] = weightedValueSum / weightSum;
#endif
}
#endif