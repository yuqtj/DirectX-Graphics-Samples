#if 1
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
// ToDo cleanup
#define HLSL
#include "RaytracingHlslCompat.h"
#include "RaytracingShaderHelper.hlsli"

Texture2D<float> g_inValues : register(t0);
Texture2D<float> g_inDepth : register(t1);  // ToDo use from normal tex directly
Texture2D<float4> g_inNormal : register(t2);
RWTexture2D<float> g_outVariance : register(u0);
RWTexture2D<float> g_outMean : register(u1);
ConstantBuffer<CalculateVariance_BilateralFilterConstantBuffer> cb: register(b0);

// ToDo add support for dxdy and perspective correct interpolation?

void AddFilterContribution(inout float weightedValueSum, inout float weightedSquaredValueSum, inout float weightSum, inout UINT numWeights, in float value, in float depth, in float3 normal, float obliqueness, in uint kernelRadius, in uint row, in uint col, in uint2 DTid)
{
    const float normalSigma = cb.normalSigma;
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
    const float depthSigma = 0.5;
#else
    const float depthSigma = cb.depthSigma;
#endif

    int2 id = int2(DTid) + (int2(row - kernelRadius, col - kernelRadius) );
    if (id.x >= 0 && id.y >= 0 && id.x < cb.textureDim.x && id.y < cb.textureDim.y)
    {
        float iValue = g_inValues[id];
#if COMPRES_NORMALS
        float4 normalBufValue = g_inNormal[id];
        float4 normal4 = float4(DecodeNormal(normalBufValue.xy), normalBufValue.z);
#else
        float4 normal4 = g_inNormal[id];
#endif 
        float3 iNormal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
        float iDepth = normal4.w;
#else
        float  iDepth = g_inDepth[id];
#endif

        // ToDO don't read from the texture if the value is not used.
        float w_d = cb.useDepthWeights ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
        float w_n = cb.useNormalWeights ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
        float w = w_n * w_d;

        float weightedValue = w * iValue;
        weightedValueSum += weightedValue;
        weightedSquaredValueSum += weightedValue * iValue;
        weightSum += w;
        numWeights += w > 0.0001f ? 1 : 0;
    }
}

// Calculates local per-pixel variance ~ Sum(X^2)/N - mean^2;
[numthreads(CalculateVariance_Bilateral::ThreadGroup::Width, CalculateVariance_Bilateral::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
#if COMPRES_NORMALS
    float4 normalBufValue = g_inNormal[DTid];
    float4 normal4 = float4(DecodeNormal(normalBufValue.xy), normalBufValue.z);
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
    float obliqueness = 1;    // ToDO
#else
    float obliqueness = max(0.0001f, pow(normalBufValue.w, 10));
#endif
#else
    float4 normal4 = g_inNormal[DTid];
    float obliqueness = max(0.0001f, pow(normal4.w, 10));
#endif 
    float3 normal = normal4.xyz;

#if PACK_NORMAL_AND_DEPTH
    float depth = normal4.w;
#else
    float  depth = g_inDepth[DTid];
#endif 

    float  value = g_inValues[DTid];

    UINT numWeights = 1;
    float weightedValueSum = value;
    float weightedSquaredValueSum = value * value;
    float weightSum = 1.f;  // ToDo check for missing value

    const uint kernelRadius = cb.kernelWidth >> 1;

   // [unroll]
    for (UINT r = 0; r < cb.kernelWidth; r++)
       // [unroll]
        for (UINT c = 0; c < cb.kernelWidth; c++)
            if (r != kernelRadius || c != kernelRadius)
                 AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, kernelRadius, r, c, DTid);

    float variance = 0;
    float mean = 0;
    if (numWeights > 1 && weightSum > FLT_EPSILON)
    {
        float invWeightSum = 1 / weightSum;
        mean = invWeightSum * weightedValueSum;
        // ToDo comment why N/N-1
        variance = (numWeights / float(numWeights - 1)) * (invWeightSum * weightedSquaredValueSum - mean * mean);

        variance = max(0, variance);    // Ensure variance doesn't go negative due to imprecision.


    }
    if (cb.outputMean)
    {
        g_outMean[DTid] = mean;
    }
    g_outVariance[DTid] = variance;
}
#elif 1
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
RWTexture2D<float> g_outVariance : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

#define SMEM_WIDTH  (CalculateVariance_Bilateral::ThreadGroup::Width + 4)
#define SMEM_HEIGHT (CalculateVariance_Bilateral::ThreadGroup::Height + 4)

groupshared float VCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float DCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheX[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheY[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheZ[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float OCache[SMEM_HEIGHT][SMEM_WIDTH];

void AddFilterContribution(inout float weightedValueSum, inout float weightedSquaredValueSum, inout float weightSum, inout UINT numWeights, in float value, in float depth, in float3 normal, float obliqueness, in int2 Cid, in int2 offset)
{
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;
    
    Cid += offset;
    float iValue = VCache[Cid.y][Cid.x];
    float iDepth = DCache[Cid.y][Cid.x];
    float3 iNormal = float3(NCacheX[Cid.y][Cid.x], NCacheY[Cid.y][Cid.x], NCacheZ[Cid.y][Cid.x]);
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
    depthSigma = 0.5;
#endif
    float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
    float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
    float w = w_n * w_d;

    float weightedValue = w * iValue;
    weightedValueSum += weightedValue;
    weightedSquaredValueSum += weightedValue * iValue;
    weightSum += w;
    numWeights += w > 0.0001f ? 1 : 0;
}

void LoadToSharedMemory(int2 pixel, int2 smemIndex)
{
    VCache[smemIndex.y][smemIndex.x] = g_inValues[pixel];
    float4 normalBufValue = g_inNormal[pixel];
    float3 normal = DecodeNormal(normalBufValue.xy);
    DCache[smemIndex.y][smemIndex.x] = normalBufValue.z;
    NCacheX[smemIndex.y][smemIndex.x] = normal.x;
    NCacheY[smemIndex.y][smemIndex.x] = normal.y;
    NCacheZ[smemIndex.y][smemIndex.x] = normal.z;
    OCache[smemIndex.y][smemIndex.x] = max(0.0001f, pow(normalBufValue.w, 10));
}

void PrefetchData(uint2 DTid, uint2 Gid, uint2 GTid)
{
    // Load data for the top-left most kernel cell.
    {
        int2 pixel = DTid - int2(2, 2);
        int2 Cid = GTid;
        LoadToSharedMemory(pixel, Cid);
    }

    // Load the rest.
    {
        int x, y;
        if (GTid.y < 4)
        {
            x = GTid.x;
            y = 8 + GTid.y;
        }
        else
        {
            x = 16 + (GTid.x % 4);
            y = (GTid.x >> 2) + ((GTid.y - 4) << 2 );
        }

        if (y < 12)
        {
            int2 offset = int2(x, y);
            int2 pixelBase = DTid - int2(2, 2) - GTid;
            int2 pixel = pixelBase + offset;
            int2 Cid = offset;
            LoadToSharedMemory(pixel, Cid);
        }
    }
}

// Calculates local per-pixel variance ~ Sum(X^2)/N - mean^2;
[numthreads(CalculateVariance_Bilateral::ThreadGroup::Width, CalculateVariance_Bilateral::ThreadGroup::Height, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID)
{
    PrefetchData(DTid, Gid, GTid);
    GroupMemoryBarrierWithGroupSync();

    // Cache ID: account for kernel neigbor in the cache.
    int2 Cid = GTid + int2(2, 2);

    float  value = VCache[Cid.y][Cid.x];
    float  depth = DCache[Cid.y][Cid.x];
    float3 normal = float3(NCacheX[Cid.y][Cid.x], NCacheY[Cid.y][Cid.x], NCacheZ[Cid.y][Cid.x]);
    float obliqueness = OCache[Cid.y][Cid.x];

    UINT numWeights = 1;
    float weightedValueSum = value;
    float weightedSquaredValueSum = value * value;
    float weightSum = 1.f;  // ToDo check for missing value


    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, - 2));
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, -1));
    [unroll]
    for (int c = -2; c < 0; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 0));


    [unroll]
    for (int c = 1; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 0));
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 1));
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 2));

    float variance;
    if (numWeights > 1)
    {
        float invWeightSum = weightSum > 0.0001f ? 1 / weightSum : 0.f;
        float mean = invWeightSum * weightedValueSum;
        variance = (numWeights / float(numWeights - 1)) * (invWeightSum * weightedSquaredValueSum - mean * mean);
    }
    else
    {
        variance = 0;
    }
    g_outVariance[DTid] = variance;
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
RWTexture2D<float> g_outVariance : register(u0);
ConstantBuffer<AtrousWaveletTransformFilterConstantBuffer> cb: register(b0);

#define SMEM_WIDTH  (CalculateVariance_Bilateral::ThreadGroup::Width + 4)
#define SMEM_HEIGHT (CalculateVariance_Bilateral::ThreadGroup::Height + 4)

groupshared float VCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float DCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheX[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheY[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float NCacheZ[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float OCache[SMEM_HEIGHT][SMEM_WIDTH];
groupshared float WCache[SMEM_HEIGHT][8][SMEM_WIDTH];

void AddFilterContribution(inout float weightedValueSum, inout float weightedSquaredValueSum, inout float weightSum, inout UINT numWeights, in float value, in float depth, in float3 normal, float obliqueness, in int2 Cid, in int2 offset)
{
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;
#if OBLIQUENESS_IS_SURFACE_PLANE_DISTANCE_FROM_ORIGIN_ALONG_SHADING_NORMAL
    depthSigma = 0.5;
#endif
    int2 NCid = Cid + offset; // Neighbor cache ID.
    if (NCid.x < CalculateVariance_Bilateral::ThreadGroup::Width + 4 && NCid.y < CalculateVariance_Bilateral::ThreadGroup::Height + 4)
    {
        float iValue = VCache[NCid.y][NCid.x];
        float iDepth = DCache[NCid.y][NCid.x];
        float3 iNormal = float3(NCacheX[NCid.y][NCid.x], NCacheY[NCid.y][NCid.x], NCacheZ[NCid.y][NCid.x]);

        float w_d = depthSigma > 0.01f ? exp(-abs(depth - iDepth) * obliqueness / (depthSigma * depthSigma)) : 1.f;
        float w_n = normalSigma > 0.01f ? pow(max(0, dot(normal, iNormal)), normalSigma) : 1.f;
        float w = w_n * w_d;

        // Cache the weight
        if (offset.x <= 0 && offset.y <= 0)
        {
            uint Wid = (2 + offset.y) * 3 + (2 + offset.x);
            WCache[Cid.y][Wid][Cid.x] = w;
        }

        float weightedValue = w * iValue;
        weightedValueSum += weightedValue;
        weightedSquaredValueSum += weightedValue * iValue;
        weightSum += w;
        numWeights += w > 0.0001f ? 1 : 0;
    }
}

void AddFilterContributionCached(inout float weightedValueSum, inout float weightedSquaredValueSum, inout float weightSum, inout UINT numWeights, in float value, in float depth, in float3 normal, float obliqueness, in int2 Cid, in int2 offset)
{
    const float normalSigma = cb.normalSigma;
    const float depthSigma = cb.depthSigma;

    if (Cid.x < CalculateVariance_Bilateral::ThreadGroup::Width && Cid.y < CalculateVariance_Bilateral::ThreadGroup::Height)
    {
        int2 NCid = Cid + offset;
        int2 offsetFromNeighbor = -offset;
        uint Wid = (2 + offsetFromNeighbor.y) * 3 + (2 + offsetFromNeighbor.x);
        float w = WCache[NCid.y][Wid][NCid.x];

        float iValue = VCache[NCid.y][NCid.x];

        float weightedValue = w * iValue;
        weightedValueSum += weightedValue;
        weightedSquaredValueSum += weightedValue * iValue;
        weightSum += w;
        numWeights += w > 0.0001f ? 1 : 0;
    }
}



void LoadToSharedMemory(int2 pixel, int2 smemIndex)
{
    VCache[smemIndex.y][smemIndex.x] = g_inValues[pixel];
    float4 normalBufValue = g_inNormal[pixel];
    float3 normal = DecodeNormal(normalBufValue.xy);
    DCache[smemIndex.y][smemIndex.x] = normalBufValue.z;
    NCacheX[smemIndex.y][smemIndex.x] = normal.x;
    NCacheY[smemIndex.y][smemIndex.x] = normal.y;
    NCacheZ[smemIndex.y][smemIndex.x] = normal.z;
    OCache[smemIndex.y][smemIndex.x] = max(0.0001f, pow(normalBufValue.w, 10));
}

void PrefetchData(uint2 DTid, uint2 GTid)
{
    // Load data for the top-left most kernel cell.
    {
        int2 pixel = DTid - int2(2, 2);
        int2 Cid = GTid;
        LoadToSharedMemory(pixel, Cid);
    }
    
    // Load the rest.
    int2 offset = GTid;
    bool load = false;

    if (GTid.x < 2 || GTid.y >= CalculateVariance_Bilateral::ThreadGroup::Height)
    {
        load = true;
        // Right two columns.
        if (GTid.y < CalculateVariance_Bilateral::ThreadGroup::Height)
        {
            offset.x += CalculateVariance_Bilateral::ThreadGroup::Width + 2;
        }
        // Bottom two rows.
        else
        {
            offset.y += 2;
        }

    }
    // Bottom right 8x4 cells.
    else if (GTid.x < 4 && GTid.y < 4)
    {
        load = true;
        offset.x += CalculateVariance_Bilateral::ThreadGroup::Width;
        offset.y += CalculateVariance_Bilateral::ThreadGroup::Height;
    }

    if (load)
    {
        int2 pixelTopLeft = DTid - int2(2, 2) - GTid;
        int2 pixel = pixelTopLeft + offset;
        int2 Cid = offset;
        LoadToSharedMemory(pixel, Cid);
    }
}

// Calculates local per-pixel variance ~ Sum(X^2)/N - mean^2;
[numthreads(CalculateVariance_Bilateral::ThreadGroup::Width + 2, CalculateVariance_Bilateral::ThreadGroup::Height + 2, 1)]
void main(uint2 Gid : SV_GroupID, uint2 GTid : SV_GroupThreadID)
{
    uint2 DTid = Gid * uint2(CalculateVariance_Bilateral::ThreadGroup::Width, CalculateVariance_Bilateral::ThreadGroup::Height) + GTid;
    PrefetchData(DTid, GTid);

    // Cache ID: account for kernel neigbor in the cache.
    int2 Cid = GTid + int2(2, 2);

    float  value = VCache[Cid.y][Cid.x];
    float  depth = DCache[Cid.y][Cid.x];
    float3 normal = float3(NCacheX[Cid.y][Cid.x], NCacheY[Cid.y][Cid.x], NCacheZ[Cid.y][Cid.x]);
    float obliqueness = OCache[Cid.y][Cid.x];

    UINT numWeights = 1;
    float weightedValueSum = value;
    float weightedSquaredValueSum = value * value;
    float weightSum = 1.f;  // ToDo check for missing value

#if 1
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, -2));
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, -1));
    [unroll]
    for (int c = -2; c < 0; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 0));
    [unroll]
    for (int c = -2; c < 0; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 1));
    [unroll]
    for (int c = -2; c < 0; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 2));


    //// Reuse cached weights for the rest.
    [unroll]
    for (int c = 1; c <= 2; c++)
        AddFilterContributionCached(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 0));
    [unroll]
    for (int c = 0; c <= 2; c++)
        AddFilterContributionCached(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 1));
    [unroll]
    for (int c = 0; c <= 2; c++)
        AddFilterContributionCached(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 2));
#else
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, -2));
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, -1));
    [unroll]
    for (int c = -2; c < 0; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 0));
    [unroll]
    for (int c = 1; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 0));
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 1));
    [unroll]
    for (int c = -2; c <= 2; c++)
        AddFilterContribution(weightedValueSum, weightedSquaredValueSum, weightSum, numWeights, value, depth, normal, obliqueness, Cid, int2(c, 2));

#endif

    if (GTid.x < CalculateVariance_Bilateral::ThreadGroup::Width && GTid.y < CalculateVariance_Bilateral::ThreadGroup::Height)
    {
        float variance;
        if (numWeights > 1)
        {
            float invWeightSum = weightSum > 0.0001f ? 1 / weightSum : 0.f;
            float mean = invWeightSum * weightedValueSum;
            variance = (numWeights / float(numWeights - 1)) * (invWeightSum * weightedSquaredValueSum - mean * mean);
        }
        else
        {
            variance = 0;
        }
        //g_outVariance[DTid] = g_inValues[DTid-int2(2,2)];
        //g_outVariance[DTid] = VCache[GTid.y][GTid.x];
        g_outVariance[DTid] = variance;
    }
}
#endif