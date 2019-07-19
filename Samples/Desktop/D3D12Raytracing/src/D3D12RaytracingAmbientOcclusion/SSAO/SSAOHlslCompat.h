#ifndef SSAOHLSLCOMPAT_H
#define SSAOHLSLCOMPAT_H

#ifdef HLSL
#include "../util/HlslCompat.h"
#else
using namespace DirectX;
#endif

#define NUM_BUFFERS 4

struct SSAORenderConstantBuffer
{
    XMFLOAT4 invThicknessTable[3];
    XMFLOAT4 sampleWeightTable[3];
    XMFLOAT2 invSliceDimension;
    float  normalMultiply;
};

struct BlurAndUpscaleConstantBuffer
{
    XMFLOAT2 invLowResolution;
    XMFLOAT2 invHighResolution;
    float noiseFilterStrength;
    float stepSize;
    float blurTolerance;
    float upsampleTolerance;
};

#endif