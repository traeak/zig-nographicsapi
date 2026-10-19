#pragma once
#include <NoGraphicsAPIUtility/shader_types.h>

static const uint32 shader_abi_thread_count = 4;

struct ShaderAbiData
{
    float scalar;
    float3 vector;
    float4 color;
    float3x3 matrix;
    float3 transformed;
};

struct ShaderAbiRoot
{
    float scalar;
    float3 vector;
    float4 color;
    ShaderAbiData* output;
    float3x3 matrix;
    float4* sampled;
    uint32 texture_base;
    uint32 sampler_base;
};
