#pragma once
#include <NoGraphicsAPIUtility/shader_types.h>

struct RootData
{
    uint32* output;
    uint32 output_index;
    uint32 seed;
    uint32 values[60];
};
