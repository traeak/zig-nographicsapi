#pragma once
#include "root_data_shared.h"

struct RootDataGenerate
{
    RootData* destination;
    uint32* output;
    uint32* dispatch_arguments;
    uint32 first_index;
    uint32 seed;
};
