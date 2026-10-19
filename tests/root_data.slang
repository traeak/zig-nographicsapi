#include <NoGraphicsAPI/shader.slang>
#include "root_data_shared.h"

#if defined(ROOT_DATA_ALIGNMENT)
GPU_ROOT(RootData, root, ROOT_DATA_ALIGNMENT);
#else
GPU_ROOT(RootData, root);
#endif

[numthreads(64, 1, 1)]
void computeMain(uint lane : SV_DispatchThreadID)
{
    uint result = root.seed + lane;
    [unroll]
    for (uint i = 0; i < 60; ++i) result = (result * 33u) ^ root.values[i];
    root.output[root.output_index + lane] = result;
}
