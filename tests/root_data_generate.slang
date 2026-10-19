#include <NoGraphicsAPI/shader.slang>
#include "root_data_generate_shared.h"

GPU_ROOT(RootDataGenerate, root);

[numthreads(1, 1, 1)]
void computeMain(uint index : SV_DispatchThreadID)
{
    root.destination[index].output = root.output;
    root.destination[index].output_index = (root.first_index + index) * 64;
    root.destination[index].seed = root.seed + index;
    for (uint i = 0; i < 60; ++i) root.destination[index].values[i] = (root.seed + index) * 7 + i;
    if (index == 0)
    {
        root.dispatch_arguments[0] = 1;
        root.dispatch_arguments[1] = 1;
        root.dispatch_arguments[2] = 1;
    }
}
