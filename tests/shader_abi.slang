#include <NoGraphicsAPI/shader.slang>
#include "shader_abi_shared.h"

GPU_ROOT(ShaderAbiRoot, root);

[numthreads(shader_abi_thread_count, 1, 1)]
void computeMain(uint lane : SV_GroupIndex)
{
    root.output[lane].scalar = root.scalar;
    root.output[lane].vector = root.vector;
    root.output[lane].color = root.color;
    root.output[lane].matrix = root.matrix;
    root.output[lane].transformed = mul(root.matrix, float3(2, 3, 5));
    Texture2D<float4> texture = gpu_texture<Texture2D<float4>>(root.texture_base + lane);
    root.sampled[lane] = texture.SampleLevel(gpu_sampler(root.sampler_base + lane), float2(1.25f, 0.5f), 0.0f);
}
