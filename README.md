# zig-nographicsapi

A Zig port of Sebastian Aaltonen's [NoGraphicsAPI](https://github.com/sebbbi/NoGraphicsAPI), implementing the ideas from his
[*No Graphics API*](https://www.sebastianaaltonen.com/blog/no-graphics-api) blog post.

zig-nographicsapi is a Zig 0.17 graphics library for the latest **Metal 4** devices and **Vulkan 1.4** devices with
**`VK_EXT_descriptor_heap`**, **`VK_KHR_device_address_commands`**, and **`VK_EXT_mesh_shader`**. It uses GPU pointers,
descriptor heaps, and shared Slang shaders.

The goal is to make GPU programming feel more like working with ordinary memory and data structures:
GPU pointers for data, heap indices for textures, and one GPU pointer to the arguments of each draw or dispatch.

Both backends are native Zig: Vulkan on Windows (Win32) and Linux (X11), and Metal 4 on macOS (AppKit), which calls Metal
through the Objective-C runtime. They share one Zig API and the same Slang sources.

**Status:** the Metal backend compiles for aarch64-macos and was checked against Apple's metal-cpp headers, but has not yet been run on a Mac.

## What changes from classic rendering?

A conventional renderer creates buffer objects, describes vertex and resource-binding layouts, and
assembles bindings before drawing. The blog asks how much of this machinery modern bindless hardware
still needs. NoGraphicsAPI makes that alternative data model the foundation of the library:

- **Memory allocation without buffer objects.** Allocate GPU heaps and partition them with an
  application-side allocator. Mapped heaps provide CPU and GPU addresses, so the CPU can write data
  directly. Vertex data, constants, and arbitrary structures are allocations, not separate public buffer types.
- **Typed GPU pointers.** Shaders follow 64-bit pointers stored in shared C/Slang structures.
  Arrays, pointer arithmetic, and nested data structures work without buffer descriptors or binding slots.
  Vertex shaders fetch their own vertices; there is no vertex-layout declaration.
- **Bindless textures and samplers.** The application owns descriptor heaps and chooses their indices.
  Materials carry those indices as data. Changing materials does not require constructing or rebinding
  per-material descriptor sets.
- **Root arguments instead of binding tables.** Each draw or dispatch receives a GPU pointer to a structure
  containing GPU pointers, texture indices, and constants. The CPU and shader share its declaration;
  there are no descriptor-set layouts or pipeline layouts to keep in agreement.
- **Less pipeline-state coupling.** Resource-binding and vertex layouts are absent from pipeline
  creation. Viewport, scissor, and depth/stencil state are set independently, reducing pipeline
  permutations. Rasterization, blending, and attachment formats still belong to pipeline objects.
- **Barriers without resource lists.** Synchronization describes which work produces and consumes
  data, not a list of buffer and image transitions. Applications do not track image layouts.

This is a low-level library: the application still owns allocation policy, resource lifetime, and
GPU synchronization. The optional `NoGraphicsAPIUtility` module supplies shared shader types, math,
allocators, upload queues, and deferred deletion without making them part of the graphics API.

Initialization and resource creation return error unions (`gpu.Error`). Ordinary frame operations, including draws,
dispatches, and descriptor writes, return no errors; native results are checked with debug assertions.
Install `gpu.setErrorCallback` on the initialization thread for backend diagnostics, such as the failing Vulkan operation,
its result and the requested allocation size. The callback is synchronous, thread-local, and disabled by default.

### What a draw's data looks like

Declare the arguments once in a shared C/Slang header:

```c
struct RootArguments
{
    Vertex* vertices;
    Material* material;
    float4x4 transform;
    uint32 texture_index;
};
```

Zig reads the header through translate-c. Allocate a root from application-owned mapped storage, fill it through the CPU
address, then pass its GPU address:

```zig
const root = frame_data.allocate(shared.RootArguments, 1).?;
root.cpu.?[0] = .{
    .vertices = vertex_memory.gpu,
    .material = material_memory.gpu,
    .transform = transform,
    .texture_index = texture_index,
};
gpu.draw(commands, root.gpu, .{ .vertex_count = vertex_count });
```

With `<NoGraphicsAPI/shader.slang>`, the shader accesses the same data directly:

```slang
GPU_ROOT(RootArguments, root);
Vertex vertex = root.vertices[vertex_id];
Material material = *root.material;
Texture2D<float4> texture = gpu_texture<Texture2D<float4>>(root.texture_index);
```

Draws and dispatches take GPU root pointers directly, including roots written by earlier GPU work. All graphics stages share the same root.
Keep each root allocation alive and stable until its GPU use completes.
See the [design comparison](docs/no-graphics-api-comparison.md) for the remaining differences and
the [shader guide](docs/slang.md) for complete examples.

## Native implementations

Metal 4 supplies native GPU addresses, draw/dispatch commands that accept those addresses, and
`MTLTextureViewPool` for indexed textures. Vulkan supplies the equivalent model through device-address
commands and descriptor heaps. See [Metal implementation](docs/metal-support.md) and
[Vulkan implementation](docs/vulkan-support.md) for how these map to NoGraphicsAPI.

## Hardware requirements

### Metal 4

Requires macOS 26+ on an Apple silicon Mac (M1 and newer), which provides Apple GPU family 7 or newer with Metal 4.
iPhone and iPad hardware meets the same GPU baseline, but iOS and iPadOS are not build targets.
See [Apple's supported devices](https://support.apple.com/en-us/102894).

Direct task/mesh draws work throughout this baseline; indirect mesh draws require M3 or newer.
BC texture compression is a separate capability. See [Metal implementation](docs/metal-support.md)
for these limits and [Metal validation](docs/metal-validation.md) for known driver and tool issues.
Intel Macs, Simulator, tvOS and visionOS are not supported targets.

### Vulkan 1.4

The Vulkan backend targets little-endian x86-64; x86-64 builds enable AVX2 and FMA.
Vulkan 1.4 alone is insufficient. Required extensions include:

- [`VK_EXT_descriptor_heap`][descriptor-heap] — application-owned descriptor heaps.
- [`VK_KHR_device_address_commands`][address-commands] — commands operating on GPU addresses.
- [`VK_EXT_mesh_shader`][mesh-shader] — task and mesh shaders.
- [`VK_KHR_shader_untyped_pointers`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_untyped_pointers.html) — descriptor-heap shader support.

[`VK_KHR_unified_image_layouts`][unified-layouts] is optional and makes the backend's common image layout efficient.
See the [Vulkan implementation](docs/vulkan-support.md) for the complete feature contract.

Missing required extensions are the main reason for unsupported GPUs below; the remaining
[requirements](docs/vulkan-support.md#vulkan-feature-surface) are more widely supported on recent GPUs.

The table preserves the driver reports checked on **5 September 2026**. These are compatibility
snapshots, not a live driver list. The checked Windows packages were
[AMD Adrenalin 26.9.1](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-9-1.html)
and [NVIDIA 616.64 WHQL](https://us.download.nvidia.com/Windows/616.64/616.64-win11-win10-release-notes.pdf).

| Architecture | Driver snapshot | Products | CPU-visible heap | Required extensions |
| --- | --- | --- | --- | --- |
| AMD RDNA 2 (dGPU) | Windows / Adrenalin 26.9.1 | [RX 6000][rdna2-rebar] | PCIe ReBAR or<br>🔴 [256 MiB fixed BAR][rdna2-fixed] | 🔴 Unsupported |
| AMD RDNA 2 (iGPU) | Windows / Adrenalin 26.9.1 | [600M](https://vulkan.gpuinfo.org/displayreport.php?id=47714) | UMA | 🔴 Unsupported |
| AMD RDNA 2 (iGPU) | Linux / Mesa RADV 26.2+ | [Steam Deck](https://vulkan.gpuinfo.org/displayreport.php?id=51189) | UMA | Supported |
| AMD RDNA 3 (dGPU) | Windows / Adrenalin 26.9.1 | [RX 7000](https://vulkan.gpuinfo.org/displayreport.php?id=51443) | PCIe ReBAR | Supported |
| AMD RDNA 3 (iGPU) | Windows / Adrenalin 26.9.1 | [700M](https://vulkan.gpuinfo.org/displayreport.php?id=49646) | UMA | Supported |
| AMD RDNA 4 (dGPU) | Windows / Adrenalin 26.9.1 | [RX 9000](https://vulkan.gpuinfo.org/displayreport.php?id=51293) | PCIe ReBAR | Supported |
| NVIDIA Turing | Windows / NVIDIA 616.64 | [GTX 16 series][gtx16] | 🔴 [256 MiB fixed BAR][turing-rebar] | Supported |
| NVIDIA Turing | Windows / NVIDIA 616.64 | [RTX 20 series][turing] | 🔴 [256 MiB fixed BAR][turing-rebar] | Supported |
| NVIDIA Ampere | Windows / NVIDIA 616.64 | [RTX 30 series](https://vulkan.gpuinfo.org/displayreport.php?id=51549) | PCIe ReBAR | Supported |
| NVIDIA Ada Lovelace | Windows / NVIDIA 616.64 | [RTX 40 series](https://vulkan.gpuinfo.org/displayreport.php?id=51469) | PCIe ReBAR | Supported |
| NVIDIA Blackwell | Windows / NVIDIA 616.64 | [RTX 50 series](https://vulkan.gpuinfo.org/displayreport.php?id=51573) | PCIe ReBAR | Supported |

🔴 marks missing extensions or a capacity-limited fixed BAR. Mapped heaps require coherent CPU-visible
GPU memory. Enable ReBAR where available on discrete GPUs; integrated GPUs use UMA. A fixed BAR can
still work, but limits mapped-heap capacity. Separate GPU-only allocations can use the remaining VRAM.
UMA heap sizes depend on system configuration.

The checked Windows RDNA 2 reports lack descriptor-heap support.
[Pascal / GTX 10](https://vulkan.gpuinfo.org/displayreport.php?id=51084) lacks the required extensions.
Intel Windows support was not verified; the checked
[Arc report](https://vulkan.gpuinfo.org/displayreport.php?id=51355) also lacks required extensions.
Mesa RADV and ANV [26.2+](https://docs.mesa3d.org/relnotes/26.2.0.html) expose the required extensions
on the reported Linux/SteamOS targets.

See [known driver issues](docs/known-driver-issues.md) for observed problems and workarounds.

## Quick start

Every platform needs [Zig 0.17](https://ziglang.org/download/) and the `slangc` compiler from
[Slang](https://github.com/shader-slang/slang/releases) on `PATH`. Then run every test and an example:

```sh
zig build test
zig build run_triangle
```

`zig build examples` installs the examples, their shaders and the cube texture into `zig-out/bin`; run them from that directory.
See [building and integration](docs/building.md) for individual test steps and using the library as a dependency.

### Linux

Install the Vulkan loader, libX11 (for the examples), Slang 2026.14.1+, SPIRV-Tools 2026.3+ (`spirv-val`), and a driver
meeting the hardware requirements above, such as Mesa 26.2+. Debug builds enable the Khronos validation layer when it is installed.

### Windows

1. Install the [Vulkan SDK 1.4.357+](https://vulkan.lunarg.com/sdk/home) and make sure `VULKAN_SDK` names its directory.
   Put the SDK's `Bin` directory, containing `spirv-val.exe` (SPIRV-Tools 2026.3+), on `PATH`.
2. Use Slang 2026.14.1+ from the Vulkan SDK, or put the `bin` directory of a standalone Windows x64 release on `PATH`.
3. Install a GPU driver meeting the hardware requirements above. The Vulkan SDK does not replace the GPU driver.

Builds target the MSVC ABI, so install the Windows SDK, for example with Visual Studio's C++ workload.

```powershell
zig build test
zig build run_cube
```

### macOS

Install Xcode 26+ with its Metal compiler (`xcrun metal`) and Slang 2026.18.2+. Shaders compile to metallib;
`zig build test` and `zig build run_triangle` work as on the other platforms.

## Examples

- [Triangle](examples/triangle/triangle.zig) — the smallest rendering example.
- [Cube](examples/cube/cube.zig) — GPU-pointer vertex fetch, bindless textures and `comath` transforms.
- [Deferred renderer](examples/deferred_renderer/deferred_renderer.zig) — compute simulation and mesh-shader rendering.

Their windows use [Win32](examples/common/window_win32.zig), [X11](examples/common/window_x11.zig) or
[AppKit](examples/common/window_cocoa.zig).

## Documentation

- [Comparison with *No Graphics API*](docs/no-graphics-api-comparison.md)
- [Vulkan implementation](docs/vulkan-support.md)
- [Metal 4 implementation](docs/metal-support.md)
- [Slang shaders and root ABI](docs/slang.md)
- [Public API](src/NoGraphicsAPI.zig)
- [Metal validation and limitations](docs/metal-validation.md)
- [Known driver issues](docs/known-driver-issues.md)
- [Building and integration](docs/building.md)

## License

zig-nographicsapi uses the [MIT License](LICENSE), retaining the copyright of Sebastian Aaltonen's NoGraphicsAPI alongside
that of the Zig port. See [third-party notices](THIRD_PARTY_NOTICES.md) for bundled assets and dependencies.

[descriptor-heap]: https://www.khronos.org/blog/vulkan-introduces-roadmap-2026-and-new-descriptor-heap-extension
[mesh-shader]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_mesh_shader.html
[address-commands]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_device_address_commands.html
[unified-layouts]: https://www.khronos.org/blog/so-long-image-layouts-simplifying-vulkan-synchronisation
[rdna2-rebar]: https://vulkan.gpuinfo.org/displayreport.php?id=42800
[rdna2-fixed]: https://vulkan.gpuinfo.org/displayreport.php?id=48951
[gtx16]: https://vulkan.gpuinfo.org/displayreport.php?id=51563
[turing]: https://vulkan.gpuinfo.org/displayreport.php?id=51475
[turing-rebar]: https://www.nvidia.com/en-us/geforce/graphics-cards/compare/?section=compare-specs
