# Building and integration

See the README for [quick starts](../README.md#quick-start) and [hardware requirements](../README.md#hardware-requirements).
The build selects the native Metal 4 backend on macOS and Vulkan elsewhere.

| Target | Backend and windowing | Toolchain |
| --- | --- | --- |
| Windows x86-64 | Vulkan, Win32 | MSVC ABI; Vulkan SDK 1.4.357+ named by `VULKAN_SDK`. |
| Linux x86-64 | Vulkan, X11 | Vulkan loader; libX11 for the examples. |
| macOS 26+ ARM64 | Metal 4, AppKit | Xcode 26+ with the Metal compiler. |

MinGW, 32-bit targets, iOS and non-Apple ARM targets are unsupported. x86-64 builds enable AVX2 and FMA.

## Build steps

Zig 0.17 builds the tests and examples from `build.zig`:

```sh
zig build test              # every test
zig build test_math         # one test; each test has a step of its own name
zig build examples          # install the examples, shaders and texture into zig-out/bin
zig build run_cube          # run one example: run_triangle, run_cube, run_deferred_renderer
zig build check             # compile the examples without linking them
```

The test steps are `test_math`, `test_allocators`, `test_device`, `test_command_context`, `test_multithreading`,
`test_texture_copy`, `test_texture_alias`, `test_placed_texture`, `test_upload_queue`, `test_upload_queue_compute`,
`test_root_data`, `test_shader_abi`, `test_task_shader`, `test_render_continuation`, `test_queue_families`, and the
Vulkan-only `test_startup_failures`. A GPU test is skipped when no suitable device is found.

Shader tests and examples need `slangc` on `PATH`: Slang 2026.14.1+ with `spirv-val` from SPIRV-Tools 2026.3+ for Vulkan,
or Slang 2026.18.2+ with `xcrun metal` for Metal. They load shaders from `shaders/` relative to the working directory:
`zig build` sets it for tests and run steps, and installed examples run from `zig-out/bin`. Shader binaries are named with
`NoGraphicsAPI.shader_extension` (`spv` or `metallib`).

Debug builds enable the Khronos validation layer when it is installed, and Vulkan validation errors fail the tests.
Metal API validation is opt-in through `MTL_DEBUG_LAYER=1`; see [Metal validation](metal-validation.md).

## Using the library

Add the package to `build.zig.zon`, for example with `zig fetch --save <url>`, then import its modules:

```zig
const nographicsapi = b.dependency("zig_nographicsapi", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("NoGraphicsAPI", nographicsapi.module("NoGraphicsAPI"));
exe.root_module.addImport("NoGraphicsAPIUtility", nographicsapi.module("NoGraphicsAPIUtility"));
```

`NoGraphicsAPI` is the core API ([`src/NoGraphicsAPI.zig`](../src/NoGraphicsAPI.zig)) with the native backend.
`NoGraphicsAPIUtility` is optional and provides `BumpAllocator`, `HeapAllocator`, `RangeAllocator`, `TextureAllocator`,
`DeleteQueue`, `UploadQueue` and `math`. [Cube](../examples/cube/cube.zig) shows typical use.

`math` works on the vector, matrix and quaternion structures of
[`shader_types.h`](../utility/include/NoGraphicsAPIUtility/shader_types.h), imported through translate-c so their layouts
match what Slang sees. `math.context` lets [comath](../deps/NOTES.md) evaluate expressions such as
`comath.eval("projection * view * position", math.context, inputs)`.

## Shared headers

Root structures and other CPU/GPU data live in C headers included by both Slang and Zig. Zig reads them with
`b.addTranslateC`, adding the package's `include` and `utility/include` directories to the include path. Shared headers
name structures without the `struct` keyword, as Slang does, so the build declares a `typedef` for each structure before
including the header. [`tools/share_shader_types.zig`](../tools/share_shader_types.zig) then makes the translated module
reuse the `shader_types` module's vector and matrix types, so `math` results assign directly into roots.
Headers that C cannot parse, such as those with C++ `enum class`, are mirrored by hand with layout checks.

Compile shaders in the build as described in [Slang compilation](slang.md#build-and-stage-artifacts); the library performs
no runtime shader compilation.
