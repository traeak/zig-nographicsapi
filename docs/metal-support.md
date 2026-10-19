# Metal 4 implementation

Metal 4 implements the same GPU-pointer and descriptor-heap model as [Vulkan](vulkan-support.md).
The backend ([`src/metal.zig`](../src/metal.zig)) is native Zig: it sends typed `objc_msgSend` messages through the
Objective-C runtime, uses native Metal 4 commands, and consumes precompiled metallibs. Applications own allocation,
descriptor indices, resource lifetime and synchronization.
See [supported devices](../README.md#metal-4) for the hardware baseline.

**Status:** the backend compiles for aarch64-macos and its constants and selectors were checked against Apple's metal-cpp
headers, but it has not yet been run on a Mac.

## GPU pointers and address-based commands

`GpuHeap` exposes real 64-bit addresses from Metal buffers. Applications partition this storage and
put typed GPU pointers in shared CPU/shader structures. Shaders dereference them directly, including
vertex fetch, without buffer descriptors or per-resource binding calls. CPU-visible heaps use shared
storage; GPU-only heaps use private storage. Textures occupy application-managed placement heaps.

`TextureDesc.aliasable` permits resident textures to share a placement while their GPU uses are
disjoint. `activateTextureAlias` issues an all-command barrier with device visibility before
the incoming alias is cleared or overwritten; prior contents are discarded by contract. Views and
descriptors stay resident. Cross-queue use requires timeline waits. This uses ordinary placement
heap aliasing, without `makeAliasable()` or sparse-resource visibility operations.

Metal 4 also accepts GPU addresses in its command API:

| NoGraphicsAPI operation | Metal 4 implementation |
| --- | --- |
| Indexed drawing | `drawIndexedPrimitives` receives the index GPU address and byte length. |
| Indirect drawing | `drawPrimitives` / `drawIndexedPrimitives` receive the argument GPU address. |
| Indirect dispatch | `dispatchThreadgroupsWithIndirectBuffer` receives the argument GPU address. |
| Indirect mesh drawing | `drawMeshThreadgroupsWithIndirectBuffer` receives the argument GPU address. |

Despite the `Buffer` names, these Metal 4 arguments are addresses, not `MTLBuffer` objects. The backend
passes them through directly. Native copy commands still require buffer objects, so copies translate
addresses through an internal index of at most 64 live GPU heaps. Lookup uses atomic snapshots without
mutexes. Metal residency-set updates are serialized; ordinary command recording and submission are not.

## Application-owned descriptor heaps

Texture descriptor heaps use `MTLTextureViewPool`, which provides application-selected slots with contiguous
resource IDs. A shader selects a texture by forming a handle from **`baseResourceID + index`**.
There is no per-texture ID-table load. Sampler heaps use an indexed array of eight-byte resource IDs.

Metal does not expose writable texture-descriptor bytes. The common API therefore exposes opaque
texture and sampler heaps with indexed write/copy operations; Vulkan keeps its mapped descriptor
storage behind the same interface. Applications still own the slots and must retain their contents
and referenced textures until GPU use finishes.

Stock Slang has no texture-pool indexing operator. The shared shader header supplies the handle
reinterpretation, following the [MSL heap experiment](https://github.com/sebbbi/msl_heap).
`gpu_texture<T>(index)` and `gpu_sampler(index)` inline on both backends; no Slang fork is required.

## Root ABI and shared shaders

Each draw or dispatch takes an application-owned GPU root pointer and places that address in the
Metal argument table. All graphics stages share the root. The backend neither allocates root storage
nor copies its contents. Roots can be CPU-written through mapped memory or produced by GPU work.
See [root allocation, lifetime and synchronization](slang.md#root-and-pointer-layout).

| Metal buffer slot | Contents |
| --- | --- |
| 0 | Root arguments |
| 1 | Texture-view-pool base resource ID |
| 2 | Sampler resource IDs |

`GPU_ROOT` preserves the shared C layout instead of Metal's ordinary constant-buffer vector alignment.
The same Slang sources compile to metallib and SPIR-V. The backend mirrors viewport Y, so vertex and
mesh clip positions need no platform wrapper. See the [shader guide](slang.md) for layout and compilation.

## Mesh work and format support

Task shaders map to Metal's object stage. Direct task/mesh drawing works throughout the supported
baseline. Native indirect mesh drawing requires Apple9 (M3) or newer and is reported through
`DeviceCaps.indirect_mesh_draw`. M1/M2 applications can launch direct task groups that read GPU-produced
counts; NoGraphicsAPI does not emulate an indirect command.

Metal 4 does not imply BC texture compression support. Query the required texture formats before
choosing assets; ASTC is available throughout the baseline.

## Presentation

`DeviceDesc.window` is an application-owned `CAMetalLayer`; the example windows create one in an AppKit view.
Presentation currently supports only `.srgb`. `createDevice` and `setSwapchainFormat` return `error.Unsupported`
for `.extended_srgb_linear`; EDR layer configuration is not implemented. After `waitIdle`, with no acquired frame,
`setSwapchainFormat` accepts the current format or switches between `.bgra8_unorm` and `.bgra8_srgb` without replacing
application resources.

## Synchronization

Resource-free barriers map to Metal producer barriers. Fragment, depth and color destinations wait
at fragment/tile stages, allowing independent vertex, object and mesh work to overlap previous passes.
Compute and copy work use their own scopes. Write dependencies request device visibility;
execution-only dependencies do not flush caches.

Command pools map to reusable Metal command allocators. Pools and queues are externally synchronized;
independent pools can record concurrently. `MTLSharedEvent` supplies cross-queue waits and CPU completion.
Resource destruction and pool reset require completion of their submitted uses.

Timestamps use native counter heaps. After the existing completion wait, `readTimestamps(pool)`
retrieves results into CPU memory before pool reset. It adds no GPU resolve, barrier or wait, preserving
the opportunity for overlap between frames. Outside-render markers are approximate all-commands timings;
inside-render markers have a [known driver limitation](known-driver-issues.md#metal-4--macos-2662-render-timestamps-leave-counter-entries-unwritten).

## References

- [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
- [Texture view pools](https://developer.apple.com/documentation/metal/mtltextureviewpool)
- [Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Validation and limitations](metal-validation.md)
