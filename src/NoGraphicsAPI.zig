//! Zig API for NoGraphicsAPI: a thin wrapper that exposes GPU memory as pointers, textures through descriptor heaps,
//! and shader inputs through one 16-byte-aligned root pointer. GPU addresses are typed pointers into GPU-visible
//! memory: shaders dereference them and the CPU only passes them on.
//!
//! Lifetime and threading:
//! - Resource destruction is immediate. Destroy resources only when no recorded or executing GPU frame uses them;
//!   NoGraphicsAPIUtility's DeleteQueue can defer destruction. Wait for submitted frames to drain before `destroyDevice`.
//! - Distinct resource creation/destruction, immutable queries, and timeline waits may run concurrently on one device.
//!   Resource lifetime changes must be synchronized with every CPU/GPU use of that resource.
//! - Concurrent descriptor updates require disjoint destinations; copy source slots must not be modified concurrently.
//! - Each (device, queue index) and each command pool, including recording its buffers, is externally synchronized.
//!   Device idle and destruction require exclusive access. Destroy command pools before their device.
//! - Initialization and resource creation failures return an `Error`; details go to the optional error callback.
//!   Programming errors are asserted, and fatal execution errors abort after reporting.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

const backend = if (builtin.os.tag.isDarwin()) @import("metal.zig") else @import("vulkan.zig");

/// File extension of the shader binaries the backend consumes: SPIR-V for Vulkan, metallib for Metal.
pub const shader_extension = if (builtin.os.tag.isDarwin()) "metallib" else "spv";

pub const Device = backend.Device;
pub const CommandPool = backend.CommandPool;
pub const CommandBuffer = backend.CommandBuffer;
pub const Texture = backend.Texture;
pub const RenderView = backend.RenderView;
pub const Pso = backend.Pso;
pub const TimelineSemaphore = backend.TimelineSemaphore;
pub const GpuHeapOwner = backend.GpuHeapOwner;
pub const TextureHeapOwner = backend.TextureHeapOwner;
pub const TextureDescriptorHeap = backend.TextureDescriptorHeap;
pub const SamplerDescriptorHeap = backend.SamplerDescriptorHeap;

pub const Error = error{ Unsupported, DeviceLost, DriverError, OutOfMemory };

/// Synchronous diagnostics on the calling thread; the message is valid only during the callback. Do not call graphics
/// APIs recursively. The thread-local callback defaults to null.
pub const ErrorCallback = *const fn (message: [:0]const u8, fatal: bool) void;

pub const Uint32x2 = extern struct {
    x: u32 = 0,
    y: u32 = 0,
};

pub const Uint32x3 = extern struct {
    x: u32 = 0,
    y: u32 = 0,
    z: u32 = 0,
};

pub const MemoryType = enum(u8) { cpu_visible, gpu_only, readback };

pub const SizeAlign = struct {
    size: u64 = 0,
    alignment: u64 = 0,
};

pub const GpuRange = extern struct {
    gpu: ?*anyopaque = null,
    size: u64 = 0,
};

/// A CPU and GPU view of the same memory. Either address may be null; `size` is in bytes, independent of `T`.
pub fn GpuCpuRange(comptime T: type) type {
    return extern struct {
        cpu: ?[*]T = null,
        gpu: ?[*]T = null,
        size: u64 = 0,

        const Self = @This();

        pub fn gpuRange(self: Self) GpuRange {
            return .{ .gpu = self.gpu, .size = self.size };
        }

        /// Returns `size` bytes starting `offset` bytes into this range, as `U`. Null addresses stay null.
        pub fn subrange(self: Self, comptime U: type, offset: u64, size: u64) GpuCpuRange(U) {
            assert(offset + size <= self.size);
            return .{ .cpu = offsetPointer(U, self.cpu, offset), .gpu = offsetPointer(U, self.gpu, offset), .size = size };
        }
    };
}

fn offsetPointer(comptime U: type, pointer: anytype, offset: u64) ?[*]U {
    const base = pointer orelse return null;
    return @ptrCast(@alignCast(@as([*]u8, @ptrCast(base)) + offset));
}

/// Copies alias the same allocation. Pass one unchanged copy to `destroyGpuHeap` exactly once.
pub const GpuHeap = struct {
    range: GpuCpuRange(u8),
    owner: *GpuHeapOwner,

    pub fn gpuRange(self: GpuHeap) GpuRange {
        return self.range.gpuRange();
    }
};

/// Copies alias the same allocation. Pass one unchanged copy to `destroyTextureHeap` exactly once.
pub const TextureHeap = struct {
    size: u64,
    owner: *TextureHeapOwner,
};

pub const TimelinePoint = struct {
    semaphore: *TimelineSemaphore,
    value: u64,
};

pub const SubmitDesc = struct {
    commands: []const *CommandBuffer = &.{},
    /// GPU waits cover all command stages.
    waits: []const TimelinePoint = &.{},
    completion: TimelinePoint,
};

pub const Format = enum(u8) {
    r8_srgb,
    rg8_srgb,
    rgba8_srgb,
    bgra8_srgb,

    rgba4_unorm,
    r5g5b5a1_unorm,
    r5g6b5_unorm,

    r8_unorm,
    rg8_unorm,
    rgba8_unorm,
    bgra8_unorm,
    r16_unorm,
    rg16_unorm,
    rgba16_unorm,

    r8_uint,
    rg8_uint,
    rgba8_uint,
    bgra8_uint,
    r16_uint,
    rg16_uint,
    rgba16_uint,
    r32_uint,
    rg32_uint,
    rgba32_uint,

    r16_float,
    rg16_float,
    rgba16_float,
    r32_float,
    rg32_float,
    rgba32_float,

    rgb10a2_unorm,
    rg11b10_float,

    d16_unorm,
    d24_unorm_s8_uint,
    d32_float,
    s8_uint,
    d32_float_s8_uint,

    eac_rg,
    astc_4x4_srgb,
    astc_4x4_unorm,
    bc3_srgb,
    bc3_unorm,
    bc5_rg,
    bc6h_ufloat,
    bc6h_sfloat,
    bc7_srgb,
    bc7_unorm,

    /// Must remain last; preceding values are concrete texture formats.
    undefined,

    pub const count = @backingInt(Format.undefined);
};

pub const TextureFormatInfo = struct {
    block_extent: Uint32x2 = .{},
    bytes_per_block: u32 = 0,
    depth: bool = false,
    stencil: bool = false,
};

pub fn getTextureFormatInfo(format: Format) TextureFormatInfo {
    const texel: Uint32x2 = .{ .x = 1, .y = 1 };
    return switch (format) {
        .r8_srgb, .r8_unorm, .r8_uint => .{ .block_extent = texel, .bytes_per_block = 1 },
        .s8_uint => .{ .block_extent = texel, .bytes_per_block = 1, .stencil = true },
        .rg8_srgb, .rgba4_unorm, .r5g5b5a1_unorm, .r5g6b5_unorm, .rg8_unorm, .r16_unorm, .rg8_uint, .r16_uint, .r16_float => .{
            .block_extent = texel,
            .bytes_per_block = 2,
        },
        .d16_unorm => .{ .block_extent = texel, .bytes_per_block = 2, .depth = true },
        .rgba8_srgb, .bgra8_srgb, .rgba8_unorm, .bgra8_unorm, .rg16_unorm, .rgba8_uint, .bgra8_uint, .rg16_uint, .r32_uint, .rg16_float => .{
            .block_extent = texel,
            .bytes_per_block = 4,
        },
        .r32_float, .rgb10a2_unorm, .rg11b10_float => .{ .block_extent = texel, .bytes_per_block = 4 },
        .d24_unorm_s8_uint => .{ .block_extent = texel, .bytes_per_block = 4, .depth = true, .stencil = true },
        .d32_float => .{ .block_extent = texel, .bytes_per_block = 4, .depth = true },
        .rgba16_unorm, .rgba16_uint, .rg32_uint, .rgba16_float, .rg32_float => .{ .block_extent = texel, .bytes_per_block = 8 },
        .d32_float_s8_uint => .{ .block_extent = texel, .bytes_per_block = 8, .depth = true, .stencil = true },
        .rgba32_uint, .rgba32_float => .{ .block_extent = texel, .bytes_per_block = 16 },
        .eac_rg, .astc_4x4_srgb, .astc_4x4_unorm, .bc3_srgb, .bc3_unorm, .bc5_rg, .bc6h_ufloat, .bc6h_sfloat, .bc7_srgb, .bc7_unorm => .{
            .block_extent = .{ .x = 4, .y = 4 },
            .bytes_per_block = 16,
        },
        .undefined => .{},
    };
}

pub const TextureType = enum(u8) { one_d, two_d, three_d, cube, two_d_array, cube_array };

pub const TextureUsage = packed struct(u32) {
    sampled: bool = false,
    storage: bool = false,
    color_attachment: bool = false,
    depth_stencil_attachment: bool = false,
    transfer_source: bool = false,
    transfer_destination: bool = false,
    _: u26 = 0,
};

pub const TextureDescriptorType = enum(u8) { sampled, storage };

pub const TextureAspect = enum(u8) { automatic, color, depth, stencil };

pub const Filter = enum(u8) { nearest, linear };

pub const AddressMode = enum(u8) { repeat, mirrored_repeat, clamp_to_edge };

pub const CompareOp = enum(u8) { never, less, equal, less_equal, greater, not_equal, greater_equal, always };

pub const CullMode = enum(u8) { none, clockwise, counter_clockwise };

pub const BlendFactor = enum(u8) {
    zero,
    one,
    source_color,
    one_minus_source_color,
    destination_color,
    one_minus_destination_color,
    source_alpha,
    one_minus_source_alpha,
    destination_alpha,
    one_minus_destination_alpha,
    source_alpha_saturate,
};

pub const BlendOp = enum(u8) { add, subtract, reverse_subtract, minimum, maximum };

pub const IndexType = enum(u8) { uint16, uint32 };

pub const LoadOp = enum(u8) { load, clear, discard };

pub const StoreOp = enum(u8) { store, discard };

pub const StencilOp = enum(u8) { keep, zero, replace, increment_clamp, decrement_clamp, invert, increment_wrap, decrement_wrap };

/// Ordered by Vulkan's logical execution order where stages are comparable. A barrier's before execution scope includes
/// the selected and logically earlier stages; its after scope includes the selected and logically later stages. Vertex
/// and task/mesh are alternative graphics branches, depth_stencil_tests spans early and late tests around fragment,
/// compute and transfer are separate pipelines, host is a pseudo-stage, and all_commands is a special mask.
pub const Stage = packed struct(u64) {
    indirect: bool = false,
    index_input: bool = false,
    vertex: bool = false,
    task: bool = false,
    mesh: bool = false,
    depth_stencil_tests: bool = false,
    fragment: bool = false,
    color_output: bool = false,
    compute: bool = false,
    transfer: bool = false,
    /// Barrier destination only, paired with `host_read`.
    host: bool = false,
    /// All GPU command stages; excludes host.
    all_commands: bool = false,
    _: u52 = 0,
};

pub const Access = packed struct(u64) {
    transfer_read: bool = false,
    transfer_write: bool = false,
    shader_read: bool = false,
    shader_write: bool = false,
    color_read: bool = false,
    color_write: bool = false,
    depth_stencil_read: bool = false,
    depth_stencil_write: bool = false,
    indirect_read: bool = false,
    index_read: bool = false,
    host_read: bool = false,
    descriptor_read: bool = false,
    _: u52 = 0,
};

pub const DeviceCaps = struct {
    device_name: [*:0]const u8 = "",
    /// Flat queue indices: general first, then compute-only, then copy-only. Queue zero supports presentation.
    queue_count: u32 = 0,
    general_queue_count: u32 = 0,
    compute_queue_count: u32 = 0,
    copy_queue_count: u32 = 0,
    /// Copy-only texture offsets/extents align to these texel-block counts, except at mip edges. Zero means whole mips.
    copy_texture_granularity: Uint32x3 = .{ .x = 1, .y = 1, .z = 1 },
    max_push_data_size: u64 = 0,
    /// Common element size for suballocating TextureHeap storage; every `SizeAlign.alignment` divides this value.
    texture_heap_alignment: u64 = 0,
    /// Nanoseconds per timestamp tick.
    timestamp_period_ns: f32 = 0,
    /// Fractional filtering precision, for conservative sampled-field bounds.
    sub_texel_precision_bits: u32 = 0,
    texture_compression_bc: bool = false,
    texture_compression_astc: bool = false,
    storage_input_output16: bool = false,
    indirect_mesh_draw: bool = false,
};

pub const ColorSpace = enum(u32) { srgb, extended_srgb_linear };

/// Windowed device lifecycle, swapchain format/extent changes, acquire, and presentation stay on the window's
/// message-pump thread. The window must outlive the device.
pub const DeviceDesc = struct {
    /// HWND on Windows, the X11 `Window` id on Linux (`@ptrFromInt`), or a `CAMetalLayer*` on Apple platforms. Null creates a
    /// headless device.
    window: ?*anyopaque = null,
    /// X11 `Display*` on Linux; required with a window there. Unused elsewhere.
    display: ?*anyopaque = null,
    swapchain_format: Format = .undefined,
    swapchain_color_space: ColorSpace = .srgb,
    /// 1..8 presentation contexts.
    desired_swapchain_image_count: u32 = 2,
    /// Counts are capped to each family's capacity. A nonzero request requires that kind of queue to be available.
    /// General graphics + compute queues; must be nonzero.
    desired_queue_count: u32 = 1,
    desired_compute_queue_count: u32 = 0,
    desired_copy_queue_count: u32 = 0,
    /// Per command buffer; zero disables timestamps.
    timestamp_query_count: u32 = 256,
};

pub const SwapchainFrame = struct {
    render_view: *RenderView,
    extent: Uint32x2,
};

pub const TextureDesc = struct {
    type: TextureType = .two_d,
    extent: Uint32x3 = .{ .x = 1, .y = 1, .z = 1 },
    mip_levels: u32 = 1,
    /// Array layers; cube faces are individual layers.
    layer_count: u32 = 1,
    format: Format = .rgba8_unorm,
    /// Allow format-compatible descriptor views, but could lose DCC.
    mutable_format: bool = false,
    usage: TextureUsage = .{ .sampled = true },
    /// Overlapping placements require activation before each period of use; contents are discarded.
    aliasable: bool = false,
};

pub const RenderViewDesc = struct {
    mip_level: u32 = 0,
    /// Physical array slice; cube faces are individual slices.
    slice: u32 = 0,
};

pub const TextureDescriptorDesc = struct {
    /// Undefined inherits the texture format.
    format: Format = .undefined,
    /// Automatic selects color, or depth before stencil.
    aspect: TextureAspect = .automatic,
    base_mip: u32 = 0,
    /// Zero selects every remaining mip level.
    mip_count: u32 = 0,
    base_layer: u32 = 0,
    /// Zero selects every remaining layer.
    layer_count: u32 = 0,
};

pub const TextureCopyDesc = struct {
    mip_level: u32 = 0,
    /// Physical array slice; cube faces are individual slices.
    base_slice: u32 = 0,
    /// Zero selects every remaining physical slice.
    slice_count: u32 = 0,
    offset: Uint32x3 = .{},
    /// Zero components select the remaining mip extent.
    extent: Uint32x3 = .{},
    /// Zero is tightly packed.
    row_pitch_bytes: u64 = 0,
    /// Zero is tightly packed.
    slice_pitch_bytes: u64 = 0,
};

pub const SamplerDesc = struct {
    min_filter: Filter = .linear,
    mag_filter: Filter = .linear,
    mip_filter: Filter = .linear,
    address_u: AddressMode = .repeat,
    address_v: AddressMode = .repeat,
    address_w: AddressMode = .repeat,
    /// Uses the API's fixed 4x profile.
    anisotropic: bool = false,
    compare_enabled: bool = false,
    compare: CompareOp = .less_equal,
};

pub const BlendComponentState = struct {
    source: BlendFactor = .one,
    destination: BlendFactor = .zero,
    operation: BlendOp = .add,
};

pub const BlendState = struct {
    enabled: bool = false,
    color: BlendComponentState = .{},
    alpha: BlendComponentState = .{},
};

pub const ColorTargetDesc = struct {
    format: Format = .undefined,
    blend: BlendState = .{},
    write_mask: u8 = 0xf,
};

pub const RasterizationState = struct {
    cull: CullMode = .none,
    depth_bias_constant: f32 = 0,
    depth_bias_clamp: f32 = 0,
    depth_bias_slope: f32 = 0,
};

pub const Viewport = struct {
    x: f32 = 0,
    y: f32 = 0,
    width: f32 = 1,
    height: f32 = 1,
    min_depth: f32 = 0,
    max_depth: f32 = 1,
};

pub const Scissor = struct {
    x: i32 = 0,
    y: i32 = 0,
    width: u32 = 1,
    height: u32 = 1,
};

pub const StencilFaceState = struct {
    compare: CompareOp = .always,
    fail: StencilOp = .keep,
    pass: StencilOp = .keep,
    depth_fail: StencilOp = .keep,
    reference: u8 = 0,
};

pub const DepthStencilState = struct {
    depth_test: bool = false,
    depth_write: bool = false,
    depth_compare: CompareOp = .less_equal,
    stencil_test: bool = false,
    stencil_read_mask: u8 = 0xff,
    stencil_write_mask: u8 = 0xff,
    front: StencilFaceState = .{},
    back: StencilFaceState = .{},
};

pub const ShaderStage = struct {
    /// Precompiled SPIR-V (4-byte-aligned words) or metallib, per `shader_extension`. Empty omits an optional stage.
    code: []const u8 = &.{},
    entry_point: [*:0]const u8 = "main",
    /// Metal compute/task/mesh dimensions must match the compiled numthreads. Vulkan uses the SPIR-V metadata.
    threadgroup_size: Uint32x3 = .{ .x = 1, .y = 1, .z = 1 },
};

pub const GraphicsPsoDesc = struct {
    vertex: ShaderStage = .{},
    /// Empty omits the fragment stage, for depth-only rasterization.
    fragment: ShaderStage = .{},
    color_targets: []const ColorTargetDesc = &.{},
    depth_format: Format = .undefined,
    stencil_format: Format = .undefined,
    rasterization: RasterizationState = .{},
};

pub const MeshPsoDesc = struct {
    /// Empty launches mesh workgroups directly; otherwise draws launch task workgroups.
    task: ShaderStage = .{},
    mesh: ShaderStage = .{},
    /// Empty omits the fragment stage, for depth-only rasterization.
    fragment: ShaderStage = .{},
    color_targets: []const ColorTargetDesc = &.{},
    depth_format: Format = .undefined,
    stencil_format: Format = .undefined,
    rasterization: RasterizationState = .{},
};

pub const ClearColor = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 1,
};

pub const ColorAttachment = struct {
    render_view: *RenderView,
    load: LoadOp = .load,
    store: StoreOp = .store,
    clear: ClearColor = .{},
};

pub const DepthAttachment = struct {
    render_view: ?*RenderView = null,
    load: LoadOp = .load,
    store: StoreOp = .store,
    clear: f32 = 1,
};

pub const StencilAttachment = struct {
    render_view: ?*RenderView = null,
    load: LoadOp = .load,
    store: StoreOp = .store,
    clear: u8 = 0,
};

/// Rendering and raster PSOs accept at most eight color attachments. A render pass needs at least one attachment to infer
/// its area.
pub const RenderingDesc = struct {
    colors: []const ColorAttachment = &.{},
    depth: DepthAttachment = .{},
    stencil: StencilAttachment = .{},
};

pub const RenderingFlags = packed struct(u32) {
    _reserved: bool = false,
    suspending: bool = false,
    resuming: bool = false,
    _: u29 = 0,
};

pub const Draw = struct {
    vertex_count: u32,
    instance_count: u32 = 1,
    first_vertex: u32 = 0,
    first_instance: u32 = 0,
};

pub const DrawIndexed = struct {
    indices: GpuRange,
    index_type: IndexType,
    index_count: u32,
    instance_count: u32 = 1,
    first_index: u32 = 0,
    vertex_offset: i32 = 0,
    first_instance: u32 = 0,
};

/// `stride` zero selects the tightly packed argument size.
pub const DrawIndirect = struct {
    arguments: GpuRange,
    draw_count: u32 = 1,
    stride: u32 = 0,
};

pub const DrawIndexedIndirect = struct {
    indices: GpuRange,
    index_type: IndexType,
    arguments: GpuRange,
    draw_count: u32 = 1,
    stride: u32 = 0,
};

// Device.

pub const setErrorCallback = backend.setErrorCallback;
pub const createDevice = backend.createDevice;
pub const destroyDevice = backend.destroyDevice;
pub const getDeviceCaps = backend.getDeviceCaps;
pub const supportsTextureFormat = backend.supportsTextureFormat;
pub const getDrawableExtent = backend.getDrawableExtent;
/// Call after waitIdle, outside an acquired frame. Unsupported pairs leave the current mode unchanged. A zero-size
/// drawable defers swapchain recreation until acquire sees a nonzero extent.
pub const setSwapchainFormat = backend.setSwapchainFormat;

// Synchronization.

pub const createTimelineSemaphore = backend.createTimelineSemaphore;
pub const destroyTimelineSemaphore = backend.destroyTimelineSemaphore;
pub const timelineCompletedValue = backend.timelineCompletedValue;
pub const waitTimeline = backend.waitTimeline;
pub const waitIdle = backend.waitIdle;

// Presentation. Acquire outside a render pass; later buffers in the same presentation submission may also access the
// returned image. Null while the drawable extent is zero. A frame must be submitted with submitAndPresent on queue zero.

pub const acquire = backend.acquire;
pub const submitAndPresent = backend.submitAndPresent;

// Memory. Every non-null returned pointer is 16-byte aligned. GPU heaps are raw blocks for application-side suballocation.

pub const createGpuHeap = backend.createGpuHeap;
pub const destroyGpuHeap = backend.destroyGpuHeap;

// Textures. Texture heaps use one device-selected GPU-only memory type and must outlive every placed texture. Placements
// must satisfy getTextureSizeAlign. Overlap is allowed only between aliasable textures with disjoint periods of GPU use.
// Reusing a placement without alias activation requires completion of its last use and destruction of the old texture.
// createTexture records initialization into commands, outside a render pass; aliasable textures defer it to activation.
// Activate an alias outside a render pass before its first use and whenever another alias has used its storage; this
// discards contents and synchronizes prior uses on this queue.

pub const createTextureHeap = backend.createTextureHeap;
pub const destroyTextureHeap = backend.destroyTextureHeap;
pub const getTextureSizeAlign = backend.getTextureSizeAlign;
pub const createTexture = backend.createTexture;
pub const destroyTexture = backend.destroyTexture;
pub const activateTextureAlias = backend.activateTextureAlias;
pub const createRenderView = backend.createRenderView;
pub const destroyRenderView = backend.destroyRenderView;

// Descriptors. Slots form application-owned namespaces. Writes and copies require indices within capacity; do not modify
// slots until all commands using their previous contents have completed. Copies may overlap.

pub const createTextureDescriptorHeap = backend.createTextureDescriptorHeap;
pub const destroyTextureDescriptorHeap = backend.destroyTextureDescriptorHeap;
pub const writeTextureDescriptor = backend.writeTextureDescriptor;
pub const copyTextureDescriptors = backend.copyTextureDescriptors;
pub const createSamplerDescriptorHeap = backend.createSamplerDescriptorHeap;
pub const destroySamplerDescriptorHeap = backend.destroySamplerDescriptorHeap;
pub const writeSamplerDescriptor = backend.writeSamplerDescriptor;
pub const copySamplerDescriptors = backend.copySamplerDescriptors;

// Pipelines.

pub const createGraphicsPso = backend.createGraphicsPso;
pub const createMeshPso = backend.createMeshPso;
pub const createComputePso = backend.createComputePso;
pub const destroyPso = backend.destroyPso;

// Command recording. Pools retain command storage until destruction. Reset only after every submitted buffer from the pool
// completes; unsubmitted buffers are discarded and every returned handle is invalidated. Use one pool per worker and
// in-flight frame. Buffers must be submitted to their pool's queue family, each exactly once before pool reset.

pub const createCommandPool = backend.createCommandPool;
pub const destroyCommandPool = backend.destroyCommandPool;
pub const resetCommandPool = backend.resetCommandPool;
pub const beginCommands = backend.beginCommands;
pub const endCommands = backend.endCommands;
pub const submit = backend.submit;

pub const setTextureDescriptorHeap = backend.setTextureDescriptorHeap;
pub const setSamplerDescriptorHeap = backend.setSamplerDescriptorHeap;

pub const copyMemory = backend.copyMemory;
/// Depth/stencil copies require a general queue. Copy-only queues also require DeviceCaps.copy_texture_granularity alignment.
pub const copyMemoryToTexture = backend.copyMemoryToTexture;
pub const copyTextureToMemory = backend.copyTextureToMemory;

pub const barrier = backend.barrier;

/// Up to DeviceDesc.timestamp_query_count markers per command buffer; stage must map to one GPU pipeline stage. Ignored
/// when timestamps are disabled or the queue lacks profiling support. Destinations must stay valid and distinct until
/// readTimestamps, which retrieves results after every submission from the pool completes and before the pool resets.
pub const writeTimestamp = backend.writeTimestamp;
pub const readTimestamps = backend.readTimestamps;

/// Each suspend/resume segment needs matching attachments, load/store operations and clear values, and its own begin/end
/// pair. Submit the chain in order in one batch, with no action or synchronization commands between segments. Resuming
/// skips load/clear operations; suspending defers stores. beginRenderPass resets a full render-area viewport and scissor
/// and disables depth/stencil tests.
pub const beginRenderPass = backend.beginRenderPass;
pub const endRenderPass = backend.endRenderPass;
pub const setViewport = backend.setViewport;
pub const setScissor = backend.setScissor;
pub const setDepthStencil = backend.setDepthStencil;
pub const bindPso = backend.bindPso;

// Draws and dispatches. `root` is a 16-byte-aligned GPU address, or null for shaders without root data. Retain its
// storage through GPU completion; these commands do not copy root bytes. drawMeshletsIndirect requires
// DeviceCaps.indirect_mesh_draw.

pub const draw = backend.draw;
pub const drawIndexed = backend.drawIndexed;
pub const drawIndirect = backend.drawIndirect;
pub const drawIndexedIndirect = backend.drawIndexedIndirect;
pub const dispatch = backend.dispatch;
pub const dispatchIndirect = backend.dispatchIndirect;
pub const drawMeshlets = backend.drawMeshlets;
pub const drawMeshletsIndirect = backend.drawMeshletsIndirect;
