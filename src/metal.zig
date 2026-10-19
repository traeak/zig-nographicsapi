//! Metal 4 backend for Apple GPU family 7 and newer. A GPU heap is a placement heap holding one buffer; textures are
//! placed in private heaps; texture descriptor heaps are texture view pools and sampler heaps are buffers of sampler
//! resource IDs. Every command buffer binds one argument table whose buffer slots hold the root pointer and the
//! texture and sampler heap addresses.

const std = @import("std");
const assert = std.debug.assert;
const api = @import("NoGraphicsAPI.zig");
const objc = @import("objc");

const Id = objc.Id;
const send = objc.send;
const release = objc.objc_release;
const Error = api.Error;
const Format = api.Format;

const gpa = std.heap.c_allocator;

const max_color_attachments = 8;
const max_gpu_heaps = 64;
/// Ring of sorted GPU heap snapshots; a reader stalled through a whole ring reuse is the accepted limitation.
const snapshot_ring_size = 4096;
const counter_page_size = 4096;
const max_counter_pages = 32;
const depth_chunk_size = 64;

const PixelFormat = enum(usize) {
    invalid = 0,
    r8_unorm = 10,
    r8_unorm_srgb = 11,
    r8_uint = 13,
    r16_unorm = 20,
    r16_uint = 23,
    r16_float = 25,
    rg8_unorm = 30,
    rg8_unorm_srgb = 31,
    rg8_uint = 33,
    b5g6r5_unorm = 40,
    a1bgr5_unorm = 41,
    abgr4_unorm = 42,
    r32_uint = 53,
    r32_float = 55,
    rg16_unorm = 60,
    rg16_uint = 63,
    rg16_float = 65,
    rgba8_unorm = 70,
    rgba8_unorm_srgb = 71,
    rgba8_uint = 73,
    bgra8_unorm = 80,
    bgra8_unorm_srgb = 81,
    rgb10a2_unorm = 90,
    rg11b10_float = 92,
    rg32_uint = 103,
    rg32_float = 105,
    rgba16_unorm = 110,
    rgba16_uint = 113,
    rgba16_float = 115,
    rgba32_uint = 123,
    rgba32_float = 125,
    bc3_rgba = 134,
    bc3_rgba_srgb = 135,
    bc5_rg_unorm = 142,
    bc6h_rgb_float = 150,
    bc6h_rgb_ufloat = 151,
    bc7_rgba_unorm = 152,
    bc7_rgba_unorm_srgb = 153,
    eac_rg11_unorm = 174,
    astc_4x4_srgb = 186,
    astc_4x4_ldr = 204,
    depth16_unorm = 250,
    depth32_float = 252,
    stencil8 = 253,
    depth32_float_stencil8 = 260,
    x32_stencil8 = 261,
    _,
};

const TextureType = enum(usize) { one_d = 0, two_d = 2, two_d_array = 3, cube = 5, cube_array = 6, three_d = 7, _ };

const texture_usage_shader_read: usize = 1;
const texture_usage_shader_write: usize = 2;
const texture_usage_render_target: usize = 4;
const texture_usage_pixel_format_view: usize = 0x10;

const storage_mode_private: usize = 2;
const hazard_tracking_untracked: usize = 1;
const resource_storage_shared: usize = 0;
const resource_storage_private: usize = storage_mode_private << 4;
const resource_cpu_cache_write_combined: usize = 1;
const resource_hazard_tracking_untracked: usize = hazard_tracking_untracked << 8;
const heap_type_placement: isize = 1;

const gpu_family_apple7: isize = 1007;
const gpu_family_apple9: isize = 1009;
const gpu_family_metal4: isize = 5002;

const stage_vertex: usize = 1 << 0;
const stage_fragment: usize = 1 << 1;
const stage_tile: usize = 1 << 2;
const stage_object: usize = 1 << 3;
const stage_mesh: usize = 1 << 4;
const stage_dispatch: usize = 1 << 27;
const stage_blit: usize = 1 << 28;
const stage_all: usize = std.math.maxInt(isize);

const render_stage_vertex: usize = 1 << 0;
const render_stage_fragment: usize = 1 << 1;
const render_stage_object: usize = 1 << 3;
const render_stage_mesh: usize = 1 << 4;
const render_stages = render_stage_vertex | render_stage_fragment | render_stage_object | render_stage_mesh;

const visibility_device: usize = 1;
const render_encoder_suspending: usize = 1;
const render_encoder_resuming: usize = 2;
const counter_heap_type_timestamp: isize = 1;
const timestamp_granularity_precise: isize = 1;
const primitive_topology_triangle: usize = 3;
const primitive_triangle: usize = 3;
const winding_counter_clockwise: usize = 1;
const dispatch_autorelease_frequency_work_item: c_ulong = 1;

const Size = extern struct { width: usize, height: usize, depth: usize };
const Origin = extern struct { x: usize, y: usize, z: usize };
const Range = extern struct { location: usize, length: usize };
const SizeAndAlign = extern struct { size: usize, alignment: usize };
const ResourceId = extern struct { impl: u64 };
const Viewport = extern struct { x: f64, y: f64, width: f64, height: f64, near: f64, far: f64 };
const ScissorRect = extern struct { x: usize, y: usize, width: usize, height: usize };
const ClearColor = extern struct { red: f64, green: f64, blue: f64, alpha: f64 };
const CGSize = extern struct { width: f64, height: f64 };

extern fn MTLCreateSystemDefaultDevice() ?Id;
extern "c" fn dispatch_queue_attr_make_with_autorelease_frequency(attr: ?*anyopaque, frequency: c_ulong) ?*anyopaque;
extern "c" fn dispatch_queue_create(label: [*:0]const u8, attr: ?*anyopaque) ?*anyopaque;
extern "c" fn dispatch_release(object: *anyopaque) void;
extern "c" fn dispatch_data_create(buffer: *const anyopaque, size: usize, queue: ?*anyopaque, destructor: ?*anyopaque) *anyopaque;

threadlocal var error_callback: ?api.ErrorCallback = null;

pub fn setErrorCallback(callback: ?api.ErrorCallback) void {
    error_callback = callback;
}

fn report(comptime format: []const u8, args: anytype) void {
    const callback = error_callback orelse return;
    var buffer: [1024]u8 = undefined;
    callback(std.mem.printSentinel(&buffer, format, args, 0) catch "graphics error message too long", false);
}

/// Reports a failed Metal creation call, with the NSError description when Metal provides one.
fn reportFailure(operation: []const u8, failure: ?Id) void {
    const pool = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(pool);
    const object = failure orelse return report("Metal {s} creation failed.", .{operation});
    report("Metal {s} creation failed: {s}", .{ operation, send([*:0]const u8, send(Id, object, "localizedDescription", .{}), "UTF8String", .{}) });
}

fn create(comptime T: type, value: T) *T {
    const pointer = gpa.create(T) catch @panic("out of memory");
    pointer.* = value;
    return pointer;
}

fn alignUp(value: u64, alignment: u64) u64 {
    return (value + alignment - 1) / alignment * alignment;
}

fn metalSize(size: api.Uint32x3) Size {
    return .{ .width = size.x, .height = size.y, .depth = size.z };
}

fn ringSlot(start: u32, offset: usize) usize {
    return (start +% @as(u32, @intCast(offset))) % snapshot_ring_size;
}

/// Maps API stages to Metal 4 barrier stages, preserving Vulkan's logical execution scopes without joining its alternative
/// graphics and compute branches.
fn barrierStages(stages: api.Stage, source: bool) usize {
    if (stages.all_commands) return stage_all;
    const raster = stage_fragment | stage_tile;
    const geometry = stage_vertex | stage_object | stage_mesh;
    var result: usize = 0;
    if (stages.indirect) result |= geometry | stage_dispatch | (if (source) 0 else raster);
    if (stages.index_input or stages.vertex) result |= stage_vertex | (if (source) 0 else raster);
    if (stages.task) result |= stage_object | (if (source) 0 else stage_mesh | raster);
    if (stages.mesh) result |= stage_mesh | (if (source) stage_object else raster);
    if (stages.depth_stencil_tests or stages.fragment or stages.color_output) result |= raster | (if (source) geometry else 0);
    if (stages.compute) result |= stage_dispatch;
    if (stages.transfer) result |= stage_blit;
    return result;
}

fn pixelFormat(format: Format) PixelFormat {
    return switch (format) {
        .r8_srgb => .r8_unorm_srgb,
        .rg8_srgb => .rg8_unorm_srgb,
        .rgba8_srgb => .rgba8_unorm_srgb,
        .bgra8_srgb => .bgra8_unorm_srgb,
        .rgba4_unorm => .abgr4_unorm,
        .r5g5b5a1_unorm => .a1bgr5_unorm,
        .r5g6b5_unorm => .b5g6r5_unorm,
        .r8_unorm => .r8_unorm,
        .rg8_unorm => .rg8_unorm,
        .rgba8_unorm => .rgba8_unorm,
        .bgra8_unorm => .bgra8_unorm,
        .r16_unorm => .r16_unorm,
        .rg16_unorm => .rg16_unorm,
        .rgba16_unorm => .rgba16_unorm,
        .r8_uint => .r8_uint,
        .rg8_uint => .rg8_uint,
        .rgba8_uint => .rgba8_uint,
        .r16_uint => .r16_uint,
        .rg16_uint => .rg16_uint,
        .rgba16_uint => .rgba16_uint,
        .r32_uint => .r32_uint,
        .rg32_uint => .rg32_uint,
        .rgba32_uint => .rgba32_uint,
        .r16_float => .r16_float,
        .rg16_float => .rg16_float,
        .rgba16_float => .rgba16_float,
        .r32_float => .r32_float,
        .rg32_float => .rg32_float,
        .rgba32_float => .rgba32_float,
        .rgb10a2_unorm => .rgb10a2_unorm,
        .rg11b10_float => .rg11b10_float,
        .d16_unorm => .depth16_unorm,
        .d32_float => .depth32_float,
        .s8_uint => .stencil8,
        .d32_float_s8_uint => .depth32_float_stencil8,
        .eac_rg => .eac_rg11_unorm,
        .astc_4x4_srgb => .astc_4x4_srgb,
        .astc_4x4_unorm => .astc_4x4_ldr,
        .bc3_srgb => .bc3_rgba_srgb,
        .bc3_unorm => .bc3_rgba,
        .bc5_rg => .bc5_rg_unorm,
        .bc6h_ufloat => .bc6h_rgb_ufloat,
        .bc6h_sfloat => .bc6h_rgb_float,
        .bc7_srgb => .bc7_rgba_unorm_srgb,
        .bc7_unorm => .bc7_rgba_unorm,
        .bgra8_uint, .d24_unorm_s8_uint, .undefined => .invalid,
    };
}

fn textureType(texture_type: api.TextureType) TextureType {
    return switch (texture_type) {
        .one_d => .one_d,
        .two_d => .two_d,
        .three_d => .three_d,
        .cube => .cube,
        .two_d_array => .two_d_array,
        .cube_array => .cube_array,
    };
}

/// The caller owns the returned MTLTextureDescriptor.
fn textureDescriptor(desc: api.TextureDesc) Id {
    const descriptor = objc.new("MTLTextureDescriptor");
    send(void, descriptor, "setTextureType:", .{textureType(desc.type)});
    send(void, descriptor, "setPixelFormat:", .{pixelFormat(desc.format)});
    send(void, descriptor, "setWidth:", .{@as(usize, desc.extent.x)});
    send(void, descriptor, "setHeight:", .{@as(usize, desc.extent.y)});
    send(void, descriptor, "setDepth:", .{@as(usize, desc.extent.z)});
    send(void, descriptor, "setMipmapLevelCount:", .{@as(usize, desc.mip_levels)});
    const array_length = switch (desc.type) {
        .cube_array => desc.layer_count / 6,
        .two_d_array => desc.layer_count,
        else => 1,
    };
    send(void, descriptor, "setArrayLength:", .{@as(usize, array_length)});
    send(void, descriptor, "setStorageMode:", .{storage_mode_private});
    send(void, descriptor, "setHazardTrackingMode:", .{hazard_tracking_untracked});
    var usage: usize = 0;
    if (desc.usage.sampled) usage |= texture_usage_shader_read;
    if (desc.usage.storage) usage |= texture_usage_shader_read | texture_usage_shader_write;
    if (desc.usage.color_attachment or desc.usage.depth_stencil_attachment) usage |= texture_usage_render_target;
    if (desc.mutable_format or api.getTextureFormatInfo(desc.format).stencil) usage |= texture_usage_pixel_format_view;
    send(void, descriptor, "setUsage:", .{usage});
    return descriptor;
}

const BufferRecord = struct {
    base: u64 = 0,
    size: u64 = 0,
    buffer: u64 = 0,
};

pub const GpuHeapOwner = struct {
    device: *Device,
    heap: Id,
    buffer: Id,
    slot: u6 = 0,
};

pub const TextureHeapOwner = struct {
    device: *Device,
    heap: Id,
};

pub const Texture = struct {
    device: *Device,
    texture: Id,
    desc: api.TextureDesc,
};

pub const RenderView = struct {
    texture: ?Id = null,
    mip: u32 = 0,
    slice: u32 = 0,
};

pub const TextureDescriptorHeap = struct {
    device: *Device,
    pool: Id,
    /// One-element buffer holding the pool's base resource ID; shaders index from it.
    base: Id,
    base_address: u64,
    base_id: u64,
    capacity: u32,
};

pub const SamplerDescriptorHeap = struct {
    device: *Device,
    buffer: Id,
    address: u64,
    ids: [*]ResourceId,
    /// Keeps each slot's sampler state alive.
    samplers: []?Id,
    capacity: u32,
};

pub const TimelineSemaphore = struct {
    device: *Device,
    event: Id,
};

pub const Pso = struct {
    device: *Device,
    render: ?Id = null,
    compute: ?Id = null,
    rasterization: api.RasterizationState = .{},
    threads: Size = .{ .width = 1, .height = 1, .depth = 1 },
    object_threads: Size = .{ .width = 1, .height = 1, .depth = 1 },
    mesh: bool = false,
};

const CounterPage = struct {
    heap: Id,
    next: ?*CounterPage = null,
    occupied: [counter_page_size / 64]u64 = @splat(0),
};

const TimestampSlot = struct {
    heap: ?Id = null,
    page: ?*CounterPage = null,
    destination: ?*u64 = null,
    index: u32 = 0,
};

/// Render pass suspension ends a native command buffer, so one API command buffer can span several native ones.
const NativeCommandBuffer = struct {
    buffer: ?Id,
    allocator: ?Id,
    next: ?*NativeCommandBuffer = null,
};

pub const CommandBuffer = struct {
    device: *Device,
    pool: *CommandPool,
    next: ?*CommandBuffer = null,
    /// The native buffer being recorded; null after a suspending render pass until the next command opens one.
    commands: ?Id = null,
    native_buffers: *NativeCommandBuffer,
    native: *NativeCommandBuffer,
    native_count: u32 = 0,
    arguments: ?Id,
    compute: ?Id = null,
    render: ?Id = null,
    pass: Id,
    timestamps: []TimestampSlot = &.{},
    timestamp_count: u32 = 0,
    retirement: ?Id = null,
    retirement_value: u64 = 0,
    recording: bool = false,
    ended: bool = false,
    render_continuation: bool = false,
    acquired: bool = false,
    pso: ?*const Pso = null,
};

const DepthStateEntry = struct {
    desc: api.DepthStencilState,
    state: ?Id,
};

const DepthStateChunk = struct {
    entries: [depth_chunk_size]DepthStateEntry = undefined,
    count: u32 = 0,
    next: ?*DepthStateChunk = null,
};

const QueueKind = enum(u8) { general, compute, copy };

const Queue = struct {
    queue: ?Id = null,
    feedback: ?*anyopaque = null,
    completion: ?Id = null,
    submitted_value: u64 = 0,
    submission: []?Id = &.{},
    kind: QueueKind = .general,
};

pub const CommandPool = struct {
    device: *Device,
    kind: QueueKind,
    first: ?*CommandBuffer = null,
    last: ?*CommandBuffer = null,
    next_buffer: ?*CommandBuffer = null,
    depth_states: ?*DepthStateChunk = null,
};

pub const Device = struct {
    metal: Id,
    queues: []Queue = &.{},
    compiler: ?Id = null,
    residency: ?Id = null,
    layer: ?Id = null,
    drawable: ?Id = null,
    acquired: ?*CommandBuffer = null,
    drawable_view: RenderView = .{},
    caps: api.DeviceCaps = .{},
    name: [255:0]u8 = @splat(0),
    residency_lock: std.c.os_unfair_lock = .{},
    /// GPU heap address ranges for resolving GPU pointers to buffers in copy commands. `buffer_snapshot` packs the count
    /// and ring start of the current sorted snapshot; writers publish new snapshots with compare-and-swap.
    buffer_records: [max_gpu_heaps]BufferRecord = @splat(.{}),
    buffer_snapshots: [snapshot_ring_size]u64 align(64) = @splat(0),
    buffer_snapshot: u64 = 0,
    occupied_buffers: u64 = 0,
    buffer_cursor: u32 = 1,
    counter_pages: ?*CounterPage = null,
    counter_page_count: u32 = 0,
    timestamp_query_count: u32 = 0,
    /// MetalTools on macOS 26.6.2 cannot enumerate placed resources through heap residency, so validation runs make
    /// resources resident individually.
    shader_validation: bool = false,
};

fn addResident(device: *Device, allocation: Id) void {
    std.c.os_unfair_lock_lock(&device.residency_lock);
    defer std.c.os_unfair_lock_unlock(&device.residency_lock);
    send(void, device.residency, "addAllocation:", .{allocation});
    send(void, device.residency, "commit", .{});
}

fn removeResident(device: *Device, allocation: Id) void {
    std.c.os_unfair_lock_lock(&device.residency_lock);
    defer std.c.os_unfair_lock_unlock(&device.residency_lock);
    send(void, device.residency, "removeAllocation:", .{allocation});
    send(void, device.residency, "commit", .{});
}

/// Claims `slots.len` timestamp slots from shared counter heap pages, creating pages as needed.
fn allocateTimestamps(device: *Device, slots: []TimestampSlot) Error!void {
    var allocated: usize = 0;
    while (true) {
        const head = @atomicLoad(?*CounterPage, &device.counter_pages, .acquire);
        var published: u32 = 0;
        var next = head;
        while (next) |page| : (next = page.next) {
            published += 1;
            for (&page.occupied, 0..) |*word, word_index| {
                var occupied = @atomicLoad(u64, word, .monotonic);
                while (occupied != std.math.maxInt(u64)) {
                    var available = ~occupied;
                    var claim: u64 = 0;
                    var wanted = slots.len - allocated;
                    while (wanted != 0 and available != 0) : (wanted -= 1) {
                        claim |= available & (0 -% available);
                        available &= available - 1;
                    }
                    if (@cmpxchgWeak(u64, word, occupied, occupied | claim, .acq_rel, .monotonic)) |current| {
                        occupied = current;
                        continue;
                    }
                    while (claim != 0) : (claim &= claim - 1) {
                        slots[allocated] = .{ .heap = page.heap, .page = page, .index = @intCast(word_index * 64 + @ctz(claim)) };
                        allocated += 1;
                    }
                    if (allocated == slots.len) return;
                    break;
                }
            }
        }
        if (@atomicLoad(?*CounterPage, &device.counter_pages, .acquire) != head) continue;
        const page_count = @atomicLoad(u32, &device.counter_page_count, .monotonic);
        if (page_count == max_counter_pages) {
            if (published != max_counter_pages) continue;
            report("Metal timestamp storage capacity exhausted.", .{});
            return error.OutOfMemory;
        }
        // Reserve native capacity before creation, and keep every successfully created page after publication races.
        if (@cmpxchgWeak(u32, &device.counter_page_count, page_count, page_count + 1, .monotonic, .monotonic) != null) continue;
        if (@atomicLoad(?*CounterPage, &device.counter_pages, .acquire) != head) {
            _ = @atomicRmw(u32, &device.counter_page_count, .Sub, 1, .monotonic);
            continue;
        }
        const autorelease = objc.objc_autoreleasePoolPush();
        defer objc.objc_autoreleasePoolPop(autorelease);
        const descriptor = objc.new("MTL4CounterHeapDescriptor");
        send(void, descriptor, "setType:", .{counter_heap_type_timestamp});
        send(void, descriptor, "setCount:", .{@as(usize, counter_page_size)});
        var failure: ?Id = null;
        const heap = send(?Id, device.metal, "newCounterHeapWithDescriptor:error:", .{ descriptor, &failure });
        release(descriptor);
        const page = create(CounterPage, .{ .heap = heap orelse {
            _ = @atomicRmw(u32, &device.counter_page_count, .Sub, 1, .monotonic);
            reportFailure("timestamp storage", failure);
            return error.OutOfMemory;
        } });
        var expected = head;
        while (true) {
            page.next = expected;
            expected = @cmpxchgWeak(?*CounterPage, &device.counter_pages, expected, page, .acq_rel, .acquire) orelse break;
        }
    }
}

fn destroyCommandBuffer(commands: *CommandBuffer) void {
    var next: ?*NativeCommandBuffer = commands.native_buffers;
    while (next) |native| {
        next = native.next;
        release(native.buffer);
        release(native.allocator);
        gpa.destroy(native);
    }
    release(commands.arguments);
    release(commands.pass);
    // Return timestamp slots to their pages one occupancy word at a time; allocation fills slots in page and word order.
    const slots = commands.timestamps;
    var i: usize = 0;
    while (i < slots.len and slots[i].page != null) {
        const start = i;
        var bits: u64 = 0;
        while (i < slots.len and slots[i].page == slots[start].page and slots[i].index / 64 == slots[start].index / 64) : (i += 1) {
            bits |= @as(u64, 1) << @intCast(slots[i].index % 64);
        }
        _ = @atomicRmw(u64, &slots[start].page.?.occupied[slots[start].index / 64], .And, ~bits, .release);
    }
    gpa.free(slots);
    gpa.destroy(commands);
}

fn newNativeCommandBuffer(device: *const Device) *NativeCommandBuffer {
    return create(NativeCommandBuffer, .{
        .buffer = send(?Id, device.metal, "newCommandBuffer", .{}),
        .allocator = send(?Id, device.metal, "newCommandAllocator", .{}),
    });
}

fn newCommandBuffer(pool: *CommandPool) Error!*CommandBuffer {
    const device = pool.device;
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const native = newNativeCommandBuffer(device);
    const descriptor = objc.new("MTL4ArgumentTableDescriptor");
    send(void, descriptor, "setMaxBufferBindCount:", .{@as(usize, 3)});
    send(void, descriptor, "setInitializeBindings:", .{true});
    var failure: ?Id = null;
    const arguments = send(?Id, device.metal, "newArgumentTableWithDescriptor:error:", .{ descriptor, &failure });
    release(descriptor);
    const commands = create(CommandBuffer, .{
        .device = device,
        .pool = pool,
        .native_buffers = native,
        .native = native,
        .arguments = arguments,
        .pass = objc.new("MTL4RenderPassDescriptor"),
    });
    if (native.buffer == null or native.allocator == null or arguments == null) {
        reportFailure("command context", failure);
        destroyCommandBuffer(commands);
        return error.DriverError;
    }
    if (device.timestamp_query_count != 0) {
        commands.timestamps = gpa.alloc(TimestampSlot, device.timestamp_query_count) catch @panic("out of memory");
        @memset(commands.timestamps, .{});
        allocateTimestamps(device, commands.timestamps) catch |err| {
            destroyCommandBuffer(commands);
            return err;
        };
    }
    return commands;
}

fn endCompute(commands: *CommandBuffer) void {
    const encoder = commands.compute orelse return;
    send(void, encoder, "endEncoding", .{});
    release(encoder);
    commands.compute = null;
}

fn nativeCommands(commands: *CommandBuffer) Id {
    if (commands.commands) |buffer| return buffer;
    const native = commands.native.next orelse next: {
        const next = newNativeCommandBuffer(commands.device);
        commands.native.next = next;
        break :next next;
    };
    const buffer = native.buffer.?;
    commands.native = native;
    commands.native_count += 1;
    commands.commands = buffer;
    send(void, buffer, "beginCommandBufferWithAllocator:", .{native.allocator});
    return buffer;
}

fn computeEncoder(commands: *CommandBuffer) Id {
    assert(commands.recording and commands.render == null);
    if (commands.compute) |encoder| return encoder;
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const encoder = objc.objc_retain(send(?Id, nativeCommands(commands), "computeCommandEncoder", .{})).?;
    commands.compute = encoder;
    send(void, encoder, "setArgumentTable:", .{commands.arguments});
    if (commands.pso) |pso| if (pso.compute) |state| send(void, encoder, "setComputePipelineState:", .{state});
    return encoder;
}

/// Inserts or removes `record` in a new sorted snapshot and publishes it.
fn updateBufferIndex(device: *Device, record: *BufferRecord, insert: bool) void {
    while (true) {
        const previous = @atomicLoad(u64, &device.buffer_snapshot, .acquire);
        const start: u32 = @truncate(previous);
        var count: u32 = @intCast(previous >> 32);
        var snapshot: [max_gpu_heaps]*BufferRecord = undefined;
        // Acquire loads order these reads before the snapshot revalidation below.
        for (snapshot[0..count], 0..) |*entry, i| entry.* = @ptrFromInt(@atomicLoad(u64, &device.buffer_snapshots[ringSlot(start, i)], .acquire));
        if (insert) {
            assert(count < max_gpu_heaps);
            const base = @atomicLoad(u64, &record.base, .monotonic);
            var index = count;
            count += 1;
            while (index != 0 and @atomicLoad(u64, &snapshot[index - 1].base, .acquire) > base) : (index -= 1) snapshot[index] = snapshot[index - 1];
            snapshot[index] = record;
        } else {
            var index: u32 = 0;
            while (index < count and snapshot[index] != record) index += 1;
            assert(index < count);
            while (index + 1 < count) : (index += 1) snapshot[index] = snapshot[index + 1];
            count -= 1;
        }
        if (@atomicLoad(u64, &device.buffer_snapshot, .monotonic) != previous) continue;
        const ticket = @atomicRmw(u32, &device.buffer_cursor, .Add, @max(count, 1), .monotonic);
        // Release stores keep the ring reservation ordered before the new entries.
        for (snapshot[0..count], 0..) |entry, i| @atomicStore(u64, &device.buffer_snapshots[ringSlot(ticket, i)], @intFromPtr(entry), .release);
        if (@cmpxchgStrong(u64, &device.buffer_snapshot, previous, (@as(u64, count) << 32) | ticket, .release, .monotonic) == null) return;
    }
}

fn registerBuffer(device: *Device, owner: *GpuHeapOwner, base: u64, size: u64) void {
    var occupied = @atomicLoad(u64, &device.occupied_buffers, .monotonic);
    const slot: u6 = while (true) {
        assert(occupied != std.math.maxInt(u64));
        const free: u6 = @intCast(@ctz(~occupied));
        occupied = @cmpxchgWeak(u64, &device.occupied_buffers, occupied, occupied | (@as(u64, 1) << free), .acq_rel, .monotonic) orelse break free;
    };
    owner.slot = slot;
    const record = &device.buffer_records[slot];
    // Release stores make a recycled record's earlier removal visible to readers that observe its new contents.
    @atomicStore(u64, &record.base, base, .release);
    @atomicStore(u64, &record.size, size, .release);
    @atomicStore(u64, &record.buffer, @intFromPtr(owner.buffer), .release);
    updateBufferIndex(device, record, true);
}

fn unregisterBuffer(device: *Device, slot: u6) void {
    updateBufferIndex(device, &device.buffer_records[slot], false);
    _ = @atomicRmw(u64, &device.occupied_buffers, .And, ~(@as(u64, 1) << slot), .release);
}

/// Returns the buffer containing `range` and the range's offset into it.
fn resolveBuffer(device: *Device, range: api.GpuRange) struct { Id, u64 } {
    const address = @intFromPtr(range.gpu);
    while (true) {
        const snapshot = @atomicLoad(u64, &device.buffer_snapshot, .acquire);
        const start: u32 = @truncate(snapshot);
        var first: u32 = 0;
        var last: u32 = @intCast(snapshot >> 32);
        while (first < last) {
            const middle = first + (last - first) / 2;
            const record: *const BufferRecord = @ptrFromInt(@atomicLoad(u64, &device.buffer_snapshots[ringSlot(start, middle)], .acquire));
            if (@atomicLoad(u64, &record.base, .acquire) <= address) first = middle + 1 else last = middle;
        }
        var base: u64 = 0;
        var size: u64 = 0;
        var buffer: u64 = 0;
        if (first != 0) {
            const record: *const BufferRecord = @ptrFromInt(@atomicLoad(u64, &device.buffer_snapshots[ringSlot(start, first - 1)], .acquire));
            base = @atomicLoad(u64, &record.base, .acquire);
            size = @atomicLoad(u64, &record.size, .acquire);
            buffer = @atomicLoad(u64, &record.buffer, .acquire);
        }
        if (@atomicLoad(u64, &device.buffer_snapshot, .monotonic) != snapshot) continue;
        assert(first != 0 and address >= base and address - base <= size and range.size <= size - (address - base));
        return .{ @ptrFromInt(buffer), address - base };
    }
}

pub fn createDevice(desc: api.DeviceDesc) Error!*Device {
    assert(desc.desired_queue_count != 0);
    if (desc.swapchain_color_space != .srgb) {
        report("Metal presentation supports only the sRGB swapchain color space.", .{});
        return error.Unsupported;
    }
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const metal = MTLCreateSystemDefaultDevice() orelse {
        report("Metal: no GPU is available.", .{});
        return error.Unsupported;
    };
    const validation = std.c.getenv("MTL_SHADER_VALIDATION");
    const device = create(Device, .{
        .metal = metal,
        .shader_validation = validation != null and (std.fmt.parseInt(i32, std.mem.span(validation.?), 10) catch 0) != 0,
    });
    errdefer destroyDevice(device);
    if (!send(bool, metal, "supportsFamily:", .{gpu_family_apple7}) or !send(bool, metal, "supportsFamily:", .{gpu_family_metal4})) {
        report("Metal: Apple7 and Metal4 GPU support is required.", .{});
        return error.Unsupported;
    }

    const queue_count = desc.desired_queue_count + desc.desired_compute_queue_count + desc.desired_copy_queue_count;
    device.queues = gpa.alloc(Queue, queue_count) catch @panic("out of memory");
    @memset(device.queues, .{});
    var failure: ?Id = null;
    for (device.queues, 0..) |*queue, i| {
        queue.kind = if (i < desc.desired_queue_count) .general else if (i < desc.desired_queue_count + desc.desired_compute_queue_count) .compute else .copy;
        const feedback_attributes = dispatch_queue_attr_make_with_autorelease_frequency(null, dispatch_autorelease_frequency_work_item);
        queue.feedback = dispatch_queue_create("NoGraphicsAPI Metal feedback", feedback_attributes);
        const descriptor = objc.new("MTL4CommandQueueDescriptor");
        send(void, descriptor, "setFeedbackQueue:", .{queue.feedback});
        queue.queue = send(?Id, metal, "newMTL4CommandQueueWithDescriptor:error:", .{ descriptor, &failure });
        // The descriptor's destructor releases its assign-only feedbackQueue; the Queue keeps ownership.
        send(void, descriptor, "setFeedbackQueue:", .{@as(?*anyopaque, null)});
        release(descriptor);
        queue.completion = send(?Id, metal, "newSharedEvent", .{});
        if (queue.queue == null or queue.completion == null) {
            reportFailure("command queue", failure);
            return error.DriverError;
        }
    }
    const compiler = objc.new("MTL4CompilerDescriptor");
    device.compiler = send(?Id, metal, "newCompilerWithDescriptor:error:", .{ compiler, &failure });
    release(compiler);
    if (device.compiler == null) {
        reportFailure("pipeline compiler", failure);
        return error.DriverError;
    }
    const residency = objc.new("MTLResidencySetDescriptor");
    send(void, residency, "setInitialCapacity:", .{@as(usize, 256)});
    device.residency = send(?Id, metal, "newResidencySetWithDescriptor:error:", .{ residency, &failure });
    release(residency);
    if (device.residency == null) {
        reportFailure("residency set", failure);
        return error.DriverError;
    }
    for (device.queues) |queue| send(void, queue.queue, "addResidencySet:", .{device.residency});

    device.timestamp_query_count = desc.timestamp_query_count;
    const name = std.mem.span(send([*:0]const u8, send(Id, metal, "name", .{}), "UTF8String", .{}));
    const name_length = @min(name.len, device.name.len);
    @memcpy(device.name[0..name_length], name[0..name_length]);
    var timebase: std.c.mach_timebase_info_data = .{ .numer = 0, .denom = 0 };
    _ = std.c.mach_timebase_info(&timebase);
    device.caps = .{
        .device_name = &device.name,
        .queue_count = queue_count,
        .general_queue_count = desc.desired_queue_count,
        .compute_queue_count = desc.desired_compute_queue_count,
        .copy_queue_count = desc.desired_copy_queue_count,
        .max_push_data_size = 256,
        .texture_heap_alignment = 16384,
        .timestamp_period_ns = @as(f32, @floatFromInt(timebase.numer)) / @as(f32, @floatFromInt(timebase.denom)),
        .sub_texel_precision_bits = 8,
        .texture_compression_bc = send(bool, metal, "supportsBCTextureCompression", .{}),
        .texture_compression_astc = true,
        .storage_input_output16 = true,
        .indirect_mesh_draw = send(bool, metal, "supportsFamily:", .{gpu_family_apple9}),
    };
    if (desc.window) |window| {
        const layer = objc.objc_retain(window).?;
        device.layer = layer;
        send(void, layer, "setDevice:", .{metal});
        send(void, layer, "setPixelFormat:", .{pixelFormat(if (desc.swapchain_format == .undefined) .bgra8_unorm else desc.swapchain_format)});
        send(void, layer, "setMaximumDrawableCount:", .{@as(usize, if (desc.desired_swapchain_image_count < 3) 2 else 3)});
        send(void, device.queues[0].queue, "addResidencySet:", .{send(Id, layer, "residencySet", .{})});
    }
    return device;
}

pub fn destroyDevice(device: *Device) void {
    assert(device.drawable == null);
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    if (device.layer) |layer| {
        send(void, device.queues[0].queue, "removeResidencySet:", .{send(Id, layer, "residencySet", .{})});
        release(layer);
    }
    for (device.queues) |queue| {
        if (std.debug.runtime_safety) if (queue.completion) |completion| assert(send(u64, completion, "signaledValue", .{}) >= queue.submitted_value);
        if (device.residency != null) send(void, queue.queue, "removeResidencySet:", .{device.residency});
        release(queue.completion);
        release(queue.queue);
        if (queue.feedback) |feedback| dispatch_release(feedback);
        gpa.free(queue.submission);
    }
    gpa.free(device.queues);
    var next = device.counter_pages;
    while (next) |page| {
        next = page.next;
        for (page.occupied) |word| assert(word == 0);
        release(page.heap);
        gpa.destroy(page);
    }
    release(device.residency);
    release(device.compiler);
    release(device.metal);
    gpa.destroy(device);
}

pub fn getDeviceCaps(device: *const Device) *const api.DeviceCaps {
    return &device.caps;
}

pub fn supportsTextureFormat(device: *const Device, format: Format, usage: api.TextureUsage) bool {
    if (pixelFormat(format) == .invalid) return false;
    const info = api.getTextureFormatInfo(format);
    if (info.depth and info.stencil and (usage.transfer_source or usage.transfer_destination)) return false;
    const compressed = info.block_extent.x > 1;
    const bc = @backingInt(format) >= @backingInt(Format.bc3_srgb) and @backingInt(format) <= @backingInt(Format.bc7_unorm);
    if (bc and !device.caps.texture_compression_bc) return false;
    if (usage.storage) switch (format) {
        .r8_srgb, .rg8_srgb, .rgba8_srgb, .bgra8_srgb, .rgba4_unorm, .r5g5b5a1_unorm, .r5g6b5_unorm => return false,
        else => if (compressed or info.depth or info.stencil) return false,
    };
    if (usage.color_attachment and (compressed or info.depth or info.stencil)) return false;
    if (usage.depth_stencil_attachment and !info.depth and !info.stencil) return false;
    return true;
}

pub fn setSwapchainFormat(device: *Device, format: Format, color_space: api.ColorSpace) Error!void {
    assert(device.drawable == null and device.acquired == null);
    const layer = device.layer orelse return error.Unsupported;
    if (color_space != .srgb) return error.Unsupported;
    if (send(PixelFormat, layer, "pixelFormat", .{}) == pixelFormat(format)) return;
    if (format != .bgra8_unorm and format != .bgra8_srgb) return error.Unsupported;
    send(void, layer, "setPixelFormat:", .{pixelFormat(format)});
}

pub fn getDrawableExtent(device: *Device) api.Uint32x2 {
    const layer = device.layer orelse return .{};
    const size = send(CGSize, layer, "drawableSize", .{});
    return .{ .x = @intFromFloat(size.width), .y = @intFromFloat(size.height) };
}

pub fn createTimelineSemaphore(device: *Device, initial_value: u64) Error!*TimelineSemaphore {
    const event = send(?Id, device.metal, "newSharedEvent", .{}) orelse {
        report("Metal timeline creation failed.", .{});
        return error.DriverError;
    };
    send(void, event, "setSignaledValue:", .{initial_value});
    return create(TimelineSemaphore, .{ .device = device, .event = event });
}

pub fn destroyTimelineSemaphore(semaphore: *TimelineSemaphore) void {
    release(semaphore.event);
    gpa.destroy(semaphore);
}

pub fn timelineCompletedValue(semaphore: *const TimelineSemaphore) u64 {
    return send(u64, semaphore.event, "signaledValue", .{});
}

pub fn waitTimeline(point: api.TimelinePoint) void {
    while (!send(bool, point.semaphore.event, "waitUntilSignaledValue:timeoutMS:", .{ point.value, @as(u64, 1000) })) {}
}

pub fn waitIdle(device: *Device) void {
    for (device.queues) |*queue| {
        queue.submitted_value += 1;
        send(void, queue.queue, "signalEvent:value:", .{ queue.completion, queue.submitted_value });
    }
    for (device.queues) |queue| {
        while (!send(bool, queue.completion, "waitUntilSignaledValue:timeoutMS:", .{ queue.submitted_value, @as(u64, 1000) })) {}
    }
}

pub fn createGpuHeap(device: *Device, byte_count: u64, memory: api.MemoryType) Error!api.GpuHeap {
    const options = resource_hazard_tracking_untracked | @as(usize, switch (memory) {
        .cpu_visible => resource_storage_shared | resource_cpu_cache_write_combined,
        .gpu_only => resource_storage_private,
        .readback => resource_storage_shared,
    });
    const requirements = send(SizeAndAlign, device.metal, "heapBufferSizeAndAlignWithLength:options:", .{ @as(usize, byte_count), options });
    if (requirements.size == 0 or requirements.alignment == 0) {
        report("Metal cannot size a GPU heap of {d} bytes.", .{byte_count});
        return error.Unsupported;
    }
    const descriptor = objc.new("MTLHeapDescriptor");
    send(void, descriptor, "setType:", .{heap_type_placement});
    send(void, descriptor, "setSize:", .{@as(usize, alignUp(requirements.size, requirements.alignment))});
    send(void, descriptor, "setResourceOptions:", .{options});
    const heap = send(?Id, device.metal, "newHeapWithDescriptor:", .{descriptor});
    release(descriptor);
    const buffer = send(?Id, heap, "newBufferWithLength:options:offset:", .{ @as(usize, byte_count), options, @as(usize, 0) }) orelse {
        release(heap);
        report("Metal GPU heap allocation of {d} bytes failed.", .{byte_count});
        return error.OutOfMemory;
    };
    const owner = create(GpuHeapOwner, .{ .device = device, .heap = heap.?, .buffer = buffer });
    const address = send(u64, buffer, "gpuAddress", .{});
    registerBuffer(device, owner, address, byte_count);
    addResident(device, if (device.shader_validation) buffer else owner.heap);
    const cpu: ?[*]u8 = if (memory == .gpu_only) null else @ptrCast(send(?*anyopaque, buffer, "contents", .{}));
    return .{ .range = .{ .cpu = cpu, .gpu = @ptrFromInt(address), .size = byte_count }, .owner = owner };
}

pub fn destroyGpuHeap(heap: api.GpuHeap) void {
    const owner = heap.owner;
    const device = owner.device;
    unregisterBuffer(device, owner.slot);
    removeResident(device, if (device.shader_validation) owner.buffer else owner.heap);
    release(owner.buffer);
    release(owner.heap);
    gpa.destroy(owner);
}

pub fn createTextureHeap(device: *Device, byte_count: u64) Error!api.TextureHeap {
    const descriptor = objc.new("MTLHeapDescriptor");
    send(void, descriptor, "setType:", .{heap_type_placement});
    send(void, descriptor, "setSize:", .{@as(usize, byte_count)});
    send(void, descriptor, "setStorageMode:", .{storage_mode_private});
    send(void, descriptor, "setHazardTrackingMode:", .{hazard_tracking_untracked});
    const heap = send(?Id, device.metal, "newHeapWithDescriptor:", .{descriptor});
    release(descriptor);
    const owner = create(TextureHeapOwner, .{ .device = device, .heap = heap orelse {
        report("Metal texture heap allocation of {d} bytes failed.", .{byte_count});
        return error.OutOfMemory;
    } });
    if (!device.shader_validation) addResident(device, owner.heap);
    return .{ .size = byte_count, .owner = owner };
}

pub fn destroyTextureHeap(heap: api.TextureHeap) void {
    if (!heap.owner.device.shader_validation) removeResident(heap.owner.device, heap.owner.heap);
    release(heap.owner.heap);
    gpa.destroy(heap.owner);
}

pub fn getTextureSizeAlign(device: *Device, desc: api.TextureDesc) Error!api.SizeAlign {
    if (!supportsTextureFormat(device, desc.format, desc.usage)) {
        report("Metal cannot create the requested texture format and usage.", .{});
        return error.Unsupported;
    }
    const descriptor = textureDescriptor(desc);
    const result = send(SizeAndAlign, device.metal, "heapTextureSizeAndAlignWithDescriptor:", .{descriptor});
    release(descriptor);
    if (result.size == 0 or result.alignment == 0) {
        report("Metal cannot size the requested texture.", .{});
        return error.Unsupported;
    }
    assert(device.caps.texture_heap_alignment % result.alignment == 0);
    return .{ .size = result.size, .alignment = result.alignment };
}

pub fn createTexture(commands: *CommandBuffer, desc: api.TextureDesc, heap: api.TextureHeap, offset: u64) Error!*Texture {
    const device = commands.device;
    assert(commands.recording and commands.render == null and heap.owner.device == device);
    const descriptor = textureDescriptor(desc);
    const texture = send(?Id, heap.owner.heap, "newTextureWithDescriptor:offset:", .{ descriptor, @as(usize, offset) });
    release(descriptor);
    const result = create(Texture, .{ .device = device, .desc = desc, .texture = texture orelse {
        report("Metal texture creation failed.", .{});
        return error.DriverError;
    } });
    if (device.shader_validation) addResident(device, result.texture);
    return result;
}

pub fn destroyTexture(texture: *Texture) void {
    if (texture.device.shader_validation) removeResident(texture.device, texture.texture);
    release(texture.texture);
    gpa.destroy(texture);
}

pub fn activateTextureAlias(commands: *CommandBuffer, texture: *Texture) void {
    assert(commands.device == texture.device and texture.desc.aliasable);
    const writes: api.Access = .{ .transfer_write = true, .shader_write = true, .color_write = true, .depth_stencil_write = true };
    barrier(commands, .{ .all_commands = true }, writes, .{ .all_commands = true }, .{});
}

pub fn createRenderView(texture: *Texture, desc: api.RenderViewDesc) Error!*RenderView {
    return create(RenderView, .{ .texture = objc.objc_retain(texture.texture), .mip = desc.mip_level, .slice = desc.slice });
}

pub fn destroyRenderView(render_view: *RenderView) void {
    release(render_view.texture);
    gpa.destroy(render_view);
}

pub fn createTextureDescriptorHeap(device: *Device, capacity: u32) Error!*TextureDescriptorHeap {
    assert(capacity != 0);
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const descriptor = objc.new("MTLResourceViewPoolDescriptor");
    send(void, descriptor, "setResourceViewCount:", .{@as(usize, capacity)});
    var failure: ?Id = null;
    const created = send(?Id, device.metal, "newTextureViewPoolWithDescriptor:error:", .{ descriptor, &failure });
    release(descriptor);
    const pool = created orelse {
        reportFailure("texture descriptor heap", failure);
        return error.OutOfMemory;
    };
    const base_id = send(ResourceId, pool, "baseResourceID", .{});
    const base_args = .{ @as(*const anyopaque, &base_id), @as(usize, @sizeOf(ResourceId)), resource_storage_shared };
    const base = send(?Id, device.metal, "newBufferWithBytes:length:options:", base_args) orelse {
        release(pool);
        report("Metal texture descriptor heap base allocation failed.", .{});
        return error.OutOfMemory;
    };
    addResident(device, base);
    return create(TextureDescriptorHeap, .{
        .device = device,
        .pool = pool,
        .base = base,
        .base_address = send(u64, base, "gpuAddress", .{}),
        .base_id = base_id.impl,
        .capacity = capacity,
    });
}

pub fn destroyTextureDescriptorHeap(heap: *TextureDescriptorHeap) void {
    removeResident(heap.device, heap.base);
    release(heap.base);
    release(heap.pool);
    gpa.destroy(heap);
}

pub fn createSamplerDescriptorHeap(device: *Device, capacity: u32) Error!*SamplerDescriptorHeap {
    assert(capacity != 0);
    const buffer = send(?Id, device.metal, "newBufferWithLength:options:", .{ @as(usize, capacity) * @sizeOf(ResourceId), resource_storage_shared }) orelse {
        report("Metal sampler heap allocation failed.", .{});
        return error.OutOfMemory;
    };
    const ids: [*]ResourceId = @ptrCast(@alignCast(send(*anyopaque, buffer, "contents", .{})));
    @memset(ids[0..capacity], .{ .impl = 0 });
    const samplers = gpa.alloc(?Id, capacity) catch @panic("out of memory");
    @memset(samplers, null);
    addResident(device, buffer);
    return create(SamplerDescriptorHeap, .{
        .device = device,
        .buffer = buffer,
        .address = send(u64, buffer, "gpuAddress", .{}),
        .ids = ids,
        .samplers = samplers,
        .capacity = capacity,
    });
}

pub fn destroySamplerDescriptorHeap(heap: *SamplerDescriptorHeap) void {
    for (heap.samplers) |sampler| release(sampler);
    gpa.free(heap.samplers);
    removeResident(heap.device, heap.buffer);
    release(heap.buffer);
    gpa.destroy(heap);
}

pub fn writeTextureDescriptor(
    heap: *TextureDescriptorHeap,
    index: u32,
    texture: *const Texture,
    descriptor_type: api.TextureDescriptorType,
    desc: api.TextureDescriptorDesc,
) void {
    assert(index < heap.capacity and texture.device == heap.device);
    const view = objc.new("MTLTextureViewDescriptor");
    defer release(view);
    const format: PixelFormat = if (desc.aspect == .stencil and texture.desc.format == .d32_float_s8_uint)
        .x32_stencil8
    else
        pixelFormat(if (desc.format == .undefined) texture.desc.format else desc.format);
    send(void, view, "setPixelFormat:", .{format});
    var view_type = send(TextureType, texture.texture, "textureType", .{});
    if (descriptor_type == .storage and (view_type == .cube or view_type == .cube_array)) view_type = .two_d_array;
    send(void, view, "setTextureType:", .{view_type});
    const mip_count = if (desc.mip_count != 0) desc.mip_count else texture.desc.mip_levels - desc.base_mip;
    const layer_count = if (desc.layer_count != 0) desc.layer_count else texture.desc.layer_count - desc.base_layer;
    send(void, view, "setLevelRange:", .{Range{ .location = desc.base_mip, .length = mip_count }});
    send(void, view, "setSliceRange:", .{Range{ .location = desc.base_layer, .length = layer_count }});
    const resource = send(ResourceId, heap.pool, "setTextureView:descriptor:atIndex:", .{ texture.texture, view, @as(usize, index) });
    assert(resource.impl == heap.base_id + index);
}

pub fn copyTextureDescriptors(
    source: *const TextureDescriptorHeap,
    source_index: u32,
    destination: *TextureDescriptorHeap,
    destination_index: u32,
    count: u32,
) void {
    assert(source.device == destination.device);
    assert(source_index + count <= source.capacity and destination_index + count <= destination.capacity);
    if (count == 0 or (source == destination and source_index == destination_index)) return;
    const selector = "copyResourceViewsFromPool:sourceRange:destinationIndex:";
    const same = source == destination;
    const forward_overlap = same and destination_index > source_index and destination_index < source_index + count;
    const backward_overlap = same and source_index > destination_index and source_index < destination_index + count;
    if (!forward_overlap and !backward_overlap) {
        send(void, destination.pool, selector, .{ source.pool, Range{ .location = source_index, .length = count }, @as(usize, destination_index) });
        return;
    }
    // Copy overlapping ranges one view at a time, in the order that reads each view before overwriting it.
    for (0..count) |n| {
        const i = if (forward_overlap) count - 1 - n else n;
        send(void, destination.pool, selector, .{ source.pool, Range{ .location = source_index + i, .length = 1 }, @as(usize, destination_index + i) });
    }
}

pub fn writeSamplerDescriptor(heap: *SamplerDescriptorHeap, index: u32, desc: api.SamplerDesc) void {
    assert(index < heap.capacity);
    // MTLSamplerAddressMode for repeat, mirrored repeat and clamp to edge.
    const address_modes = [_]usize{ 2, 3, 0 };
    const sampler = objc.new("MTLSamplerDescriptor");
    send(void, sampler, "setMinFilter:", .{@as(usize, @backingInt(desc.min_filter))});
    send(void, sampler, "setMagFilter:", .{@as(usize, @backingInt(desc.mag_filter))});
    send(void, sampler, "setMipFilter:", .{@as(usize, @backingInt(desc.mip_filter)) + 1});
    send(void, sampler, "setSAddressMode:", .{address_modes[@backingInt(desc.address_u)]});
    send(void, sampler, "setTAddressMode:", .{address_modes[@backingInt(desc.address_v)]});
    send(void, sampler, "setRAddressMode:", .{address_modes[@backingInt(desc.address_w)]});
    send(void, sampler, "setMaxAnisotropy:", .{@as(usize, if (desc.anisotropic) 4 else 1)});
    send(void, sampler, "setCompareFunction:", .{@as(usize, if (desc.compare_enabled) @backingInt(desc.compare) else 0)});
    send(void, sampler, "setSupportArgumentBuffers:", .{true});
    const state = send(?Id, heap.device.metal, "newSamplerStateWithDescriptor:", .{sampler}).?;
    release(sampler);
    release(heap.samplers[index]);
    heap.samplers[index] = state;
    heap.ids[index] = send(ResourceId, state, "gpuResourceID", .{});
}

pub fn copySamplerDescriptors(
    source: *const SamplerDescriptorHeap,
    source_index: u32,
    destination: *SamplerDescriptorHeap,
    destination_index: u32,
    count: u32,
) void {
    assert(source.device == destination.device);
    assert(source_index + count <= source.capacity and destination_index + count <= destination.capacity);
    if (count == 0 or (source == destination and source_index == destination_index)) return;
    const backwards = source == destination and destination_index > source_index;
    for (0..count) |n| {
        const i = if (backwards) count - 1 - n else n;
        const state = objc.objc_retain(source.samplers[source_index + i]);
        release(destination.samplers[destination_index + i]);
        destination.samplers[destination_index + i] = state;
        destination.ids[destination_index + i] = source.ids[source_index + i];
    }
}

/// Returns an owned MTL4LibraryFunctionDescriptor, or null for an omitted stage. Call inside an autorelease pool.
fn shaderFunction(device: *Device, stage: api.ShaderStage) Error!?Id {
    if (stage.code.len == 0) return null;
    assert(stage.code.len >= 4 and std.mem.eql(u8, stage.code[0..4], "MTLB"));
    const data = dispatch_data_create(stage.code.ptr, stage.code.len, null, null);
    var failure: ?Id = null;
    const library = send(?Id, device.metal, "newLibraryWithData:error:", .{ data, &failure });
    dispatch_release(data);
    if (library == null) {
        reportFailure("shader library", failure);
        return error.DriverError;
    }
    const function = objc.new("MTL4LibraryFunctionDescriptor");
    send(void, function, "setLibrary:", .{library});
    send(void, function, "setName:", .{send(Id, objc.class("NSString"), "stringWithUTF8String:", .{stage.entry_point})});
    release(library);
    return function;
}

fn blendFactor(factor: api.BlendFactor) usize {
    return switch (factor) {
        .zero => 0,
        .one => 1,
        .source_color => 2,
        .one_minus_source_color => 3,
        .source_alpha => 4,
        .one_minus_source_alpha => 5,
        .destination_color => 6,
        .one_minus_destination_color => 7,
        .destination_alpha => 8,
        .one_minus_destination_alpha => 9,
        .source_alpha_saturate => 10,
    };
}

fn colorTargets(attachments: Id, targets: []const api.ColorTargetDesc) void {
    assert(targets.len <= max_color_attachments);
    for (targets, 0..) |target, i| {
        const attachment = send(Id, attachments, "objectAtIndexedSubscript:", .{@as(usize, i)});
        const mask = target.write_mask;
        send(void, attachment, "setPixelFormat:", .{pixelFormat(target.format)});
        send(void, attachment, "setWriteMask:", .{@as(usize, (mask & 1) << 3 | (mask & 2) << 1 | (mask & 4) >> 1 | (mask & 8) >> 3)});
        send(void, attachment, "setBlendingState:", .{@as(isize, @intFromBool(target.blend.enabled))});
        send(void, attachment, "setSourceRGBBlendFactor:", .{blendFactor(target.blend.color.source)});
        send(void, attachment, "setDestinationRGBBlendFactor:", .{blendFactor(target.blend.color.destination)});
        send(void, attachment, "setRgbBlendOperation:", .{@as(usize, @backingInt(target.blend.color.operation))});
        send(void, attachment, "setSourceAlphaBlendFactor:", .{blendFactor(target.blend.alpha.source)});
        send(void, attachment, "setDestinationAlphaBlendFactor:", .{blendFactor(target.blend.alpha.destination)});
        send(void, attachment, "setAlphaBlendOperation:", .{@as(usize, @backingInt(target.blend.alpha.operation))});
    }
}

fn compileRenderPso(device: *Device, pipeline: Id, operation: []const u8, pso: Pso) Error!*Pso {
    var failure: ?Id = null;
    const selector = "newRenderPipelineStateWithDescriptor:compilerTaskOptions:error:";
    const state = send(?Id, device.compiler, selector, .{ pipeline, @as(?Id, null), &failure }) orelse {
        reportFailure(operation, failure);
        return error.DriverError;
    };
    addResident(device, state);
    var result = pso;
    result.render = state;
    return create(Pso, result);
}

pub fn createGraphicsPso(device: *Device, desc: api.GraphicsPsoDesc) Error!*Pso {
    assert(desc.vertex.code.len != 0);
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const vertex = try shaderFunction(device, desc.vertex);
    defer release(vertex);
    const fragment = try shaderFunction(device, desc.fragment);
    defer release(fragment);
    const pipeline = objc.new("MTL4RenderPipelineDescriptor");
    defer release(pipeline);
    send(void, pipeline, "setVertexFunctionDescriptor:", .{vertex});
    send(void, pipeline, "setFragmentFunctionDescriptor:", .{fragment});
    send(void, pipeline, "setInputPrimitiveTopology:", .{primitive_topology_triangle});
    colorTargets(send(Id, pipeline, "colorAttachments", .{}), desc.color_targets);
    return compileRenderPso(device, pipeline, "graphics pipeline", .{ .device = device, .rasterization = desc.rasterization });
}

pub fn createMeshPso(device: *Device, desc: api.MeshPsoDesc) Error!*Pso {
    assert(desc.mesh.code.len != 0);
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const task = try shaderFunction(device, desc.task);
    defer release(task);
    const mesh = try shaderFunction(device, desc.mesh);
    defer release(mesh);
    const fragment = try shaderFunction(device, desc.fragment);
    defer release(fragment);
    const pipeline = objc.new("MTL4MeshRenderPipelineDescriptor");
    defer release(pipeline);
    send(void, pipeline, "setObjectFunctionDescriptor:", .{task});
    send(void, pipeline, "setMeshFunctionDescriptor:", .{mesh});
    send(void, pipeline, "setFragmentFunctionDescriptor:", .{fragment});
    send(void, pipeline, "setRequiredThreadsPerMeshThreadgroup:", .{metalSize(desc.mesh.threadgroup_size)});
    if (task != null) send(void, pipeline, "setRequiredThreadsPerObjectThreadgroup:", .{metalSize(desc.task.threadgroup_size)});
    colorTargets(send(Id, pipeline, "colorAttachments", .{}), desc.color_targets);
    return compileRenderPso(device, pipeline, "mesh pipeline", .{
        .device = device,
        .rasterization = desc.rasterization,
        .threads = metalSize(desc.mesh.threadgroup_size),
        .object_threads = metalSize(desc.task.threadgroup_size),
        .mesh = true,
    });
}

pub fn createComputePso(device: *Device, compute: api.ShaderStage) Error!*Pso {
    assert(compute.code.len != 0);
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const function = try shaderFunction(device, compute);
    defer release(function);
    const pipeline = objc.new("MTL4ComputePipelineDescriptor");
    defer release(pipeline);
    send(void, pipeline, "setComputeFunctionDescriptor:", .{function});
    send(void, pipeline, "setRequiredThreadsPerThreadgroup:", .{metalSize(compute.threadgroup_size)});
    var failure: ?Id = null;
    const selector = "newComputePipelineStateWithDescriptor:compilerTaskOptions:error:";
    const state = send(?Id, device.compiler, selector, .{ pipeline, @as(?Id, null), &failure }) orelse {
        reportFailure("compute pipeline", failure);
        return error.DriverError;
    };
    addResident(device, state);
    return create(Pso, .{ .device = device, .compute = state, .threads = metalSize(compute.threadgroup_size) });
}

pub fn destroyPso(pso: *Pso) void {
    if (pso.render) |state| removeResident(pso.device, state);
    if (pso.compute) |state| removeResident(pso.device, state);
    release(pso.render);
    release(pso.compute);
    gpa.destroy(pso);
}

fn sameStencil(a: api.StencilFaceState, b: api.StencilFaceState) bool {
    return a.compare == b.compare and a.fail == b.fail and a.pass == b.pass and a.depth_fail == b.depth_fail;
}

/// Compares the parts of a depth-stencil state baked into MTLDepthStencilState; references are dynamic.
fn sameDepth(a: api.DepthStencilState, b: api.DepthStencilState) bool {
    return a.depth_test == b.depth_test and a.depth_write == b.depth_write and a.depth_compare == b.depth_compare and
        a.stencil_test == b.stencil_test and a.stencil_read_mask == b.stencil_read_mask and a.stencil_write_mask == b.stencil_write_mask and
        sameStencil(a.front, b.front) and sameStencil(a.back, b.back);
}

/// The caller owns the returned MTLStencilDescriptor.
fn stencilDescriptor(face: api.StencilFaceState, state: api.DepthStencilState) Id {
    const descriptor = objc.new("MTLStencilDescriptor");
    send(void, descriptor, "setStencilCompareFunction:", .{@as(usize, @backingInt(face.compare))});
    send(void, descriptor, "setStencilFailureOperation:", .{@as(usize, @backingInt(face.fail))});
    send(void, descriptor, "setDepthFailureOperation:", .{@as(usize, @backingInt(face.depth_fail))});
    send(void, descriptor, "setDepthStencilPassOperation:", .{@as(usize, @backingInt(face.pass))});
    send(void, descriptor, "setReadMask:", .{@as(u32, state.stencil_read_mask)});
    send(void, descriptor, "setWriteMask:", .{@as(u32, state.stencil_write_mask)});
    return descriptor;
}

/// Returns the pool's cached MTLDepthStencilState for `state`, creating it on first use.
fn depthState(pool: *CommandPool, state: api.DepthStencilState) ?Id {
    var link = &pool.depth_states;
    while (link.*) |chunk| {
        for (chunk.entries[0..chunk.count]) |entry| if (sameDepth(entry.desc, state)) return entry.state;
        if (chunk.count < depth_chunk_size) break;
        link = &chunk.next;
    }
    const chunk = link.* orelse chunk: {
        const chunk = create(DepthStateChunk, .{});
        link.* = chunk;
        break :chunk chunk;
    };
    const descriptor = objc.new("MTLDepthStencilDescriptor");
    const compare: api.CompareOp = if (state.depth_test) state.depth_compare else .always;
    send(void, descriptor, "setDepthCompareFunction:", .{@as(usize, @backingInt(compare))});
    send(void, descriptor, "setDepthWriteEnabled:", .{state.depth_write and state.depth_test});
    if (state.stencil_test) {
        const front = stencilDescriptor(state.front, state);
        const back = stencilDescriptor(state.back, state);
        send(void, descriptor, "setFrontFaceStencil:", .{front});
        send(void, descriptor, "setBackFaceStencil:", .{back});
        release(front);
        release(back);
    }
    const result = send(?Id, pool.device.metal, "newDepthStencilStateWithDescriptor:", .{descriptor});
    release(descriptor);
    chunk.entries[chunk.count] = .{ .desc = state, .state = result };
    chunk.count += 1;
    return result;
}

pub fn createCommandPool(device: *Device, queue_index: u32) Error!*CommandPool {
    assert(queue_index < device.queues.len);
    const pool = create(CommandPool, .{ .device = device, .kind = device.queues[queue_index].kind });
    // Create the default state and every plain depth test up front, so recording does not create them.
    const created = created: {
        if (depthState(pool, .{}) == null) break :created false;
        for (0..8) |compare| for ([_]bool{ false, true }) |write| {
            if (depthState(pool, .{ .depth_test = true, .depth_write = write, .depth_compare = @fromBackingInt(@as(u8, @intCast(compare))) }) == null) {
                break :created false;
            }
        };
        break :created true;
    };
    if (!created) {
        report("Metal command pool depth state creation failed.", .{});
        destroyCommandPool(pool);
        return error.DriverError;
    }
    return pool;
}

pub fn destroyCommandPool(pool: *CommandPool) void {
    resetCommandPool(pool);
    while (pool.first) |commands| {
        pool.first = commands.next;
        destroyCommandBuffer(commands);
    }
    while (pool.depth_states) |chunk| {
        pool.depth_states = chunk.next;
        for (chunk.entries[0..chunk.count]) |entry| release(entry.state);
        gpa.destroy(chunk);
    }
    gpa.destroy(pool);
}

pub fn resetCommandPool(pool: *CommandPool) void {
    var next = pool.first;
    while (next) |commands| : (next = commands.next) {
        assert(!commands.acquired);
        if (std.debug.runtime_safety) {
            if (commands.retirement) |retirement| assert(send(u64, retirement, "signaledValue", .{}) >= commands.retirement_value);
        }
        if (commands.render != null) endRenderPass(commands);
        endCompute(commands);
        if (commands.recording) send(void, commands.commands, "endCommandBuffer", .{});
        commands.recording = false;
        commands.ended = false;
        commands.retirement = null;
        commands.retirement_value = 0;
        var native: ?*NativeCommandBuffer = commands.native_buffers;
        while (native) |buffer| : (native = buffer.next) send(void, buffer.allocator, "reset", .{});
    }
    pool.next_buffer = pool.first;
}

pub fn beginCommands(pool: *CommandPool) Error!*CommandBuffer {
    const commands = if (pool.next_buffer) |reused| reused: {
        pool.next_buffer = reused.next;
        break :reused reused;
    } else created: {
        const created = try newCommandBuffer(pool);
        if (pool.last) |last| last.next = created else pool.first = created;
        pool.last = created;
        break :created created;
    };
    assert(!commands.recording and !commands.ended);
    commands.native = commands.native_buffers;
    commands.commands = commands.native.buffer;
    commands.native_count = 1;
    send(void, commands.commands, "beginCommandBufferWithAllocator:", .{commands.native.allocator});
    for (0..3) |index| send(void, commands.arguments, "setAddress:atIndex:", .{ @as(u64, 0), @as(usize, index) });
    commands.timestamp_count = 0;
    commands.pso = null;
    commands.recording = true;
    commands.render_continuation = false;
    return commands;
}

pub fn endCommands(commands: *CommandBuffer) void {
    assert(commands.recording and commands.render == null);
    endCompute(commands);
    send(void, commands.commands, "endCommandBuffer", .{});
    commands.recording = false;
    commands.ended = true;
}

pub fn submit(device: *Device, desc: api.SubmitDesc, queue_index: u32) void {
    assert(queue_index < device.queues.len and desc.completion.semaphore.device == device);
    const queue = &device.queues[queue_index];
    var count: usize = 0;
    for (desc.commands) |commands| count += commands.native_count;
    if (queue.submission.len < count) queue.submission = gpa.realloc(queue.submission, count) catch @panic("out of memory");
    var native_index: usize = 0;
    for (desc.commands) |commands| {
        assert(commands.device == device and commands.pool.kind == queue.kind and commands.ended and commands.retirement == null);
        commands.retirement = queue.completion;
        commands.retirement_value = queue.submitted_value + 1;
        var native: ?*NativeCommandBuffer = commands.native_buffers;
        for (0..commands.native_count) |_| {
            queue.submission[native_index] = native.?.buffer;
            native_index += 1;
            native = native.?.next;
        }
    }
    for (desc.waits) |wait| {
        assert(wait.semaphore.device == device);
        send(void, queue.queue, "waitForEvent:value:", .{ wait.semaphore.event, wait.value });
    }
    // MetalTools enumerates residency on the CPU during commit without synchronizing concurrent updates.
    if (device.shader_validation) std.c.os_unfair_lock_lock(&device.residency_lock);
    if (count != 0) send(void, queue.queue, "commit:count:", .{ queue.submission.ptr, count });
    if (device.shader_validation) std.c.os_unfair_lock_unlock(&device.residency_lock);
    queue.submitted_value += 1;
    send(void, queue.queue, "signalEvent:value:", .{ queue.completion, queue.submitted_value });
    send(void, queue.queue, "signalEvent:value:", .{ desc.completion.semaphore.event, desc.completion.value });
}

pub fn acquire(commands: *CommandBuffer) ?api.SwapchainFrame {
    const device = commands.device;
    assert(commands.recording and commands.render == null and commands.pool.kind == .general);
    assert(device.layer != null and device.drawable == null);
    const extent = getDrawableExtent(device);
    if (extent.x == 0 or extent.y == 0) return null;
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const drawable = objc.objc_retain(send(?Id, device.layer, "nextDrawable", .{})) orelse return null;
    const texture = send(Id, drawable, "texture", .{});
    device.drawable = drawable;
    device.acquired = commands;
    commands.acquired = true;
    device.drawable_view = .{ .texture = texture };
    return .{
        .render_view = &device.drawable_view,
        .extent = .{ .x = @intCast(send(usize, texture, "width", .{})), .y = @intCast(send(usize, texture, "height", .{})) },
    };
}

pub fn submitAndPresent(device: *Device, desc: api.SubmitDesc) void {
    const drawable = device.drawable.?;
    const acquired = device.acquired.?;
    var found = false;
    for (desc.commands) |commands| found = found or commands == acquired;
    assert(found);
    send(void, device.queues[0].queue, "waitForDrawable:", .{drawable});
    submit(device, desc, 0);
    send(void, device.queues[0].queue, "signalDrawable:", .{drawable});
    send(void, drawable, "present", .{});
    release(drawable);
    acquired.acquired = false;
    device.acquired = null;
    device.drawable = null;
    device.drawable_view = .{};
}

pub fn setTextureDescriptorHeap(commands: *CommandBuffer, heap: *TextureDescriptorHeap) void {
    assert(commands.recording and heap.device == commands.device);
    send(void, commands.arguments, "setAddress:atIndex:", .{ heap.base_address, @as(usize, 1) });
}

pub fn setSamplerDescriptorHeap(commands: *CommandBuffer, heap: *SamplerDescriptorHeap) void {
    assert(commands.recording and heap.device == commands.device);
    send(void, commands.arguments, "setAddress:atIndex:", .{ heap.address, @as(usize, 2) });
}

pub fn copyMemory(commands: *CommandBuffer, source: api.GpuRange, destination: api.GpuRange) void {
    assert(source.size <= destination.size);
    const source_buffer, const source_offset = resolveBuffer(commands.device, source);
    const destination_buffer, const destination_offset = resolveBuffer(commands.device, .{ .gpu = destination.gpu, .size = source.size });
    const selector = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
    const size: usize = source.size;
    send(void, computeEncoder(commands), selector, .{ source_buffer, @as(usize, source_offset), destination_buffer, @as(usize, destination_offset), size });
}

const TextureCopy = struct {
    origin: Origin,
    extent: Size,
    row_pitch: u64,
    image_pitch: u64,
    slice_pitch: u64,
    size: u64,
    slices: u32,
};

fn textureCopy(texture: *const Texture, desc: api.TextureCopyDesc) TextureCopy {
    const format = api.getTextureFormatInfo(texture.desc.format);
    const mip: u5 = @intCast(desc.mip_level);
    const extent: Size = .{
        .width = if (desc.extent.x != 0) desc.extent.x else @max(texture.desc.extent.x >> mip, 1) - desc.offset.x,
        .height = if (desc.extent.y != 0) desc.extent.y else @max(texture.desc.extent.y >> mip, 1) - desc.offset.y,
        .depth = if (desc.extent.z != 0) desc.extent.z else @max(texture.desc.extent.z >> mip, 1) - desc.offset.z,
    };
    const slices = if (desc.slice_count != 0) desc.slice_count else texture.desc.layer_count - desc.base_slice;
    const row_bytes: u64 = (extent.width + format.block_extent.x - 1) / format.block_extent.x * format.bytes_per_block;
    const rows: u64 = (extent.height + format.block_extent.y - 1) / format.block_extent.y;
    const row_pitch = if (desc.row_pitch_bytes != 0) desc.row_pitch_bytes else row_bytes;
    const packed_image_pitch = row_pitch * rows;
    const slice_pitch = if (desc.slice_pitch_bytes != 0) desc.slice_pitch_bytes else packed_image_pitch * extent.depth;
    // A 3D texture's slice pitch is the pitch between its depth images.
    const image_pitch = if (texture.desc.type == .three_d and desc.slice_pitch_bytes != 0) desc.slice_pitch_bytes else packed_image_pitch;
    return .{
        .origin = .{ .x = desc.offset.x, .y = desc.offset.y, .z = desc.offset.z },
        .extent = extent,
        .row_pitch = row_pitch,
        .image_pitch = image_pitch,
        .slice_pitch = slice_pitch,
        .size = slice_pitch * (slices - 1) + image_pitch * (extent.depth - 1) + row_pitch * (rows - 1) + row_bytes,
        .slices = slices,
    };
}

pub fn copyMemoryToTexture(commands: *CommandBuffer, source: api.GpuRange, destination: *Texture, desc: api.TextureCopyDesc) void {
    const copy = textureCopy(destination, desc);
    assert(source.size >= copy.size);
    const buffer, const offset = resolveBuffer(commands.device, .{ .gpu = source.gpu, .size = copy.size });
    const encoder = computeEncoder(commands);
    const selector = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:" ++
        "toTexture:destinationSlice:destinationLevel:destinationOrigin:";
    for (0..copy.slices) |slice| send(void, encoder, selector, .{
        buffer,
        @as(usize, offset + slice * copy.slice_pitch),
        @as(usize, copy.row_pitch),
        @as(usize, copy.image_pitch),
        copy.extent,
        destination.texture,
        @as(usize, desc.base_slice + slice),
        @as(usize, desc.mip_level),
        copy.origin,
    });
}

pub fn copyTextureToMemory(commands: *CommandBuffer, source: *Texture, destination: api.GpuRange, desc: api.TextureCopyDesc) void {
    const copy = textureCopy(source, desc);
    assert(destination.size >= copy.size);
    const buffer, const offset = resolveBuffer(commands.device, .{ .gpu = destination.gpu, .size = copy.size });
    const encoder = computeEncoder(commands);
    const selector = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:" ++
        "toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:";
    for (0..copy.slices) |slice| send(void, encoder, selector, .{
        source.texture,
        @as(usize, desc.base_slice + slice),
        @as(usize, desc.mip_level),
        copy.origin,
        copy.extent,
        buffer,
        @as(usize, offset + slice * copy.slice_pitch),
        @as(usize, copy.row_pitch),
        @as(usize, copy.image_pitch),
    });
}

pub fn barrier(commands: *CommandBuffer, before: api.Stage, before_access: api.Access, after: api.Stage, after_access: api.Access) void {
    _ = after_access;
    assert(commands.render == null);
    const source = barrierStages(before, true);
    const destination = barrierStages(after, false);
    // Shared-event completion provides host visibility; a host-only destination blocks no future GPU stages.
    if (source == 0 or destination == 0) return;
    const writes = before_access.transfer_write or before_access.shader_write or before_access.color_write or before_access.depth_stencil_write;
    const selector = "barrierAfterStages:beforeQueueStages:visibilityOptions:";
    send(void, computeEncoder(commands), selector, .{ source, destination, if (writes) visibility_device else 0 });
    endCompute(commands);
}

pub fn writeTimestamp(commands: *CommandBuffer, cpu_destination: *u64, stage: api.Stage) void {
    if (commands.device.timestamp_query_count == 0) return;
    assert(commands.recording and commands.timestamp_count < commands.device.timestamp_query_count);
    const slot = &commands.timestamps[commands.timestamp_count];
    commands.timestamp_count += 1;
    slot.destination = cpu_destination;
    if (commands.render) |encoder| {
        const after = if (stage.vertex or stage.index_input)
            render_stage_vertex
        else if (stage.task)
            render_stage_object
        else if (stage.mesh)
            render_stage_mesh
        else
            render_stage_fragment;
        const selector = "writeTimestampWithGranularity:afterStage:intoHeap:atIndex:";
        send(void, encoder, selector, .{ timestamp_granularity_precise, after, slot.heap, @as(usize, slot.index) });
    } else {
        endCompute(commands);
        send(void, nativeCommands(commands), "writeTimestampIntoHeap:atIndex:", .{ slot.heap, @as(usize, slot.index) });
    }
}

pub fn readTimestamps(pool: *CommandPool) void {
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    var next = pool.first;
    while (next) |commands| : (next = commands.next) {
        const retirement = commands.retirement orelse continue;
        if (commands.timestamp_count == 0) continue;
        if (std.debug.runtime_safety) assert(send(u64, retirement, "signaledValue", .{}) >= commands.retirement_value);
        // Resolve runs of consecutive slots in the same counter heap with one call.
        const slots = commands.timestamps[0..commands.timestamp_count];
        var i: usize = 0;
        while (i < slots.len) {
            const first = slots[i];
            var count: usize = 1;
            while (i + count < slots.len and slots[i + count].heap == first.heap and slots[i + count].index == first.index + count) count += 1;
            const data = send(Id, first.heap, "resolveCounterRange:", .{Range{ .location = first.index, .length = count }});
            assert(send(usize, data, "length", .{}) == count * @sizeOf(u64));
            const values: [*]const u64 = @ptrCast(@alignCast(send(*const anyopaque, data, "bytes", .{})));
            for (slots[i..][0..count], values[0..count]) |slot, value| slot.destination.?.* = value;
            i += count;
        }
        commands.timestamp_count = 0;
    }
}

fn renderAttachment(attachment: Id, render_view: ?*const RenderView, load: api.LoadOp, store: api.StoreOp) void {
    const view = render_view orelse return send(void, attachment, "setTexture:", .{@as(?Id, null)});
    // MTLLoadAction for load, clear and discard.
    const load_actions = [_]usize{ 1, 2, 0 };
    const volume = send(TextureType, view.texture, "textureType", .{}) == .three_d;
    send(void, attachment, "setTexture:", .{view.texture});
    send(void, attachment, "setLevel:", .{@as(usize, view.mip)});
    send(void, attachment, "setSlice:", .{@as(usize, if (volume) 0 else view.slice)});
    send(void, attachment, "setDepthPlane:", .{@as(usize, if (volume) view.slice else 0)});
    send(void, attachment, "setLoadAction:", .{load_actions[@backingInt(load)]});
    send(void, attachment, "setStoreAction:", .{@as(usize, @intFromBool(store == .store))});
}

fn bindRenderState(encoder: Id, pso: *const Pso) void {
    // Front faces wind counter-clockwise, so clockwise culling removes back faces.
    const cull: usize = switch (pso.rasterization.cull) {
        .none => 0,
        .counter_clockwise => 1,
        .clockwise => 2,
    };
    send(void, encoder, "setRenderPipelineState:", .{pso.render});
    send(void, encoder, "setCullMode:", .{cull});
    const bias = pso.rasterization;
    send(void, encoder, "setDepthBias:slopeScale:clamp:", .{ bias.depth_bias_constant, bias.depth_bias_slope, bias.depth_bias_clamp });
}

pub fn beginRenderPass(commands: *CommandBuffer, desc: api.RenderingDesc, flags: api.RenderingFlags) void {
    assert(commands.recording and commands.render == null and commands.pool.kind == .general and desc.colors.len <= max_color_attachments);
    endCompute(commands);
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const pass = commands.pass;
    const colors = send(Id, pass, "colorAttachments", .{});
    for (0..max_color_attachments) |i| {
        const attachment = send(Id, colors, "objectAtIndexedSubscript:", .{@as(usize, i)});
        if (i >= desc.colors.len) {
            send(void, attachment, "setTexture:", .{@as(?Id, null)});
            continue;
        }
        const color = desc.colors[i];
        renderAttachment(attachment, color.render_view, color.load, color.store);
        send(void, attachment, "setClearColor:", .{ClearColor{ .red = color.clear.x, .green = color.clear.y, .blue = color.clear.z, .alpha = color.clear.w }});
    }
    const depth = send(Id, pass, "depthAttachment", .{});
    renderAttachment(depth, desc.depth.render_view, desc.depth.load, desc.depth.store);
    send(void, depth, "setClearDepth:", .{@as(f64, desc.depth.clear)});
    const stencil = send(Id, pass, "stencilAttachment", .{});
    renderAttachment(stencil, desc.stencil.render_view, desc.stencil.load, desc.stencil.store);
    send(void, stencil, "setClearStencil:", .{@as(u32, desc.stencil.clear)});

    const area: *const RenderView = if (desc.colors.len != 0) desc.colors[0].render_view else desc.depth.render_view orelse desc.stencil.render_view.?;
    const width = @max(send(usize, area.texture, "width", .{}) >> @intCast(area.mip), 1);
    const height = @max(send(usize, area.texture, "height", .{}) >> @intCast(area.mip), 1);
    send(void, pass, "setRenderTargetWidth:", .{width});
    send(void, pass, "setRenderTargetHeight:", .{height});
    commands.render_continuation = @as(u32, @bitCast(flags)) != 0;
    var options: usize = 0;
    if (flags.suspending) options |= render_encoder_suspending;
    if (flags.resuming) options |= render_encoder_resuming;
    const encoder = objc.objc_retain(send(?Id, nativeCommands(commands), "renderCommandEncoderWithDescriptor:options:", .{ pass, options })).?;
    commands.render = encoder;
    send(void, encoder, "setArgumentTable:atStages:", .{ commands.arguments, render_stages });
    send(void, encoder, "setFrontFacingWinding:", .{winding_counter_clockwise});
    // Metal's framebuffer y axis points down; a negative-height viewport keeps Vulkan's clip-space orientation.
    const full_height: f64 = @floatFromInt(height);
    send(void, encoder, "setViewport:", .{Viewport{ .x = 0, .y = full_height, .width = @floatFromInt(width), .height = -full_height, .near = 0, .far = 1 }});
    send(void, encoder, "setScissorRect:", .{ScissorRect{ .x = 0, .y = 0, .width = width, .height = height }});
    send(void, encoder, "setDepthStencilState:", .{depthState(commands.pool, .{})});
    if (commands.pso) |pso| if (pso.render != null) bindRenderState(encoder, pso);
}

pub fn endRenderPass(commands: *CommandBuffer) void {
    const encoder = commands.render.?;
    send(void, encoder, "endEncoding", .{});
    release(encoder);
    commands.render = null;
    if (commands.render_continuation) {
        send(void, commands.commands, "endCommandBuffer", .{});
        commands.commands = null;
    }
    const autorelease = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(autorelease);
    const colors = send(Id, commands.pass, "colorAttachments", .{});
    for (0..max_color_attachments) |i| send(void, send(Id, colors, "objectAtIndexedSubscript:", .{@as(usize, i)}), "setTexture:", .{@as(?Id, null)});
    send(void, send(Id, commands.pass, "depthAttachment", .{}), "setTexture:", .{@as(?Id, null)});
    send(void, send(Id, commands.pass, "stencilAttachment", .{}), "setTexture:", .{@as(?Id, null)});
}

pub fn setViewport(commands: *CommandBuffer, viewport: api.Viewport) void {
    assert(commands.recording);
    const encoder = commands.render orelse return;
    const metal_viewport: Viewport = .{
        .x = viewport.x,
        .y = @as(f64, viewport.y) + viewport.height,
        .width = viewport.width,
        .height = -@as(f64, viewport.height),
        .near = viewport.min_depth,
        .far = viewport.max_depth,
    };
    send(void, encoder, "setViewport:", .{metal_viewport});
}

pub fn setScissor(commands: *CommandBuffer, scissor: api.Scissor) void {
    assert(commands.recording and scissor.x >= 0 and scissor.y >= 0);
    const encoder = commands.render orelse return;
    const rect: ScissorRect = .{ .x = @intCast(scissor.x), .y = @intCast(scissor.y), .width = scissor.width, .height = scissor.height };
    send(void, encoder, "setScissorRect:", .{rect});
}

pub fn setDepthStencil(commands: *CommandBuffer, state: api.DepthStencilState) void {
    assert(commands.recording);
    const depth = depthState(commands.pool, state);
    const encoder = commands.render orelse return;
    send(void, encoder, "setDepthStencilState:", .{depth});
    send(void, encoder, "setStencilFrontReferenceValue:backReferenceValue:", .{ @as(u32, state.front.reference), @as(u32, state.back.reference) });
}

pub fn bindPso(commands: *CommandBuffer, pso: *const Pso) void {
    assert(commands.recording and pso.device == commands.device);
    commands.pso = pso;
    if (pso.compute) |state| {
        assert(commands.render == null);
        if (commands.compute) |encoder| send(void, encoder, "setComputePipelineState:", .{state}) else _ = computeEncoder(commands);
    } else if (commands.render) |encoder| {
        bindRenderState(encoder, pso);
    }
}

fn setRoot(commands: *CommandBuffer, root: ?*const anyopaque) void {
    const pointer = root orelse return;
    assert(@intFromPtr(pointer) % 16 == 0);
    send(void, commands.arguments, "setAddress:atIndex:", .{ @as(u64, @intFromPtr(pointer)), @as(usize, 0) });
}

fn renderEncoder(commands: *CommandBuffer, root: ?*const anyopaque, mesh: bool) Id {
    assert(commands.pso != null and commands.pso.?.mesh == mesh);
    setRoot(commands, root);
    return commands.render.?;
}

pub fn draw(commands: *CommandBuffer, root: ?*const anyopaque, args: api.Draw) void {
    const encoder = renderEncoder(commands, root, false);
    const selector = "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:";
    send(void, encoder, selector, .{
        primitive_triangle,
        @as(usize, args.first_vertex),
        @as(usize, args.vertex_count),
        @as(usize, args.instance_count),
        @as(usize, args.first_instance),
    });
}

pub fn drawIndexed(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndexed) void {
    const encoder = renderEncoder(commands, root, false);
    const offset = @as(u64, args.first_index) * @as(u64, if (args.index_type == .uint16) 2 else 4);
    const selector = "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferLength:instanceCount:baseVertex:baseInstance:";
    send(void, encoder, selector, .{
        primitive_triangle,
        @as(usize, args.index_count),
        @as(usize, @backingInt(args.index_type)),
        @as(u64, @intFromPtr(args.indices.gpu)) + offset,
        @as(usize, args.indices.size - offset),
        @as(usize, args.instance_count),
        @as(isize, args.vertex_offset),
        @as(usize, args.first_instance),
    });
}

pub fn drawIndirect(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndirect) void {
    const encoder = renderEncoder(commands, root, false);
    // MTLDrawPrimitivesIndirectArguments.
    const stride: u64 = if (args.stride != 0) args.stride else 16;
    const arguments: u64 = @intFromPtr(args.arguments.gpu);
    for (0..args.draw_count) |i| send(void, encoder, "drawPrimitives:indirectBuffer:", .{ primitive_triangle, arguments + i * stride });
}

pub fn drawIndexedIndirect(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndexedIndirect) void {
    const encoder = renderEncoder(commands, root, false);
    // MTLDrawIndexedPrimitivesIndirectArguments.
    const stride: u64 = if (args.stride != 0) args.stride else 20;
    const selector = "drawIndexedPrimitives:indexType:indexBuffer:indexBufferLength:indirectBuffer:";
    for (0..args.draw_count) |i| send(void, encoder, selector, .{
        primitive_triangle,
        @as(usize, @backingInt(args.index_type)),
        @as(u64, @intFromPtr(args.indices.gpu)),
        @as(usize, args.indices.size),
        @as(u64, @intFromPtr(args.arguments.gpu)) + i * stride,
    });
}

pub fn dispatch(commands: *CommandBuffer, root: ?*const anyopaque, group_count: api.Uint32x3) void {
    const pso = commands.pso.?;
    assert(pso.compute != null);
    setRoot(commands, root);
    send(void, computeEncoder(commands), "dispatchThreadgroups:threadsPerThreadgroup:", .{ metalSize(group_count), pso.threads });
}

pub fn dispatchIndirect(commands: *CommandBuffer, root: ?*const anyopaque, arguments: api.GpuRange) void {
    const pso = commands.pso.?;
    assert(pso.compute != null);
    setRoot(commands, root);
    const selector = "dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:";
    send(void, computeEncoder(commands), selector, .{ @as(u64, @intFromPtr(arguments.gpu)), pso.threads });
}

pub fn drawMeshlets(commands: *CommandBuffer, root: ?*const anyopaque, group_count: api.Uint32x3) void {
    const encoder = renderEncoder(commands, root, true);
    const pso = commands.pso.?;
    const selector = "drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:";
    send(void, encoder, selector, .{ metalSize(group_count), pso.object_threads, pso.threads });
}

pub fn drawMeshletsIndirect(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndirect) void {
    assert(commands.device.caps.indirect_mesh_draw);
    const encoder = renderEncoder(commands, root, true);
    const pso = commands.pso.?;
    // MTLDispatchThreadgroupsIndirectArguments.
    const stride: u64 = if (args.stride != 0) args.stride else 12;
    const selector = "drawMeshThreadgroupsWithIndirectBuffer:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:";
    for (0..args.draw_count) |i| send(void, encoder, selector, .{ @as(u64, @intFromPtr(args.arguments.gpu)) + i * stride, pso.object_threads, pso.threads });
}
