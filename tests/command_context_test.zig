const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("NoGraphicsAPI");
const testing = std.testing;

const batch_command_count = 40;
const batch_submission_count = 4;
const test_gpu_heap_size = 1024 * 1024;
const max_timestamp_capacity = 4097;
const sentinel = ~@as(u64, 0);
const all_commands: gpu.Stage = .{ .all_commands = true };

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

fn createDevice(desc: gpu.DeviceDesc) !*gpu.Device {
    gpu.setErrorCallback(printError);
    return gpu.createDevice(desc) catch |err| {
        if (err == error.Unsupported) return error.SkipZigTest;
        return err;
    };
}

fn validSizeAlign(size_align: gpu.SizeAlign, common_alignment: u64) bool {
    return size_align.size != 0 and size_align.alignment != 0 and std.math.isPowerOfTwo(size_align.alignment) and common_alignment >= size_align.alignment and
        common_alignment % size_align.alignment == 0;
}

/// Returns the placement offset for `size_align` and advances `heap_size` by whole common-alignment elements.
fn place(heap_size: *u64, size_align: gpu.SizeAlign, common_alignment: u64) !u64 {
    try testing.expect(validSizeAlign(size_align, common_alignment));
    const offset = heap_size.*;
    heap_size.* += (1 + (size_align.size - 1) / common_alignment) * common_alignment;
    return offset;
}

fn submitAndWait(device: *gpu.Device, commands: []const *gpu.CommandBuffer, timeline: *gpu.TimelineSemaphore, value: *u64) void {
    value.* += 1;
    const completion: gpu.TimelinePoint = .{ .semaphore = timeline, .value = value.* };
    gpu.submit(device, .{ .commands = commands, .completion = completion }, 0);
    gpu.waitTimeline(completion);
}

fn testGpuHeaps(device: *gpu.Device) !void {
    const sizes = [_]u64{ 1, 15, 16, 17, test_gpu_heap_size };
    for (sizes) |size| {
        const heap = try gpu.createGpuHeap(device, size, .cpu_visible);
        defer gpu.destroyGpuHeap(heap);
        const range = heap.gpuRange();
        const nested_range = heap.range.gpuRange();
        try testing.expect(heap.range.cpu != null and heap.range.gpu != null and heap.range.size == size);
        try testing.expect(@intFromPtr(heap.range.cpu) % 16 == 0 and @intFromPtr(heap.range.gpu) % 16 == 0);
        try testing.expect(range.gpu == @as(?*anyopaque, heap.range.gpu) and range.size == heap.range.size);
        try testing.expect(nested_range.gpu == range.gpu and nested_range.size == range.size);
    }
    const readback = try gpu.createGpuHeap(device, 1, .readback);
    defer gpu.destroyGpuHeap(readback);
    const gpu_only = try gpu.createGpuHeap(device, 1, .gpu_only);
    defer gpu.destroyGpuHeap(gpu_only);
    try testing.expect(readback.range.cpu != null and readback.range.gpu != null and readback.range.size == 1);
    try testing.expect(@intFromPtr(readback.range.cpu) % 16 == 0 and @intFromPtr(readback.range.gpu) % 16 == 0);
    try testing.expect(gpu_only.range.cpu == null and gpu_only.range.gpu != null and @intFromPtr(gpu_only.range.gpu) % 16 == 0);
    try testing.expect(gpu_only.range.size == 1);
}

fn testDescriptorHeaps(device: *gpu.Device, timeline: *gpu.TimelineSemaphore, next_timeline_value: *u64) !void {
    const texture_heap = try gpu.createTextureDescriptorHeap(device, 2);
    defer gpu.destroyTextureDescriptorHeap(texture_heap);
    const sampler_heap = try gpu.createSamplerDescriptorHeap(device, 2);
    defer gpu.destroySamplerDescriptorHeap(sampler_heap);
    gpu.writeSamplerDescriptor(sampler_heap, 0, .{
        .min_filter = .nearest,
        .mip_filter = .nearest,
        .address_u = .clamp_to_edge,
        .address_w = .clamp_to_edge,
        .anisotropic = true,
        .compare_enabled = true,
        .compare = .greater_equal,
    });

    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const commands = try gpu.beginCommands(pool);
    gpu.copySamplerDescriptors(sampler_heap, 0, sampler_heap, 1, 1);
    gpu.setTextureDescriptorHeap(commands, texture_heap);
    gpu.setSamplerDescriptorHeap(commands, sampler_heap);
    gpu.setViewport(commands, .{ .x = 1, .y = 2, .width = 3, .height = 4, .min_depth = 0.25, .max_depth = 0.75 });
    gpu.setScissor(commands, .{ .x = 5, .y = 6, .width = 7, .height = 8 });
    gpu.setDepthStencil(commands, .{
        .depth_test = true,
        .depth_write = true,
        .depth_compare = .greater_equal,
        .stencil_test = true,
        .stencil_read_mask = 0x7f,
        .stencil_write_mask = 0x3f,
        .front = .{ .compare = .equal, .fail = .replace, .pass = .increment_wrap, .depth_fail = .decrement_clamp, .reference = 11 },
        .back = .{ .compare = .not_equal, .fail = .zero, .pass = .invert, .depth_fail = .increment_clamp, .reference = 17 },
    });
    gpu.endCommands(commands);
    submitAndWait(device, &.{commands}, timeline, next_timeline_value);
}

fn testBatchGrowthAndReuse(device: *gpu.Device, timeline: *gpu.TimelineSemaphore, next_timeline_value: *u64) !void {
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    var high_water_commands: [batch_command_count]*gpu.CommandBuffer = undefined;
    var valid = true;
    for (0..batch_submission_count) |submission| {
        var commands: [batch_command_count]*gpu.CommandBuffer = undefined;
        for (&commands, &high_water_commands) |*command, *high_water| {
            command.* = try gpu.beginCommands(pool);
            if (submission == 0) high_water.* = command.* else if (submission == 1) valid = valid and command.* == high_water.*;
            gpu.endCommands(command.*);
        }
        submitAndWait(device, &commands, timeline, next_timeline_value);
        gpu.resetCommandPool(pool);
    }
    try testing.expect(valid);
}

fn testTimestampReadback(device: *gpu.Device, timeline: *gpu.TimelineSemaphore, next_timeline_value: *u64) !void {
    const context_count = 3;
    const slot_words = 514;
    const word_count = context_count * slot_words;
    const counts = [4][context_count]u32{ .{ 256, 4, 17 }, .{ 1, 0, 3 }, .{ 9, 256, 2 }, .{ 0, 2, 1 } };
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    var cpu: [word_count]u64 = undefined;
    var second_cpu: [4]u64 = undefined;
    var discarded: u64 = sentinel;
    var previous_batch_end: u64 = 0;
    var valid = true;
    for (0..4) |batch| {
        @memset(&cpu, sentinel);
        const unsubmitted = try gpu.beginCommands(pool);
        gpu.writeTimestamp(unsubmitted, &discarded, all_commands);
        if (batch & 1 != 0) gpu.endCommands(unsubmitted);
        var submitted: [context_count]*gpu.CommandBuffer = undefined;
        const first = 1 + (batch & 1);
        for (0..context_count) |context| {
            const commands = try gpu.beginCommands(pool);
            submitted[context_count - context - 1] = commands;
            const stride = 1 + ((batch + context) & 1);
            for (0..counts[batch][context]) |index| gpu.writeTimestamp(commands, &cpu[context * slot_words + first + index * stride], all_commands);
            gpu.endCommands(commands);
        }
        // All contexts remain recorded together, and execute in the opposite order. No caller query object or host barrier is needed.
        submitAndWait(device, &submitted, timeline, next_timeline_value);
        var batch_valid = discarded == sentinel;
        for (cpu) |word| batch_valid = word == sentinel and batch_valid;
        gpu.readTimestamps(pool);
        batch_valid = discarded == sentinel and batch_valid;
        for (cpu, 0..) |word, index| {
            const context = index / slot_words;
            const offset = index % slot_words;
            const stride = 1 + ((batch + context) & 1);
            const written = offset >= first and (offset - first) % stride == 0 and (offset - first) / stride < counts[batch][context];
            batch_valid = batch_valid and (if (written) word != sentinel else word == sentinel);
        }
        var first_tick: u64 = 0;
        var previous_tick: u64 = 0;
        var have_tick = false;
        for (0..context_count) |order| {
            const context = context_count - order - 1;
            const stride = 1 + ((batch + context) & 1);
            for (0..counts[batch][context]) |index| {
                const tick = cpu[context * slot_words + first + index * stride];
                if (have_tick) batch_valid = batch_valid and tick >= previous_tick else first_tick = tick;
                previous_tick = tick;
                have_tick = true;
            }
        }
        batch_valid = batch_valid and previous_tick > first_tick and (batch == 0 or first_tick != previous_batch_end);
        previous_batch_end = previous_tick;
        if (!batch_valid) std.debug.print("Timestamp readback failed in batch {}.\n", .{batch});
        valid = valid and batch_valid;
        @memset(&cpu, sentinel);
        gpu.readTimestamps(pool);
        for (cpu) |word| valid = word == sentinel and valid;
        gpu.resetCommandPool(pool);
    }
    @memset(&cpu, sentinel);
    @memset(&second_cpu, sentinel);
    var commands = try gpu.beginCommands(pool);
    gpu.writeTimestamp(commands, &cpu[0], all_commands);
    gpu.writeTimestamp(commands, &second_cpu[0], all_commands);
    gpu.writeTimestamp(commands, &cpu[1], all_commands);
    gpu.endCommands(commands);
    submitAndWait(device, &.{commands}, timeline, next_timeline_value);
    gpu.readTimestamps(pool);
    var separate_destinations_valid = cpu[0] != sentinel and second_cpu[0] != sentinel and cpu[1] != sentinel and cpu[0] <= second_cpu[0] and
        second_cpu[0] <= cpu[1];
    for (cpu[2..]) |word| separate_destinations_valid = separate_destinations_valid and word == sentinel;
    for (second_cpu[1..]) |word| separate_destinations_valid = separate_destinations_valid and word == sentinel;
    if (!separate_destinations_valid) std.debug.print("Timestamp readback failed across separate CPU arrays.\n", .{});
    valid = valid and separate_destinations_valid;
    gpu.resetCommandPool(pool);
    commands = try gpu.beginCommands(pool);
    gpu.writeTimestamp(commands, &discarded, all_commands);
    gpu.endCommands(commands);
    submitAndWait(device, &.{commands}, timeline, next_timeline_value);
    gpu.resetCommandPool(pool);
    gpu.readTimestamps(pool);
    try testing.expect(valid and discarded == sentinel);
}

fn testTimestampCapacity(count: u32) !void {
    const device = try createDevice(.{ .timestamp_query_count = count });
    defer gpu.destroyDevice(device);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    var storage: [max_timestamp_capacity + 2]u64 = undefined;
    const cpu = storage[0 .. count + 2];
    var timeline_value: u64 = 0;
    var previous_result: u64 = 0;
    var valid = true;
    for (0..4) |batch| {
        @memset(cpu, sentinel);
        const commands = try gpu.beginCommands(pool);
        for (cpu[1 .. count + 1]) |*destination| gpu.writeTimestamp(commands, destination, all_commands);
        gpu.endCommands(commands);
        submitAndWait(device, &.{commands}, timeline, &timeline_value);
        gpu.readTimestamps(pool);
        gpu.resetCommandPool(pool);
        var batch_valid = cpu[0] == sentinel and cpu[count + 1] == sentinel and (batch == 0 or cpu[1] != previous_result);
        for (1..count + 1) |index| batch_valid = batch_valid and cpu[index] != sentinel and (index == 1 or cpu[index] >= cpu[index - 1]);
        previous_result = cpu[1];
        if (!batch_valid) std.debug.print("Timestamp capacity {} failed in batch {}.\n", .{ count, batch });
        valid = valid and batch_valid;
        gpu.waitIdle(device);
    }
    try testing.expect(valid);
}

fn testWithoutTimestamps(query_count: u32) !void {
    const device = try createDevice(.{ .timestamp_query_count = query_count });
    defer gpu.destroyDevice(device);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const source = try gpu.createGpuHeap(device, @sizeOf(u64) * batch_command_count, .cpu_visible);
    defer gpu.destroyGpuHeap(source);
    const readback = try gpu.createGpuHeap(device, @sizeOf(u64) * (batch_command_count * 2 + 1), .readback);
    defer gpu.destroyGpuHeap(readback);
    const source_cpu: [*]u64 = @ptrCast(@alignCast(source.range.cpu.?));
    const readback_cpu: [*]u64 = @ptrCast(@alignCast(readback.range.cpu.?));
    var first_commands: [batch_command_count]*gpu.CommandBuffer = undefined;
    var timeline_value: u64 = 0;
    var valid = true;
    for (0..batch_submission_count) |batch| {
        for (0..batch_command_count) |index| source_cpu[index] = (batch + 1) * 1000 + index;
        @memset(readback_cpu[0 .. batch_command_count * 2 + 1], sentinel);
        var commands: [batch_command_count]*gpu.CommandBuffer = undefined;
        var batch_valid = true;
        for (&commands, &first_commands, 0..) |*command, *first, index| {
            command.* = try gpu.beginCommands(pool);
            if (batch == 0) first.* = command.* else batch_valid = batch_valid and command.* == first.*;
            gpu.writeTimestamp(command.*, &readback_cpu[index * 2], all_commands);
            gpu.copyMemory(command.*, .{ .gpu = source.range.gpu.? + index * 8, .size = 8 }, .{ .gpu = readback.range.gpu.? + (index * 2 + 1) * 8, .size = 8 });
            gpu.writeTimestamp(command.*, &readback_cpu[index * 2 + 2], all_commands);
            gpu.barrier(command.*, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
            gpu.endCommands(command.*);
        }
        submitAndWait(device, &commands, timeline, &timeline_value);
        gpu.readTimestamps(pool);
        gpu.resetCommandPool(pool);
        for (0..batch_command_count) |index|
            batch_valid = batch_valid and readback_cpu[index * 2] == sentinel and readback_cpu[index * 2 + 1] == source_cpu[index];
        batch_valid = batch_valid and readback_cpu[batch_command_count * 2] == sentinel;
        if (!batch_valid) std.debug.print("Command readback with timestamps disabled failed in batch {}.\n", .{batch});
        valid = valid and batch_valid;
        gpu.waitIdle(device);
    }
    try testing.expect(valid);
}

fn combinedDepthStencilFormat(device: *gpu.Device) gpu.Format {
    const usage: gpu.TextureUsage = .{ .depth_stencil_attachment = true };
    if (gpu.supportsTextureFormat(device, .d24_unorm_s8_uint, usage)) return .d24_unorm_s8_uint;
    if (gpu.supportsTextureFormat(device, .d32_float_s8_uint, usage)) return .d32_float_s8_uint;
    return .undefined;
}

fn recordAttachmentSubresourcePasses(commands: *gpu.CommandBuffer, color_view: ?*gpu.RenderView, depth_stencil_view: ?*gpu.RenderView) void {
    if (color_view) |view| {
        gpu.beginRenderPass(commands, .{ .colors = &.{.{ .render_view = view, .load = .clear, .clear = .{ .x = 0.1, .y = 0.2, .z = 0.3, .w = 1 } }} }, .{});
        gpu.endRenderPass(commands);
    }
    if (depth_stencil_view) |view| {
        gpu.beginRenderPass(commands, .{
            .depth = .{ .render_view = view, .load = .clear, .clear = 0.5 },
            .stencil = .{ .render_view = view, .load = .clear, .clear = 7 },
        }, .{});
        gpu.endRenderPass(commands);
    }
}

fn testPlacedTextures(device: *gpu.Device, caps: *const gpu.DeviceCaps, timeline: *gpu.TimelineSemaphore, next_timeline_value: *u64) !void {
    const alignment = caps.texture_heap_alignment;
    const first_desc: gpu.TextureDesc = .{ .extent = .{ .x = 17, .y = 9, .z = 1 } };
    const second_desc: gpu.TextureDesc = .{ .extent = .{ .x = 31, .y = 15, .z = 1 }, .mip_levels = 3 };
    const first_size_align = try gpu.getTextureSizeAlign(device, first_desc);
    var heap_size: u64 = 0;
    _ = try place(&heap_size, first_size_align, alignment);
    const second_offset = try place(&heap_size, try gpu.getTextureSizeAlign(device, second_desc), alignment);

    const broad_usage: gpu.TextureUsage = .{ .sampled = true, .storage = true, .transfer_destination = true };
    const has_broad_3d = gpu.supportsTextureFormat(device, .rgba32_float, broad_usage);
    const broad_3d_desc: gpu.TextureDesc = .{
        .type = .three_d,
        .extent = .{ .x = 16, .y = 8, .z = 4 },
        .mip_levels = 4,
        .format = .rgba32_float,
        .usage = broad_usage,
    };
    const broad_3d_offset = if (has_broad_3d) try place(&heap_size, try gpu.getTextureSizeAlign(device, broad_3d_desc), alignment) else 0;

    const mutable_desc: gpu.TextureDesc = .{ .extent = .{ .x = 19, .y = 11, .z = 1 }, .mip_levels = 3, .mutable_format = true };
    const mutable_offset = try place(&heap_size, try gpu.getTextureSizeAlign(device, mutable_desc), alignment);

    var compressed_format: gpu.Format = .undefined;
    if (caps.texture_compression_bc and gpu.supportsTextureFormat(device, .bc7_unorm, .{ .sampled = true })) {
        compressed_format = .bc7_unorm;
    } else if (caps.texture_compression_astc and gpu.supportsTextureFormat(device, .astc_4x4_unorm, .{ .sampled = true })) {
        compressed_format = .astc_4x4_unorm;
    }
    const has_compressed = compressed_format != .undefined;
    const compressed_desc: gpu.TextureDesc = .{ .extent = .{ .x = 20, .y = 12, .z = 1 }, .mip_levels = 4, .format = compressed_format };
    const compressed_offset = if (has_compressed) try place(&heap_size, try gpu.getTextureSizeAlign(device, compressed_desc), alignment) else 0;

    const color_usage: gpu.TextureUsage = .{ .color_attachment = true };
    const has_color = gpu.supportsTextureFormat(device, .rgba8_unorm, color_usage);
    const color_desc: gpu.TextureDesc = .{ .type = .cube, .extent = .{ .x = 8, .y = 8, .z = 1 }, .mip_levels = 3, .layer_count = 6, .usage = color_usage };
    const color_offset = if (has_color) try place(&heap_size, try gpu.getTextureSizeAlign(device, color_desc), alignment) else 0;

    const depth_stencil_format = combinedDepthStencilFormat(device);
    const has_depth_stencil = depth_stencil_format != .undefined;
    const sampled_depth_stencil_usage: gpu.TextureUsage = .{ .sampled = true, .depth_stencil_attachment = true };
    const sampled_depth_stencil = has_depth_stencil and gpu.supportsTextureFormat(device, depth_stencil_format, sampled_depth_stencil_usage);
    const depth_stencil_desc: gpu.TextureDesc = .{
        .type = .two_d_array,
        .extent = .{ .x = 8, .y = 8, .z = 1 },
        .mip_levels = 3,
        .layer_count = 2,
        .format = depth_stencil_format,
        .usage = if (sampled_depth_stencil) sampled_depth_stencil_usage else .{ .depth_stencil_attachment = true },
    };
    const depth_stencil_offset = if (has_depth_stencil) try place(&heap_size, try gpu.getTextureSizeAlign(device, depth_stencil_desc), alignment) else 0;

    const texture_heap = try gpu.createTextureHeap(device, heap_size);
    defer gpu.destroyTextureHeap(texture_heap);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const commands = try gpu.beginCommands(pool);
    const first = try gpu.createTexture(commands, first_desc, texture_heap, 0);
    defer gpu.destroyTexture(first);
    const second = try gpu.createTexture(commands, second_desc, texture_heap, second_offset);
    defer gpu.destroyTexture(second);
    const broad_3d = if (has_broad_3d) try gpu.createTexture(commands, broad_3d_desc, texture_heap, broad_3d_offset) else null;
    defer if (broad_3d) |texture| gpu.destroyTexture(texture);
    const mutable_texture = try gpu.createTexture(commands, mutable_desc, texture_heap, mutable_offset);
    defer gpu.destroyTexture(mutable_texture);
    const compressed = if (has_compressed) try gpu.createTexture(commands, compressed_desc, texture_heap, compressed_offset) else null;
    defer if (compressed) |texture| gpu.destroyTexture(texture);
    const color = if (has_color) try gpu.createTexture(commands, color_desc, texture_heap, color_offset) else null;
    defer if (color) |texture| gpu.destroyTexture(texture);
    const depth_stencil = if (has_depth_stencil) try gpu.createTexture(commands, depth_stencil_desc, texture_heap, depth_stencil_offset) else null;
    defer if (depth_stencil) |texture| gpu.destroyTexture(texture);
    if (builtin.mode != .debug and sampled_depth_stencil) {
        const descriptor_heap = try gpu.createTextureDescriptorHeap(device, 1);
        defer gpu.destroyTextureDescriptorHeap(descriptor_heap);
        gpu.writeTextureDescriptor(descriptor_heap, 0, depth_stencil.?, .sampled, .{});
        gpu.writeTextureDescriptor(descriptor_heap, 0, depth_stencil.?, .sampled, .{ .aspect = .stencil });
    }
    const recording_heap = try gpu.createTextureHeap(device, first_size_align.size);
    defer gpu.destroyTextureHeap(recording_heap);
    const color_render_view = if (color) |texture| try gpu.createRenderView(texture, .{ .mip_level = 2, .slice = 5 }) else null;
    defer if (color_render_view) |view| gpu.destroyRenderView(view);
    const depth_stencil_render_view = if (depth_stencil) |texture| try gpu.createRenderView(texture, .{ .mip_level = 1, .slice = 1 }) else null;
    defer if (depth_stencil_render_view) |view| gpu.destroyRenderView(view);
    recordAttachmentSubresourcePasses(commands, color_render_view, depth_stencil_render_view);
    gpu.endCommands(commands);
    submitAndWait(device, &.{commands}, timeline, next_timeline_value);
}

test "command context" {
    const device = try createDevice(.{});
    defer gpu.destroyDevice(device);
    const caps = gpu.getDeviceCaps(device);
    try testing.expect(caps.texture_heap_alignment != 0 and std.math.isPowerOfTwo(caps.texture_heap_alignment));
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    defer gpu.waitIdle(device);
    var next_timeline_value: u64 = 0;
    try testGpuHeaps(device);
    try testDescriptorHeaps(device, timeline, &next_timeline_value);
    try testPlacedTextures(device, caps, timeline, &next_timeline_value);
    try testTimestampReadback(device, timeline, &next_timeline_value);
    try testBatchGrowthAndReuse(device, timeline, &next_timeline_value);
}

test "timestamp capacity" {
    try testTimestampCapacity(1);
    try testTimestampCapacity(513);
    try testTimestampCapacity(4097);
}

test "timestamps disabled" {
    try testWithoutTimestamps(0);
}
