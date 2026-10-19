const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const testing = std.testing;

const thread_count = 4;
const textures_per_thread = 8;
const texture_bytes = 8 * 8 * 4;
const texture_words = texture_bytes / @sizeOf(u32);
const all_commands: gpu.Stage = .{ .all_commands = true };
const texture_desc: gpu.TextureDesc = .{
    .extent = .{ .x = 8, .y = 8, .z = 1 },
    .usage = .{ .sampled = true, .color_attachment = true, .transfer_source = true, .transfer_destination = true },
};

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

fn waitForStart(start: *const std.atomic.Value(bool)) void {
    while (!start.load(.acquire)) std.Thread.yield() catch {};
}

/// Spawns one worker per context, releases them together, and joins every started worker.
fn runThreads(comptime function: anytype, contexts: anytype) !void {
    var start: std.atomic.Value(bool) = .init(false);
    var threads: [thread_count]std.Thread = undefined;
    var started_count: usize = 0;
    const result = while (started_count != contexts.len) : (started_count += 1) {
        threads[started_count] = std.Thread.spawn(.{}, function, .{ &contexts[started_count], &start }) catch |err| break err;
    } else {};
    start.store(true, .release);
    for (threads[0..started_count]) |thread| thread.join();
    result catch |err| {
        std.debug.print("Started {} of {} worker threads.\n", .{ started_count, contexts.len });
        return err;
    };
}

const RecordingContext = struct {
    device: *gpu.Device,
    texture_heap: gpu.TextureHeap,
    texture_stride: u64,
    index: u32,
    result: gpu.Error!void = {},
    pool: ?*gpu.CommandPool = null,
    upload_commands: *gpu.CommandBuffer = undefined,
    readback_commands: *gpu.CommandBuffer = undefined,
    timeline: ?*gpu.TimelineSemaphore = null,
    upload: ?gpu.GpuHeap = null,
    readback: ?gpu.GpuHeap = null,
    texture_descriptors: ?*gpu.TextureDescriptorHeap = null,
    sampler_descriptors: ?*gpu.SamplerDescriptorHeap = null,
    textures: [textures_per_thread]?*gpu.Texture = @splat(null),
    views: [textures_per_thread]?*gpu.RenderView = @splat(null),

    fn timestamps(context: *const RecordingContext) [*]u64 {
        return context.readback.?.range.subrange(u64, textures_per_thread * texture_bytes, @sizeOf(u64) * 2).cpu.?;
    }

    fn uploadWords(context: *const RecordingContext) []u32 {
        return context.upload.?.range.subrange(u32, 0, textures_per_thread * texture_bytes).cpu.?[0 .. textures_per_thread * texture_words];
    }

    fn readbackWords(context: *const RecordingContext) []u32 {
        return context.readback.?.range.subrange(u32, 0, textures_per_thread * texture_bytes).cpu.?[0 .. textures_per_thread * texture_words];
    }
};

fn recordTextures(context: *RecordingContext, start: *const std.atomic.Value(bool)) void {
    waitForStart(start);
    gpu.setErrorCallback(printError);
    context.result = tryRecordTextures(context);
}

fn tryRecordTextures(context: *RecordingContext) gpu.Error!void {
    const device = context.device;
    const pool = try gpu.createCommandPool(device, 0);
    context.pool = pool;
    context.upload_commands = try gpu.beginCommands(pool);
    context.timeline = try gpu.createTimelineSemaphore(device, 0);
    context.upload = try gpu.createGpuHeap(device, textures_per_thread * texture_bytes, .cpu_visible);
    context.readback = try gpu.createGpuHeap(device, textures_per_thread * texture_bytes + @sizeOf(u64) * 2, .readback);
    const texture_descriptors = try gpu.createTextureDescriptorHeap(device, textures_per_thread);
    context.texture_descriptors = texture_descriptors;
    const sampler_descriptors = try gpu.createSamplerDescriptorHeap(device, textures_per_thread);
    context.sampler_descriptors = sampler_descriptors;
    for (context.uploadWords(), 0..) |*word, index| word.* = 0x12340000 + context.index * 0x1000 + @as(u32, @intCast(index));
    @memset(context.readback.?.range.cpu.?[0 .. textures_per_thread * texture_bytes], 0xcd);
    @memset(context.timestamps()[0..2], std.math.maxInt(u64));

    const upload = context.upload.?.range;
    for (0..textures_per_thread) |index| {
        const offset = (context.index * textures_per_thread + index) * context.texture_stride;
        const texture = try gpu.createTexture(context.upload_commands, texture_desc, context.texture_heap, offset);
        context.textures[index] = texture;
        context.views[index] = try gpu.createRenderView(texture, .{});
        gpu.writeTextureDescriptor(texture_descriptors, @intCast(index), texture, .sampled, .{});
        gpu.writeSamplerDescriptor(sampler_descriptors, @intCast(index), .{});
        gpu.copyMemoryToTexture(context.upload_commands, upload.subrange(u8, index * texture_bytes, texture_bytes).gpuRange(), texture, .{});
    }
    gpu.writeTimestamp(context.upload_commands, &context.timestamps()[0], all_commands);
    gpu.endCommands(context.upload_commands);
    context.readback_commands = try gpu.beginCommands(pool);
    const readback = context.readback.?.range;
    for (context.textures, 0..) |texture, index|
        gpu.copyTextureToMemory(context.readback_commands, texture.?, readback.subrange(u8, index * texture_bytes, texture_bytes).gpuRange(), .{});
    gpu.barrier(context.readback_commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
    gpu.writeTimestamp(context.readback_commands, &context.timestamps()[1], all_commands);
    gpu.endCommands(context.readback_commands);
}

fn destroyRecordingResources(context: *RecordingContext) void {
    if (context.pool) |pool| gpu.destroyCommandPool(pool);
    for (context.views, context.textures) |view, texture| {
        if (view) |v| gpu.destroyRenderView(v);
        if (texture) |t| gpu.destroyTexture(t);
    }
    if (context.sampler_descriptors) |heap| gpu.destroySamplerDescriptorHeap(heap);
    if (context.texture_descriptors) |heap| gpu.destroyTextureDescriptorHeap(heap);
    if (context.readback) |heap| gpu.destroyGpuHeap(heap);
    if (context.upload) |heap| gpu.destroyGpuHeap(heap);
    if (context.timeline) |timeline| gpu.destroyTimelineSemaphore(timeline);
}

fn testParallelRecording(device: *gpu.Device) !void {
    const requirements = try gpu.getTextureSizeAlign(device, texture_desc);
    const texture_stride = std.mem.alignForward(u64, requirements.size, requirements.alignment);
    const texture_heap = try gpu.createTextureHeap(device, thread_count * textures_per_thread * texture_stride);
    defer gpu.destroyTextureHeap(texture_heap);
    var contexts: [thread_count]RecordingContext = undefined;
    for (&contexts, 0..) |*context, index|
        context.* = .{ .device = device, .texture_heap = texture_heap, .texture_stride = texture_stride, .index = @intCast(index) };
    defer for (&contexts) |*context| destroyRecordingResources(context);
    try runThreads(recordTextures, &contexts);
    for (contexts) |context| try context.result;

    const pool = try gpu.createCommandPool(device, 0);
    errdefer gpu.destroyCommandPool(pool);
    const held_commands = try gpu.beginCommands(pool);
    const completion: gpu.TimelinePoint = .{ .semaphore = contexts[thread_count - 1].timeline.?, .value = 3 };
    const waiter = try std.Thread.spawn(.{}, gpu.waitTimeline, .{gpu.TimelinePoint{ .semaphore = contexts[0].timeline.?, .value = 2 }});
    for (contexts) |context| {
        // Submit independent ended buffers while held_commands is still recording.
        const timeline = context.timeline.?;
        gpu.submit(device, .{ .commands = &.{context.upload_commands}, .completion = .{ .semaphore = timeline, .value = 1 } }, 0);
        gpu.submit(device, .{
            .commands = &.{context.readback_commands},
            .waits = &.{.{ .semaphore = timeline, .value = 1 }},
            .completion = .{ .semaphore = timeline, .value = 2 },
        }, 0);
    }
    gpu.endCommands(held_commands);
    gpu.submit(device, .{ .commands = &.{held_commands}, .completion = completion }, 0);
    waiter.join();
    gpu.waitTimeline(completion);
    gpu.destroyCommandPool(pool);

    var valid = true;
    for (contexts, 0..) |context, index| {
        gpu.waitTimeline(.{ .semaphore = context.timeline.?, .value = 2 });
        gpu.readTimestamps(context.pool.?);
        const timestamps = context.timestamps();
        if (timestamps[0] == std.math.maxInt(u64) or timestamps[1] == std.math.maxInt(u64) or timestamps[0] > timestamps[1]) {
            std.debug.print("Worker {} timestamp readback failed.\n", .{index});
            valid = false;
        }
        const expected = context.uploadWords();
        const received = context.readbackWords();
        if (std.mem.indexOfDiff(u32, expected, received)) |word| {
            std.debug.print("Worker {} texture {} word {}: expected {x:0>8}, received {x:0>8}.\n", .{
                index, word / texture_words, word, expected[word], received[word],
            });
            std.debug.print("Readback words 0={x:0>8} 64={x:0>8} last={x:0>8}.\n", .{ received[0], received[64], received[received.len - 1] });
            valid = false;
        }
    }
    try testing.expect(valid);
}

const QueueContext = struct {
    device: *gpu.Device,
    index: u32,
    result: anyerror!void = {},
};

fn submitCopies(context: *QueueContext, start: *const std.atomic.Value(bool)) void {
    waitForStart(start);
    gpu.setErrorCallback(printError);
    context.result = trySubmitCopies(context);
}

fn trySubmitCopies(context: *QueueContext) !void {
    const device = context.device;
    const pool = try gpu.createCommandPool(device, context.index);
    defer gpu.destroyCommandPool(pool);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    const upload = try gpu.createGpuHeap(device, texture_bytes, .cpu_visible);
    defer gpu.destroyGpuHeap(upload);
    const scratch = try gpu.createGpuHeap(device, texture_bytes, .gpu_only);
    defer gpu.destroyGpuHeap(scratch);
    const readback = try gpu.createGpuHeap(device, texture_bytes + @sizeOf(u64), .readback);
    defer gpu.destroyGpuHeap(readback);
    const upload_words = upload.range.subrange(u32, 0, texture_bytes).cpu.?[0..texture_words];
    const timestamp = &readback.range.subrange(u64, texture_bytes, @sizeOf(u64)).cpu.?[0];
    var valid = true;
    for (1..33) |iteration| {
        for (upload_words, 0..) |*word, index| word.* = context.index * 0x10000 + @as(u32, @intCast(iteration * 0x100 + index));
        timestamp.* = std.math.maxInt(u64);
        const commands = try gpu.beginCommands(pool);
        gpu.copyMemory(commands, upload.gpuRange(), scratch.gpuRange());
        gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
        gpu.copyMemory(commands, scratch.gpuRange(), .{ .gpu = readback.range.gpu, .size = texture_bytes });
        gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.writeTimestamp(commands, timestamp, all_commands);
        gpu.endCommands(commands);
        gpu.submit(device, .{ .commands = &.{commands}, .completion = .{ .semaphore = timeline, .value = iteration } }, context.index);
        gpu.waitTimeline(.{ .semaphore = timeline, .value = iteration });
        gpu.readTimestamps(pool);
        valid = std.mem.eql(u8, upload.range.cpu.?[0..texture_bytes], readback.range.cpu.?[0..texture_bytes]) and valid;
        valid = timestamp.* != std.math.maxInt(u64) and valid;
        gpu.resetCommandPool(pool);
    }
    if (!valid) return error.TestUnexpectedResult;
}

fn testMultipleQueues(device: *gpu.Device, queue_families: bool) !void {
    var contexts: [3]QueueContext = .{ .{ .device = device, .index = 0 }, .{ .device = device, .index = 1 }, .{ .device = device, .index = 2 } };
    try runThreads(submitCopies, contexts[0..if (queue_families) 3 else 2]);
    for (contexts) |context| try context.result;

    const producer_pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(producer_pool);
    const consumer_pool = try gpu.createCommandPool(device, 1);
    defer gpu.destroyCommandPool(consumer_pool);
    const produced = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(produced);
    const consumed = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(consumed);
    const upload = try gpu.createGpuHeap(device, texture_bytes, .cpu_visible);
    defer gpu.destroyGpuHeap(upload);
    const scratch = try gpu.createGpuHeap(device, texture_bytes, .gpu_only);
    defer gpu.destroyGpuHeap(scratch);
    const readback = try gpu.createGpuHeap(device, texture_bytes, .readback);
    defer gpu.destroyGpuHeap(readback);
    for (upload.range.subrange(u32, 0, texture_bytes).cpu.?[0..texture_words], 0..) |*word, index| word.* = 0xcafe0000 + @as(u32, @intCast(index));
    @memset(readback.range.cpu.?[0..texture_bytes], 0);
    const producer = try gpu.beginCommands(producer_pool);
    gpu.copyMemory(producer, upload.gpuRange(), scratch.gpuRange());
    gpu.endCommands(producer);
    const consumer = try gpu.beginCommands(consumer_pool);
    gpu.copyMemory(consumer, scratch.gpuRange(), readback.gpuRange());
    gpu.barrier(consumer, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
    gpu.endCommands(consumer);
    // Submit the consumer first: its cross-queue wait supplies the memory dependency.
    gpu.submit(device, .{
        .commands = &.{consumer},
        .waits = &.{.{ .semaphore = produced, .value = 1 }},
        .completion = .{ .semaphore = consumed, .value = 1 },
    }, 1);
    gpu.submit(device, .{ .commands = &.{producer}, .completion = .{ .semaphore = produced, .value = 1 } }, 0);
    gpu.waitTimeline(.{ .semaphore = consumed, .value = 1 });
    gpu.waitTimeline(.{ .semaphore = produced, .value = 1 });
    try testing.expectEqualSlices(u8, upload.range.cpu.?[0..texture_bytes], readback.range.cpu.?[0..texture_bytes]);
}

fn createTestDevice(queue_families: bool) !*gpu.Device {
    gpu.setErrorCallback(printError);
    const device = gpu.createDevice(.{
        .desired_queue_count = if (queue_families) 1 else 32,
        .desired_compute_queue_count = if (queue_families) 1 else 0,
        .desired_copy_queue_count = if (queue_families) 1 else 0,
    }) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
    const queue_count = gpu.getDeviceCaps(device).queue_count;
    std.debug.print("Device exposes {} queue(s).\n", .{queue_count});
    if (queue_count < 1 or queue_count > 32) {
        gpu.destroyDevice(device);
        return error.TestUnexpectedResult;
    }
    return device;
}

fn runMultipleQueues(queue_families: bool) !void {
    const device = try createTestDevice(queue_families);
    defer {
        gpu.waitIdle(device);
        gpu.destroyDevice(device);
    }
    const queue_count = gpu.getDeviceCaps(device).queue_count;
    if (queue_count < 2) {
        std.debug.print("Multiple-queue test skipped: device exposes {} compatible queue(s); parallel recording is tested separately.\n", .{queue_count});
        return error.SkipZigTest;
    }
    try testMultipleQueues(device, queue_families);
}

test "parallel recording" {
    const device = try createTestDevice(false);
    defer {
        gpu.waitIdle(device);
        gpu.destroyDevice(device);
    }
    try testParallelRecording(device);
}

test "multiple queues" {
    try runMultipleQueues(false);
}

test "queue family threads" {
    try runMultipleQueues(true);
}
