const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const shared = @import("queue_family_shared");
const testing = std.testing;

const width = shared.queue_test_width;
const height = shared.queue_test_height;
const slot_bytes = 2048;
const data_bytes = width * height * 4;
const timestamp_offset = 3 * slot_bytes;
const guard: u64 = 0xa5a5a5a5a5a5a5a5;
const all_commands: gpu.Stage = .{ .all_commands = true };

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

/// Producer (copy queue when present), compute (compute queue when present) and graphics work chained by timelines.
const Fixture = struct {
    device: *gpu.Device,
    /// Copy, compute and general queue indices; missing dedicated queues fall back to the general queue 0.
    queues: [3]u32,
    pools: [3]*gpu.CommandPool,
    signals: [4]*gpu.TimelineSemaphore,
    value: u64 = 1,
    compute: *gpu.Pso,
    upload: gpu.GpuHeap,
    data: gpu.GpuHeap,
    readback: gpu.GpuHeap,
    descriptors: *gpu.TextureDescriptorHeap,
    texture_heap: gpu.TextureHeap,
    textures: [2]*gpu.Texture,
    target: *gpu.RenderView,

    fn init(self: *Fixture, device: *gpu.Device) !void {
        const path = "shaders/queue_family.comp." ++ gpu.shader_extension;
        const code = try std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .limited(16 << 20), .@"4", null);
        defer testing.allocator.free(code);
        const caps = gpu.getDeviceCaps(device);
        self.* = .{
            .device = device,
            .queues = .{
                if (caps.copy_queue_count != 0) caps.general_queue_count + caps.compute_queue_count else 0,
                if (caps.compute_queue_count != 0) caps.general_queue_count else 0,
                0,
            },
            .pools = undefined,
            .signals = undefined,
            .compute = try gpu.createComputePso(device, .{ .code = code, .entry_point = "computeMain", .threadgroup_size = .{ .x = 8, .y = 8, .z = 1 } }),
            .upload = undefined,
            .data = undefined,
            .readback = undefined,
            .descriptors = undefined,
            .texture_heap = undefined,
            .textures = undefined,
            .target = undefined,
        };
        for (&self.pools, self.queues) |*pool, queue| pool.* = try gpu.createCommandPool(device, queue);
        for (&self.signals) |*signal| signal.* = try gpu.createTimelineSemaphore(device, 0);
        self.upload = try gpu.createGpuHeap(device, 2 * slot_bytes + @sizeOf(shared.QueueFamilyRoot), .cpu_visible);
        self.data = try gpu.createGpuHeap(device, 2 * slot_bytes, .gpu_only);
        self.readback = try gpu.createGpuHeap(device, 4 * slot_bytes, .readback);
        self.descriptors = try gpu.createTextureDescriptorHeap(device, 2);
        const description: gpu.TextureDesc = .{
            .extent = .{ .x = width, .y = height, .z = 1 },
            .usage = .{ .sampled = true, .storage = true, .color_attachment = true, .transfer_source = true, .transfer_destination = true },
        };
        const placement = try gpu.getTextureSizeAlign(device, description);
        const second_offset = std.mem.alignForward(u64, placement.size, placement.alignment);
        self.texture_heap = try gpu.createTextureHeap(device, second_offset + placement.size);
        const commands = try gpu.beginCommands(self.pools[0]);
        self.textures = .{
            try gpu.createTexture(commands, description, self.texture_heap, 0),
            try gpu.createTexture(commands, description, self.texture_heap, second_offset),
        };
        self.target = try gpu.createRenderView(self.textures[1], .{});
        gpu.writeTextureDescriptor(self.descriptors, 0, self.textures[0], .sampled, .{});
        gpu.writeTextureDescriptor(self.descriptors, 1, self.textures[1], .storage, .{});
        gpu.endCommands(commands);
        const point: gpu.TimelinePoint = .{ .semaphore = self.signals[0], .value = self.value };
        gpu.submit(device, .{ .commands = &.{commands}, .completion = point }, self.queues[0]);
        gpu.waitTimeline(point);
    }

    fn deinit(self: *Fixture) void {
        gpu.waitIdle(self.device);
        for (self.pools) |pool| gpu.destroyCommandPool(pool);
        for (self.signals) |signal| gpu.destroyTimelineSemaphore(signal);
        gpu.destroyRenderView(self.target);
        for (self.textures) |texture| gpu.destroyTexture(texture);
        gpu.destroyTextureHeap(self.texture_heap);
        gpu.destroyTextureDescriptorHeap(self.descriptors);
        gpu.destroyGpuHeap(self.readback);
        gpu.destroyGpuHeap(self.data);
        gpu.destroyGpuHeap(self.upload);
        gpu.destroyPso(self.compute);
    }

    fn range(heap: gpu.GpuHeap, offset: u64, size: u64) gpu.GpuRange {
        return .{ .gpu = heap.range.gpu.? + offset, .size = size };
    }

    fn runCase(self: *Fixture, iteration: u32) !void {
        self.value += 1;
        for (self.pools) |pool| gpu.resetCommandPool(pool);
        const upload = self.upload.range.cpu.?;
        const source: [*]u32 = @ptrCast(@alignCast(upload));
        for (0..data_bytes / 4) |i| {
            const index: u32 = @intCast(i);
            source[index] = iteration * 1000 + index;
            const pixel = upload[slot_bytes + index * 4 ..][0..4];
            pixel.* = .{ @truncate(index + iteration), @truncate(index * 3 + iteration), @truncate(index * 7 + iteration), 255 };
        }
        const readback = self.readback.range.cpu.?[0..self.readback.range.size];
        @memset(readback, 0xa5);
        const timestamps: *[8]u64 = @ptrCast(@alignCast(readback[timestamp_offset..]));

        const producer = try gpu.beginCommands(self.pools[0]);
        gpu.writeTimestamp(producer, &timestamps[0], all_commands);
        gpu.copyMemory(producer, range(self.upload, 0, data_bytes), range(self.data, 0, data_bytes));
        gpu.copyMemoryToTexture(producer, range(self.upload, slot_bytes, data_bytes), self.textures[0], .{});
        gpu.writeTimestamp(producer, &timestamps[1], all_commands);
        gpu.endCommands(producer);

        const compute = try gpu.beginCommands(self.pools[1]);
        gpu.writeTimestamp(compute, &timestamps[2], all_commands);
        gpu.bindPso(compute, self.compute);
        gpu.setTextureDescriptorHeap(compute, self.descriptors);
        const root: *shared.QueueFamilyRoot = @ptrCast(@alignCast(upload + 2 * slot_bytes));
        const data: [*]u8 = @ptrCast(self.data.range.gpu.?);
        root.* = .{ .source = @ptrCast(@alignCast(data)), .destination = @ptrCast(@alignCast(data + slot_bytes)), .destination_texture = 1 };
        gpu.dispatch(compute, self.upload.range.gpu.? + 2 * slot_bytes, .{ .x = (width + 7) / 8, .y = (height + 7) / 8, .z = 1 });
        gpu.writeTimestamp(compute, &timestamps[3], all_commands);
        gpu.endCommands(compute);

        const graphics = try gpu.beginCommands(self.pools[2]);
        gpu.writeTimestamp(graphics, &timestamps[4], all_commands);
        gpu.copyMemory(graphics, range(self.data, slot_bytes, data_bytes), range(self.readback, 0, data_bytes));
        gpu.copyTextureToMemory(graphics, self.textures[1], range(self.readback, slot_bytes, data_bytes), .{});
        gpu.barrier(graphics, .{ .transfer = true }, .{ .transfer_read = true }, .{ .color_output = true }, .{ .color_write = true });
        gpu.beginRenderPass(graphics, .{ .colors = &.{.{ .render_view = self.target, .load = .clear, .clear = .{ .x = 1, .w = 1 } }} }, .{});
        gpu.endRenderPass(graphics);
        gpu.writeTimestamp(graphics, &timestamps[5], all_commands);
        gpu.endCommands(graphics);

        const copy_back = try gpu.beginCommands(self.pools[0]);
        gpu.writeTimestamp(copy_back, &timestamps[6], all_commands);
        gpu.copyTextureToMemory(copy_back, self.textures[1], range(self.readback, 2 * slot_bytes, data_bytes), .{});
        gpu.barrier(copy_back, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.writeTimestamp(copy_back, &timestamps[7], all_commands);
        gpu.endCommands(copy_back);

        const produced: gpu.SubmitDesc = .{
            .commands = &.{producer},
            .waits = &.{.{ .semaphore = self.signals[if (iteration == 1) 0 else 3], .value = self.value - 1 }},
            .completion = .{ .semaphore = self.signals[0], .value = self.value },
        };
        const computed: gpu.SubmitDesc = .{
            .commands = &.{compute},
            .waits = &.{.{ .semaphore = self.signals[0], .value = self.value }},
            .completion = .{ .semaphore = self.signals[1], .value = self.value },
        };
        const rendered: gpu.SubmitDesc = .{
            .commands = &.{graphics},
            .waits = &.{.{ .semaphore = self.signals[1], .value = self.value }},
            .completion = .{ .semaphore = self.signals[2], .value = self.value },
        };
        // All three families can submit consumers before their producer without blocking the producer's queue.
        if (self.queues[0] != 0 and self.queues[1] != 0) {
            gpu.submit(self.device, rendered, 0);
            gpu.submit(self.device, computed, self.queues[1]);
            gpu.submit(self.device, produced, self.queues[0]);
        } else {
            gpu.submit(self.device, produced, self.queues[0]);
            gpu.submit(self.device, computed, self.queues[1]);
            gpu.submit(self.device, rendered, 0);
        }
        const done: gpu.TimelinePoint = .{ .semaphore = self.signals[3], .value = self.value };
        gpu.submit(self.device, .{
            .commands = &.{copy_back},
            .waits = &.{.{ .semaphore = self.signals[2], .value = self.value }},
            .completion = done,
        }, self.queues[0]);
        gpu.waitTimeline(done);
        for (self.pools) |pool| gpu.readTimestamps(pool);

        var valid = true;
        const results: [*]const u32 = @ptrCast(@alignCast(readback.ptr));
        for (0..data_bytes / 4) |i| {
            const index: u32 = @intCast(i);
            valid = valid and results[index] == (iteration * 1000 + index) * 3 + 7;
            const input = upload[slot_bytes + index * 4 ..][0..4];
            const pixel = readback[slot_bytes + index * 4 ..][0..4];
            valid = valid and std.mem.eql(u8, pixel, &.{ input[2], input[1], input[0], input[3] });
            valid = valid and std.mem.eql(u8, readback[2 * slot_bytes + index * 4 ..][0..4], &.{ 255, 0, 0, 255 });
        }
        for (0..4) |pair| {
            const begin = timestamps[pair * 2];
            const end = timestamps[pair * 2 + 1];
            valid = valid and begin != guard and end != guard and end >= begin;
        }
        for (readback, 0..) |byte, index| {
            const written = if (index < timestamp_offset) index % slot_bytes < data_bytes else index < timestamp_offset + 8 * @sizeOf(u64);
            valid = valid and (written or byte == 0xa5);
        }
        if (!valid) {
            std.debug.print("Queue-family data, texture, timestamp, or guard check failed at iteration {d}.\n", .{iteration});
            return error.TestUnexpectedResult;
        }
    }
};

fn run(compute: bool, copy: bool) !void {
    gpu.setErrorCallback(printError);
    const device = gpu.createDevice(.{
        .desired_compute_queue_count = if (compute) 32 else 0,
        .desired_copy_queue_count = if (copy) 32 else 0,
        .timestamp_query_count = 2,
    }) catch |err| {
        std.debug.print("Dedicated queue device creation: {t}.\n", .{err});
        return if (err == error.Unsupported) error.SkipZigTest else err;
    };
    defer gpu.destroyDevice(device);
    const caps = gpu.getDeviceCaps(device);
    std.debug.print("{s}: {d} general, {d} compute, {d} copy queues; copy granularity {d} x {d} x {d}.\n", .{
        caps.device_name,
        caps.general_queue_count,
        caps.compute_queue_count,
        caps.copy_queue_count,
        caps.copy_texture_granularity.x,
        caps.copy_texture_granularity.y,
        caps.copy_texture_granularity.z,
    });
    try testing.expect(caps.general_queue_count == 1 and caps.queue_count == 1 + caps.compute_queue_count + caps.copy_queue_count);
    try testing.expect(if (compute) caps.compute_queue_count >= 1 and caps.compute_queue_count <= 32 else caps.compute_queue_count == 0);
    try testing.expect(if (copy) caps.copy_queue_count >= 1 and caps.copy_queue_count <= 32 else caps.copy_queue_count == 0);
    var fixture: Fixture = undefined;
    try fixture.init(device);
    defer fixture.deinit();
    for (1..9) |iteration| try fixture.runCase(@intCast(iteration));
}

test "copy, compute and general queue families" {
    try run(true, true);
}

test "dedicated compute queue family" {
    try run(true, false);
}

test "dedicated copy queue family" {
    try run(false, true);
}
