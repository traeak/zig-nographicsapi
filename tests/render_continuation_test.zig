const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const testing = std.testing;

const segment_count = 3;
const width = segment_count * 8;
const height = 8;
const pixel_bytes = width * height * 4;
const timestamp_bytes = segment_count * 2 * @sizeOf(u64);
const all_commands: gpu.Stage = .{ .all_commands = true };

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

const Worker = struct {
    pool: *gpu.CommandPool,
    commands: *gpu.CommandBuffer = undefined,
    pso: *gpu.Pso = undefined,
    rendering: *const gpu.RenderingDesc = undefined,
    timestamps: *[2]u64 = undefined,
    index: u32 = 0,
    iteration: u32 = 0,
};

/// Segment `index` of the render pass draws a 4-pixel-wide red column whose position alternates per iteration;
/// even iterations also time the segment.
fn drawSegment(commands: *gpu.CommandBuffer, worker: *const Worker) void {
    const timed = worker.iteration & 1 == 0;
    gpu.beginRenderPass(commands, worker.rendering.*, .{ .resuming = worker.index != 0, .suspending = worker.index != segment_count - 1 });
    if (timed) gpu.writeTimestamp(commands, &worker.timestamps[0], all_commands);
    gpu.bindPso(commands, worker.pso);
    gpu.setScissor(commands, .{ .x = @intCast(worker.index * 8 + (worker.iteration & 1) * 4), .width = 4, .height = height });
    gpu.draw(commands, null, .{ .vertex_count = 3 });
    if (timed) gpu.writeTimestamp(commands, &worker.timestamps[1], all_commands);
    gpu.endRenderPass(commands);
}

fn recordSegment(worker: *Worker, start: *const std.atomic.Value(bool)) void {
    while (!start.load(.acquire)) std.Thread.yield() catch {};
    gpu.resetCommandPool(worker.pool);
    worker.commands = gpu.beginCommands(worker.pool) catch @panic("command recording failed");
    if (worker.index == 0) gpu.barrier(worker.commands, .{ .transfer = true }, .{ .transfer_read = true }, .{ .color_output = true }, .{ .color_write = true });
    drawSegment(worker.commands, worker);
    gpu.endCommands(worker.commands);
}

/// Records every segment on its own thread, released together, and joins every started worker.
fn recordWorkers(workers: *[segment_count]Worker) !void {
    var start: std.atomic.Value(bool) = .init(false);
    var threads: [segment_count]std.Thread = undefined;
    var started: usize = 0;
    const result = while (started != workers.len) : (started += 1) {
        threads[started] = std.Thread.spawn(.{}, recordSegment, .{ &workers[started], &start }) catch |err| break err;
    } else {};
    start.store(true, .release);
    for (threads[0..started]) |thread| thread.join();
    result catch |err| {
        std.debug.print("Render continuation worker creation failed.\n", .{});
        return err;
    };
}

fn loadShader(path: []const u8) ![]align(4) u8 {
    return std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .limited(16 * 1024 * 1024), .@"4", null);
}

test "parallel render continuation, timestamp readback and pool reuse" {
    gpu.setErrorCallback(printError);
    const device = gpu.createDevice(.{ .timestamp_query_count = segment_count * 2 }) catch |err|
        return if (err == error.Unsupported) error.SkipZigTest else err;
    defer gpu.destroyDevice(device);
    defer gpu.waitIdle(device);

    const vertex = try loadShader("shaders/render_continuation.vert." ++ gpu.shader_extension);
    defer testing.allocator.free(vertex);
    const fragment = try loadShader("shaders/render_continuation.frag." ++ gpu.shader_extension);
    defer testing.allocator.free(fragment);
    const pso = try gpu.createGraphicsPso(device, .{
        .vertex = .{ .code = vertex, .entry_point = "vertexMain" },
        .fragment = .{ .code = fragment, .entry_point = "fragmentMain" },
        .color_targets = &.{.{ .format = .rgba8_unorm }},
    });
    defer gpu.destroyPso(pso);
    const readback_pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(readback_pool);
    var workers: [segment_count]Worker = undefined;
    for (&workers, 0..) |*worker, index| {
        errdefer for (workers[0..index]) |created| gpu.destroyCommandPool(created.pool);
        worker.* = .{ .pool = try gpu.createCommandPool(device, 0) };
    }
    defer for (workers) |worker| gpu.destroyCommandPool(worker.pool);
    var completion: gpu.TimelinePoint = .{ .semaphore = try gpu.createTimelineSemaphore(device, 0), .value = 0 };
    defer gpu.destroyTimelineSemaphore(completion.semaphore);
    const readback = try gpu.createGpuHeap(device, pixel_bytes + timestamp_bytes, .readback);
    defer gpu.destroyGpuHeap(readback);
    const target_desc: gpu.TextureDesc = .{ .extent = .{ .x = width, .y = height, .z = 1 }, .usage = .{ .color_attachment = true, .transfer_source = true } };
    const texture_heap = try gpu.createTextureHeap(device, (try gpu.getTextureSizeAlign(device, target_desc)).size);
    defer gpu.destroyTextureHeap(texture_heap);
    const setup = try gpu.beginCommands(readback_pool);
    const target = try gpu.createTexture(setup, target_desc, texture_heap, 0);
    defer gpu.destroyTexture(target);
    const view = try gpu.createRenderView(target, .{});
    defer gpu.destroyRenderView(view);
    gpu.endCommands(setup);
    completion.value += 1;
    gpu.submit(device, .{ .commands = &.{setup}, .completion = completion }, 0);
    gpu.waitTimeline(completion);

    const bytes = readback.range.cpu.?[0 .. pixel_bytes + timestamp_bytes];
    const timestamps: *[segment_count * 2]u64 = @ptrCast(@alignCast(bytes[pixel_bytes..]));
    const rendering: gpu.RenderingDesc = .{ .colors = &.{.{ .render_view = view, .load = .clear }} };
    var previous_timestamp: u64 = std.math.maxInt(u64);
    for (0..6) |i| {
        const iteration: u32 = @intCast(i);
        @memset(bytes, 0xff);
        for (&workers, 0..) |*worker, index| {
            worker.pso = pso;
            worker.rendering = &rendering;
            worker.timestamps = timestamps[index * 2 ..][0..2];
            worker.index = @intCast(index);
            worker.iteration = iteration;
        }
        // The first iterations record one segment per pool on worker threads; later ones record every segment into one buffer.
        const parallel = iteration < 4;
        if (parallel) {
            try recordWorkers(&workers);
        } else {
            gpu.resetCommandPool(workers[0].pool);
            workers[0].commands = try gpu.beginCommands(workers[0].pool);
            gpu.barrier(workers[0].commands, .{ .transfer = true }, .{ .transfer_read = true }, .{ .color_output = true }, .{ .color_write = true });
            for (&workers) |*worker| drawSegment(workers[0].commands, worker);
            gpu.endCommands(workers[0].commands);
        }
        gpu.resetCommandPool(readback_pool);
        const copy = try gpu.beginCommands(readback_pool);
        gpu.barrier(copy, .{ .color_output = true }, .{ .color_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
        gpu.copyTextureToMemory(copy, target, .{ .gpu = readback.range.gpu, .size = pixel_bytes }, .{});
        gpu.barrier(copy, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.endCommands(copy);
        completion.value += 1;
        const commands = [_]*gpu.CommandBuffer{ workers[0].commands, workers[1].commands, workers[2].commands, copy };
        const submitted: []const *gpu.CommandBuffer = if (parallel) &commands else &.{ workers[0].commands, copy };
        gpu.submit(device, .{ .commands = submitted, .completion = completion }, 0);
        gpu.waitTimeline(completion);
        for (workers[0..if (parallel) segment_count else 1]) |worker| gpu.readTimestamps(worker.pool);

        var valid = true;
        for (0..height) |y| for (0..width) |x| {
            const pixel = bytes[(y * width + x) * 4 ..][0..4];
            const covered = x % 8 / 4 == iteration & 1;
            valid = valid and std.mem.eql(u8, pixel, &.{ if (covered) 255 else 0, 0, 0, 255 });
        };
        for (timestamps, 0..) |timestamp, index| {
            if (iteration & 1 != 0) {
                valid = valid and timestamp == std.math.maxInt(u64);
            } else {
                valid = valid and timestamp != std.math.maxInt(u64) and (index == 0 or timestamp >= timestamps[index - 1]);
            }
        }
        if (iteration & 1 == 0) {
            valid = valid and timestamps[0] != previous_timestamp;
            previous_timestamp = timestamps[0];
        }
        if (!valid) {
            std.debug.print("Render continuation pixel or timestamp readback failed in iteration {d}.\n", .{iteration});
            return error.TestUnexpectedResult;
        }
    }
}
