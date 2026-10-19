const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const shader_types = @import("shader_types");
const testing = std.testing;

// Mirrors tests/task_shader_common.h, whose array sizes are not valid C and cannot go through translate-c.
const task_test_batch_size = 32;
const task_test_capacity = 96;
const task_test_batches = task_test_capacity / task_test_batch_size;
const task_test_width = task_test_capacity * 4;
const task_test_height = 8;

/// Mirrors TaskTestData in tests/task_shader_common.h.
const TaskTestData = extern struct {
    visibility: [task_test_capacity]u32,
    batch_counts: [task_test_batches]u32,
    visits: [task_test_capacity + 1]u32,
};

/// Mirrors TaskShaderRoot in tests/task_shader_shared.h and TaskMeshRoot in tests/task_mesh_shared.h.
const TaskRoot = extern struct {
    data: ?*TaskTestData,
    count: u32,
    visible_mask: u32,
    color: shader_types.float4,
};

comptime {
    std.debug.assert(@sizeOf(TaskRoot) == 32 and @offsetOf(TaskRoot, "count") == 8 and @offsetOf(TaskRoot, "visible_mask") == 12);
    std.debug.assert(@offsetOf(TaskRoot, "color") == 16 and @sizeOf(TaskTestData) == 784);
}

const pixel_offset = 1024;
const pixel_bytes = task_test_width * task_test_height * 4;
const data_bytes = @sizeOf(TaskTestData) + @sizeOf(gpu.Uint32x3);
const root_offset = std.mem.alignForward(u64, data_bytes, 16);

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

fn loadShader(path: []const u8) ![]align(4) u8 {
    const code = try std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .unlimited, .@"4", null);
    if (code.len < 20 or code.len % 4 != 0 or std.mem.bytesAsSlice(u32, code)[0] != 0x07230203) {
        testing.allocator.free(code);
        return error.InvalidShader;
    }
    return code;
}

const Fixture = struct {
    device: *gpu.Device,
    pool: *gpu.CommandPool,
    pso: *gpu.Pso,
    target: *gpu.Texture,
    view: *gpu.RenderView,
    seed: gpu.GpuHeap,
    data: gpu.GpuHeap,
    readback: gpu.GpuHeap,
    completion: gpu.TimelinePoint,

    fn renderCase(self: *Fixture, count: u32, visible_mask: u32, scenario: u32, indirect: bool) !void {
        const seed: *TaskTestData = @ptrCast(@alignCast(self.seed.range.cpu.?));
        seed.* = std.mem.zeroes(TaskTestData);
        for (&seed.visibility, 0..) |*visibility, id| visibility.* = @as(u32, 1) << @intCast(id % 3);
        @memset(&seed.batch_counts, ~@as(u32, 0));
        const batches: u32 = if (count != 0) std.math.divCeil(u32, count, task_test_batch_size) catch unreachable else 1;
        const group_count: gpu.Uint32x3 = .{ .x = batches, .y = 1, .z = 1 };
        @as(*align(1) gpu.Uint32x3, @ptrCast(self.seed.range.cpu.? + @sizeOf(TaskTestData))).* = group_count;
        const readback = self.readback.range.cpu.?[0..self.readback.range.size];
        @memset(readback, 0xa5);
        const seed_gpu = self.seed.range.gpu.?;
        const data_gpu = self.data.range.gpu.?;

        gpu.resetCommandPool(self.pool);
        const commands = try gpu.beginCommands(self.pool);
        gpu.barrier(commands, .{ .task = true, .mesh = true, .transfer = true }, .{ .shader_write = true, .transfer_read = true }, .{ .transfer = true }, .{
            .transfer_write = true,
        });
        @as(*TaskRoot, @ptrCast(@alignCast(self.seed.range.cpu.? + root_offset))).* = .{
            .data = @ptrCast(@alignCast(data_gpu)),
            .count = count,
            .visible_mask = visible_mask,
            .color = .{ .x = 1, .y = @floatFromInt(scenario & 1), .z = @floatFromInt(@intFromBool(indirect)), .w = 1 },
        };
        gpu.copyMemory(commands, .{ .gpu = seed_gpu, .size = data_bytes }, self.data.gpuRange());
        gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .task = true, .mesh = true, .indirect = true }, .{
            .shader_read = true,
            .shader_write = true,
            .indirect_read = true,
        });
        gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_read = true }, .{ .color_output = true }, .{ .color_write = true });
        gpu.beginRenderPass(commands, .{ .colors = &.{.{ .render_view = self.view, .load = .clear }} }, .{});
        gpu.setViewport(commands, .{ .width = task_test_width, .height = task_test_height });
        gpu.setScissor(commands, .{ .width = task_test_width, .height = task_test_height });
        gpu.bindPso(commands, self.pso);
        if (indirect) {
            const arguments: gpu.GpuRange = .{ .gpu = data_gpu + @sizeOf(TaskTestData), .size = @sizeOf(gpu.Uint32x3) };
            gpu.drawMeshletsIndirect(commands, seed_gpu + root_offset, .{ .arguments = arguments });
        } else {
            gpu.drawMeshlets(commands, seed_gpu + root_offset, group_count);
        }
        gpu.endRenderPass(commands);
        gpu.barrier(commands, .{ .task = true, .mesh = true }, .{ .shader_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
        gpu.barrier(commands, .{ .color_output = true }, .{ .color_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
        const readback_gpu = self.readback.range.gpu.?;
        gpu.copyMemory(commands, .{ .gpu = data_gpu, .size = @sizeOf(TaskTestData) }, .{ .gpu = readback_gpu, .size = @sizeOf(TaskTestData) });
        gpu.copyTextureToMemory(commands, self.target, .{ .gpu = readback_gpu + pixel_offset, .size = pixel_bytes }, .{});
        gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.endCommands(commands);
        self.completion.value += 1;
        gpu.submit(self.device, .{ .commands = &.{commands}, .completion = self.completion }, 0);
        gpu.waitTimeline(self.completion);

        const actual: *const TaskTestData = @ptrCast(@alignCast(readback.ptr));
        var valid = std.mem.eql(u32, &actual.visibility, &seed.visibility) and actual.visits[task_test_capacity] == 0;
        for (0..task_test_batches) |batch| {
            var expected: u32 = 0;
            for (0..task_test_batch_size) |lane| {
                const id = batch * task_test_batch_size + lane;
                const visible: u32 = @intFromBool(id < count and seed.visibility[id] & visible_mask != 0);
                expected += visible;
                valid = valid and actual.visits[id] == visible;
            }
            valid = valid and actual.batch_counts[batch] == if (batch < batches) expected else ~@as(u32, 0);
        }
        for (readback[@sizeOf(TaskTestData)..pixel_offset]) |byte| valid = valid and byte == 0xa5;
        const pixels = readback[pixel_offset..][0..pixel_bytes];
        for (0..task_test_height) |y| for (0..task_test_width) |x| {
            const covered = x / 4 < count and seed.visibility[x / 4] & visible_mask != 0 and (x % 4 == 1 or x % 4 == 2) and y >= 2 and y < 6;
            const pixel = pixels[(y * task_test_width + x) * 4 ..][0..4];
            const expected = [4]u8{ if (covered) 255 else 0, if (covered and scenario & 1 != 0) 255 else 0, if (covered and indirect) 255 else 0, 255 };
            valid = valid and std.mem.eql(u8, pixel, &expected);
        };
        if (!valid) {
            const kind = if (indirect) "indirect" else "direct";
            std.debug.print("Task shader {s} case {d} failed: count={d} mask={d}.\n", .{ kind, scenario, count, visible_mask });
            return error.TestUnexpectedResult;
        }
    }
};

test "task shader compaction, culling and root ABI" {
    gpu.setErrorCallback(printError);
    const device = gpu.createDevice(.{ .timestamp_query_count = 0 }) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
    defer gpu.destroyDevice(device);
    defer gpu.waitIdle(device);
    const test_indirect = gpu.getDeviceCaps(device).indirect_mesh_draw;
    if (!test_indirect) std.debug.print("Native indirect mesh draws unavailable; testing direct task/mesh draws.\n", .{});

    const task = try loadShader("shaders/task_shader.task." ++ gpu.shader_extension);
    defer testing.allocator.free(task);
    const mesh = try loadShader("shaders/task_mesh.mesh." ++ gpu.shader_extension);
    defer testing.allocator.free(mesh);
    const fragment = try loadShader("shaders/task_mesh.frag." ++ gpu.shader_extension);
    defer testing.allocator.free(fragment);
    const threadgroup_size: gpu.Uint32x3 = .{ .x = task_test_batch_size, .y = 1, .z = 1 };
    const pso = try gpu.createMeshPso(device, .{
        .task = .{ .code = task, .entry_point = "taskMain", .threadgroup_size = threadgroup_size },
        .mesh = .{ .code = mesh, .entry_point = "meshMain", .threadgroup_size = threadgroup_size },
        .fragment = .{ .code = fragment, .entry_point = "fragmentMain" },
        .color_targets = &.{.{ .format = .rgba8_unorm }},
    });
    defer gpu.destroyPso(pso);

    const target_desc: gpu.TextureDesc = .{
        .extent = .{ .x = task_test_width, .y = task_test_height, .z = 1 },
        .usage = .{ .color_attachment = true, .transfer_source = true },
    };
    const texture_heap = try gpu.createTextureHeap(device, (try gpu.getTextureSizeAlign(device, target_desc)).size);
    defer gpu.destroyTextureHeap(texture_heap);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const commands = try gpu.beginCommands(pool);
    const target = try gpu.createTexture(commands, target_desc, texture_heap, 0);
    defer gpu.destroyTexture(target);
    const view = try gpu.createRenderView(target, .{});
    defer gpu.destroyRenderView(view);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    const seed = try gpu.createGpuHeap(device, root_offset + @sizeOf(TaskRoot), .cpu_visible);
    defer gpu.destroyGpuHeap(seed);
    const data = try gpu.createGpuHeap(device, data_bytes, .gpu_only);
    defer gpu.destroyGpuHeap(data);
    const readback = try gpu.createGpuHeap(device, pixel_offset + pixel_bytes, .readback);
    defer gpu.destroyGpuHeap(readback);
    gpu.endCommands(commands);
    var fixture: Fixture = .{
        .device = device,
        .pool = pool,
        .pso = pso,
        .view = view,
        .target = target,
        .seed = seed,
        .data = data,
        .readback = readback,
        .completion = .{ .semaphore = timeline, .value = 1 },
    };
    gpu.submit(device, .{ .commands = &.{commands}, .completion = fixture.completion }, 0);
    gpu.waitTimeline(fixture.completion);

    const counts = [_]u32{ 0, 1, 32, 33, 70, 70, 70 };
    const masks = [_]u32{ 7, 7, 7, 5, 7, 2, 0 };
    for (0..if (test_indirect) 2 else 1) |indirect| {
        for (counts, masks, 0..) |count, mask, scenario| try fixture.renderCase(count, mask, @intCast(scenario), indirect != 0);
    }
    std.debug.print("Task shader compaction, culling and root ABI: passed\n", .{});
}
