const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const BumpAllocator = @import("NoGraphicsAPIUtility").BumpAllocator;
const shared = @import("root_data_generate_shared");
const testing = std.testing;
const RootData = shared.RootData;
const RootDataGenerate = shared.RootDataGenerate;

comptime {
    std.debug.assert(@sizeOf(RootData) == 256 and @offsetOf(RootData, "values") == 16);
    std.debug.assert(@sizeOf(RootDataGenerate) == 32);
}

fn loadCompute(device: *gpu.Device, path: []const u8, threads: u32) !*gpu.Pso {
    const code = try std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .unlimited, .@"4", null);
    defer testing.allocator.free(code);
    return gpu.createComputePso(device, .{ .code = code, .entry_point = "computeMain", .threadgroup_size = .{ .x = threads, .y = 1, .z = 1 } });
}

/// Application root allocation, interleaved recording, arena reset, GPU-written roots and indirect arguments.
fn run(consume_path: []const u8) !void {
    const device = gpu.createDevice(.{}) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
    defer gpu.destroyDevice(device);
    const consume = try loadCompute(device, consume_path, 64);
    defer gpu.destroyPso(consume);
    const generate = try loadCompute(device, "shaders/root_data_generate.comp." ++ gpu.shader_extension, 1);
    defer gpu.destroyPso(generate);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const external_pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(external_pool);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    const cpu_roots = try gpu.createGpuHeap(device, 3 * @sizeOf(RootData) + @sizeOf(RootDataGenerate), .cpu_visible);
    defer gpu.destroyGpuHeap(cpu_roots);
    var arena: BumpAllocator = .init(cpu_roots.range);
    const roots = try gpu.createGpuHeap(device, 2 * @sizeOf(RootData) + 16, .gpu_only);
    defer gpu.destroyGpuHeap(roots);
    const output = try gpu.createGpuHeap(device, 5 * 64 * @sizeOf(u32), .gpu_only);
    defer gpu.destroyGpuHeap(output);
    const readback = try gpu.createGpuHeap(device, output.range.size, .readback);
    defer gpu.destroyGpuHeap(readback);
    defer gpu.waitIdle(device);

    const roots_gpu = roots.range.gpu.?;
    const output_gpu: [*c]u32 = @ptrCast(@alignCast(output.range.gpu.?));
    for (0..4) |round_index| {
        const round: u32 = @intCast(round_index);
        const first = try gpu.beginCommands(pool);
        const second = try gpu.beginCommands(pool);
        gpu.bindPso(first, consume);
        gpu.bindPso(second, consume);
        var root: RootData = .{ .output = output_gpu };
        for (0..3) |i| {
            root.output_index = @intCast(i * 64);
            root.seed = round * 100 + @as(u32, @intCast(i));
            for (&root.values, 0..) |*value, j| value.* = root.seed * 7 + @as(u32, @intCast(j));
            const copied = arena.allocate(RootData, 1).?;
            copied.cpu.?[0] = root;
            gpu.dispatch(if (i == 1) second else first, copied.gpu, .{ .x = 1, .y = 1, .z = 1 });
        }
        gpu.bindPso(second, generate);
        const generated = arena.allocateAtomic(RootDataGenerate, 1).?;
        generated.cpu.?[0] = .{
            .destination = @ptrCast(@alignCast(roots_gpu)),
            .output = output_gpu,
            .dispatch_arguments = @ptrCast(@alignCast(roots_gpu + 2 * @sizeOf(RootData))),
            .first_index = 3,
            .seed = round * 100 + 3,
        };
        gpu.dispatch(second, generated.gpu, .{ .x = 2, .y = 1, .z = 1 });
        gpu.endCommands(first);
        gpu.endCommands(second);
        const third = try gpu.beginCommands(external_pool);
        const consumers: gpu.Stage = .{ .compute = true, .indirect = true };
        gpu.barrier(third, .{ .compute = true }, .{ .shader_write = true }, consumers, .{ .shader_read = true, .indirect_read = true });
        gpu.bindPso(third, consume);
        gpu.dispatch(third, roots_gpu, .{ .x = 1, .y = 1, .z = 1 });
        gpu.dispatchIndirect(third, roots_gpu + @sizeOf(RootData), .{ .gpu = roots_gpu + 2 * @sizeOf(RootData), .size = 12 });
        gpu.barrier(third, .{ .compute = true }, .{ .shader_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
        gpu.copyMemory(third, output.gpuRange(), readback.gpuRange());
        gpu.barrier(third, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.endCommands(third);
        gpu.submit(device, .{ .commands = &.{ first, second, third }, .completion = .{ .semaphore = timeline, .value = round + 1 } }, 0);
        gpu.waitTimeline(.{ .semaphore = timeline, .value = round + 1 });
        const actual: [*]const u32 = @ptrCast(@alignCast(readback.range.cpu.?));
        for (0..5) |i| {
            const seed = round * 100 + @as(u32, @intCast(i));
            for (0..64) |lane| {
                var expected = seed + @as(u32, @intCast(lane));
                for (0..60) |j| expected = (expected *% 33) ^ (seed * 7 + @as(u32, @intCast(j)));
                try testing.expectEqual(expected, actual[i * 64 + lane]);
            }
        }
        gpu.resetCommandPool(pool);
        gpu.resetCommandPool(external_pool);
        arena.reset();
    }
}

test "root data from application arenas and GPU-generated roots" {
    try run("shaders/root_data.comp." ++ gpu.shader_extension);
}

test "root data with aligned root layout" {
    try run("shaders/root_data_aligned.comp." ++ gpu.shader_extension);
}
