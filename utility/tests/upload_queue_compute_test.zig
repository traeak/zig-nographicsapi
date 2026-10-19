const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const base = @import("upload_queue_test.zig");
const testing = std.testing;
const Fixture = base.Fixture;
const guard_bytes = base.guard_bytes;

/// Mirrors UploadQueueRoot in tests/shaders/upload_queue_shared.h.
const UploadQueueRoot = extern struct {
    source: ?[*]u32,
    destination: ?[*]u32,
    count: u32,
};

/// Mirrors upload_queue_thread_count in tests/shaders/upload_queue_shared.h.
const upload_queue_thread_count = 64;

/// Fills staging with `first + index * step` and dispatches the shader that adds it to `count` values of the fixture buffer.
fn accumulate(
    fixture: *Fixture,
    pso: *gpu.Pso,
    callbacks: *u32,
    count: u32,
    offset: u64,
    first: u32,
    step: u32,
    commands: *gpu.CommandBuffer,
    staging: gpu.GpuCpuRange(u8),
) void {
    callbacks.* += 1;
    fixture.check(staging.size == count * @sizeOf(u32) + 32 and @intFromPtr(staging.cpu) % 16 == 0 and
        @intFromPtr(staging.gpu) % 16 == 0, "callback receives the requested aligned staging range");
    const values: [*]u32 = @ptrCast(@alignCast(staging.cpu.?));
    for (values[0..count], 0..) |*value, index| value.* = first + @as(u32, @intCast(index)) * step;
    gpu.bindPso(commands, pso);
    const root: *UploadQueueRoot = @ptrCast(@alignCast(staging.cpu.? + count * @sizeOf(u32)));
    root.* = .{ .source = @ptrCast(@alignCast(staging.gpu.?)), .destination = @ptrCast(@alignCast(fixture.buffer.range.gpu.? + offset)), .count = count };
    gpu.dispatch(commands, staging.gpu.? + count * @sizeOf(u32), .{ .x = 1, .y = 1, .z = 1 });
}

fn runComputeCallbacks(fixture: *Fixture, pso: *gpu.Pso, queue_index: u32) !void {
    try fixture.replaceRing(.{ .capacity = 288, .queue_index = queue_index });
    const ring = &fixture.ring.?;
    try fixture.orderQueue(queue_index);
    var initial: [16]u32 = undefined;
    for (&initial, 0..) |*value, index| value.* = @as(u32, @intCast(index)) * 23 + 7;
    @memcpy(fixture.expected[guard_bytes..][0..@sizeOf(@TypeOf(initial))], std.mem.asBytes(&initial));
    ring.uploadBuffer(fixture.buffer.range.subrange(u8, 0, @sizeOf(@TypeOf(initial))).gpuRange(), std.mem.asBytes(&initial));
    var callbacks: u32 = 0;
    for (0..2) |i| {
        const iteration: u32 = @intCast(i);
        ring.uploadWithCompute(@sizeOf(@TypeOf(initial)) + 32, accumulate, .{ fixture, pso, &callbacks, initial.len, 0, iteration * 31 + 1, 1 });
        fixture.check(callbacks == iteration + 1, "compute callback executes synchronously exactly once");
        for (&initial, 0..) |*value, index| value.* += iteration * 31 + @as(u32, @intCast(index)) + 1;
    }
    @memcpy(fixture.expected[guard_bytes..][0..@sizeOf(@TypeOf(initial))], std.mem.asBytes(&initial));
    const replacement = [4]u32{ 0x12345678, 0x9abcdef0, 0x31415926, 0x27182818 };
    ring.uploadBuffer(fixture.buffer.range.subrange(u8, 0, @sizeOf(@TypeOf(replacement))).gpuRange(), std.mem.asBytes(&replacement));
    @memcpy(fixture.expected[guard_bytes..][0..@sizeOf(@TypeOf(replacement))], std.mem.asBytes(&replacement));
    fixture.check(ring.stats().submissions == 0 and
        ring.stats().pending_operations == 4, "transfer-compute-compute-transfer ordering remains within one batch");
    try fixture.readResults();

    var values: [64]u32 = undefined;
    for (&values, 0..) |*value, index| value.* = @as(u32, @intCast(index)) * 19 + 3;
    ring.uploadBuffer(fixture.buffer.range.subrange(u8, 1024, @sizeOf(@TypeOf(values))).gpuRange(), std.mem.asBytes(&values));
    const submissions = ring.stats().submissions;
    for (0..5) |i| {
        const iteration: u32 = @intCast(i);
        ring.uploadWithCompute(@sizeOf(@TypeOf(values)) + 32, accumulate, .{ fixture, pso, &callbacks, values.len, 1024, iteration * 37 + 11, 5 });
        for (&values, 0..) |*value, index| value.* += iteration * 37 + @as(u32, @intCast(index)) * 5 + 11;
    }
    @memcpy(fixture.expected[guard_bytes + 1024 ..][0..@sizeOf(@TypeOf(values))], std.mem.asBytes(&values));
    fixture.check(callbacks == 7 and
        ring.stats().submissions == submissions + 5, "full staging reservations flush before exposing each callback's command buffer");
    try fixture.readResults();
    fixture.check(ring.stats().peak_bytes <= 288, "compute callbacks preserve bounded ring storage across wraps");
}

fn run(queue_families: bool) !void {
    var fixture: Fixture = undefined;
    try fixture.init(queue_families);
    defer fixture.deinit();
    const path = "shaders/upload_queue.comp." ++ gpu.shader_extension;
    const code = try std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .unlimited, .@"4", null);
    defer testing.allocator.free(code);
    const compute = try gpu.createComputePso(fixture.device, .{
        .code = code,
        .entry_point = "computeMain",
        .threadgroup_size = .{ .x = upload_queue_thread_count, .y = 1, .z = 1 },
    });
    defer gpu.destroyPso(compute);
    // Seeds every texture so readResults compares known contents.
    try fixture.runSmallRing();
    try runComputeCallbacks(&fixture, compute, if (queue_families) gpu.getDeviceCaps(fixture.device).general_queue_count else 0);
    std.debug.print("Upload queue compute: {d} failures.\n", .{fixture.failures});
    try testing.expectEqual(0, fixture.failures);
}

test "upload queue compute callbacks" {
    try run(false);
}

test "upload queue compute callbacks on a dedicated compute queue" {
    try run(true);
}
