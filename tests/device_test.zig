const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const testing = std.testing;

test "every declaration compiles" {
    testing.refAllDecls(gpu);
}

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

test "copy memory round trip" {
    gpu.setErrorCallback(printError);
    const device = gpu.createDevice(.{}) catch |err| {
        if (err == error.Unsupported) return error.SkipZigTest;
        return err;
    };
    defer gpu.destroyDevice(device);
    const caps = gpu.getDeviceCaps(device);
    try testing.expect(caps.queue_count >= 1 and caps.general_queue_count >= 1 and caps.max_push_data_size >= 8);
    try testing.expect(caps.texture_heap_alignment != 0 and std.math.isPowerOfTwo(caps.texture_heap_alignment));

    const source = try gpu.createGpuHeap(device, 256, .cpu_visible);
    defer gpu.destroyGpuHeap(source);
    const destination = try gpu.createGpuHeap(device, 256, .readback);
    defer gpu.destroyGpuHeap(destination);
    try testing.expect(@intFromPtr(source.range.gpu) % 16 == 0 and @intFromPtr(destination.range.cpu) % 16 == 0);
    for (source.range.cpu.?[0..256], 0..) |*byte, index| byte.* = @truncate(index * 7);

    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const commands = try gpu.beginCommands(pool);
    gpu.copyMemory(commands, source.gpuRange(), destination.gpuRange());
    gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
    gpu.endCommands(commands);
    gpu.submit(device, .{ .commands = &.{commands}, .completion = .{ .semaphore = timeline, .value = 1 } }, 0);
    gpu.waitTimeline(.{ .semaphore = timeline, .value = 1 });
    try testing.expectEqual(1, gpu.timelineCompletedValue(timeline));
    try testing.expectEqualSlices(u8, source.range.cpu.?[0..256], destination.range.cpu.?[0..256]);
    gpu.waitIdle(device);
}
