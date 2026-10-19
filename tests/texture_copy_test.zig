const std = @import("std");
const gpu = @import("NoGraphicsAPI");

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

fn check(device: *gpu.Device, timeline: *gpu.TimelineSemaphore, value: *u64, texture_type: gpu.TextureType, format: gpu.Format) !void {
    const compressed = format == .bc7_unorm;
    const volume = texture_type == .three_d;
    const desc: gpu.TextureDesc = .{
        .type = texture_type,
        .extent = .{ .x = 8, .y = 8, .z = if (volume) 4 else 1 },
        .layer_count = if (volume) 1 else if (texture_type == .cube) 6 else 3,
        .format = format,
        .usage = .{ .transfer_source = true, .transfer_destination = true },
    };
    const required = try gpu.getTextureSizeAlign(device, desc);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    var commands = try gpu.beginCommands(pool);
    const heap = try gpu.createTextureHeap(device, required.size);
    defer gpu.destroyTextureHeap(heap);
    const texture = try gpu.createTexture(commands, desc, heap, 0);
    defer gpu.destroyTexture(texture);
    const upload = try gpu.createGpuHeap(device, 4096, .cpu_visible);
    defer gpu.destroyGpuHeap(upload);
    const readback = try gpu.createGpuHeap(device, 4096, .readback);
    defer gpu.destroyGpuHeap(readback);
    const upload_bytes = upload.range.cpu.?[0..4096];
    const readback_bytes = readback.range.cpu.?[0..4096];
    for (upload_bytes, 0..) |*byte, i| byte.* = @truncate(i * 17 + i / 16);
    @memset(readback_bytes, 0xa5);
    const row_pitch: u32 = if (compressed) 48 else 16;
    const slice_pitch: u32 = if (compressed) 144 else 64;
    const row_bytes: u32 = if (compressed) 32 else 3;
    const copy: gpu.TextureCopyDesc = .{
        .base_slice = if (volume) 0 else 1,
        .slice_count = if (volume) 1 else 2,
        .offset = .{ .x = if (compressed) 0 else 1, .y = if (compressed) 0 else 1, .z = if (volume) 1 else 0 },
        .extent = .{ .x = if (compressed) 8 else 3, .y = if (compressed) 8 else 2, .z = if (volume) 2 else 1 },
        .row_pitch_bytes = row_pitch,
        .slice_pitch_bytes = slice_pitch,
    };
    gpu.copyMemoryToTexture(commands, upload.gpuRange(), texture, copy);
    // NVIDIA 596.99 needs a timeline wait before address-based image readback; see docs/known-driver-issues.md.
    gpu.endCommands(commands);
    value.* += 1;
    gpu.submit(device, .{ .commands = &.{commands}, .completion = .{ .semaphore = timeline, .value = value.* } }, 0);
    gpu.waitTimeline(.{ .semaphore = timeline, .value = value.* });
    commands = try gpu.beginCommands(pool);
    gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
    gpu.copyTextureToMemory(commands, texture, readback.gpuRange(), copy);
    gpu.endCommands(commands);
    value.* += 1;
    gpu.submit(device, .{ .commands = &.{commands}, .completion = .{ .semaphore = timeline, .value = value.* } }, 0);
    gpu.waitTimeline(.{ .semaphore = timeline, .value = value.* });
    for (readback_bytes, upload_bytes, 0..) |got, uploaded, i| {
        const slice = i / slice_pitch;
        const in_slice = i % slice_pitch;
        const copied = slice < 2 and in_slice / row_pitch < 2 and in_slice % row_pitch < row_bytes;
        const expected: u8 = if (copied) uploaded else 0xa5;
        if (got != expected) {
            std.debug.print("Texture copy type={t} format={t} at byte {d}: got {d}, expected {d}.\n", .{ texture_type, format, i, got, expected });
            return error.TestUnexpectedResult;
        }
    }
}

test "3D/array/cube/BC7 pitch roundtrips" {
    gpu.setErrorCallback(printError);
    const device = gpu.createDevice(.{ .timestamp_query_count = 0 }) catch |err| {
        if (err == error.Unsupported) return error.SkipZigTest;
        return err;
    };
    defer gpu.destroyDevice(device);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);
    defer gpu.waitIdle(device);
    var value: u64 = 0;
    try check(device, timeline, &value, .three_d, .r8_unorm);
    try check(device, timeline, &value, .two_d_array, .r8_unorm);
    try check(device, timeline, &value, .cube, .r8_unorm);
    if (gpu.supportsTextureFormat(device, .bc7_unorm, .{ .transfer_source = true, .transfer_destination = true }))
        try check(device, timeline, &value, .two_d_array, .bc7_unorm);
}
