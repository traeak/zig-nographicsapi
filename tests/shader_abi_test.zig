const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const shared = @import("shader_abi_shared");
const testing = std.testing;
const ShaderAbiData = shared.ShaderAbiData;
const ShaderAbiRoot = shared.ShaderAbiRoot;
const thread_count = shared.shader_abi_thread_count;

comptime {
    std.debug.assert(@sizeOf(ShaderAbiData) == 80 and @sizeOf(ShaderAbiRoot) == 96);
    std.debug.assert(@offsetOf(ShaderAbiRoot, "vector") == 4 and @offsetOf(ShaderAbiRoot, "color") == 16);
    std.debug.assert(@offsetOf(ShaderAbiRoot, "output") == 32 and @offsetOf(ShaderAbiRoot, "matrix") == 40 and @offsetOf(ShaderAbiRoot, "sampled") == 80);
}

/// Vulkan guarantees 4,000 sampler descriptors plus its reserved heap range.
const sampler_count = 4000;

fn bytesEqual(a: anytype, b: @TypeOf(a)) bool {
    return std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b));
}

test "shared shader root layout, matrices, GPU pointers, divergent texture slots and sampler indices" {
    const device = gpu.createDevice(.{}) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
    defer gpu.destroyDevice(device);
    const path = "shaders/shader_abi.comp." ++ gpu.shader_extension;
    const code = try std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .unlimited, .@"4", null);
    defer testing.allocator.free(code);
    const pso = try gpu.createComputePso(device, .{ .code = code, .entry_point = "computeMain", .threadgroup_size = .{ .x = thread_count, .y = 1, .z = 1 } });
    defer gpu.destroyPso(pso);

    const desc: gpu.TextureDesc = .{ .extent = .{ .x = 2, .y = 1, .z = 1 }, .usage = .{ .sampled = true, .transfer_destination = true } };
    const placement = try gpu.getTextureSizeAlign(device, desc);
    const stride = std.mem.alignForward(u64, placement.size, placement.alignment);
    const texture_heap = try gpu.createTextureHeap(device, stride * thread_count);
    defer gpu.destroyTextureHeap(texture_heap);
    const views = try gpu.createTextureDescriptorHeap(device, thread_count);
    defer gpu.destroyTextureDescriptorHeap(views);
    const samplers = try gpu.createSamplerDescriptorHeap(device, sampler_count);
    defer gpu.destroySamplerDescriptorHeap(samplers);
    const upload = try gpu.createGpuHeap(device, 128, .cpu_visible);
    defer gpu.destroyGpuHeap(upload);
    const output = try gpu.createGpuHeap(device, 1024, .gpu_only);
    defer gpu.destroyGpuHeap(output);
    const readback = try gpu.createGpuHeap(device, 1024, .readback);
    defer gpu.destroyGpuHeap(readback);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);

    const upload_bytes = upload.range.cpu.?;
    const commands = try gpu.beginCommands(pool);
    var textures: [thread_count]*gpu.Texture = undefined;
    var texture_count: usize = 0;
    defer for (textures[0..texture_count]) |texture| gpu.destroyTexture(texture);
    defer gpu.waitIdle(device);
    for (&textures, 0..) |*texture, i| {
        texture.* = try gpu.createTexture(commands, desc, texture_heap, stride * i);
        texture_count += 1;
        gpu.writeTextureDescriptor(views, @intCast(i), texture.*, .sampled, .{});
        gpu.writeSamplerDescriptor(samplers, @intCast(sampler_count - thread_count + i), .{
            .min_filter = .nearest,
            .mag_filter = .nearest,
            .address_u = if (i & 1 != 0) .repeat else .clamp_to_edge,
        });
        for (0..8) |component| upload_bytes[i * 8 + component] = @intCast(17 * i + component + 1);
    }

    const output_gpu = output.range.gpu.?;
    const root: ShaderAbiRoot = .{
        .scalar = 3.0,
        .vector = .{ .x = 5.0, .y = 6.0, .z = 7.0 },
        .color = .{ .x = 8.0, .y = 9.0, .z = 10.0, .w = 11.0 },
        .output = @ptrCast(@alignCast(output_gpu)),
        .matrix = .{ .rows = .{ .{ .x = 12.0, .y = 13.0, .z = 14.0 }, .{ .x = 15.0, .y = 16.0, .z = 17.0 }, .{ .x = 18.0, .y = 19.0, .z = 20.0 } } },
        .sampled = @ptrCast(@alignCast(output_gpu + 512)),
        .sampler_base = sampler_count - thread_count,
    };
    for (textures, 0..) |texture, i| gpu.copyMemoryToTexture(commands, .{ .gpu = upload.range.gpu.? + i * 8, .size = 8 }, texture, .{});
    gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .compute = true }, .{ .shader_read = true });
    gpu.setTextureDescriptorHeap(commands, views);
    gpu.setSamplerDescriptorHeap(commands, samplers);
    gpu.bindPso(commands, pso);
    @memcpy(upload_bytes[32..][0..@sizeOf(ShaderAbiRoot)], std.mem.asBytes(&root));
    gpu.dispatch(commands, upload.range.gpu.? + 32, .{ .x = 1, .y = 1, .z = 1 });
    gpu.barrier(commands, .{ .compute = true }, .{ .shader_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
    gpu.copyMemory(commands, output.gpuRange(), readback.gpuRange());
    gpu.endCommands(commands);
    gpu.submit(device, .{ .commands = &.{commands}, .completion = .{ .semaphore = timeline, .value = 1 } }, 0);
    gpu.waitTimeline(.{ .semaphore = timeline, .value = 1 });

    const values: [*]const ShaderAbiData = @ptrCast(@alignCast(readback.range.cpu.?));
    const sampled: [*]const [4]f32 = @ptrCast(@alignCast(readback.range.cpu.? + 512));
    for (0..thread_count) |i| {
        const value = values[i];
        try testing.expect(value.scalar == root.scalar and bytesEqual(value.vector, root.vector));
        try testing.expect(bytesEqual(value.color, root.color) and bytesEqual(value.matrix, root.matrix));
        const transformed = [3]f32{ value.transformed.x, value.transformed.y, value.transformed.z };
        for (root.matrix.rows, transformed) |row, actual| try testing.expectEqual(row.x * 2 + row.y * 3 + row.z * 5, actual);
        for (sampled[i], 0..) |actual, component| {
            const expected = @as(f32, @floatFromInt(upload_bytes[i * 8 + @as(usize, if (i & 1 != 0) 0 else 4) + component])) / 255.0;
            try testing.expectApproxEqAbs(expected, actual, 0.00001);
        }
    }
    std.debug.print("shared shader root layout, matrices, GPU pointers, divergent texture slots and sampler indices {d}..{d}: passed\n", .{
        sampler_count - thread_count,
        sampler_count - 1,
    });
}
