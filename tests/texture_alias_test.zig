const std = @import("std");
const gpu = @import("NoGraphicsAPI");

const width = 256;
const height = 192;
const passes = 8;
const readback_stride: u64 = width * height * 8;

fn printError(message: [:0]const u8, fatal: bool) void {
    std.debug.print("{s}{s}\n", .{ if (fatal) "fatal: " else "", message });
}

fn recordAlias(commands: *gpu.CommandBuffer, textures: *const [2]*gpu.Texture, views: *const [2]*gpu.RenderView, readback: *const gpu.GpuHeap, pass: u32) void {
    const mode = pass & 1;
    const alternate = pass & 2 != 0;
    gpu.activateTextureAlias(commands, textures[mode]);
    const clear: gpu.ClearColor = if (mode != 0)
        .{ .x = if (alternate) 0.5 else 2.5, .y = if (alternate) 1.0 else 12.5, .z = 0.5, .w = 1.0 }
    else
        .{ .x = if (alternate) 0.0 else 1.0, .y = if (alternate) 1.0 else 0.0, .z = 1.0, .w = 1.0 };
    gpu.beginRenderPass(commands, .{ .colors = &.{.{ .render_view = views[mode], .load = .clear, .clear = clear }} }, .{});
    gpu.endRenderPass(commands);
    gpu.barrier(commands, .{ .color_output = true }, .{ .color_write = true }, .{ .transfer = true }, .{ .transfer_read = true });
    gpu.copyTextureToMemory(commands, textures[mode], readback.range.subrange(u8, pass * readback_stride, readback_stride).gpuRange(), .{
        .extent = .{ .x = width, .y = height, .z = 1 },
    });
}

fn checkAlias(readback: *const gpu.GpuHeap, pass: u32) bool {
    const alternate = pass & 2 != 0;
    const data = readback.range.cpu.? + pass * readback_stride;
    for (0..width * height) |pixel| {
        const valid = if (pass & 1 != 0) blk: {
            const color: [*]align(1) const u16 = @ptrCast(data + pixel * 8);
            break :blk color[0] == @as(u16, if (alternate) 0x3800 else 0x4100) and color[1] == @as(u16, if (alternate) 0x3c00 else 0x4a40) and
                color[2] == 0x3800 and color[3] == 0x3c00;
        } else blk: {
            const color = data + pixel * 4;
            break :blk color[0] == 255 and color[1] == @as(u8, if (alternate) 255 else 0) and color[2] == @as(u8, if (alternate) 0 else 255) and
                color[3] == 255;
        };
        if (!valid) {
            std.debug.print("Aliased texture pass {d} pixel {d} has unexpected color.\n", .{ pass, pixel });
            return false;
        }
    }
    return true;
}

test "resident SDR/HDR texture aliases, batched and idle handoffs" {
    gpu.setErrorCallback(printError);
    const device = gpu.createDevice(.{ .timestamp_query_count = 0 }) catch |err| {
        if (err == error.Unsupported) return error.SkipZigTest;
        return err;
    };
    defer gpu.destroyDevice(device);
    const usage: gpu.TextureUsage = .{ .sampled = true, .color_attachment = true, .transfer_source = true };
    const descriptions = [2]gpu.TextureDesc{
        .{ .extent = .{ .x = width, .y = height, .z = 1 }, .format = .bgra8_srgb, .usage = usage, .aliasable = true },
        .{ .extent = .{ .x = width, .y = height, .z = 1 }, .format = .rgba16_float, .usage = usage, .aliasable = true },
    };
    const sdr = try gpu.getTextureSizeAlign(device, descriptions[0]);
    const hdr = try gpu.getTextureSizeAlign(device, descriptions[1]);
    const offset = @max(sdr.alignment, hdr.alignment);
    const heap = try gpu.createTextureHeap(device, offset + @max(sdr.size, hdr.size));
    defer gpu.destroyTextureHeap(heap);
    const readback = try gpu.createGpuHeap(device, passes * readback_stride, .readback);
    defer gpu.destroyGpuHeap(readback);
    const descriptors = try gpu.createTextureDescriptorHeap(device, 2);
    defer gpu.destroyTextureDescriptorHeap(descriptors);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    var completion: gpu.TimelinePoint = .{ .semaphore = try gpu.createTimelineSemaphore(device, 0), .value = 0 };
    defer gpu.destroyTimelineSemaphore(completion.semaphore);
    defer gpu.waitIdle(device);
    var commands = try gpu.beginCommands(pool);
    var textures: [2]*gpu.Texture = undefined;
    var views: [2]*gpu.RenderView = undefined;
    for (0..2) |mode| {
        textures[mode] = try gpu.createTexture(commands, descriptions[mode], heap, offset);
        views[mode] = try gpu.createRenderView(textures[mode], .{});
        gpu.writeTextureDescriptor(descriptors, @intCast(mode), textures[mode], .sampled, .{});
    }
    defer for (0..2) |mode| {
        gpu.destroyRenderView(views[mode]);
        gpu.destroyTexture(textures[mode]);
    };
    // Alternate without CPU waits first, then repeat handoffs between idle submissions using the same handles and descriptors.
    for (0..passes) |pass| recordAlias(commands, &textures, &views, &readback, @intCast(pass));
    gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
    gpu.endCommands(commands);
    completion.value += 1;
    gpu.submit(device, .{ .commands = &.{commands}, .completion = completion }, 0);
    gpu.waitTimeline(completion);
    var valid = true;
    for (0..passes) |pass| valid = checkAlias(&readback, @intCast(pass)) and valid;
    for (0..passes) |pass| {
        gpu.waitIdle(device);
        gpu.resetCommandPool(pool);
        commands = try gpu.beginCommands(pool);
        recordAlias(commands, &textures, &views, &readback, @intCast(pass));
        gpu.barrier(commands, .{ .transfer = true }, .{ .transfer_write = true }, .{ .host = true }, .{ .host_read = true });
        gpu.endCommands(commands);
        completion.value += 1;
        gpu.submit(device, .{ .commands = &.{commands}, .completion = completion }, 0);
        gpu.waitTimeline(completion);
        valid = checkAlias(&readback, @intCast(pass)) and valid;
    }
    try std.testing.expect(valid);
}
