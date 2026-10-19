//! Injects native Vulkan failures into resource creation and checks the returned error, the diagnostic and that no
//! partially created native object leaks.

const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("NoGraphicsAPI");
const utility = @import("NoGraphicsAPIUtility");
const hooks = @import("vulkan_hooks");
const c = @import("vulkan");
const testing = std.testing;

var failures: u32 = 0;
var reports: u32 = 0;
var reported_fatal = false;
var diagnostic_buffer: [1024]u8 = undefined;
var diagnostic: []const u8 = "";

fn check(condition: bool, message: []const u8) void {
    if (condition) return;
    std.debug.print("FAIL: {s} [{s}]\n", .{ message, diagnostic });
    failures += 1;
}

fn receiveError(message: [:0]const u8, fatal: bool) void {
    reports += 1;
    reported_fatal = reported_fatal or fatal;
    const length = @min(message.len, diagnostic_buffer.len);
    @memcpy(diagnostic_buffer[0..length], message[0..length]);
    diagnostic = diagnostic_buffer[0..length];
}

fn armResult(operation: ?[]const u8, result: c.VkResult, skip: u32) void {
    hooks.arm(operation, result, skip);
    reports = 0;
    reported_fatal = false;
    diagnostic = "";
}

fn arm(operation: ?[]const u8) void {
    armResult(operation, c.VK_ERROR_OUT_OF_DEVICE_MEMORY, 0);
}

fn unchanged(before: hooks.NativeResources) void {
    check(std.meta.eql(hooks.live, before), "partial native resources are released");
}

/// Checks that the armed failure was reached and reported once, nonfatally, naming the operation and result.
fn reported(operation: []const u8, before: hooks.NativeResources, bytes: bool) void {
    var buffer: [64]u8 = undefined;
    const result = std.fmt.bufPrint(&buffer, "Vulkan result {d}", .{hooks.failure_result}) catch unreachable;
    check(hooks.failure_operation == null, "the selected native failure was reached");
    check(reports == 1 and !reported_fatal, "one nonfatal diagnostic accompanies the returned failure");
    const named = std.mem.indexOf(u8, diagnostic, operation) != null and std.mem.indexOf(u8, diagnostic, result) != null;
    check(named, "diagnostic identifies the failing native operation and result");
    if (bytes) check(std.mem.indexOf(u8, diagnostic, "MiB requested") != null, "allocation diagnostics include the requested memory size");
    unchanged(before);
}

fn heapFailures(device: *gpu.Device) !void {
    for ([_][]const u8{ "vkCreateBuffer", "vkAllocateMemory", "vkBindBufferMemory", "vkMapMemory" }) |operation| {
        const before = hooks.live;
        const map = std.mem.eql(u8, operation, "vkMapMemory");
        armResult(operation, if (map) c.VK_ERROR_MEMORY_MAP_FAILED else c.VK_ERROR_OUT_OF_DEVICE_MEMORY, 0);
        const failed = gpu.createGpuHeap(device, 1024 * 1024, .cpu_visible);
        check(failed == error.OutOfMemory or (map and failed == error.DriverError), "failed mapped heap returns an error");
        reported(operation, before, true);
        const retry = try gpu.createGpuHeap(device, 1024 * 1024, .cpu_visible);
        check(retry.range.cpu != null and retry.range.gpu != null and retry.range.size == 1024 * 1024, "mapped heap creation succeeds after failure");
        gpu.destroyGpuHeap(retry);
        unchanged(before);
    }
    const before = hooks.live;
    arm("vkAllocateMemory");
    check(std.meta.isError(gpu.createTextureDescriptorHeap(device, 4)), "descriptor heap propagates backing allocation failure");
    reported("vkAllocateMemory", before, true);
    gpu.destroyTextureDescriptorHeap(try gpu.createTextureDescriptorHeap(device, 4));
    armResult("vkMapMemory", c.VK_ERROR_MEMORY_MAP_FAILED, 0);
    check(std.meta.isError(gpu.createSamplerDescriptorHeap(device, 4)), "sampler heap propagates backing map failure");
    reported("vkMapMemory", before, true);
    gpu.destroySamplerDescriptorHeap(try gpu.createSamplerDescriptorHeap(device, 4));
    unchanged(before);
}

fn textureFailures(device: *gpu.Device) !void {
    const desc: gpu.TextureDesc = .{ .extent = .{ .x = 16, .y = 16, .z = 1 }, .usage = .{ .sampled = true, .color_attachment = true } };
    const size = try gpu.getTextureSizeAlign(device, desc);
    check(size.size != 0, "ordinary texture requirements are available");
    const initial = hooks.live;
    arm("vkAllocateMemory");
    check(std.meta.isError(gpu.createTextureHeap(device, size.size)), "texture heap allocation failure returns an error");
    reported("vkAllocateMemory", initial, true);
    const heap = try gpu.createTextureHeap(device, size.size);
    const pool = try gpu.createCommandPool(device, 0);
    var commands = try gpu.beginCommands(pool);
    const before = hooks.live;
    for ([_][]const u8{ "vkCreateImage", "vkBindImageMemory" }) |operation| {
        arm(operation);
        check(std.meta.isError(gpu.createTexture(commands, desc, heap, 0)), "image creation or binding failure returns an error");
        reported(operation, before, false);
        const retry = try gpu.createTexture(commands, desc, heap, 0);
        gpu.endCommands(commands);
        gpu.resetCommandPool(pool);
        gpu.destroyTexture(retry);
        commands = try gpu.beginCommands(pool);
        unchanged(before);
    }
    const texture = try gpu.createTexture(commands, desc, heap, 0);
    const before_view = hooks.live;
    arm("vkCreateImageView");
    check(std.meta.isError(gpu.createRenderView(texture, .{})), "render view creation failure returns an error");
    reported("vkCreateImageView", before_view, false);
    gpu.destroyRenderView(try gpu.createRenderView(texture, .{}));
    gpu.endCommands(commands);
    gpu.resetCommandPool(pool);
    gpu.destroyTexture(texture);
    gpu.endCommands(try gpu.beginCommands(pool));
    gpu.destroyCommandPool(pool);
    gpu.destroyTextureHeap(heap);
    unchanged(initial);
}

fn commandFailures(device: *gpu.Device) !void {
    const before = hooks.live;
    arm("vkCreateSemaphore");
    check(std.meta.isError(gpu.createTimelineSemaphore(device, 0)), "timeline creation failure returns an error");
    reported("vkCreateSemaphore", before, false);
    gpu.destroyTimelineSemaphore(try gpu.createTimelineSemaphore(device, 0));
    arm("vkCreateCommandPool");
    check(std.meta.isError(gpu.createCommandPool(device, 0)), "command pool creation failure returns an error");
    reported("vkCreateCommandPool", before, false);
    gpu.destroyCommandPool(try gpu.createCommandPool(device, 0));
    for ([_][]const u8{ "vkAllocateCommandBuffers", "vkCreateQueryPool" }) |operation| {
        const pool = try gpu.createCommandPool(device, 0);
        const before_commands = hooks.live;
        const allocations = hooks.command_allocations;
        const frees = hooks.command_frees;
        arm(operation);
        check(std.meta.isError(gpu.beginCommands(pool)), "first command recording failure returns an error");
        reported(operation, before_commands, false);
        check(hooks.command_allocations - allocations == hooks.command_frees - frees, "failed first command context frees its native command buffer");
        gpu.endCommands(try gpu.beginCommands(pool));
        gpu.destroyCommandPool(pool);
        unchanged(before);
    }
}

fn utilityFailures(device: *gpu.Device) !void {
    const initial = hooks.live;
    const upload_operations = [_][]const u8{
        "vkCreateBuffer", "vkAllocateMemory", "vkMapMemory", "vkCreateSemaphore", "vkCreateCommandPool", "vkAllocateCommandBuffers", "vkCreateQueryPool",
    };
    for (upload_operations, 0..) |operation, index| {
        // The second batch's pool and command buffer fail, so the first batch must be released as well.
        armResult(operation, if (index == 2) c.VK_ERROR_MEMORY_MAP_FAILED else c.VK_ERROR_OUT_OF_DEVICE_MEMORY, if (index >= 4) 1 else 0);
        check(std.meta.isError(utility.UploadQueue.init(testing.allocator, device, .{ .capacity = 65536 })), "failed uploader initialization returns an error");
        reported(operation, initial, index <= 2);
        var retry = try utility.UploadQueue.init(testing.allocator, device, .{ .capacity = 65536 });
        check(retry.stats().capacity == 65536, "uploader construction succeeds after native failure");
        retry.deinit(testing.allocator);
        unchanged(initial);
    }

    const desc: gpu.TextureDesc = .{ .extent = .{ .x = 16, .y = 16, .z = 1 } };
    const size = try gpu.getTextureSizeAlign(device, desc);
    const heap = try gpu.createTextureHeap(device, std.mem.alignForward(u64, size.size, gpu.getDeviceCaps(device).texture_heap_alignment));
    const pool = try gpu.createCommandPool(device, 0);
    var allocator: utility.TextureAllocator = .init(testing.allocator, device, heap, 1);
    for ([_][]const u8{ "vkCreateImage", "vkBindImageMemory" }) |operation| {
        const commands = try gpu.beginCommands(pool);
        const before = hooks.live;
        arm(operation);
        check(std.meta.isError(allocator.allocate(commands, desc)), "failed texture creation returns an error");
        reported(operation, before, false);
        const retry = (try allocator.allocate(commands, desc)) orelse return error.TestUnexpectedResult;
        check(try allocator.allocate(commands, desc) == null, "successful retry consumes the only texture reservation");
        gpu.endCommands(commands);
        gpu.resetCommandPool(pool);
        allocator.free(retry);
        unchanged(before);
    }
    allocator.deinit(testing.allocator);
    gpu.destroyCommandPool(pool);
    gpu.destroyTextureHeap(heap);
    unchanged(initial);
}

fn readShader(path: []const u8) ![]align(4) u8 {
    return std.Io.Dir.cwd().readFileAllocOptions(testing.io, path, testing.allocator, .unlimited, .@"4", null);
}

fn pipelineFailures(device: *gpu.Device) !void {
    const compute = try readShader("shaders/queue_family.comp.spv");
    defer testing.allocator.free(compute);
    const vertex = try readShader("shaders/render_continuation.vert.spv");
    defer testing.allocator.free(vertex);
    const fragment = try readShader("shaders/render_continuation.frag.spv");
    defer testing.allocator.free(fragment);
    const before = hooks.live;

    const stage: gpu.ShaderStage = .{ .code = compute, .entry_point = "computeMain" };
    arm("vkCreateComputePipelines");
    check(std.meta.isError(gpu.createComputePso(device, stage)), "compute pipeline failure returns an error");
    reported("vkCreateComputePipelines", before, false);
    gpu.destroyPso(try gpu.createComputePso(device, stage));

    const desc: gpu.GraphicsPsoDesc = .{
        .vertex = .{ .code = vertex, .entry_point = "vertexMain" },
        .fragment = .{ .code = fragment, .entry_point = "fragmentMain" },
        .color_targets = &.{.{ .format = .rgba8_unorm }},
    };
    arm("vkCreateGraphicsPipelines");
    check(std.meta.isError(gpu.createGraphicsPso(device, desc)), "graphics pipeline failure returns an error");
    reported("vkCreateGraphicsPipelines", before, false);
    gpu.destroyPso(try gpu.createGraphicsPso(device, desc));
    unchanged(before);
}

fn callbackThread(result: *bool) void {
    armResult("vkCreateInstance", c.VK_ERROR_OUT_OF_HOST_MEMORY, 0);
    result.* = gpu.createDevice(.{}) == error.OutOfMemory and hooks.failure_operation == null;
}

test "startup failures return errors and release native resources" {
    gpu.setErrorCallback(receiveError);
    defer gpu.setErrorCallback(null);
    armResult("vkCreateInstance", c.VK_ERROR_OUT_OF_HOST_MEMORY, 0);
    check(gpu.createDevice(.{}) == error.OutOfMemory, "instance allocation failure returns an error instead of terminating");
    reported("vkCreateInstance", .{}, false);
    arm(null);
    const device = gpu.createDevice(.{ .timestamp_query_count = 8 }) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
    const before_device = hooks.live;
    arm("vkCreateDevice");
    check(gpu.createDevice(.{}) == error.OutOfMemory, "logical device allocation failure returns an error");
    reported("vkCreateDevice", before_device, false);
    try heapFailures(device);
    try textureFailures(device);
    try commandFailures(device);
    try utilityFailures(device);
    try pipelineFailures(device);

    arm("vkCreateBuffer");
    gpu.setErrorCallback(null);
    const silent = std.meta.isError(gpu.createGpuHeap(device, 1024, .cpu_visible));
    check(silent and hooks.failure_operation == null and reports == 0, "clearing the callback suppresses diagnostics without changing failure returns");
    gpu.setErrorCallback(receiveError);

    // The error callback is per thread, so a worker without one reports nothing.
    arm(null);
    var worker_result = false;
    (try std.Thread.spawn(.{}, callbackThread, .{&worker_result})).join();
    check(worker_result and reports == 0, "startup callback is not inherited by another thread");
    arm("vkCreateBuffer");
    check(std.meta.isError(gpu.createGpuHeap(device, 1024, .cpu_visible)), "main thread callback remains installed after the worker finishes");
    reported("vkCreateBuffer", before_device, true);

    gpu.waitIdle(device);
    gpu.destroyDevice(device);
    unchanged(.{});
    std.debug.print("Startup failure returns and cleanup: {d} failures\n", .{failures});
    try testing.expectEqual(0, failures);
}
