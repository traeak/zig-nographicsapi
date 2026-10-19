const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const utility = @import("NoGraphicsAPIUtility");
const testing = std.testing;
const DeleteQueue = utility.DeleteQueue;
const TextureAllocator = utility.TextureAllocator;

fn addCount(count: *u32, increment: u32) void {
    count.* += increment;
}

test "placed textures and deferred deletion" {
    const device = gpu.createDevice(.{}) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
    defer gpu.destroyDevice(device);

    const desc: gpu.TextureDesc = .{};
    const size_align = try gpu.getTextureSizeAlign(device, desc);
    const element_size = gpu.getDeviceCaps(device).texture_heap_alignment;
    try testing.expect(size_align.size != 0 and size_align.alignment != 0 and element_size >= size_align.alignment);
    try testing.expect(element_size % size_align.alignment == 0);

    const texture_heap = try gpu.createTextureHeap(device, std.mem.alignForward(u64, size_align.size, element_size));
    defer gpu.destroyTextureHeap(texture_heap);
    var allocator: TextureAllocator = .init(testing.allocator, device, texture_heap, 1);
    defer allocator.deinit(testing.allocator);
    const pool = try gpu.createCommandPool(device, 0);
    defer gpu.destroyCommandPool(pool);
    const timeline = try gpu.createTimelineSemaphore(device, 0);
    defer gpu.destroyTimelineSemaphore(timeline);

    var commands = try gpu.beginCommands(pool);
    const first = (try allocator.allocate(commands, desc)).?;
    try testing.expectEqual(null, try allocator.allocate(commands, desc));
    gpu.endCommands(commands);
    gpu.resetCommandPool(pool);
    allocator.free(first);

    commands = try gpu.beginCommands(pool);
    var texture = (try allocator.allocate(commands, desc)).?;
    var callback_count: u32 = 0;
    var delete_queue: DeleteQueue = .init(testing.allocator, timeline, 2);
    defer delete_queue.deinit(testing.allocator);
    delete_queue.enqueue(1, TextureAllocator.free, .{ &allocator, texture });
    delete_queue.enqueue(std.math.maxInt(u64), addCount, .{ &callback_count, 1 });

    // The texture stays allocated until the timeline reaches its retire value.
    delete_queue.tick();
    try testing.expectEqual(null, try allocator.allocate(commands, desc));
    gpu.endCommands(commands);
    gpu.submit(device, .{ .commands = &.{commands}, .completion = .{ .semaphore = timeline, .value = 1 } }, 0);
    gpu.waitTimeline(.{ .semaphore = timeline, .value = 1 });
    gpu.resetCommandPool(pool);
    commands = try gpu.beginCommands(pool);
    try testing.expectEqual(null, try allocator.allocate(commands, desc));
    gpu.endCommands(commands);
    gpu.resetCommandPool(pool);

    commands = try gpu.beginCommands(pool);
    delete_queue.tick();
    texture = (try allocator.allocate(commands, desc)).?;
    try testing.expectEqual(0, callback_count);
    gpu.endCommands(commands);
    gpu.submit(device, .{ .commands = &.{commands}, .completion = .{ .semaphore = timeline, .value = 2 } }, 0);
    gpu.waitIdle(device);
    gpu.resetCommandPool(pool);

    delete_queue.drain();
    delete_queue.enqueue(2, addCount, .{ &callback_count, 1 });
    delete_queue.enqueue(2, addCount, .{ &callback_count, 1 });
    delete_queue.tick();
    try testing.expectEqual(3, callback_count);
    allocator.free(texture);
}

fn countWithLargeArguments(count: *u32, padding: [DeleteQueue.storage_size - @sizeOf(*u32)]u8) void {
    count.* += padding[0] + 1;
}

test "delete queue runs callbacks in order and wraps" {
    const device = gpu.createDevice(.{}) catch |err| return if (err == error.Unsupported) error.SkipZigTest else err;
    defer gpu.destroyDevice(device);
    const timeline = try gpu.createTimelineSemaphore(device, 5);
    defer gpu.destroyTimelineSemaphore(timeline);
    var delete_queue: DeleteQueue = .init(testing.allocator, timeline, 2);
    defer delete_queue.deinit(testing.allocator);

    var count: u32 = 0;
    for (0..3) |round| {
        delete_queue.enqueue(5, countWithLargeArguments, .{ &count, @as([DeleteQueue.storage_size - @sizeOf(*u32)]u8, @splat(1)) });
        delete_queue.enqueue(6, addCount, .{ &count, 10 });
        delete_queue.tick();
        try testing.expectEqual(round * 12 + 2, count);
        delete_queue.drain();
        try testing.expectEqual(round * 12 + 12, count);
    }
}
