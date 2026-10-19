const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const utility = @import("NoGraphicsAPIUtility");
const testing = std.testing;
const BumpAllocator = utility.BumpAllocator;
const HeapAllocator = utility.HeapAllocator;

fn storage(cpu: ?[]align(16) u8, gpu_memory: ?[]align(16) u8) gpu.GpuCpuRange(u8) {
    return .{
        .cpu = if (cpu) |memory| memory.ptr else null,
        .gpu = if (gpu_memory) |memory| memory.ptr else null,
        .size = if (cpu) |memory| memory.len else gpu_memory.?.len,
    };
}

fn at(memory: []align(16) u8, offset: usize) ?[*]u8 {
    return memory[offset..].ptr;
}

test "range defaults" {
    const range: gpu.GpuCpuRange(u8) = .{};
    try testing.expect(range.cpu == null and range.gpu == null and range.size == 0);
    try testing.expectEqual(gpu.GpuRange{ .gpu = null, .size = 0 }, range.gpuRange());
    comptime std.debug.assert(@sizeOf(gpu.GpuCpuRange(u32)) == 24 and @sizeOf(HeapAllocator.Allocation(u8)) == 32);
}

test "heap allocator basics" {
    var cpu: [4096]u8 align(64) = @splat(0);
    var gpu_memory: [4096]u8 align(64) = @splat(0);
    var allocator: HeapAllocator = .init(testing.allocator, storage(&cpu, &gpu_memory), 8);
    defer allocator.deinit(testing.allocator);

    const first = allocator.allocate(u8, 1).?;
    const second = allocator.allocate(u8, 17).?;
    const third = allocator.allocate(u8, 32).?;
    try testing.expect(first.range.cpu == at(&cpu, 0) and first.range.gpu == at(&gpu_memory, 0) and first.range.size == 1);
    try testing.expect(second.range.cpu == at(&cpu, 16) and second.range.gpu == at(&gpu_memory, 16) and second.range.size == 17);
    try testing.expect(third.range.cpu == at(&cpu, 48) and third.range.gpu == at(&gpu_memory, 48) and third.range.size == 32);
    try testing.expect(first.token != second.token and first.token != third.token and second.token != third.token);

    allocator.free(second);
    allocator.free(first);
    allocator.free(third);
    const whole = allocator.allocate(u8, cpu.len).?;
    try testing.expect(whole.range.cpu == at(&cpu, 0) and whole.range.gpu == at(&gpu_memory, 0) and whole.range.size == cpu.len);
    try testing.expectEqual(null, allocator.allocate(u8, 16));
    allocator.free(whole);

    allocator.reset();
    try testing.expectEqual(null, allocator.allocate(u8, cpu.len + 1));

    var partial: [17]u8 align(16) = @splat(0);
    var partial_allocator: HeapAllocator = .init(testing.allocator, storage(&partial, null), 2);
    defer partial_allocator.deinit(testing.allocator);
    try testing.expect(partial_allocator.allocate(u8, 1).?.range.cpu == at(&partial, 0));
    try testing.expectEqual(null, partial_allocator.allocate(u8, 1));
}

test "heap allocator address domains" {
    var cpu: [128]u8 align(64) = @splat(0);
    var gpu_memory: [128]u8 align(64) = @splat(0);
    var cpu_only: HeapAllocator = .init(testing.allocator, storage(&cpu, null), 1);
    defer cpu_only.deinit(testing.allocator);
    const cpu_allocation = cpu_only.allocate(u8, 1).?;
    try testing.expect(cpu_allocation.range.cpu == at(&cpu, 0) and cpu_allocation.range.gpu == null and cpu_allocation.range.size == 1);
    cpu_only.free(cpu_allocation);

    var gpu_only: HeapAllocator = .init(testing.allocator, storage(null, &gpu_memory), 1);
    defer gpu_only.deinit(testing.allocator);
    const gpu_allocation = gpu_only.allocate(u8, 1).?;
    try testing.expect(gpu_allocation.range.cpu == null and gpu_allocation.range.gpu == at(&gpu_memory, 0) and gpu_allocation.range.size == 1);
    gpu_only.free(gpu_allocation);
}

test "heap allocator limit" {
    var cpu: [128]u8 align(16) = @splat(0);
    var gpu_memory: [128]u8 align(16) = @splat(0);
    var allocator: HeapAllocator = .init(testing.allocator, storage(&cpu, &gpu_memory), 2);
    defer allocator.deinit(testing.allocator);
    const first = allocator.allocate(u8, 16).?;
    const second = allocator.allocate(u8, 16).?;
    try testing.expectEqual(null, allocator.allocate(u8, 16));
    allocator.free(first);
    const replacement = allocator.allocate(u8, 16).?;
    try testing.expect(replacement.range.gpu == first.range.gpu and replacement.token == first.token);
    allocator.free(replacement);
    allocator.free(second);
    try testing.expect(allocator.allocate(u8, cpu.len).?.range.gpu == at(&gpu_memory, 0));
}

test "typed allocations" {
    var cpu: [64]u8 align(16) = @splat(0);
    var gpu_memory: [64]u8 align(16) = @splat(0);
    var heap_allocator: HeapAllocator = .init(testing.allocator, storage(&cpu, &gpu_memory), 1);
    defer heap_allocator.deinit(testing.allocator);
    const heap_allocation = heap_allocator.allocate(u32, 3).?;
    try testing.expect(@intFromPtr(heap_allocation.range.cpu) == @intFromPtr(&cpu) and @intFromPtr(heap_allocation.range.gpu) == @intFromPtr(&gpu_memory));
    try testing.expectEqual(12, heap_allocation.range.size);
    try testing.expect(heap_allocation.range.gpuRange().gpu == @as(*anyopaque, &gpu_memory) and heap_allocation.range.gpuRange().size == 12);
    heap_allocator.free(heap_allocation);

    var bump_allocator: BumpAllocator = .init(storage(&cpu, &gpu_memory));
    const bump_allocation = bump_allocator.allocate(u16, 5).?;
    try testing.expect(@intFromPtr(bump_allocation.cpu) == @intFromPtr(&cpu) and @intFromPtr(bump_allocation.gpu) == @intFromPtr(&gpu_memory));
    try testing.expectEqual(10, bump_allocation.size);
}

fn rangesOverlap(lhs: HeapAllocator.Allocation(u8), rhs: HeapAllocator.Allocation(u8)) bool {
    const lhs_begin = @intFromPtr(lhs.range.gpu);
    const rhs_begin = @intFromPtr(rhs.range.gpu);
    const lhs_size = std.mem.alignForward(u64, lhs.range.size, 16);
    const rhs_size = std.mem.alignForward(u64, rhs.range.size, 16);
    return lhs_begin < rhs_begin + rhs_size and rhs_begin < lhs_begin + lhs_size;
}

test "heap allocator fragmentation" {
    const allocation_count = 64;
    var gpu_memory: [4096]u8 align(16) = @splat(0);
    var allocator: HeapAllocator = .init(testing.allocator, storage(null, &gpu_memory), allocation_count);
    defer allocator.deinit(testing.allocator);
    var allocations: [allocation_count]?HeapAllocator.Allocation(u8) = @splat(null);
    var random: u32 = 0x12345678;

    for (0..20000) |_| {
        random = random *% 1664525 +% 1013904223;
        const slot = random % allocation_count;
        if (allocations[slot]) |allocation| {
            allocator.free(allocation);
            allocations[slot] = null;
            continue;
        }
        random = random *% 1664525 +% 1013904223;
        const size = random % 97 + 1;
        allocations[slot] = allocator.allocate(u8, size);
        const allocation = allocations[slot] orelse continue;
        try testing.expectEqual(size, allocation.range.size);
        const begin = @intFromPtr(allocation.range.gpu);
        try testing.expect(begin >= @intFromPtr(&gpu_memory) and begin + allocation.range.size <= @intFromPtr(&gpu_memory) + gpu_memory.len);
        for (allocations, 0..) |other, index| {
            if (index != slot) if (other) |other_allocation| try testing.expect(!rangesOverlap(allocation, other_allocation));
        }
    }
    for (allocations) |allocation| if (allocation) |live| allocator.free(live);
    try testing.expect(allocator.allocate(u8, gpu_memory.len).?.range.gpu == at(&gpu_memory, 0));
}

test "bump allocator" {
    var cpu: [64]u8 align(16) = @splat(0);
    var gpu_memory: [64]u8 align(16) = @splat(0);
    var allocator: BumpAllocator = .init(storage(&cpu, &gpu_memory));
    const first = allocator.allocate(u8, 1).?;
    const second = allocator.allocate(u8, 17).?;
    const third = allocator.allocate(u8, 16).?;
    try testing.expect(first.cpu == at(&cpu, 0) and first.gpu == at(&gpu_memory, 0) and first.size == 1);
    try testing.expect(second.cpu == at(&cpu, 16) and second.gpu == at(&gpu_memory, 16) and second.size == 17);
    try testing.expect(third.cpu == at(&cpu, 48) and third.gpu == at(&gpu_memory, 48) and third.size == 16);
    try testing.expectEqual(null, allocator.allocate(u8, 1));
    allocator.reset();
    try testing.expect(allocator.allocate(u8, cpu.len).?.gpu == at(&gpu_memory, 0));

    var partial: [17]u8 align(16) = @splat(0);
    var partial_allocator: BumpAllocator = .init(storage(&partial, null));
    try testing.expect(partial_allocator.allocate(u8, 1).?.cpu == at(&partial, 0));
    try testing.expect(partial_allocator.allocate(u8, 1).?.cpu == at(&partial, 16));
    try testing.expectEqual(null, partial_allocator.allocate(u8, 1));
}

fn allocateAtomicRanges(allocator: *BumpAllocator, allocations: []?gpu.GpuCpuRange(u8)) void {
    for (allocations) |*allocation| allocation.* = allocator.allocateAtomic(u8, 1);
}

test "atomic bump allocator" {
    const thread_count = 8;
    const allocations_per_thread = 64;
    const allocation_count = thread_count * allocations_per_thread;
    var cpu: [allocation_count * BumpAllocator.alignment]u8 align(16) = @splat(0);
    var gpu_memory: [cpu.len]u8 align(16) = @splat(0);
    var allocations: [allocation_count]?gpu.GpuCpuRange(u8) = @splat(null);
    var allocator: BumpAllocator = .init(storage(&cpu, &gpu_memory));

    var threads: [thread_count]std.Thread = undefined;
    for (&threads, 0..) |*thread, index| {
        const slice = allocations[index * allocations_per_thread ..][0..allocations_per_thread];
        thread.* = try std.Thread.spawn(.{}, allocateAtomicRanges, .{ &allocator, slice });
    }
    for (threads) |thread| thread.join();

    for (allocations, 0..) |maybe_allocation, index| {
        const allocation = maybe_allocation.?;
        const cpu_offset = @intFromPtr(allocation.cpu) - @intFromPtr(&cpu);
        const gpu_offset = @intFromPtr(allocation.gpu) - @intFromPtr(&gpu_memory);
        try testing.expect(allocation.size == 1 and cpu_offset == gpu_offset and cpu_offset % BumpAllocator.alignment == 0 and cpu_offset < cpu.len);
        for (allocations[0..index]) |other| try testing.expect(other.?.cpu != allocation.cpu and other.?.gpu != allocation.gpu);
    }
    try testing.expectEqual(null, allocator.allocateAtomic(u8, 1));
    allocator.reset();
    try testing.expect(allocator.allocateAtomic(u8, cpu.len).?.cpu == at(&cpu, 0));
}

test "maximum heap range" {
    const gpu_address: [*]u8 = @ptrFromInt(HeapAllocator.alignment);
    var allocator: HeapAllocator = .init(testing.allocator, .{ .gpu = gpu_address, .size = HeapAllocator.maximum_size }, 1);
    defer allocator.deinit(testing.allocator);
    const allocation = allocator.allocate(u8, HeapAllocator.maximum_size).?;
    try testing.expect(allocation.range.gpu == gpu_address and allocation.range.size == HeapAllocator.maximum_size);
    allocator.free(allocation);
}
