//! General-purpose suballocation of a GPU/CPU range with 16-byte granularity, backed by `RangeAllocator`.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("NoGraphicsAPI");
const RangeAllocator = @import("RangeAllocator.zig");

const HeapAllocator = @This();

pub const alignment = 16;
pub const maximum_size = @as(u64, 0xffffffff) * alignment;
pub const maximum_allocation_count = RangeAllocator.maximum_allocation_count;

pub fn Allocation(comptime T: type) type {
    return extern struct {
        /// Size is in bytes.
        range: gpu.GpuCpuRange(T),
        token: u32,
    };
}

storage: gpu.GpuCpuRange(u8),
ranges: RangeAllocator,

/// Storage must expose at least one address, align every exposed address to 16 bytes, and contain 16..maximum_size bytes.
/// `max_allocations` must be 1..maximum_allocation_count. A trailing partial element is unused.
pub fn init(allocator: std.mem.Allocator, storage: gpu.GpuCpuRange(u8), max_allocations: u32) HeapAllocator {
    assert(storage.cpu != null or storage.gpu != null);
    assert(storage.size <= maximum_size);
    assert(@intFromPtr(storage.cpu) % alignment == 0 and @intFromPtr(storage.gpu) % alignment == 0);
    return .{ .storage = storage, .ranges = .init(allocator, storage.size, max_allocations, alignment) };
}

pub fn deinit(self: *HeapAllocator, allocator: std.mem.Allocator) void {
    self.ranges.deinit(allocator);
    self.* = undefined;
}

/// The request must be nonzero. Returns null when storage or the allocation limit is exhausted.
pub fn allocate(self: *HeapAllocator, comptime T: type, count: u64) ?Allocation(T) {
    comptime assert(@alignOf(T) <= alignment);
    const size = count * @sizeOf(T);
    const range = self.ranges.allocate(size) orelse return null;
    return .{ .range = self.storage.subrange(T, @as(u64, range.offset) * alignment, size), .token = range.token };
}

/// The allocation must come from this allocator and not have been freed since the last reset.
pub fn free(self: *HeapAllocator, allocation: anytype) void {
    self.ranges.free(allocation.token);
}

/// Invalidates every allocation.
pub fn reset(self: *HeapAllocator) void {
    self.ranges.reset();
}
