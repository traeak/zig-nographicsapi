//! Linear suballocation of a GPU/CPU range. `allocateAtomic` may run concurrently with itself from worker threads and
//! returns disjoint ranges; every other operation, including `allocate` and `reset`, requires exclusive access.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("NoGraphicsAPI");

const BumpAllocator = @This();

pub const alignment = 16;

storage: gpu.GpuCpuRange(u8),
offset: u64 = 0,

/// Storage must be nonempty, expose at least one address, and align every exposed address to 16 bytes.
pub fn init(storage: gpu.GpuCpuRange(u8)) BumpAllocator {
    assert(storage.size != 0);
    assert(storage.cpu != null or storage.gpu != null);
    assert(@intFromPtr(storage.cpu) % alignment == 0 and @intFromPtr(storage.gpu) % alignment == 0);
    return .{ .storage = storage };
}

/// Reserves `count` elements, rounded up to 16 bytes. The request must be nonzero. Returns null when storage is exhausted.
pub fn allocate(self: *BumpAllocator, comptime T: type, count: u64) ?gpu.GpuCpuRange(T) {
    comptime assert(@alignOf(T) <= alignment);
    const size = count * @sizeOf(T);
    assert(size != 0);
    const remaining = self.storage.size - self.offset;
    if (size > remaining) return null;
    const allocation = self.storage.subrange(T, self.offset, size);
    self.offset += @min(std.mem.alignForward(u64, size, alignment), remaining);
    return allocation;
}

/// Thread-safe `allocate`.
pub fn allocateAtomic(self: *BumpAllocator, comptime T: type, count: u64) ?gpu.GpuCpuRange(T) {
    comptime assert(@alignOf(T) <= alignment);
    const size = count * @sizeOf(T);
    assert(size != 0);
    var offset = @atomicLoad(u64, &self.offset, .monotonic);
    while (true) {
        const remaining = self.storage.size - offset;
        if (size > remaining) return null;
        const next = offset + @min(std.mem.alignForward(u64, size, alignment), remaining);
        offset = @cmpxchgWeak(u64, &self.offset, offset, next, .monotonic, .monotonic) orelse return self.storage.subrange(T, offset, size);
    }
}

/// Invalidates every previous allocation.
pub fn reset(self: *BumpAllocator) void {
    self.offset = 0;
}
