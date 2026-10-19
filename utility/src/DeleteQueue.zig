//! Defers callbacks, typically resource destruction, until a timeline reaches each callback's retire value.
//! Call `drain` after `waitIdle` and before `destroyDevice`; every callback must run before `deinit`.

const std = @import("std");
const assert = std.debug.assert;
const gpu = @import("NoGraphicsAPI");

const DeleteQueue = @This();

/// Inline bytes available to each callback's arguments.
pub const storage_size = 128;

const Entry = struct {
    retire_value: u64,
    storage: [storage_size]u8 align(16),
    invoke: *const fn (storage: *const anyopaque) void,
};

timeline: *gpu.TimelineSemaphore,
entries: []Entry,
head: u32 = 0,
tail: u32 = 0,
count: u32 = 0,
ticking: bool = false,

/// Capacity must be nonzero. The timeline must stay live while `tick` is called.
pub fn init(allocator: std.mem.Allocator, timeline: *gpu.TimelineSemaphore, capacity: u32) DeleteQueue {
    assert(capacity != 0);
    return .{ .timeline = timeline, .entries = allocator.alloc(Entry, capacity) catch @panic("out of memory") };
}

pub fn deinit(self: *DeleteQueue, allocator: std.mem.Allocator) void {
    assert(self.count == 0);
    allocator.free(self.entries);
    self.* = undefined;
}

/// Calls `function` with `args` once the timeline reaches `retire_value`. Retire values must be nondecreasing, the queue
/// must have free capacity, and `args` must fit `storage_size` bytes.
pub fn enqueue(self: *DeleteQueue, retire_value: u64, comptime function: anytype, args: anytype) void {
    const Args = @TypeOf(args);
    comptime assert(@sizeOf(Args) <= storage_size and @alignOf(Args) <= 16);
    assert(self.count < self.entries.len);
    assert(self.count == 0 or self.entries[if (self.tail == 0) self.entries.len - 1 else self.tail - 1].retire_value <= retire_value);
    const entry = &self.entries[self.tail];
    entry.retire_value = retire_value;
    @as(*Args, @ptrCast(&entry.storage)).* = args;
    entry.invoke = struct {
        fn invoke(storage: *const anyopaque) void {
            @call(.auto, function, @as(*const Args, @ptrCast(@alignCast(storage))).*);
        }
    }.invoke;
    self.tail = if (self.tail + 1 == self.entries.len) 0 else self.tail + 1;
    self.count += 1;
}

/// Runs every callback whose retire value the timeline has reached.
pub fn tick(self: *DeleteQueue) void {
    if (self.count != 0) self.collect(gpu.timelineCompletedValue(self.timeline));
}

/// Runs every queued callback without checking the timeline.
pub fn drain(self: *DeleteQueue) void {
    self.collect(std.math.maxInt(u64));
}

/// Callbacks cannot recursively collect the same queue.
fn collect(self: *DeleteQueue, completed_value: u64) void {
    assert(!self.ticking);
    self.ticking = true;
    while (self.count != 0) {
        const entry = &self.entries[self.head];
        if (entry.retire_value > completed_value) break;
        entry.invoke(&entry.storage);
        self.head = if (self.head + 1 == self.entries.len) 0 else self.head + 1;
        self.count -= 1;
    }
    self.ticking = false;
}
