//! Two-level segregated-fit allocator over element offsets. Each free block lives in one of 256 bins indexed by a
//! 3-bit-mantissa floating-point size class; bitmasks find the smallest nonempty bin in constant time.

const std = @import("std");
const assert = std.debug.assert;

const RangeAllocator = @This();

pub const maximum_allocation_count = (0xffffffff - 1) / 2;

const unused_node = 0xffffffff;
const top_bin_count = 32;
const bins_per_leaf = 8;
const leaf_bin_count = top_bin_count * bins_per_leaf;
const mantissa_bits = 3;

pub const Range = struct {
    offset: u32,
    token: u32,
};

const Node = struct {
    offset: u32 = 0,
    size: u32 = 0,
    bin_previous: u32 = unused_node,
    bin_next: u32 = unused_node,
    neighbor_previous: u32 = unused_node,
    neighbor_next: u32 = unused_node,
    used: bool = false,
};

element_size: u64,
capacity: u32,
max_allocations: u32,
allocation_count: u32 = 0,
free_node_count: u32 = 0,
used_top_bins: u32 = 0,
used_leaf_bins: [top_bin_count]u8 = @splat(0),
bin_indices: [leaf_bin_count]u32 = @splat(unused_node),
nodes: []Node,
free_nodes: []u32,

/// `element_size` must be a power of two no larger than `byte_size`; `max_allocations` must be 1..maximum_allocation_count.
pub fn init(allocator: std.mem.Allocator, byte_size: u64, max_allocations: u32, element_size: u64) RangeAllocator {
    assert(std.math.isPowerOfTwo(element_size) and byte_size >= element_size);
    assert(max_allocations != 0 and max_allocations <= maximum_allocation_count);
    assert(byte_size / element_size <= 0xffffffff);
    const node_capacity = max_allocations * 2 + 1;
    var self: RangeAllocator = .{
        .element_size = element_size,
        .capacity = @intCast(byte_size / element_size),
        .max_allocations = max_allocations,
        .nodes = allocator.alloc(Node, node_capacity) catch @panic("out of memory"),
        .free_nodes = allocator.alloc(u32, node_capacity) catch @panic("out of memory"),
    };
    self.reset();
    return self;
}

pub fn deinit(self: *RangeAllocator, allocator: std.mem.Allocator) void {
    allocator.free(self.nodes);
    allocator.free(self.free_nodes);
    self.* = undefined;
}

/// The request must be nonzero. Returns null when no block fits or the allocation limit is reached.
pub fn allocate(self: *RangeAllocator, byte_size: u64) ?Range {
    assert(byte_size != 0);
    if (self.allocation_count == self.max_allocations) return null;
    const requested = 1 + (byte_size - 1) / self.element_size;
    if (requested > self.capacity) return null;
    const size: u32 = @intCast(requested);

    // Search the bin that may hold a large enough block, then any larger nonempty bin.
    const approximate_bin = binRoundDown(size);
    var node_index = self.bin_indices[approximate_bin];
    while (node_index != unused_node and self.nodes[node_index].size < size) node_index = self.nodes[node_index].bin_next;
    if (node_index == unused_node) {
        const minimum_bin = approximate_bin + 1;
        var top_bin = minimum_bin / bins_per_leaf;
        var leaf_bin: ?u32 = null;
        if (top_bin < top_bin_count) leaf_bin = lowestSetBit(self.used_leaf_bins[top_bin], minimum_bin % bins_per_leaf);
        if (leaf_bin == null) {
            top_bin = lowestSetBit(self.used_top_bins, top_bin + 1) orelse return null;
            leaf_bin = @ctz(self.used_leaf_bins[top_bin]);
        }
        node_index = self.bin_indices[top_bin * bins_per_leaf + leaf_bin.?];
    }

    const node = &self.nodes[node_index];
    const original_size = node.size;
    const original_next = node.neighbor_next;
    assert(original_size >= size);
    self.removeFreeNode(node_index);
    node.size = size;
    node.used = true;
    self.allocation_count += 1;

    if (original_size != size) {
        const remainder_index = self.acquireNode();
        self.nodes[remainder_index] = .{
            .offset = node.offset + size,
            .size = original_size - size,
            .neighbor_previous = node_index,
            .neighbor_next = original_next,
        };
        if (original_next != unused_node) self.nodes[original_next].neighbor_previous = remainder_index;
        node.neighbor_next = remainder_index;
        self.insertFreeNode(remainder_index);
    }
    return .{ .offset = node.offset, .token = node_index };
}

/// The token must come from an allocation of this allocator that has not been freed since the last reset.
pub fn free(self: *RangeAllocator, token: u32) void {
    assert(token < self.nodes.len);
    const node = &self.nodes[token];
    assert(node.used);

    if (node.neighbor_previous != unused_node and !self.nodes[node.neighbor_previous].used) {
        const previous_index = node.neighbor_previous;
        const previous = self.nodes[previous_index];
        self.removeFreeNode(previous_index);
        node.offset = previous.offset;
        node.size += previous.size;
        node.neighbor_previous = previous.neighbor_previous;
        self.releaseNode(previous_index);
    }
    if (node.neighbor_next != unused_node and !self.nodes[node.neighbor_next].used) {
        const next_index = node.neighbor_next;
        const next = self.nodes[next_index];
        self.removeFreeNode(next_index);
        node.size += next.size;
        node.neighbor_next = next.neighbor_next;
        self.releaseNode(next_index);
    }
    if (node.neighbor_previous != unused_node) self.nodes[node.neighbor_previous].neighbor_next = token;
    if (node.neighbor_next != unused_node) self.nodes[node.neighbor_next].neighbor_previous = token;

    self.insertFreeNode(token);
    self.allocation_count -= 1;
}

/// Invalidates every allocation.
pub fn reset(self: *RangeAllocator) void {
    self.allocation_count = 0;
    self.free_node_count = @intCast(self.nodes.len);
    self.used_top_bins = 0;
    self.used_leaf_bins = @splat(0);
    self.bin_indices = @splat(unused_node);
    for (self.nodes, self.free_nodes, 0..) |*node, *free_node, index| {
        node.* = .{};
        free_node.* = @intCast(index);
    }
    const node_index = self.acquireNode();
    self.nodes[node_index].size = self.capacity;
    self.insertFreeNode(node_index);
}

fn binRoundDown(size: u32) u32 {
    if (size < 1 << mantissa_bits) return size;
    const mantissa_start: u5 = @intCast(31 - @clz(size) - mantissa_bits);
    return ((@as(u32, mantissa_start) + 1) << mantissa_bits) | ((size >> mantissa_start) & ((1 << mantissa_bits) - 1));
}

fn lowestSetBit(bits: u32, first_bit: u32) ?u32 {
    if (first_bit >= 32) return null;
    const masked = bits & (@as(u32, 0xffffffff) << @intCast(first_bit));
    return if (masked == 0) null else @ctz(masked);
}

fn acquireNode(self: *RangeAllocator) u32 {
    assert(self.free_node_count != 0);
    self.free_node_count -= 1;
    return self.free_nodes[self.free_node_count];
}

fn releaseNode(self: *RangeAllocator, node_index: u32) void {
    assert(self.free_node_count < self.nodes.len);
    self.nodes[node_index] = .{};
    self.free_nodes[self.free_node_count] = node_index;
    self.free_node_count += 1;
}

fn insertFreeNode(self: *RangeAllocator, node_index: u32) void {
    const node = &self.nodes[node_index];
    const bin = binRoundDown(node.size);
    node.used = false;
    node.bin_previous = unused_node;
    node.bin_next = self.bin_indices[bin];
    if (node.bin_next != unused_node) self.nodes[node.bin_next].bin_previous = node_index;
    self.bin_indices[bin] = node_index;
    self.used_leaf_bins[bin / bins_per_leaf] |= @as(u8, 1) << @intCast(bin % bins_per_leaf);
    self.used_top_bins |= @as(u32, 1) << @intCast(bin / bins_per_leaf);
}

fn removeFreeNode(self: *RangeAllocator, node_index: u32) void {
    const node = &self.nodes[node_index];
    assert(!node.used);
    if (node.bin_previous != unused_node) {
        self.nodes[node.bin_previous].bin_next = node.bin_next;
        if (node.bin_next != unused_node) self.nodes[node.bin_next].bin_previous = node.bin_previous;
    } else {
        const bin = binRoundDown(node.size);
        const top_bin = bin / bins_per_leaf;
        assert(self.bin_indices[bin] == node_index);
        self.bin_indices[bin] = node.bin_next;
        if (node.bin_next != unused_node) {
            self.nodes[node.bin_next].bin_previous = unused_node;
        } else {
            self.used_leaf_bins[top_bin] &= ~(@as(u8, 1) << @intCast(bin % bins_per_leaf));
            if (self.used_leaf_bins[top_bin] == 0) self.used_top_bins &= ~(@as(u32, 1) << @intCast(top_bin));
        }
    }
    node.bin_previous = unused_node;
    node.bin_next = unused_node;
}
