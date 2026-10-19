//! Places textures in one texture heap, using DeviceCaps.texture_heap_alignment as the allocation granularity.
//! Externally synchronize the allocator and the command pool used for allocation.

const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const RangeAllocator = @import("RangeAllocator.zig");

const TextureAllocator = @This();

pub const PlacedTexture = struct {
    texture: *gpu.Texture,
    token: u32,
};

device: *gpu.Device,
heap: gpu.TextureHeap,
ranges: RangeAllocator,

/// `max_textures` must be 1..RangeAllocator.maximum_allocation_count.
pub fn init(allocator: std.mem.Allocator, device: *gpu.Device, heap: gpu.TextureHeap, max_textures: u32) TextureAllocator {
    return .{ .device = device, .heap = heap, .ranges = .init(allocator, heap.size, max_textures, gpu.getDeviceCaps(device).texture_heap_alignment) };
}

pub fn deinit(self: *TextureAllocator, allocator: std.mem.Allocator) void {
    self.ranges.deinit(allocator);
    self.* = undefined;
}

/// Records texture initialization into `commands`. Returns null when the heap is exhausted. Free the texture with this
/// allocator once all views and GPU use have finished.
pub fn allocate(self: *TextureAllocator, commands: *gpu.CommandBuffer, desc: gpu.TextureDesc) gpu.Error!?PlacedTexture {
    const requirements = try gpu.getTextureSizeAlign(self.device, desc);
    const range = self.ranges.allocate(requirements.size) orelse return null;
    errdefer self.ranges.free(range.token);
    const texture = try gpu.createTexture(commands, desc, self.heap, @as(u64, range.offset) * self.ranges.element_size);
    return .{ .texture = texture, .token = range.token };
}

pub fn free(self: *TextureAllocator, placed: PlacedTexture) void {
    self.ranges.free(placed.token);
    gpu.destroyTexture(placed.texture);
}
