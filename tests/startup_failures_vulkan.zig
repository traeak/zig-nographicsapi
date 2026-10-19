//! Replaces the backend's `vulkan_native` bindings: forwards each native create/destroy call to Vulkan, counts live objects
//! and fails the armed operation once.

const std = @import("std");
const c = @import("vulkan");

pub const NativeResources = struct {
    instances: i32 = 0,
    devices: i32 = 0,
    buffers: i32 = 0,
    memory: i32 = 0,
    semaphores: i32 = 0,
    images: i32 = 0,
    views: i32 = 0,
    pools: i32 = 0,
    queries: i32 = 0,
    pipelines: i32 = 0,
};

pub var live: NativeResources = .{};
pub var command_allocations: u32 = 0;
pub var command_frees: u32 = 0;
pub threadlocal var failure_operation: ?[]const u8 = null;
pub threadlocal var failure_result: c.VkResult = c.VK_ERROR_OUT_OF_DEVICE_MEMORY;
pub threadlocal var failure_skip: u32 = 0;

/// Fails the `skip + 1`th call of `operation` with `result`; null disarms.
pub fn arm(operation: ?[]const u8, result: c.VkResult, skip: u32) void {
    failure_operation = operation;
    failure_result = result;
    failure_skip = skip;
}

fn inject(comptime name: []const u8) bool {
    const operation = failure_operation orelse return false;
    if (!std.mem.eql(u8, operation, name)) return false;
    if (failure_skip != 0) {
        failure_skip -= 1;
        return false;
    }
    failure_operation = null;
    return true;
}

fn create(comptime name: []const u8, counter: *i32, args: anytype) c.VkResult {
    if (inject(name)) return failure_result;
    const result = @call(.auto, @field(c, name), args);
    if (result == c.VK_SUCCESS) counter.* += 1;
    return result;
}

fn destroy(comptime name: []const u8, counter: *i32, handle: anytype, args: anytype) void {
    if (handle != null) counter.* -= 1;
    @call(.auto, @field(c, name), args);
}

fn forward(comptime name: []const u8, args: anytype) c.VkResult {
    return if (inject(name)) failure_result else @call(.auto, @field(c, name), args);
}

pub fn vkCreateInstance(info: anytype, allocator: anytype, instance: anytype) c.VkResult {
    return create("vkCreateInstance", &live.instances, .{ info, allocator, instance });
}
pub fn vkCreateDevice(physical: anytype, info: anytype, allocator: anytype, device: anytype) c.VkResult {
    return create("vkCreateDevice", &live.devices, .{ physical, info, allocator, device });
}
pub fn vkCreateBuffer(device: anytype, info: anytype, allocator: anytype, buffer: anytype) c.VkResult {
    return create("vkCreateBuffer", &live.buffers, .{ device, info, allocator, buffer });
}
pub fn vkAllocateMemory(device: anytype, info: anytype, allocator: anytype, memory: anytype) c.VkResult {
    return create("vkAllocateMemory", &live.memory, .{ device, info, allocator, memory });
}
pub fn vkCreateSemaphore(device: anytype, info: anytype, allocator: anytype, semaphore: anytype) c.VkResult {
    return create("vkCreateSemaphore", &live.semaphores, .{ device, info, allocator, semaphore });
}
pub fn vkCreateImage(device: anytype, info: anytype, allocator: anytype, image: anytype) c.VkResult {
    return create("vkCreateImage", &live.images, .{ device, info, allocator, image });
}
pub fn vkCreateImageView(device: anytype, info: anytype, allocator: anytype, view: anytype) c.VkResult {
    return create("vkCreateImageView", &live.views, .{ device, info, allocator, view });
}
pub fn vkCreateCommandPool(device: anytype, info: anytype, allocator: anytype, pool: anytype) c.VkResult {
    return create("vkCreateCommandPool", &live.pools, .{ device, info, allocator, pool });
}
pub fn vkCreateQueryPool(device: anytype, info: anytype, allocator: anytype, pool: anytype) c.VkResult {
    return create("vkCreateQueryPool", &live.queries, .{ device, info, allocator, pool });
}
pub fn vkCreateComputePipelines(device: anytype, cache: anytype, count: anytype, info: anytype, allocator: anytype, pipelines: anytype) c.VkResult {
    return create("vkCreateComputePipelines", &live.pipelines, .{ device, cache, count, info, allocator, pipelines });
}
pub fn vkCreateGraphicsPipelines(device: anytype, cache: anytype, count: anytype, info: anytype, allocator: anytype, pipelines: anytype) c.VkResult {
    return create("vkCreateGraphicsPipelines", &live.pipelines, .{ device, cache, count, info, allocator, pipelines });
}
pub fn vkBindBufferMemory(device: anytype, buffer: anytype, memory: anytype, offset: anytype) c.VkResult {
    return forward("vkBindBufferMemory", .{ device, buffer, memory, offset });
}
pub fn vkBindImageMemory(device: anytype, image: anytype, memory: anytype, offset: anytype) c.VkResult {
    return forward("vkBindImageMemory", .{ device, image, memory, offset });
}
pub fn vkMapMemory(device: anytype, memory: anytype, offset: anytype, size: anytype, flags: anytype, data: anytype) c.VkResult {
    return forward("vkMapMemory", .{ device, memory, offset, size, flags, data });
}
pub fn vkAllocateCommandBuffers(device: c.VkDevice, info: *const c.VkCommandBufferAllocateInfo, commands: anytype) c.VkResult {
    if (inject("vkAllocateCommandBuffers")) return failure_result;
    const allocated = c.vkAllocateCommandBuffers(device, info, commands);
    if (allocated == c.VK_SUCCESS) command_allocations += info.commandBufferCount;
    return allocated;
}
pub fn vkFreeCommandBuffers(device: anytype, pool: anytype, count: u32, commands: anytype) void {
    command_frees += count;
    c.vkFreeCommandBuffers(device, pool, count, commands);
}

pub fn vkDestroyInstance(instance: anytype, allocator: anytype) void {
    destroy("vkDestroyInstance", &live.instances, instance, .{ instance, allocator });
}
pub fn vkDestroyDevice(device: anytype, allocator: anytype) void {
    destroy("vkDestroyDevice", &live.devices, device, .{ device, allocator });
}
pub fn vkDestroyBuffer(device: anytype, buffer: anytype, allocator: anytype) void {
    destroy("vkDestroyBuffer", &live.buffers, buffer, .{ device, buffer, allocator });
}
pub fn vkFreeMemory(device: anytype, memory: anytype, allocator: anytype) void {
    destroy("vkFreeMemory", &live.memory, memory, .{ device, memory, allocator });
}
pub fn vkDestroySemaphore(device: anytype, semaphore: anytype, allocator: anytype) void {
    destroy("vkDestroySemaphore", &live.semaphores, semaphore, .{ device, semaphore, allocator });
}
pub fn vkDestroyImage(device: anytype, image: anytype, allocator: anytype) void {
    destroy("vkDestroyImage", &live.images, image, .{ device, image, allocator });
}
pub fn vkDestroyImageView(device: anytype, view: anytype, allocator: anytype) void {
    destroy("vkDestroyImageView", &live.views, view, .{ device, view, allocator });
}
pub fn vkDestroyCommandPool(device: anytype, pool: anytype, allocator: anytype) void {
    destroy("vkDestroyCommandPool", &live.pools, pool, .{ device, pool, allocator });
}
pub fn vkDestroyQueryPool(device: anytype, pool: anytype, allocator: anytype) void {
    destroy("vkDestroyQueryPool", &live.queries, pool, .{ device, pool, allocator });
}
pub fn vkDestroyPipeline(device: anytype, pipeline: anytype, allocator: anytype) void {
    destroy("vkDestroyPipeline", &live.pipelines, pipeline, .{ device, pipeline, allocator });
}
