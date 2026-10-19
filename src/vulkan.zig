//! Vulkan 1.4 backend. Requires VK_EXT_descriptor_heap, VK_KHR_device_address_commands, VK_KHR_shader_untyped_pointers and
//! VK_EXT_mesh_shader. Buffers are one device-address range per GPU heap; textures live in GENERAL layout; shaders read
//! their root pointer from push data.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const api = @import("NoGraphicsAPI.zig");
const c = @import("vulkan");
/// Native object creation and destruction. The same bindings as `c`; the startup failure test replaces them to inject failures.
const native = @import("vulkan_native");

const Error = api.Error;
const Format = api.Format;

const gpa = std.heap.c_allocator;
const validation = builtin.mode == .debug;
const windows = builtin.os.tag == .windows;
const x11 = builtin.os.tag == .linux;
const surface_extension_name = if (windows) "VK_KHR_win32_surface" else c.VK_KHR_XLIB_SURFACE_EXTENSION_NAME;

const max_device_extensions = 512;
const max_instance_extensions = 256;
const max_instance_layers = 64;
const max_physical_devices = 32;
const max_queue_families = 64;
/// General, compute-only, copy-only.
const queue_type_count = 3;
const max_color_attachments = 8;
const max_swapchain_images = 8;
const max_surface_formats = 64;
const swapchain_present_mode = c.VK_PRESENT_MODE_FIFO_KHR;
const gpu_allocation_alignment = 16;
const address_flags = c.VK_ADDRESS_COMMAND_FULLY_BOUND_BIT_KHR;

threadlocal var error_callback: ?api.ErrorCallback = null;

pub fn setErrorCallback(callback: ?api.ErrorCallback) void {
    error_callback = callback;
}

fn report(fatal: bool, comptime format: []const u8, args: anytype) void {
    const callback = error_callback orelse return;
    var buffer: [512]u8 = undefined;
    callback(std.mem.printSentinel(&buffer, format, args, 0) catch "graphics error message too long", fatal);
}

fn reportVkFailure(result: c.VkResult, operation: []const u8, fatal: bool, byte_count: u64) void {
    const reason = switch (result) {
        c.VK_ERROR_INITIALIZATION_FAILED => "VK_ERROR_INITIALIZATION_FAILED (graphics initialization failed)",
        c.VK_ERROR_OUT_OF_HOST_MEMORY => "VK_ERROR_OUT_OF_HOST_MEMORY (not enough system memory)",
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY => "VK_ERROR_OUT_OF_DEVICE_MEMORY (not enough GPU memory or CPU-visible GPU memory)",
        c.VK_ERROR_TOO_MANY_OBJECTS => "VK_ERROR_TOO_MANY_OBJECTS (graphics resource limit reached)",
        c.VK_ERROR_DEVICE_LOST => "VK_ERROR_DEVICE_LOST (graphics device lost)",
        c.VK_ERROR_MEMORY_MAP_FAILED => "VK_ERROR_MEMORY_MAP_FAILED (GPU memory could not be mapped for CPU access)",
        c.VK_ERROR_LAYER_NOT_PRESENT => "VK_ERROR_LAYER_NOT_PRESENT (required Vulkan layer unavailable)",
        c.VK_ERROR_EXTENSION_NOT_PRESENT => "VK_ERROR_EXTENSION_NOT_PRESENT (required Vulkan extension unavailable)",
        c.VK_ERROR_FEATURE_NOT_PRESENT => "VK_ERROR_FEATURE_NOT_PRESENT (required graphics feature unavailable)",
        c.VK_ERROR_INCOMPATIBLE_DRIVER => "VK_ERROR_INCOMPATIBLE_DRIVER (incompatible graphics driver)",
        c.VK_ERROR_FORMAT_NOT_SUPPORTED => "VK_ERROR_FORMAT_NOT_SUPPORTED (required texture or display format unsupported)",
        c.VK_ERROR_INVALID_SHADER_NV => "VK_ERROR_INVALID_SHADER_NV (the graphics driver rejected the shader)",
        c.VK_ERROR_UNKNOWN => "VK_ERROR_UNKNOWN (unspecified graphics driver error)",
        else => "graphics driver error",
    };
    if (byte_count != 0) {
        const mib = @as(f64, @floatFromInt(byte_count)) / (1024.0 * 1024.0);
        report(fatal, "{s} failed: {s} (Vulkan result {d}, {d:.2} MiB requested).", .{ operation, reason, result, mib });
    } else {
        report(fatal, "{s} failed: {s} (Vulkan result {d}).", .{ operation, reason, result });
    }
}

fn abortVkFailure(result: c.VkResult, operation: []const u8) noreturn {
    reportVkFailure(result, operation, true, 0);
    @panic("unexpected Vulkan failure");
}

/// Reports a failed Vulkan call and maps it to an API error.
fn check(result: c.VkResult, operation: []const u8, byte_count: u64) Error!void {
    if (result == c.VK_SUCCESS) return;
    reportVkFailure(result, operation, false, byte_count);
    return switch (result) {
        c.VK_ERROR_OUT_OF_HOST_MEMORY, c.VK_ERROR_OUT_OF_DEVICE_MEMORY, c.VK_ERROR_TOO_MANY_OBJECTS => error.OutOfMemory,
        c.VK_ERROR_DEVICE_LOST => error.DeviceLost,
        c.VK_ERROR_LAYER_NOT_PRESENT,
        c.VK_ERROR_EXTENSION_NOT_PRESENT,
        c.VK_ERROR_FEATURE_NOT_PRESENT,
        c.VK_ERROR_INCOMPATIBLE_DRIVER,
        c.VK_ERROR_FORMAT_NOT_SUPPORTED,
        => error.Unsupported,
        else => error.DriverError,
    };
}

fn requireOk(result: Error!void, operation: []const u8) void {
    result catch |err| {
        report(true, "{s} failed (graphics error {s}).", .{ operation, @errorName(err) });
        @panic("unexpected graphics API failure");
    };
}

fn create(comptime T: type, value: T) *T {
    const pointer = gpa.create(T) catch @panic("out of memory");
    pointer.* = value;
    return pointer;
}

fn alignUp(value: u64, alignment: u64) u64 {
    assert(alignment != 0);
    return (value + alignment - 1) / alignment * alignment;
}

fn address(pointer: ?*const anyopaque) c.VkDeviceAddress {
    return @intFromPtr(pointer);
}

fn hasName(names: anytype, name: []const u8) bool {
    for (names) |properties| {
        const field = if (@hasField(@TypeOf(properties), "extensionName")) &properties.extensionName else &properties.layerName;
        if (std.mem.eql(u8, std.mem.sliceTo(field, 0), name)) return true;
    }
    return false;
}

fn debugCallback(
    severity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
    _: c.VkDebugUtilsMessageTypeFlagsEXT,
    data: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque,
) callconv(.c) c.VkBool32 {
    // Keep the library callback dependency-free; it makes validation failures visible on stderr.
    if (severity >= c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT and data != null and data.*.pMessage != null)
        std.debug.print("NoGraphicsAPI validation: {s}\n", .{data.*.pMessage});
    return c.VK_FALSE;
}

fn toVkFormat(format: Format) c.VkFormat {
    return switch (format) {
        .r8_srgb => c.VK_FORMAT_R8_SRGB,
        .rg8_srgb => c.VK_FORMAT_R8G8_SRGB,
        .rgba8_srgb => c.VK_FORMAT_R8G8B8A8_SRGB,
        .bgra8_srgb => c.VK_FORMAT_B8G8R8A8_SRGB,
        .rgba4_unorm => c.VK_FORMAT_R4G4B4A4_UNORM_PACK16,
        .r5g5b5a1_unorm => c.VK_FORMAT_R5G5B5A1_UNORM_PACK16,
        .r5g6b5_unorm => c.VK_FORMAT_R5G6B5_UNORM_PACK16,
        .r8_unorm => c.VK_FORMAT_R8_UNORM,
        .rg8_unorm => c.VK_FORMAT_R8G8_UNORM,
        .rgba8_unorm => c.VK_FORMAT_R8G8B8A8_UNORM,
        .bgra8_unorm => c.VK_FORMAT_B8G8R8A8_UNORM,
        .r16_unorm => c.VK_FORMAT_R16_UNORM,
        .rg16_unorm => c.VK_FORMAT_R16G16_UNORM,
        .rgba16_unorm => c.VK_FORMAT_R16G16B16A16_UNORM,
        .r8_uint => c.VK_FORMAT_R8_UINT,
        .rg8_uint => c.VK_FORMAT_R8G8_UINT,
        .rgba8_uint => c.VK_FORMAT_R8G8B8A8_UINT,
        .bgra8_uint => c.VK_FORMAT_B8G8R8A8_UINT,
        .r16_uint => c.VK_FORMAT_R16_UINT,
        .rg16_uint => c.VK_FORMAT_R16G16_UINT,
        .rgba16_uint => c.VK_FORMAT_R16G16B16A16_UINT,
        .r32_uint => c.VK_FORMAT_R32_UINT,
        .rg32_uint => c.VK_FORMAT_R32G32_UINT,
        .rgba32_uint => c.VK_FORMAT_R32G32B32A32_UINT,
        .r16_float => c.VK_FORMAT_R16_SFLOAT,
        .rg16_float => c.VK_FORMAT_R16G16_SFLOAT,
        .rgba16_float => c.VK_FORMAT_R16G16B16A16_SFLOAT,
        .r32_float => c.VK_FORMAT_R32_SFLOAT,
        .rg32_float => c.VK_FORMAT_R32G32_SFLOAT,
        .rgba32_float => c.VK_FORMAT_R32G32B32A32_SFLOAT,
        .rgb10a2_unorm => c.VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        .rg11b10_float => c.VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        .d16_unorm => c.VK_FORMAT_D16_UNORM,
        .d24_unorm_s8_uint => c.VK_FORMAT_D24_UNORM_S8_UINT,
        .d32_float => c.VK_FORMAT_D32_SFLOAT,
        .s8_uint => c.VK_FORMAT_S8_UINT,
        .d32_float_s8_uint => c.VK_FORMAT_D32_SFLOAT_S8_UINT,
        .eac_rg => c.VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
        .astc_4x4_srgb => c.VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
        .astc_4x4_unorm => c.VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
        .bc3_srgb => c.VK_FORMAT_BC3_SRGB_BLOCK,
        .bc3_unorm => c.VK_FORMAT_BC3_UNORM_BLOCK,
        .bc5_rg => c.VK_FORMAT_BC5_UNORM_BLOCK,
        .bc6h_ufloat => c.VK_FORMAT_BC6H_UFLOAT_BLOCK,
        .bc6h_sfloat => c.VK_FORMAT_BC6H_SFLOAT_BLOCK,
        .bc7_srgb => c.VK_FORMAT_BC7_SRGB_BLOCK,
        .bc7_unorm => c.VK_FORMAT_BC7_UNORM_BLOCK,
        .undefined => c.VK_FORMAT_UNDEFINED,
    };
}

fn compatibleViewFormats(image_format: Format, view_format: Format) bool {
    if (image_format == view_format) return true;
    const image = api.getTextureFormatInfo(image_format);
    const view = api.getTextureFormatInfo(view_format);
    if (image.depth or image.stencil or view.depth or view.stencil) return false;
    if (image.block_extent.x == 1 and view.block_extent.x == 1) return image.bytes_per_block != 0 and image.bytes_per_block == view.bytes_per_block;
    const pairs = [_][2]Format{ .{ .astc_4x4_unorm, .astc_4x4_srgb }, .{ .bc3_unorm, .bc3_srgb }, .{ .bc6h_ufloat, .bc6h_sfloat }, .{ .bc7_unorm, .bc7_srgb } };
    for (pairs) |pair| {
        if ((image_format == pair[0] and view_format == pair[1]) or (image_format == pair[1] and view_format == pair[0])) return true;
    }
    return false;
}

fn imageAspects(format: Format) c.VkImageAspectFlags {
    const info = api.getTextureFormatInfo(format);
    var result: c.VkImageAspectFlags = 0;
    if (info.depth) result |= c.VK_IMAGE_ASPECT_DEPTH_BIT;
    if (info.stencil) result |= c.VK_IMAGE_ASPECT_STENCIL_BIT;
    return if (result == 0) c.VK_IMAGE_ASPECT_COLOR_BIT else result;
}

fn toVkImageType(texture_type: api.TextureType) c.VkImageType {
    return switch (texture_type) {
        .one_d => c.VK_IMAGE_TYPE_1D,
        .three_d => c.VK_IMAGE_TYPE_3D,
        .two_d, .cube, .two_d_array, .cube_array => c.VK_IMAGE_TYPE_2D,
    };
}

fn toVkViewType(texture_type: api.TextureType) c.VkImageViewType {
    return switch (texture_type) {
        .one_d => c.VK_IMAGE_VIEW_TYPE_1D,
        .two_d => c.VK_IMAGE_VIEW_TYPE_2D,
        .three_d => c.VK_IMAGE_VIEW_TYPE_3D,
        .cube => c.VK_IMAGE_VIEW_TYPE_CUBE,
        .two_d_array => c.VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .cube_array => c.VK_IMAGE_VIEW_TYPE_CUBE_ARRAY,
    };
}

fn requiredFormatFeatures(usage: api.TextureUsage) c.VkFormatFeatureFlags2 {
    var result: c.VkFormatFeatureFlags2 = 0;
    if (usage.sampled) result |= c.VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
    if (usage.storage) result |= c.VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT | c.VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
        c.VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
    if (usage.color_attachment) result |= c.VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT;
    if (usage.depth_stencil_attachment) result |= c.VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (usage.transfer_source) result |= c.VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT;
    if (usage.transfer_destination) result |= c.VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
    return result;
}

fn imageUsage(usage: api.TextureUsage) c.VkImageUsageFlags {
    var result: c.VkImageUsageFlags = 0;
    if (usage.sampled) result |= c.VK_IMAGE_USAGE_SAMPLED_BIT;
    if (usage.storage) result |= c.VK_IMAGE_USAGE_STORAGE_BIT;
    if (usage.color_attachment) result |= c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (usage.depth_stencil_attachment) result |= c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (usage.transfer_source) result |= c.VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (usage.transfer_destination) result |= c.VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return result;
}

const TextureCompression = enum { none, etc2, astc, bc };

fn textureCompression(format: Format) TextureCompression {
    return switch (format) {
        .bc3_srgb, .bc3_unorm, .bc5_rg, .bc6h_ufloat, .bc6h_sfloat, .bc7_srgb, .bc7_unorm => .bc,
        .astc_4x4_srgb, .astc_4x4_unorm => .astc,
        .eac_rg => .etc2,
        else => .none,
    };
}

fn optimalFormatFeatures(physical_device: c.VkPhysicalDevice, format: Format) c.VkFormatFeatureFlags2 {
    var properties3: c.VkFormatProperties3 = .{ .sType = c.VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3 };
    var properties2: c.VkFormatProperties2 = .{ .sType = c.VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &properties3 };
    c.vkGetPhysicalDeviceFormatProperties2(physical_device, toVkFormat(format), &properties2);
    return properties3.optimalTilingFeatures;
}

fn toVkBlendFactor(factor: api.BlendFactor) c.VkBlendFactor {
    return switch (factor) {
        .zero => c.VK_BLEND_FACTOR_ZERO,
        .one => c.VK_BLEND_FACTOR_ONE,
        .source_color => c.VK_BLEND_FACTOR_SRC_COLOR,
        .one_minus_source_color => c.VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        .destination_color => c.VK_BLEND_FACTOR_DST_COLOR,
        .one_minus_destination_color => c.VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
        .source_alpha => c.VK_BLEND_FACTOR_SRC_ALPHA,
        .one_minus_source_alpha => c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .destination_alpha => c.VK_BLEND_FACTOR_DST_ALPHA,
        .one_minus_destination_alpha => c.VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
        .source_alpha_saturate => c.VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
    };
}

fn toVkStages(stages: api.Stage) c.VkPipelineStageFlags2 {
    var result: c.VkPipelineStageFlags2 = 0;
    if (stages.indirect) result |= c.VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    if (stages.index_input) result |= c.VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
    if (stages.vertex) result |= c.VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    if (stages.task) result |= c.VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
    if (stages.mesh) result |= c.VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    if (stages.depth_stencil_tests) result |= c.VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | c.VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    if (stages.fragment) result |= c.VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    if (stages.color_output) result |= c.VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (stages.compute) result |= c.VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (stages.transfer) result |= c.VK_PIPELINE_STAGE_2_COPY_BIT;
    if (stages.host) result |= c.VK_PIPELINE_STAGE_2_HOST_BIT;
    if (stages.all_commands) result |= c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    return result;
}

fn toVkAccess(accesses: api.Access) c.VkAccessFlags2 {
    var result: c.VkAccessFlags2 = 0;
    if (accesses.transfer_read) result |= c.VK_ACCESS_2_TRANSFER_READ_BIT;
    if (accesses.transfer_write) result |= c.VK_ACCESS_2_TRANSFER_WRITE_BIT;
    if (accesses.shader_read) result |= c.VK_ACCESS_2_SHADER_STORAGE_READ_BIT | c.VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | c.VK_ACCESS_2_UNIFORM_READ_BIT;
    if (accesses.shader_write) result |= c.VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    if (accesses.color_read) result |= c.VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    if (accesses.color_write) result |= c.VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    if (accesses.depth_stencil_read) result |= c.VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    if (accesses.depth_stencil_write) result |= c.VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    if (accesses.indirect_read) result |= c.VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    if (accesses.index_read) result |= c.VK_ACCESS_2_INDEX_READ_BIT;
    if (accesses.host_read) result |= c.VK_ACCESS_2_HOST_READ_BIT;
    if (accesses.descriptor_read) result |= c.VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT | c.VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT;
    return result;
}

comptime {
    assert(toVkStages(.{ .all_commands = true }) == c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    assert(toVkStages(.{ .task = true }) == c.VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT);
    assert(@as(u32, @bitCast(api.RenderingFlags{ .suspending = true, .resuming = true })) ==
        c.VK_RENDERING_SUSPENDING_BIT | c.VK_RENDERING_RESUMING_BIT);
}

const universal_buffer_usage: c.VkBufferUsageFlags = c.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    c.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
const cpu_visible_memory_properties: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
const forbidden_memory_properties: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT | c.VK_MEMORY_PROPERTY_PROTECTED_BIT |
    c.VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | c.VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;

fn isUsableMemoryType(properties: *const c.VkPhysicalDeviceMemoryProperties, index: u32) bool {
    const memory_type = properties.memoryTypes[index];
    if (memory_type.propertyFlags & forbidden_memory_properties != 0) return false;
    return properties.memoryHeaps[memory_type.heapIndex].flags & c.VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM == 0;
}

/// Extension entry points, named after their Vulkan functions and loaded from the device.
const DeviceFunctions = struct {
    vkWriteSamplerDescriptorsEXT: c.PFN_vkWriteSamplerDescriptorsEXT = null,
    vkWriteResourceDescriptorsEXT: c.PFN_vkWriteResourceDescriptorsEXT = null,
    vkCmdBindSamplerHeapEXT: c.PFN_vkCmdBindSamplerHeapEXT = null,
    vkCmdBindResourceHeapEXT: c.PFN_vkCmdBindResourceHeapEXT = null,
    vkCmdPushDataEXT: c.PFN_vkCmdPushDataEXT = null,
    vkCmdBindIndexBuffer3KHR: c.PFN_vkCmdBindIndexBuffer3KHR = null,
    vkCmdDrawIndirect2KHR: c.PFN_vkCmdDrawIndirect2KHR = null,
    vkCmdDrawIndexedIndirect2KHR: c.PFN_vkCmdDrawIndexedIndirect2KHR = null,
    vkCmdDispatchIndirect2KHR: c.PFN_vkCmdDispatchIndirect2KHR = null,
    vkCmdDrawMeshTasksEXT: c.PFN_vkCmdDrawMeshTasksEXT = null,
    vkCmdDrawMeshTasksIndirect2EXT: c.PFN_vkCmdDrawMeshTasksIndirect2EXT = null,
    vkCmdCopyMemoryKHR: c.PFN_vkCmdCopyMemoryKHR = null,
    vkCmdCopyMemoryToImageKHR: c.PFN_vkCmdCopyMemoryToImageKHR = null,
    vkCmdCopyImageToMemoryKHR: c.PFN_vkCmdCopyImageToMemoryKHR = null,
};

const BackingBuffer = struct {
    buffer: c.VkBuffer = null,
    memory: c.VkDeviceMemory = null,
    mapped: ?*anyopaque = null,
    address: c.VkDeviceAddress = 0,
};

const DeferredSwapchainImage = struct {
    retire_value: u64 = 0,
    swapchain: c.VkSwapchainKHR = null,
    view: c.VkImageView = null,
};

/// Swapchain images whose views and handles retire once the presentation timeline reaches their value.
const SwapchainDeleteQueue = struct {
    entries: []DeferredSwapchainImage = &.{},
    first: usize = 0,
    count: usize = 0,

    fn push(self: *SwapchainDeleteQueue, retire_value: u64, swapchain: c.VkSwapchainKHR, view: c.VkImageView) void {
        assert(swapchain != null and view != null);
        if (self.count == self.entries.len) {
            const old_size = self.entries.len;
            self.entries = gpa.realloc(self.entries, @max(1, old_size * 2)) catch @panic("out of memory");
            for (0..self.first) |index| {
                self.entries[old_size + index] = self.entries[index];
                self.entries[index] = .{};
            }
        }
        assert(self.count == 0 or self.entries[(self.first + self.count - 1) % self.entries.len].retire_value <= retire_value);
        self.entries[(self.first + self.count) % self.entries.len] = .{ .retire_value = retire_value, .swapchain = swapchain, .view = view };
        self.count += 1;
    }

    fn collect(self: *SwapchainDeleteQueue, device: c.VkDevice, completed_value: u64) void {
        while (self.count != 0 and self.entries[self.first].retire_value <= completed_value) {
            const entry = self.entries[self.first];
            const final_image = self.count == 1 or self.entries[(self.first + 1) % self.entries.len].swapchain != entry.swapchain;
            native.vkDestroyImageView(device, entry.view, null);
            if (final_image) c.vkDestroySwapchainKHR(device, entry.swapchain, null);
            self.entries[self.first] = .{};
            self.first = (self.first + 1) % self.entries.len;
            self.count -= 1;
        }
        if (self.count == 0) self.first = 0;
    }
};

const RetiredSwapchain = struct {
    handle: c.VkSwapchainKHR = null,
    views: [max_swapchain_images]c.VkImageView = @splat(null),
    view_count: u32 = 0,
};

pub const GpuHeapOwner = struct {
    device: *Device,
    backing: BackingBuffer,
};

pub const TextureDescriptorHeap = struct {
    device: *Device,
    storage: api.GpuHeap,
    capacity: u32,
};

pub const SamplerDescriptorHeap = struct {
    device: *Device,
    storage: api.GpuHeap,
    capacity: u32,
};

pub const TextureHeapOwner = struct {
    device: *Device,
    memory: c.VkDeviceMemory,
};

pub const TimelineSemaphore = struct {
    device: *Device,
    semaphore: c.VkSemaphore,
};

pub const CommandBuffer = struct {
    device: *Device,
    next: ?*CommandBuffer = null,
    command_pool: c.VkCommandPool,
    command_buffer: c.VkCommandBuffer = null,
    epilogue: c.VkCommandBuffer = null,
    timestamp_pool: c.VkQueryPool = null,
    timestamp_destinations: []*u64 = &.{},
    timestamp_results: []u64 = &.{},
    timestamp_count: u32 = 0,
    swapchain: ?*Swapchain = null,
    submitted: bool = false,
    has_epilogue: bool = false,
};

pub const CommandPool = struct {
    device: *Device,
    command_pool: c.VkCommandPool = null,
    first: ?*CommandBuffer = null,
    last: ?*CommandBuffer = null,
    next_buffer: ?*CommandBuffer = null,
    timestamps: bool,
};

const Queue = struct {
    queue: c.VkQueue = null,
    family_index: u32 = 0,
    timestamps: bool = false,
    command_submit_infos: []c.VkCommandBufferSubmitInfo = &.{},
    wait_submit_infos: []c.VkSemaphoreSubmitInfo = &.{},
};

const PresentContext = struct {
    acquired: c.VkSemaphore = null,
    rendered: c.VkSemaphore = null,
    presented: c.VkFence = null,
    swapchain: c.VkSwapchainKHR = null,
    present_pending: bool = false,
};

pub const Texture = struct {
    device: *Device,
    image: c.VkImage = null,
    width: u32,
    height: u32,
    depth: u32,
    layer_count: u32,
    type: api.TextureType,
    format: Format,
};

pub const RenderView = struct {
    device: ?*Device = null,
    view: c.VkImageView = null,
    width: u32 = 0,
    height: u32 = 0,
    swapchain_view: bool = false,
};

const Swapchain = struct {
    device: *Device,
    handle: c.VkSwapchainKHR = null,
    images: [max_swapchain_images]c.VkImage = @splat(null),
    render_views: [max_swapchain_images]RenderView = @splat(.{}),
    initialized: [max_swapchain_images]bool = @splat(false),
    image_count: u32 = 0,
    image_index: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,
    format: Format,
    color_space: c.VkColorSpaceKHR,
    transform: c.VkSurfaceTransformFlagBitsKHR = c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    composite_alpha: c.VkCompositeAlphaFlagBitsKHR = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    present_context: ?*PresentContext = null,
    transition_commands: ?*CommandBuffer = null,
    acquired: bool = false,
    recreate_required: bool = false,
};

pub const Pso = struct {
    device: *Device,
    pso: c.VkPipeline = null,
    bind_point: c.VkPipelineBindPoint,
};

pub const Device = struct {
    instance: c.VkInstance = null,
    debug_messenger: c.VkDebugUtilsMessengerEXT = null,
    destroy_debug_messenger: c.PFN_vkDestroyDebugUtilsMessengerEXT = null,
    physical_device: c.VkPhysicalDevice = null,
    device: c.VkDevice = null,
    queues: []Queue = &.{},
    surface: c.VkSurfaceKHR = null,
    queue_families: [queue_type_count]u32 = @splat(0),
    queue_family_count: u32 = 0,
    timestamp_query_count: u32 = 0,
    memory_properties: c.VkPhysicalDeviceMemoryProperties = .{},
    physical_properties: c.VkPhysicalDeviceProperties = .{},
    heap_properties: c.VkPhysicalDeviceDescriptorHeapPropertiesEXT = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT },
    max_timeline_value_difference: u64 = 0,
    texture_heap_alignment: u64 = 16,
    texture_memory_type: u32 = c.VK_MAX_MEMORY_TYPES,
    fns: DeviceFunctions = .{},
    caps: api.DeviceCaps = .{},
    format_features: [Format.count]c.VkFormatFeatureFlags2 = @splat(0),
    texture_compression_etc2: bool = false,
    swapchain_color_space_enabled: bool = false,
    presentation_retirement: c.VkSemaphore = null,
    presentation_retirement_value: u64 = 0,
    completed_presentation_retirement: u64 = 0,
    swapchain_delete_queue: SwapchainDeleteQueue = .{},
    present_contexts: [max_swapchain_images]PresentContext = @splat(.{}),
    retired_swapchains: [max_swapchain_images]RetiredSwapchain = @splat(.{}),
    swapchain: ?*Swapchain = null,
    acquired_swapchain: ?*Swapchain = null,
    present_context_count: u32 = 0,
    next_present_context: u32 = 0,

    fn sharingMode(self: *const Device) c.VkSharingMode {
        return if (self.queue_family_count > 1) c.VK_SHARING_MODE_CONCURRENT else c.VK_SHARING_MODE_EXCLUSIVE;
    }

    fn findMemoryType(
        self: *const Device,
        bits: u32,
        required: c.VkMemoryPropertyFlags,
        preferred: c.VkMemoryPropertyFlags,
        minimum_heap_size: u64,
        avoided: c.VkMemoryPropertyFlags,
    ) ?u32 {
        var best: ?u32 = null;
        var best_is_avoided = false;
        var best_score: u32 = 0;
        var best_heap_size: u64 = 0;
        for (0..self.memory_properties.memoryTypeCount) |i| {
            const index: u32 = @intCast(i);
            if (bits & (@as(u32, 1) << @intCast(index)) == 0) continue;
            const flags = self.memory_properties.memoryTypes[index].propertyFlags;
            if (flags & required != required or !isUsableMemoryType(&self.memory_properties, index)) continue;
            const heap = self.memory_properties.memoryHeaps[self.memory_properties.memoryTypes[index].heapIndex];
            if (heap.size < minimum_heap_size) continue;
            const is_avoided = flags & avoided != 0;
            const score: u32 = @popCount(flags & preferred);
            if (best == null or (best_is_avoided and !is_avoided) or
                (best_is_avoided == is_avoided and (score > best_score or (score == best_score and heap.size > best_heap_size))))
            {
                best = index;
                best_is_avoided = is_avoided;
                best_score = score;
                best_heap_size = heap.size;
            }
        }
        return best;
    }

    fn createBackingBuffer(
        self: *const Device,
        size: u64,
        usage: c.VkBufferUsageFlags,
        required: c.VkMemoryPropertyFlags,
        preferred: c.VkMemoryPropertyFlags,
        avoided: c.VkMemoryPropertyFlags,
    ) Error!BackingBuffer {
        var result: BackingBuffer = .{};
        const buffer_info: c.VkBufferCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = self.sharingMode(),
            .queueFamilyIndexCount = self.queue_family_count,
            .pQueueFamilyIndices = &self.queue_families,
        };
        try check(native.vkCreateBuffer(self.device, &buffer_info, null, &result.buffer), "vkCreateBuffer", size);
        errdefer native.vkDestroyBuffer(self.device, result.buffer, null);

        var requirements: c.VkMemoryRequirements = .{};
        c.vkGetBufferMemoryRequirements(self.device, result.buffer, &requirements);
        const host_visible = required & c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0;
        const memory_type = self.findMemoryType(requirements.memoryTypeBits, required, preferred, requirements.size, avoided) orelse {
            const mib = @as(f64, @floatFromInt(requirements.size)) / (1024.0 * 1024.0);
            report(false, "No compatible {s} GPU memory heap can hold {d:.2} MiB.", .{ if (host_visible) "CPU-visible" else "device-local", mib });
            return error.OutOfMemory;
        };

        const flags_info: c.VkMemoryAllocateFlagsInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .flags = c.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        };
        const allocate_info: c.VkMemoryAllocateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &flags_info,
            .allocationSize = requirements.size,
            .memoryTypeIndex = memory_type,
        };
        const operation = if (host_visible) "vkAllocateMemory (CPU-visible GPU heap)" else "vkAllocateMemory (GPU-only heap)";
        try check(native.vkAllocateMemory(self.device, &allocate_info, null, &result.memory), operation, requirements.size);
        errdefer native.vkFreeMemory(self.device, result.memory, null);
        try check(native.vkBindBufferMemory(self.device, result.buffer, result.memory, 0), "vkBindBufferMemory", requirements.size);
        if (host_visible) try check(native.vkMapMemory(self.device, result.memory, 0, c.VK_WHOLE_SIZE, 0, &result.mapped), "vkMapMemory", requirements.size);

        const address_info: c.VkBufferDeviceAddressInfo = .{ .sType = c.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = result.buffer };
        result.address = c.vkGetBufferDeviceAddress(self.device, &address_info);
        return result;
    }

    fn allocateDescriptorHeap(self: *Device, size: u64, texture: bool) Error!api.GpuHeap {
        const properties = &self.heap_properties;
        const resource_alignment = @max(properties.imageDescriptorAlignment, properties.bufferDescriptorAlignment);
        const reserved_alignment = if (texture) resource_alignment else properties.samplerDescriptorAlignment;
        const heap_alignment = if (texture) properties.resourceHeapAlignment else properties.samplerHeapAlignment;
        const reserved_size = if (texture) properties.minResourceHeapReservedRange else properties.minSamplerHeapReservedRange;
        const allocation_alignment = @max(heap_alignment, gpu_allocation_alignment);
        const backing_size = alignUp(size, reserved_alignment) + reserved_size + allocation_alignment - 1;

        const backing = try self.createBackingBuffer(
            backing_size,
            universal_buffer_usage | c.VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT,
            cpu_visible_memory_properties,
            0,
            0,
        );
        const gpu_address = alignUp(backing.address, allocation_alignment);
        const offset = gpu_address - backing.address;
        return .{
            .range = .{ .cpu = @as([*]u8, @ptrCast(backing.mapped.?)) + offset, .gpu = @ptrFromInt(gpu_address), .size = size },
            .owner = create(GpuHeapOwner, .{ .device = self, .backing = backing }),
        };
    }

    fn pollPresentationRetirement(self: *Device) void {
        if (self.presentation_retirement == null or self.completed_presentation_retirement == self.presentation_retirement_value) return;
        var completed: u64 = 0;
        assert(c.vkGetSemaphoreCounterValue(self.device, self.presentation_retirement, &completed) == c.VK_SUCCESS);
        assert(completed >= self.completed_presentation_retirement and completed <= self.presentation_retirement_value);
        if (completed == self.completed_presentation_retirement) return;
        self.completed_presentation_retirement = completed;
        self.swapchain_delete_queue.collect(self.device, completed);
    }

    fn waitPresentationRetirement(self: *Device, value: u64) void {
        assert(value <= self.presentation_retirement_value);
        if (value > self.completed_presentation_retirement) {
            const wait_info: c.VkSemaphoreWaitInfo = .{
                .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .semaphoreCount = 1,
                .pSemaphores = &self.presentation_retirement,
                .pValues = &value,
            };
            assert(c.vkWaitSemaphores(self.device, &wait_info, std.math.maxInt(u64)) == c.VK_SUCCESS);
            self.completed_presentation_retirement = value;
        }
        self.swapchain_delete_queue.collect(self.device, self.completed_presentation_retirement);
    }

    fn nextPresentationRetirement(self: *Device) u64 {
        const next = self.presentation_retirement_value + 1;
        if (next - self.completed_presentation_retirement > self.max_timeline_value_difference) {
            self.pollPresentationRetirement();
            if (next - self.completed_presentation_retirement > self.max_timeline_value_difference)
                self.waitPresentationRetirement(next - self.max_timeline_value_difference);
            assert(next - self.completed_presentation_retirement <= self.max_timeline_value_difference);
        }
        self.presentation_retirement_value = next;
        return next;
    }

    fn createPresentContext(self: *Device, context: *PresentContext) Error!void {
        assert(context.acquired == null and context.rendered == null and context.presented == null and !context.present_pending);
        errdefer self.destroyPresentContext(context);
        const semaphore_info: c.VkSemaphoreCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        try check(native.vkCreateSemaphore(self.device, &semaphore_info, null, &context.acquired), "vkCreateSemaphore (acquire)", 0);
        try check(native.vkCreateSemaphore(self.device, &semaphore_info, null, &context.rendered), "vkCreateSemaphore (render)", 0);
        const fence_info: c.VkFenceCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        try check(c.vkCreateFence(self.device, &fence_info, null, &context.presented), "vkCreateFence", 0);
    }

    fn destroyPresentContext(self: *Device, context: *PresentContext) void {
        assert(!context.present_pending);
        if (context.acquired != null) native.vkDestroySemaphore(self.device, context.acquired, null);
        if (context.rendered != null) native.vkDestroySemaphore(self.device, context.rendered, null);
        if (context.presented != null) c.vkDestroyFence(self.device, context.presented, null);
        context.* = .{};
    }

    fn finishPresentContext(self: *Device, context: *PresentContext) void {
        assert(!context.present_pending and context.swapchain != null);
        const completed_swapchain = context.swapchain;
        context.swapchain = null;
        for (&self.retired_swapchains) |*retired| {
            if (retired.handle != completed_swapchain) continue;
            for (self.present_contexts) |pending| {
                if (pending.present_pending and pending.swapchain == completed_swapchain) return;
            }
            self.queueRetiredSwapchain(retired);
            return;
        }
    }

    fn waitPresentContext(self: *Device, context: *PresentContext) void {
        if (!context.present_pending) return;
        assert(c.vkWaitForFences(self.device, 1, &context.presented, c.VK_TRUE, std.math.maxInt(u64)) == c.VK_SUCCESS);
        context.present_pending = false;
        self.finishPresentContext(context);
    }

    fn pollPresentContexts(self: *Device) void {
        for (self.present_contexts[0..self.present_context_count]) |*context| {
            if (!context.present_pending) continue;
            const result = c.vkGetFenceStatus(self.device, context.presented);
            if (result == c.VK_NOT_READY) continue;
            assert(result == c.VK_SUCCESS);
            context.present_pending = false;
            self.finishPresentContext(context);
        }
    }

    fn queueRetiredSwapchain(self: *Device, retired: *RetiredSwapchain) void {
        assert(retired.handle != null and retired.view_count != 0);
        for (retired.views[0..retired.view_count]) |view| self.swapchain_delete_queue.push(self.presentation_retirement_value, retired.handle, view);
        retired.* = .{};
        self.swapchain_delete_queue.collect(self.device, self.completed_presentation_retirement);
    }

    fn drainContexts(self: *Device) void {
        self.waitPresentationRetirement(self.presentation_retirement_value);
        for (self.present_contexts[0..self.present_context_count]) |*context| self.waitPresentContext(context);
        for (self.retired_swapchains) |retired| assert(retired.handle == null);
        self.swapchain_delete_queue.collect(self.device, self.completed_presentation_retirement);
    }

    /// Releases everything created so far; also cleans up a partially initialized device.
    fn deinit(self: *Device) void {
        assert(self.acquired_swapchain == null);
        if (self.device != null) {
            self.drainContexts();
            if (self.swapchain) |swapchain| {
                assert(!swapchain.acquired);
                retireSwapchainHandle(swapchain);
                gpa.destroy(swapchain);
            }
            self.swapchain_delete_queue.collect(self.device, self.completed_presentation_retirement);
            assert(self.swapchain_delete_queue.count == 0);
            if (self.presentation_retirement != null) native.vkDestroySemaphore(self.device, self.presentation_retirement, null);
            for (self.present_contexts[0..self.present_context_count]) |*context| self.destroyPresentContext(context);
        }
        for (self.queues) |queue| {
            gpa.free(queue.command_submit_infos);
            gpa.free(queue.wait_submit_infos);
        }
        gpa.free(self.queues);
        gpa.free(self.swapchain_delete_queue.entries);
        if (self.device != null) native.vkDestroyDevice(self.device, null);
        if (self.instance != null) {
            if (self.surface != null) c.vkDestroySurfaceKHR(self.instance, self.surface, null);
            if (self.debug_messenger != null) self.destroy_debug_messenger.?(self.instance, self.debug_messenger, null);
            native.vkDestroyInstance(self.instance, null);
        }
        gpa.destroy(self);
    }
};

fn bufferMemoryRequirements(device: *const Device, usage: c.VkBufferUsageFlags) c.VkMemoryRequirements {
    const buffer_info: c.VkBufferCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 1,
        .usage = usage,
        .sharingMode = device.sharingMode(),
        .queueFamilyIndexCount = device.queue_family_count,
        .pQueueFamilyIndices = &device.queue_families,
    };
    const requirements_info: c.VkDeviceBufferMemoryRequirements = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
        .pCreateInfo = &buffer_info,
    };
    var requirements: c.VkMemoryRequirements2 = .{ .sType = c.VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
    c.vkGetDeviceBufferMemoryRequirements(device.device, &requirements_info, &requirements);
    return requirements.memoryRequirements;
}

fn supportsGpuHeapMemory(device: *const Device) bool {
    const ordinary = bufferMemoryRequirements(device, universal_buffer_usage);
    if (device.findMemoryType(ordinary.memoryTypeBits, cpu_visible_memory_properties, 0, ordinary.size, 0) == null) return false;
    if (device.findMemoryType(ordinary.memoryTypeBits, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, ordinary.size, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == null)
        return false;
    const descriptor = bufferMemoryRequirements(device, universal_buffer_usage | c.VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT);
    return device.findMemoryType(descriptor.memoryTypeBits, cpu_visible_memory_properties, 0, descriptor.size, 0) != null;
}

fn fitsImageFormatProperties(image_info: *const c.VkImageCreateInfo, properties: *const c.VkImageFormatProperties) bool {
    return image_info.extent.width <= properties.maxExtent.width and image_info.extent.height <= properties.maxExtent.height and
        image_info.extent.depth <= properties.maxExtent.depth and image_info.mipLevels <= properties.maxMipLevels and
        image_info.arrayLayers <= properties.maxArrayLayers;
}

/// Returns the image format limits when the device can create `image_info`.
fn imageFormatProperties(device: *const Device, image_info: *const c.VkImageCreateInfo) ?c.VkImageFormatProperties {
    const format_info: c.VkPhysicalDeviceImageFormatInfo2 = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .format = image_info.format,
        .type = image_info.imageType,
        .tiling = image_info.tiling,
        .usage = image_info.usage,
        .flags = image_info.flags,
    };
    var properties: c.VkImageFormatProperties2 = .{ .sType = c.VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
    const result = c.vkGetPhysicalDeviceImageFormatProperties2(device.physical_device, &format_info, &properties);
    if (result == c.VK_ERROR_FORMAT_NOT_SUPPORTED) return null;
    check(result, "vkGetPhysicalDeviceImageFormatProperties2", 0) catch return null;
    if (!fitsImageFormatProperties(image_info, &properties.imageFormatProperties)) return null;
    return properties.imageFormatProperties;
}

fn imageMemoryRequirements(device: *const Device, image_info: *const c.VkImageCreateInfo) c.VkMemoryRequirements {
    const requirements_info: c.VkDeviceImageMemoryRequirements = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
        .pCreateInfo = image_info,
    };
    var requirements: c.VkMemoryRequirements2 = .{ .sType = c.VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
    c.vkGetDeviceImageMemoryRequirements(device.device, &requirements_info, &requirements);
    return requirements.memoryRequirements;
}

/// Picks one memory type for every texture heap. Probes cover DCC-capable color, broad 3D, and sampled depth layouts.
/// Resource Memory Association makes the color mask common to ordinary optimal-tiled images; depth/stencil formats are
/// intersected explicitly.
fn selectTextureMemoryType(device: *Device) bool {
    const color_features = device.format_features[@backingInt(Format.rgba8_unorm)];
    if (color_features & c.VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT == 0) return false;
    const limits = &device.physical_properties.limits;
    const probe_2d_size: u32 = @min(limits.maxImageDimension2D, 2048);
    var image_info: c.VkImageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = c.VK_IMAGE_TYPE_2D,
        .format = c.VK_FORMAT_R8G8B8A8_UNORM,
        .extent = .{ .width = probe_2d_size, .height = probe_2d_size, .depth = 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .tiling = c.VK_IMAGE_TILING_OPTIMAL,
        .usage = c.VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = device.sharingMode(),
        .queueFamilyIndexCount = device.queue_family_count,
        .pQueueFamilyIndices = &device.queue_families,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (color_features & c.VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT != 0) image_info.usage |= c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (imageFormatProperties(device, &image_info) == null) {
        image_info.usage = c.VK_IMAGE_USAGE_SAMPLED_BIT;
        if (imageFormatProperties(device, &image_info) == null) return false;
    }
    const color_requirements = imageMemoryRequirements(device, &image_info);
    var memory_type_bits = color_requirements.memoryTypeBits;
    device.texture_heap_alignment = @max(device.texture_heap_alignment, color_requirements.alignment);

    const broad_features = device.format_features[@backingInt(Format.rgba32_float)];
    const broad_required = requiredFormatFeatures(.{ .sampled = true, .storage = true, .transfer_destination = true });
    if (broad_features & broad_required == broad_required) {
        const probe_3d_size: u32 = @min(limits.maxImageDimension3D, 2048);
        image_info.imageType = c.VK_IMAGE_TYPE_3D;
        image_info.format = c.VK_FORMAT_R32G32B32A32_SFLOAT;
        image_info.extent = .{ .width = probe_3d_size, .height = probe_3d_size, .depth = @min(probe_3d_size, 4) };
        image_info.mipLevels = 32 - @clz(probe_3d_size);
        image_info.usage = c.VK_IMAGE_USAGE_SAMPLED_BIT | c.VK_IMAGE_USAGE_STORAGE_BIT | c.VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (imageFormatProperties(device, &image_info) != null)
            device.texture_heap_alignment = @max(device.texture_heap_alignment, imageMemoryRequirements(device, &image_info).alignment);
    }

    const storage_features = requiredFormatFeatures(.{ .storage = true });
    for ([_]Format{ .d16_unorm, .d24_unorm_s8_uint, .d32_float, .s8_uint, .d32_float_s8_uint }) |format| {
        const features = device.format_features[@backingInt(format)];
        const info = api.getTextureFormatInfo(format);
        const combined = info.depth and info.stencil;
        const compatibility_usage: c.VkImageUsageFlags = if (features & c.VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT != 0)
            c.VK_IMAGE_USAGE_SAMPLED_BIT
        else if (features & c.VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT != 0)
            c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
        else if (features & storage_features == storage_features)
            c.VK_IMAGE_USAGE_STORAGE_BIT
        else if (!combined and features & c.VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT != 0)
            c.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        else if (!combined and features & c.VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT != 0)
            c.VK_IMAGE_USAGE_TRANSFER_DST_BIT
        else
            0;
        if (compatibility_usage == 0) continue;
        image_info.imageType = c.VK_IMAGE_TYPE_2D;
        image_info.format = toVkFormat(format);
        image_info.extent = .{ .width = 1, .height = 1, .depth = 1 };
        image_info.mipLevels = 1;
        image_info.usage = compatibility_usage;
        const compatibility_properties = imageFormatProperties(device, &image_info) orelse continue;
        image_info.extent = .{ .width = 512, .height = 512, .depth = 1 };
        if (features & c.VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT != 0 and features & c.VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT != 0)
            image_info.usage = c.VK_IMAGE_USAGE_SAMPLED_BIT | c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        var supported = if (image_info.usage == compatibility_usage)
            fitsImageFormatProperties(&image_info, &compatibility_properties)
        else
            imageFormatProperties(device, &image_info) != null;
        if (!supported and image_info.usage != compatibility_usage) {
            image_info.usage = compatibility_usage;
            supported = fitsImageFormatProperties(&image_info, &compatibility_properties);
        }
        if (!supported) {
            image_info.extent = .{ .width = 1, .height = 1, .depth = 1 };
            image_info.usage = compatibility_usage;
        }
        const requirements = imageMemoryRequirements(device, &image_info);
        memory_type_bits &= requirements.memoryTypeBits;
        device.texture_heap_alignment = @max(device.texture_heap_alignment, requirements.alignment);
    }
    device.texture_memory_type = device.findMemoryType(
        memory_type_bits,
        c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        0,
        1,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    ) orelse
        return false;
    return true;
}

fn recordImageBarrier(command_buffer: c.VkCommandBuffer, image_barrier: *const c.VkImageMemoryBarrier2) void {
    const dependency: c.VkDependencyInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = image_barrier,
    };
    c.vkCmdPipelineBarrier2(command_buffer, &dependency);
}

fn recordMemoryBarrier(command_buffer: c.VkCommandBuffer, memory_barrier: *const c.VkMemoryBarrier2) void {
    const dependency: c.VkDependencyInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = memory_barrier,
    };
    c.vkCmdPipelineBarrier2(command_buffer, &dependency);
}

fn heapBindInfo(heap: api.GpuRange, reserved_alignment: u64, reserved_size: u64) c.VkBindHeapInfoEXT {
    const reserved_offset = alignUp(heap.size, reserved_alignment);
    return .{
        .sType = c.VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
        .heapRange = .{ .address = address(heap.gpu), .size = reserved_offset + reserved_size },
        .reservedRangeOffset = reserved_offset,
        .reservedRangeSize = reserved_size,
    };
}

fn enumerated(result: c.VkResult, operation: []const u8) Error!void {
    if (result == c.VK_INCOMPLETE) return error.Unsupported;
    return check(result, operation, 0);
}

/// Feature structures chained in place; the chain points into this struct, so it must not move after `init`.
const QueriedFeatures = struct {
    core: c.VkPhysicalDeviceFeatures2 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 },
    vulkan11: c.VkPhysicalDeviceVulkan11Features = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES },
    vulkan12: c.VkPhysicalDeviceVulkan12Features = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES },
    vulkan13: c.VkPhysicalDeviceVulkan13Features = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES },
    vulkan14: c.VkPhysicalDeviceVulkan14Features = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES },
    descriptor_heap: c.VkPhysicalDeviceDescriptorHeapFeaturesEXT = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT },
    address_commands: c.VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR,
    },
    untyped_pointers: c.VkPhysicalDeviceShaderUntypedPointersFeaturesKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
    },
    unified_image_layouts: c.VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR,
    },
    mesh_shader: c.VkPhysicalDeviceMeshShaderFeaturesEXT = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT },
    swapchain_maintenance1: c.VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
    },

    fn init(self: *QueriedFeatures, presentation: bool, include_unified_image_layouts: bool) void {
        self.* = .{};
        self.core.pNext = &self.vulkan11;
        self.vulkan11.pNext = &self.vulkan12;
        self.vulkan12.pNext = &self.vulkan13;
        self.vulkan13.pNext = &self.vulkan14;
        self.vulkan14.pNext = &self.descriptor_heap;
        self.descriptor_heap.pNext = &self.address_commands;
        self.address_commands.pNext = &self.untyped_pointers;
        self.untyped_pointers.pNext = if (include_unified_image_layouts) @ptrCast(&self.unified_image_layouts) else @ptrCast(&self.mesh_shader);
        self.unified_image_layouts.pNext = &self.mesh_shader;
        self.mesh_shader.pNext = if (presentation) @ptrCast(&self.swapchain_maintenance1) else null;
    }
};

const Candidate = struct {
    physical_device: c.VkPhysicalDevice,
    queue_families: [queue_type_count]u32 = @splat(0),
    queue_counts: [queue_type_count]u32 = @splat(0),
    timestamps: [queue_type_count]bool = @splat(false),
    copy_texture_granularity: api.Uint32x3 = .{ .x = 1, .y = 1, .z = 1 },
    properties: c.VkPhysicalDeviceProperties = .{},
    memory_properties: c.VkPhysicalDeviceMemoryProperties = .{},
    unified_image_layouts: bool = false,
    image_cube_array: bool = false,
    texture_compression_bc: bool = false,
    texture_compression_astc: bool = false,
    texture_compression_etc2: bool = false,
    storage_input_output16: bool = false,
    khr_swapchain_maintenance1: bool = false,
    heap_properties: c.VkPhysicalDeviceDescriptorHeapPropertiesEXT = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT },
    vulkan12_properties: c.VkPhysicalDeviceVulkan12Properties = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES },
};

/// Returns null when the physical device lacks a required capability.
fn inspectCandidate(
    physical_device: c.VkPhysicalDevice,
    surface: c.VkSurfaceKHR,
    khr_surface_maintenance1: bool,
    ext_surface_maintenance1: bool,
    desc: api.DeviceDesc,
) Error!?Candidate {
    var extension_buffer: [max_device_extensions]c.VkExtensionProperties = undefined;
    var extension_count: u32 = max_device_extensions;
    enumerated(
        c.vkEnumerateDeviceExtensionProperties(physical_device, null, &extension_count, &extension_buffer),
        "vkEnumerateDeviceExtensionProperties",
    ) catch |err|
        return if (err == error.Unsupported) null else err;
    const extensions = extension_buffer[0..extension_count];
    const unified_image_layouts_extension = hasName(extensions, c.VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
    for ([_][]const u8{
        c.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME,
        c.VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME,
        c.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME,
        c.VK_EXT_MESH_SHADER_EXTENSION_NAME,
    }) |name| {
        if (!hasName(extensions, name)) return null;
    }
    const khr_swapchain_maintenance1 = khr_surface_maintenance1 and hasName(extensions, c.VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
    const ext_swapchain_maintenance1 = ext_surface_maintenance1 and hasName(extensions, c.VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
    if (surface != null and (!hasName(extensions, c.VK_KHR_SWAPCHAIN_EXTENSION_NAME) or (!khr_swapchain_maintenance1 and !ext_swapchain_maintenance1)))
        return null;

    var result: Candidate = .{ .physical_device = physical_device };
    result.heap_properties.pNext = &result.vulkan12_properties;
    var properties2: c.VkPhysicalDeviceProperties2 = .{ .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &result.heap_properties };
    c.vkGetPhysicalDeviceProperties2(physical_device, &properties2);
    result.properties = properties2.properties;
    result.heap_properties.pNext = null;
    if (result.properties.apiVersion < c.VK_API_VERSION_1_4) return null;

    c.vkGetPhysicalDeviceMemoryProperties(physical_device, &result.memory_properties);
    const cpu_visible_memory = for (0..result.memory_properties.memoryTypeCount) |index| {
        const flags = result.memory_properties.memoryTypes[index].propertyFlags;
        const usable = isUsableMemoryType(&result.memory_properties, @intCast(index));
        if (usable and flags & cpu_visible_memory_properties == cpu_visible_memory_properties) break true;
    } else false;
    if (!cpu_visible_memory) return null;

    var features: QueriedFeatures = undefined;
    features.init(surface != null, unified_image_layouts_extension);
    c.vkGetPhysicalDeviceFeatures2(physical_device, &features.core);
    const core = &features.core.features;
    const required_features = core.shaderInt16 == c.VK_TRUE and core.samplerAnisotropy == c.VK_TRUE and core.depthBiasClamp == c.VK_TRUE and
        core.independentBlend == c.VK_TRUE and core.fragmentStoresAndAtomics == c.VK_TRUE and core.vertexPipelineStoresAndAtomics == c.VK_TRUE and
        core.shaderStorageImageReadWithoutFormat == c.VK_TRUE and core.shaderStorageImageWriteWithoutFormat == c.VK_TRUE and
        core.multiDrawIndirect == c.VK_TRUE and core.drawIndirectFirstInstance == c.VK_TRUE and
        features.vulkan11.storageBuffer16BitAccess == c.VK_TRUE and features.vulkan11.storagePushConstant16 == c.VK_TRUE and
        features.vulkan11.shaderDrawParameters == c.VK_TRUE and features.vulkan12.shaderFloat16 == c.VK_TRUE and
        features.vulkan12.scalarBlockLayout == c.VK_TRUE and features.vulkan12.bufferDeviceAddress == c.VK_TRUE and
        features.vulkan12.timelineSemaphore == c.VK_TRUE and features.vulkan13.synchronization2 == c.VK_TRUE and
        features.vulkan13.dynamicRendering == c.VK_TRUE and features.vulkan13.maintenance4 == c.VK_TRUE and
        features.vulkan14.maintenance5 == c.VK_TRUE and features.descriptor_heap.descriptorHeap == c.VK_TRUE and
        features.address_commands.deviceAddressCommands == c.VK_TRUE and features.untyped_pointers.shaderUntypedPointers == c.VK_TRUE and
        features.mesh_shader.meshShader == c.VK_TRUE and (core.textureCompressionBC == c.VK_TRUE or core.textureCompressionASTC_LDR == c.VK_TRUE) and
        (surface == null or features.swapchain_maintenance1.swapchainMaintenance1 == c.VK_TRUE);
    if (!required_features) return null;
    result.unified_image_layouts = unified_image_layouts_extension and features.unified_image_layouts.unifiedImageLayouts == c.VK_TRUE;
    result.image_cube_array = core.imageCubeArray == c.VK_TRUE;
    result.texture_compression_bc = core.textureCompressionBC == c.VK_TRUE;
    result.texture_compression_astc = core.textureCompressionASTC_LDR == c.VK_TRUE;
    result.texture_compression_etc2 = core.textureCompressionETC2 == c.VK_TRUE;
    result.storage_input_output16 = features.vulkan11.storageInputOutput16 == c.VK_TRUE;
    result.khr_swapchain_maintenance1 = khr_swapchain_maintenance1;

    var family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, null);
    if (family_count > max_queue_families) return null;
    var families: [max_queue_families]c.VkQueueFamilyProperties = undefined;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, &families);
    for (families[0..family_count], 0..) |family, family_index| {
        if (family.queueCount == 0) continue;
        const flags = family.queueFlags;
        const queue_type: usize = if (flags & c.VK_QUEUE_GRAPHICS_BIT != 0)
            (if (flags & c.VK_QUEUE_COMPUTE_BIT != 0) 0 else continue)
        else if (flags & c.VK_QUEUE_COMPUTE_BIT != 0)
            1
        else if (flags & ~@as(c.VkQueueFlags, c.VK_QUEUE_SPARSE_BINDING_BIT | c.VK_QUEUE_PROTECTED_BIT) == c.VK_QUEUE_TRANSFER_BIT)
            2
        else
            continue;
        if (result.queue_counts[queue_type] != 0) continue;
        var presentation_supported: c.VkBool32 = c.VK_TRUE;
        if (surface != null and queue_type == 0)
            try check(
                c.vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, @intCast(family_index), surface, &presentation_supported),
                "vkGetPhysicalDeviceSurfaceSupportKHR",
                0,
            );
        if (presentation_supported != c.VK_TRUE) continue;
        result.queue_families[queue_type] = @intCast(family_index);
        result.queue_counts[queue_type] = family.queueCount;
        result.timestamps[queue_type] = desc.timestamp_query_count != 0 and family.timestampValidBits == 64 and features.vulkan12.hostQueryReset == c.VK_TRUE;
        if (queue_type == 2) {
            const granularity = family.minImageTransferGranularity;
            result.copy_texture_granularity = .{ .x = granularity.width, .y = granularity.height, .z = granularity.depth };
        }
    }
    if (result.queue_counts[0] == 0 or (desc.desired_compute_queue_count != 0 and result.queue_counts[1] == 0) or
        (desc.desired_copy_queue_count != 0 and result.queue_counts[2] == 0)) return null;
    return result;
}

pub fn createDevice(desc: api.DeviceDesc) Error!*Device {
    assert(desc.desired_queue_count != 0);
    assert(desc.desired_swapchain_image_count != 0 and desc.desired_swapchain_image_count <= max_swapchain_images);
    const presentation = desc.window != null;
    assert(!presentation or !x11 or desc.display != null);
    if (presentation and !windows and !x11) return error.Unsupported;

    var loader_version: u32 = c.VK_API_VERSION_1_0;
    try check(c.vkEnumerateInstanceVersion(&loader_version), "vkEnumerateInstanceVersion", 0);
    if (loader_version < c.VK_API_VERSION_1_4) {
        report(false, "Vulkan 1.4 or newer is required. Update the graphics driver.", .{});
        return error.Unsupported;
    }

    const device = create(Device, .{
        .timestamp_query_count = desc.timestamp_query_count,
        .present_context_count = if (presentation) desc.desired_swapchain_image_count else 0,
    });
    var reported = false;
    initDevice(device, desc, &reported) catch |err| {
        if (err == error.Unsupported and !reported) report(false, "Required Vulkan device, memory, or display capabilities are unavailable.", .{});
        device.deinit();
        return err;
    };
    return device;
}

fn initDevice(device: *Device, desc: api.DeviceDesc, reported: *bool) Error!void {
    const presentation = desc.window != null;
    var instance_extension_buffer: [max_instance_extensions]c.VkExtensionProperties = undefined;
    var instance_extension_count: u32 = max_instance_extensions;
    try enumerated(
        c.vkEnumerateInstanceExtensionProperties(null, &instance_extension_count, &instance_extension_buffer),
        "vkEnumerateInstanceExtensionProperties",
    );
    const instance_extensions = instance_extension_buffer[0..instance_extension_count];

    var khr_surface_maintenance1 = false;
    var ext_surface_maintenance1 = false;
    if (presentation) {
        device.swapchain_color_space_enabled = hasName(instance_extensions, c.VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
        if (desc.swapchain_color_space != .srgb and !device.swapchain_color_space_enabled) return error.Unsupported;
        khr_surface_maintenance1 = hasName(instance_extensions, c.VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        ext_surface_maintenance1 = hasName(instance_extensions, c.VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        if (!hasName(instance_extensions, c.VK_KHR_SURFACE_EXTENSION_NAME) or !hasName(instance_extensions, surface_extension_name) or
            !hasName(instance_extensions, c.VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME) or (!khr_surface_maintenance1 and !ext_surface_maintenance1))
            return error.Unsupported;
    }

    var debug_utils_available = false;
    var validation_available = false;
    if (validation) {
        var layer_buffer: [max_instance_layers]c.VkLayerProperties = undefined;
        var layer_count: u32 = max_instance_layers;
        try enumerated(c.vkEnumerateInstanceLayerProperties(&layer_count, &layer_buffer), "vkEnumerateInstanceLayerProperties");
        debug_utils_available = hasName(instance_extensions, c.VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        validation_available = hasName(layer_buffer[0..layer_count], "VK_LAYER_KHRONOS_validation");
    }

    var enabled_instance_extensions: [7][*:0]const u8 = undefined;
    var enabled_instance_extension_count: u32 = 0;
    const enabled_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
    if (debug_utils_available) {
        enabled_instance_extensions[enabled_instance_extension_count] = c.VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        enabled_instance_extension_count += 1;
    }
    if (presentation) {
        const names = [_]struct { bool, [*:0]const u8 }{
            .{ device.swapchain_color_space_enabled, c.VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME },
            .{ true, c.VK_KHR_SURFACE_EXTENSION_NAME },
            .{ true, surface_extension_name },
            .{ true, c.VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME },
            .{ khr_surface_maintenance1, c.VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME },
            .{ ext_surface_maintenance1, c.VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME },
        };
        for (names) |name| {
            if (!name[0]) continue;
            enabled_instance_extensions[enabled_instance_extension_count] = name[1];
            enabled_instance_extension_count += 1;
        }
    }

    const app_info: c.VkApplicationInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "NoGraphicsAPI application",
        .applicationVersion = c.VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "NoGraphicsAPI",
        .engineVersion = c.VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = c.VK_API_VERSION_1_4,
    };
    const instance_info: c.VkInstanceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = if (validation_available) 1 else 0,
        .ppEnabledLayerNames = if (validation_available) @ptrCast(&enabled_layers) else null,
        .enabledExtensionCount = enabled_instance_extension_count,
        .ppEnabledExtensionNames = if (enabled_instance_extension_count != 0) @ptrCast(&enabled_instance_extensions) else null,
    };
    try check(native.vkCreateInstance(&instance_info, null, &device.instance), "vkCreateInstance", 0);

    if (debug_utils_available) {
        const create_debug: c.PFN_vkCreateDebugUtilsMessengerEXT = @ptrCast(c.vkGetInstanceProcAddr(device.instance, "vkCreateDebugUtilsMessengerEXT"));
        device.destroy_debug_messenger = @ptrCast(c.vkGetInstanceProcAddr(device.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (create_debug == null or device.destroy_debug_messenger == null) return error.DriverError;
        const debug_info: c.VkDebugUtilsMessengerCreateInfoEXT = .{
            .sType = c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
        };
        try check(create_debug.?(device.instance, &debug_info, null, &device.debug_messenger), "vkCreateDebugUtilsMessengerEXT", 0);
    }

    if (windows and presentation) {
        const surface_info: c.VkWin32SurfaceCreateInfoKHR = .{
            .sType = c.VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .hinstance = c.GetModuleHandleW(null),
            .hwnd = @ptrCast(@alignCast(desc.window)),
        };
        try check(c.vkCreateWin32SurfaceKHR(device.instance, &surface_info, null, &device.surface), "vkCreateWin32SurfaceKHR", 0);
    }
    if (x11 and presentation) {
        const surface_info: c.VkXlibSurfaceCreateInfoKHR = .{
            .sType = c.VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            .dpy = @ptrCast(desc.display),
            .window = @intFromPtr(desc.window),
        };
        try check(c.vkCreateXlibSurfaceKHR(device.instance, &surface_info, null, &device.surface), "vkCreateXlibSurfaceKHR", 0);
    }

    var physical_device_buffer: [max_physical_devices]c.VkPhysicalDevice = undefined;
    var physical_device_count: u32 = max_physical_devices;
    try enumerated(c.vkEnumeratePhysicalDevices(device.instance, &physical_device_count, &physical_device_buffer), "vkEnumeratePhysicalDevices");
    var selected: ?Candidate = null;
    for (physical_device_buffer[0..physical_device_count]) |physical_device| {
        const candidate = try inspectCandidate(physical_device, device.surface, khr_surface_maintenance1, ext_surface_maintenance1, desc) orelse continue;
        const discrete = candidate.properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (selected == null or discrete) selected = candidate;
        if (discrete) break;
    }
    const chosen = selected orelse {
        report(false, "No compatible GPU found. Vulkan 1.4, VK_EXT_descriptor_heap, VK_KHR_device_address_commands, VK_EXT_mesh_shader, " ++
            "VK_KHR_shader_untyped_pointers, and the required memory, shader, and presentation features are needed.", .{});
        reported.* = true;
        return error.Unsupported;
    };

    device.physical_device = chosen.physical_device;
    device.physical_properties = chosen.properties;
    device.heap_properties = chosen.heap_properties;
    device.max_timeline_value_difference = chosen.vulkan12_properties.maxTimelineSemaphoreValueDifference;
    device.memory_properties = chosen.memory_properties;
    for (&device.format_features, 0..) |*features, value| features.* = optimalFormatFeatures(device.physical_device, @fromBackingInt(@intCast(value)));

    var enabled: QueriedFeatures = undefined;
    enabled.init(presentation, chosen.unified_image_layouts);
    device.texture_compression_etc2 = chosen.texture_compression_etc2;
    const core = &enabled.core.features;
    core.imageCubeArray = @intFromBool(chosen.image_cube_array);
    core.samplerAnisotropy = c.VK_TRUE;
    core.shaderInt16 = c.VK_TRUE;
    core.depthBiasClamp = c.VK_TRUE;
    core.independentBlend = c.VK_TRUE;
    core.textureCompressionBC = @intFromBool(chosen.texture_compression_bc);
    core.textureCompressionASTC_LDR = @intFromBool(chosen.texture_compression_astc);
    core.textureCompressionETC2 = @intFromBool(chosen.texture_compression_etc2);
    core.fragmentStoresAndAtomics = c.VK_TRUE;
    core.vertexPipelineStoresAndAtomics = c.VK_TRUE;
    core.shaderStorageImageReadWithoutFormat = c.VK_TRUE;
    core.shaderStorageImageWriteWithoutFormat = c.VK_TRUE;
    core.multiDrawIndirect = c.VK_TRUE;
    core.drawIndirectFirstInstance = c.VK_TRUE;
    enabled.vulkan11.storageBuffer16BitAccess = c.VK_TRUE;
    enabled.vulkan11.storagePushConstant16 = c.VK_TRUE;
    enabled.vulkan11.storageInputOutput16 = @intFromBool(chosen.storage_input_output16);
    enabled.vulkan11.shaderDrawParameters = c.VK_TRUE;
    enabled.vulkan12.shaderFloat16 = c.VK_TRUE;
    enabled.vulkan12.scalarBlockLayout = c.VK_TRUE;
    enabled.vulkan12.timelineSemaphore = c.VK_TRUE;
    enabled.vulkan12.hostQueryReset = @intFromBool(chosen.timestamps[0] or chosen.timestamps[1] or chosen.timestamps[2]);
    enabled.vulkan12.bufferDeviceAddress = c.VK_TRUE;
    enabled.vulkan13.synchronization2 = c.VK_TRUE;
    enabled.vulkan13.dynamicRendering = c.VK_TRUE;
    enabled.vulkan13.maintenance4 = c.VK_TRUE;
    enabled.vulkan14.maintenance5 = c.VK_TRUE;
    enabled.descriptor_heap.descriptorHeap = c.VK_TRUE;
    enabled.address_commands.deviceAddressCommands = c.VK_TRUE;
    enabled.untyped_pointers.shaderUntypedPointers = c.VK_TRUE;
    enabled.unified_image_layouts.unifiedImageLayouts = @intFromBool(chosen.unified_image_layouts);
    enabled.mesh_shader.taskShader = c.VK_TRUE;
    enabled.mesh_shader.meshShader = c.VK_TRUE;
    enabled.swapchain_maintenance1.swapchainMaintenance1 = c.VK_TRUE;

    const requested_counts = [queue_type_count]u32{ desc.desired_queue_count, desc.desired_compute_queue_count, desc.desired_copy_queue_count };
    var queue_counts: [queue_type_count]u32 = undefined;
    var queue_count: u32 = 0;
    for (&queue_counts, requested_counts, chosen.queue_counts) |*count, requested, available| {
        count.* = @min(requested, available);
        queue_count += count.*;
    }
    device.queues = gpa.alloc(Queue, queue_count) catch @panic("out of memory");
    @memset(device.queues, .{});
    const queue_priorities = gpa.alloc(f32, queue_count) catch @panic("out of memory");
    defer gpa.free(queue_priorities);
    @memset(queue_priorities, 1.0);
    var queue_infos: [queue_type_count]c.VkDeviceQueueCreateInfo = undefined;
    var first_queue: u32 = 0;
    for (queue_counts, chosen.queue_families) |count, family| {
        if (count == 0) continue;
        device.queue_families[device.queue_family_count] = family;
        queue_infos[device.queue_family_count] = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount = count,
            .pQueuePriorities = queue_priorities[first_queue..].ptr,
        };
        device.queue_family_count += 1;
        first_queue += count;
    }

    var enabled_device_extensions: [7][*:0]const u8 = undefined;
    var enabled_device_extension_count: u32 = 0;
    const device_extension_names = [_]struct { bool, [*:0]const u8 }{
        .{ true, c.VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME },
        .{ true, c.VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME },
        .{ true, c.VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME },
        .{ chosen.unified_image_layouts, c.VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME },
        .{ true, c.VK_EXT_MESH_SHADER_EXTENSION_NAME },
        .{ presentation, c.VK_KHR_SWAPCHAIN_EXTENSION_NAME },
        .{ presentation and chosen.khr_swapchain_maintenance1, c.VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME },
        .{ presentation and !chosen.khr_swapchain_maintenance1, c.VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME },
    };
    for (device_extension_names) |name| {
        if (!name[0]) continue;
        enabled_device_extensions[enabled_device_extension_count] = name[1];
        enabled_device_extension_count += 1;
    }
    const device_info: c.VkDeviceCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &enabled.core,
        .queueCreateInfoCount = device.queue_family_count,
        .pQueueCreateInfos = &queue_infos,
        .enabledExtensionCount = enabled_device_extension_count,
        .ppEnabledExtensionNames = @ptrCast(&enabled_device_extensions),
    };
    try check(native.vkCreateDevice(device.physical_device, &device_info, null, &device.device), "vkCreateDevice", 0);
    var queue_index: usize = 0;
    for (queue_counts, chosen.queue_families, chosen.timestamps) |count, family, timestamps| {
        for (0..count) |index| {
            const queue = &device.queues[queue_index];
            queue.family_index = family;
            queue.timestamps = timestamps;
            c.vkGetDeviceQueue(device.device, family, @intCast(index), &queue.queue);
            queue_index += 1;
        }
    }
    if (!supportsGpuHeapMemory(device) or !selectTextureMemoryType(device)) return error.Unsupported;

    inline for (comptime std.meta.fieldNames(DeviceFunctions)) |name| {
        @field(device.fns, name) = @ptrCast(c.vkGetDeviceProcAddr(device.device, name));
        if (@field(device.fns, name) == null) {
            report(false, "The graphics driver did not provide the required Vulkan command entry points.", .{});
            return error.DriverError;
        }
    }

    if (presentation) {
        const type_info: c.VkSemaphoreTypeCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = c.VK_SEMAPHORE_TYPE_TIMELINE,
        };
        const semaphore_info: c.VkSemaphoreCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type_info };
        try check(native.vkCreateSemaphore(device.device, &semaphore_info, null, &device.presentation_retirement), "vkCreateSemaphore (presentation)", 0);
        for (device.present_contexts[0..device.present_context_count]) |*context| try device.createPresentContext(context);
    }

    device.caps = .{
        .device_name = @ptrCast(&device.physical_properties.deviceName),
        .queue_count = queue_count,
        .general_queue_count = queue_counts[0],
        .compute_queue_count = queue_counts[1],
        .copy_queue_count = queue_counts[2],
        .copy_texture_granularity = chosen.copy_texture_granularity,
        .max_push_data_size = device.heap_properties.maxPushDataSize,
        .texture_heap_alignment = device.texture_heap_alignment,
        .timestamp_period_ns = chosen.properties.limits.timestampPeriod,
        .sub_texel_precision_bits = chosen.properties.limits.subTexelPrecisionBits,
        .texture_compression_bc = chosen.texture_compression_bc,
        .texture_compression_astc = chosen.texture_compression_astc,
        .storage_input_output16 = chosen.storage_input_output16,
        .indirect_mesh_draw = true,
    };
    if (presentation) {
        device.swapchain = create(Swapchain, .{
            .device = device,
            .format = desc.swapchain_format,
            .color_space = toVkColorSpace(desc.swapchain_color_space),
        });
        try recreateSwapchain(device.swapchain.?);
    }
}

pub fn destroyDevice(device: *Device) void {
    device.deinit();
}

pub fn getDeviceCaps(device: *const Device) *const api.DeviceCaps {
    return &device.caps;
}

pub fn supportsTextureFormat(device: *const Device, format: Format, usage: api.TextureUsage) bool {
    assert(format != .undefined and @as(u32, @bitCast(usage)) != 0);
    const info = api.getTextureFormatInfo(format);
    if ((info.depth or info.stencil) and usage.color_attachment) return false;
    if (!info.depth and !info.stencil and usage.depth_stencil_attachment) return false;
    if (info.depth and info.stencil and (usage.transfer_source or usage.transfer_destination)) return false;
    const compression_supported = switch (textureCompression(format)) {
        .none => true,
        .etc2 => device.texture_compression_etc2,
        .astc => device.caps.texture_compression_astc,
        .bc => device.caps.texture_compression_bc,
    };
    if (!compression_supported) return false;
    const required = requiredFormatFeatures(usage);
    return device.format_features[@backingInt(format)] & required == required;
}

pub fn createTimelineSemaphore(device: *Device, initial_value: u64) Error!*TimelineSemaphore {
    const timeline_info: c.VkSemaphoreTypeCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = c.VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = initial_value,
    };
    const create_info: c.VkSemaphoreCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timeline_info };
    var semaphore: c.VkSemaphore = null;
    try check(native.vkCreateSemaphore(device.device, &create_info, null, &semaphore), "vkCreateSemaphore (timeline)", 0);
    return create(TimelineSemaphore, .{ .device = device, .semaphore = semaphore });
}

pub fn destroyTimelineSemaphore(semaphore: *TimelineSemaphore) void {
    native.vkDestroySemaphore(semaphore.device.device, semaphore.semaphore, null);
    gpa.destroy(semaphore);
}

pub fn timelineCompletedValue(semaphore: *const TimelineSemaphore) u64 {
    var value: u64 = 0;
    assert(c.vkGetSemaphoreCounterValue(semaphore.device.device, semaphore.semaphore, &value) == c.VK_SUCCESS);
    return value;
}

pub fn waitTimeline(point: api.TimelinePoint) void {
    const wait_info: c.VkSemaphoreWaitInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &point.semaphore.semaphore,
        .pValues = &point.value,
    };
    assert(c.vkWaitSemaphores(point.semaphore.device.device, &wait_info, std.math.maxInt(u64)) == c.VK_SUCCESS);
}

pub fn waitIdle(device: *Device) void {
    assert(device.acquired_swapchain == null);
    assert(c.vkDeviceWaitIdle(device.device) == c.VK_SUCCESS);
    device.drainContexts();
    device.next_present_context = 0;
}

pub fn writeTimestamp(commands: *CommandBuffer, cpu_destination: *u64, stage: api.Stage) void {
    if (commands.timestamp_pool == null) return;
    assert(commands.timestamp_count < commands.device.timestamp_query_count);
    commands.timestamp_destinations[commands.timestamp_count] = cpu_destination;
    c.vkCmdWriteTimestamp2(commands.command_buffer, toVkStages(stage), commands.timestamp_pool, commands.timestamp_count);
    commands.timestamp_count += 1;
}

pub fn readTimestamps(pool: *CommandPool) void {
    var next = pool.first;
    while (next) |commands| : (next = commands.next) {
        if (!commands.submitted or commands.timestamp_count == 0) continue;
        const count = commands.timestamp_count;
        const results = commands.timestamp_results[0..count];
        const size = @sizeOf(u64) * count;
        const result = c.vkGetQueryPoolResults(pool.device.device, commands.timestamp_pool, 0, count, size, results.ptr, 8, c.VK_QUERY_RESULT_64_BIT);
        assert(result == c.VK_SUCCESS);
        for (commands.timestamp_destinations[0..count], commands.timestamp_results[0..count]) |destination, value| destination.* = value;
        commands.timestamp_count = 0;
    }
}

pub fn createGpuHeap(device: *Device, byte_count: u64, memory: api.MemoryType) Error!api.GpuHeap {
    const required: c.VkMemoryPropertyFlags, const preferred: c.VkMemoryPropertyFlags, const avoided: c.VkMemoryPropertyFlags = switch (memory) {
        .cpu_visible => .{ cpu_visible_memory_properties, 0, 0 },
        .gpu_only => .{ c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT },
        .readback => .{ c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, c.VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0 },
    };
    const backing = try device.createBackingBuffer(byte_count, universal_buffer_usage, required, preferred, avoided);
    return .{
        .range = .{ .cpu = @ptrCast(backing.mapped), .gpu = @ptrFromInt(backing.address), .size = byte_count },
        .owner = create(GpuHeapOwner, .{ .device = device, .backing = backing }),
    };
}

pub fn destroyGpuHeap(heap: api.GpuHeap) void {
    const device = heap.owner.device.device;
    const backing = heap.owner.backing;
    if (backing.mapped != null) c.vkUnmapMemory(device, backing.memory);
    native.vkDestroyBuffer(device, backing.buffer, null);
    native.vkFreeMemory(device, backing.memory, null);
    gpa.destroy(heap.owner);
}

fn retireSwapchainHandle(swapchain: *Swapchain) void {
    const device = swapchain.device;
    if (swapchain.handle == null) {
        assert(swapchain.image_count == 0);
        swapchain.width = 0;
        swapchain.height = 0;
        return;
    }
    device.pollPresentContexts();
    var retired: RetiredSwapchain = .{ .handle = swapchain.handle, .view_count = swapchain.image_count };
    for (0..swapchain.image_count) |index| {
        retired.views[index] = swapchain.render_views[index].view;
        swapchain.render_views[index] = .{};
        swapchain.images[index] = null;
        swapchain.initialized[index] = false;
    }
    swapchain.image_count = 0;
    swapchain.handle = null;
    swapchain.width = 0;
    swapchain.height = 0;

    const contexts = device.present_contexts[0..device.present_context_count];
    const present_pending = for (contexts) |context| {
        if (context.present_pending and context.swapchain == retired.handle) break true;
    } else false;
    if (present_pending) {
        for (&device.retired_swapchains) |*slot| {
            if (slot.handle == null) {
                slot.* = retired;
                return;
            }
        }
        for (contexts) |*context| {
            if (context.present_pending and context.swapchain == retired.handle) device.waitPresentContext(context);
        }
    }
    device.queueRetiredSwapchain(&retired);
}

fn chooseCompositeAlpha(supported: c.VkCompositeAlphaFlagsKHR) c.VkCompositeAlphaFlagBitsKHR {
    for ([_]c.VkCompositeAlphaFlagBitsKHR{
        c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        c.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        c.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        c.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    }) |choice| {
        if (supported & choice != 0) return choice;
    }
    return 0;
}

fn toVkColorSpace(color_space: api.ColorSpace) c.VkColorSpaceKHR {
    return switch (color_space) {
        .srgb => c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        .extended_srgb_linear => c.VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT,
    };
}

fn surfaceCapabilities(device: *const Device) Error!c.VkSurfaceCapabilitiesKHR {
    var capabilities: c.VkSurfaceCapabilitiesKHR = .{};
    try check(
        c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical_device, device.surface, &capabilities),
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        0,
    );
    return capabilities;
}

fn swapchainSurfaceConfigurationChanged(swapchain: *const Swapchain) bool {
    const capabilities = surfaceCapabilities(swapchain.device) catch |err| {
        requireOk(err, "Display surface query");
        unreachable;
    };
    const extent = capabilities.currentExtent;
    const variable_extent = std.math.maxInt(u32);
    return extent.width == variable_extent or extent.height == variable_extent or extent.width != swapchain.width or extent.height != swapchain.height or
        capabilities.currentTransform != swapchain.transform or chooseCompositeAlpha(capabilities.supportedCompositeAlpha) != swapchain.composite_alpha;
}

fn recreateSwapchain(swapchain: *Swapchain) Error!void {
    const device = swapchain.device;
    assert(!swapchain.acquired and device.acquired_swapchain == null);
    const present_mode_info: c.VkSurfacePresentModeKHR = .{ .sType = c.VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_KHR, .presentMode = swapchain_present_mode };
    const surface_info: c.VkPhysicalDeviceSurfaceInfo2KHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
        .pNext = &present_mode_info,
        .surface = device.surface,
    };
    var capabilities_info: c.VkSurfaceCapabilities2KHR = .{ .sType = c.VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR };
    try check(
        c.vkGetPhysicalDeviceSurfaceCapabilities2KHR(device.physical_device, &surface_info, &capabilities_info),
        "vkGetPhysicalDeviceSurfaceCapabilities2KHR",
        0,
    );
    const capabilities = capabilities_info.surfaceCapabilities;
    const extent = capabilities.currentExtent;
    if (extent.width == std.math.maxInt(u32) or capabilities.supportedUsageFlags & c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT == 0) return error.Unsupported;

    var formats: [max_surface_formats]c.VkSurfaceFormatKHR = undefined;
    var surface_format_count: u32 = 0;
    try check(
        c.vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical_device, device.surface, &surface_format_count, null),
        "vkGetPhysicalDeviceSurfaceFormatsKHR",
        0,
    );
    if (surface_format_count == 0 or surface_format_count > max_surface_formats) return error.Unsupported;
    try check(
        c.vkGetPhysicalDeviceSurfaceFormatsKHR(device.physical_device, device.surface, &surface_format_count, &formats),
        "vkGetPhysicalDeviceSurfaceFormatsKHR",
        0,
    );
    const requested_format = toVkFormat(swapchain.format);
    const format_supported = for (formats[0..surface_format_count]) |format| {
        if ((format.format == requested_format or format.format == c.VK_FORMAT_UNDEFINED) and format.colorSpace == swapchain.color_space) break true;
    } else false;
    if (!format_supported) return error.Unsupported;

    if (extent.width == 0 or extent.height == 0) {
        swapchain.width = 0;
        swapchain.height = 0;
        swapchain.recreate_required = true;
        return;
    }
    var image_count = @max(device.present_context_count, capabilities.minImageCount);
    if (capabilities.maxImageCount != 0) image_count = @min(image_count, capabilities.maxImageCount);
    if (image_count == 0 or image_count > max_swapchain_images) return error.Unsupported;
    const composite_alpha = chooseCompositeAlpha(capabilities.supportedCompositeAlpha);
    if (composite_alpha == 0) {
        report(false, "The display surface exposes no supported composite alpha mode.", .{});
        return error.Unsupported;
    }

    const present_mode: c.VkPresentModeKHR = swapchain_present_mode;
    const present_modes_info: c.VkSwapchainPresentModesCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_KHR,
        .presentModeCount = 1,
        .pPresentModes = &present_mode,
    };
    const create_info: c.VkSwapchainCreateInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = &present_modes_info,
        .surface = device.surface,
        .minImageCount = image_count,
        .imageFormat = requested_format,
        .imageColorSpace = swapchain.color_space,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = device.sharingMode(),
        .queueFamilyIndexCount = device.queue_family_count,
        .pQueueFamilyIndices = &device.queue_families,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = composite_alpha,
        .presentMode = swapchain_present_mode,
        .clipped = c.VK_TRUE,
        .oldSwapchain = swapchain.handle,
    };
    // vkCreateSwapchainKHR retires oldSwapchain even when creation fails.
    errdefer retireSwapchainHandle(swapchain);
    var new_handle: c.VkSwapchainKHR = null;
    try check(c.vkCreateSwapchainKHR(device.device, &create_info, null, &new_handle), "vkCreateSwapchainKHR", 0);
    errdefer c.vkDestroySwapchainKHR(device.device, new_handle, null);

    var images: [max_swapchain_images]c.VkImage = undefined;
    var swapchain_image_count: u32 = 0;
    try check(c.vkGetSwapchainImagesKHR(device.device, new_handle, &swapchain_image_count, null), "vkGetSwapchainImagesKHR", 0);
    if (swapchain_image_count == 0 or swapchain_image_count > max_swapchain_images) return error.Unsupported;
    try check(c.vkGetSwapchainImagesKHR(device.device, new_handle, &swapchain_image_count, &images), "vkGetSwapchainImagesKHR", 0);

    var views: [max_swapchain_images]c.VkImageView = @splat(null);
    errdefer for (views[0..swapchain_image_count]) |view| if (view != null) native.vkDestroyImageView(device.device, view, null);
    for (images[0..swapchain_image_count], views[0..swapchain_image_count]) |image, *view| {
        const view_info: c.VkImageViewCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = requested_format,
            .subresourceRange = .{ .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
        };
        try check(native.vkCreateImageView(device.device, &view_info, null, view), "vkCreateImageView (swapchain)", 0);
    }

    retireSwapchainHandle(swapchain);
    swapchain.handle = new_handle;
    swapchain.image_count = swapchain_image_count;
    swapchain.width = extent.width;
    swapchain.height = extent.height;
    swapchain.transform = capabilities.currentTransform;
    swapchain.composite_alpha = composite_alpha;
    swapchain.recreate_required = false;
    for (0..swapchain_image_count) |index| {
        swapchain.images[index] = images[index];
        swapchain.render_views[index] = .{ .device = device, .view = views[index], .width = extent.width, .height = extent.height, .swapchain_view = true };
    }
}

pub fn setSwapchainFormat(device: *Device, format: Format, color_space: api.ColorSpace) Error!void {
    assert(device.acquired_swapchain == null);
    const swapchain = device.swapchain orelse return error.Unsupported;
    if (color_space != .srgb and !device.swapchain_color_space_enabled) return error.Unsupported;
    const requested_color_space = toVkColorSpace(color_space);
    if (swapchain.format == format and swapchain.color_space == requested_color_space) return;

    const old_format = swapchain.format;
    const old_color_space = swapchain.color_space;
    swapchain.format = format;
    swapchain.color_space = requested_color_space;
    recreateSwapchain(swapchain) catch |err| {
        swapchain.format = old_format;
        swapchain.color_space = old_color_space;
        if (swapchain.handle == null) requireOk(recreateSwapchain(swapchain), "Swapchain recreation");
        return err;
    };
}

pub fn getDrawableExtent(device: *Device) api.Uint32x2 {
    assert(device.acquired_swapchain == null);
    const swapchain = device.swapchain orelse return .{};
    const capabilities = surfaceCapabilities(device) catch return .{};
    const extent = capabilities.currentExtent;
    if (extent.width == std.math.maxInt(u32) or extent.height == std.math.maxInt(u32)) {
        report(false, "The display surface does not provide a fixed drawable extent.", .{});
        return .{};
    }
    if (swapchain.handle != null and (swapchain.width != extent.width or swapchain.height != extent.height)) swapchain.recreate_required = true;
    return .{ .x = extent.width, .y = extent.height };
}

pub fn acquire(commands: *CommandBuffer) ?api.SwapchainFrame {
    assert(commands.swapchain == null);
    const device = commands.device;
    const swapchain = device.swapchain.?;
    assert(!swapchain.acquired and device.acquired_swapchain == null);
    device.pollPresentationRetirement();

    var present_context: ?*PresentContext = null;
    while (true) {
        if (swapchain.handle == null or swapchain.recreate_required) {
            requireOk(recreateSwapchain(swapchain), "Swapchain recreation");
            if (swapchain.handle == null or swapchain.width == 0 or swapchain.height == 0) return null;
        }
        const context = present_context orelse context: {
            const next = &device.present_contexts[device.next_present_context];
            device.waitPresentContext(next);
            assert(!next.present_pending and next.swapchain == null);
            present_context = next;
            break :context next;
        };

        var image_index: u32 = 0;
        const result = c.vkAcquireNextImageKHR(device.device, swapchain.handle, std.math.maxInt(u64), context.acquired, null, &image_index);
        if (result == c.VK_ERROR_OUT_OF_DATE_KHR) {
            swapchain.recreate_required = true;
            continue;
        }
        if (result != c.VK_SUCCESS and result != c.VK_SUBOPTIMAL_KHR) abortVkFailure(result, "vkAcquireNextImageKHR");
        assert(image_index < swapchain.image_count);

        swapchain.image_index = image_index;
        swapchain.present_context = context;
        swapchain.acquired = true;
        swapchain.recreate_required = result == c.VK_SUBOPTIMAL_KHR and swapchainSurfaceConfigurationChanged(swapchain);
        device.acquired_swapchain = swapchain;
        const image_barrier: c.VkImageMemoryBarrier2 = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = c.VK_PIPELINE_STAGE_2_NONE,
            .dstStageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = c.VK_ACCESS_2_MEMORY_READ_BIT | c.VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = if (swapchain.initialized[image_index]) c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR else c.VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = c.VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain.images[image_index],
            .subresourceRange = .{ .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
        };
        recordImageBarrier(commands.command_buffer, &image_barrier);
        commands.swapchain = swapchain;
        swapchain.transition_commands = commands;
        return .{ .render_view = &swapchain.render_views[image_index], .extent = .{ .x = swapchain.width, .y = swapchain.height } };
    }
}

/// Image creation parameters; `format_list` is referenced from `image_info`, so the struct must not move after `prepare`.
const PreparedTexture = struct {
    view_formats: [Format.count]c.VkFormat = undefined,
    format_list: c.VkImageFormatListCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO },
    image_info: c.VkImageCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO },

    fn prepare(self: *PreparedTexture, device: *const Device, desc: api.TextureDesc) void {
        const format = toVkFormat(desc.format);
        const sampled_features = requiredFormatFeatures(.{ .sampled = true });
        const storage_features = requiredFormatFeatures(.{ .storage = true });
        var view_format_count: u32 = 1;
        self.view_formats[0] = format;
        if (desc.mutable_format) {
            for (device.format_features, 0..) |view_features, value| {
                const view_format: Format = @fromBackingInt(@intCast(value));
                if (view_format == desc.format or !compatibleViewFormats(desc.format, view_format)) continue;
                const sampled = desc.usage.sampled and view_features & sampled_features == sampled_features;
                const storage = desc.usage.storage and view_features & storage_features == storage_features;
                if (!sampled and !storage) continue;
                self.view_formats[view_format_count] = toVkFormat(view_format);
                view_format_count += 1;
            }
        }
        var flags: c.VkImageCreateFlags = 0;
        if (desc.aliasable) flags |= c.VK_IMAGE_CREATE_ALIAS_BIT;
        if (desc.type == .cube or desc.type == .cube_array) flags |= c.VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        if (view_format_count > 1) flags |= c.VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        self.format_list = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            .viewFormatCount = view_format_count,
            .pViewFormats = &self.view_formats,
        };
        self.image_info = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = if (view_format_count > 1) &self.format_list else null,
            .flags = flags,
            .imageType = toVkImageType(desc.type),
            .format = format,
            .extent = .{ .width = desc.extent.x, .height = desc.extent.y, .depth = desc.extent.z },
            .mipLevels = desc.mip_levels,
            .arrayLayers = desc.layer_count,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .tiling = c.VK_IMAGE_TILING_OPTIMAL,
            .usage = imageUsage(desc.usage),
            .sharingMode = device.sharingMode(),
            .queueFamilyIndexCount = device.queue_family_count,
            .pQueueFamilyIndices = &device.queue_families,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }
};

pub fn createTextureHeap(device: *Device, byte_count: u64) Error!api.TextureHeap {
    const allocate_info: c.VkMemoryAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = byte_count,
        .memoryTypeIndex = device.texture_memory_type,
    };
    var memory: c.VkDeviceMemory = null;
    try check(native.vkAllocateMemory(device.device, &allocate_info, null, &memory), "vkAllocateMemory (texture heap)", byte_count);
    return .{ .size = byte_count, .owner = create(TextureHeapOwner, .{ .device = device, .memory = memory }) };
}

pub fn destroyTextureHeap(heap: api.TextureHeap) void {
    native.vkFreeMemory(heap.owner.device.device, heap.owner.memory, null);
    gpa.destroy(heap.owner);
}

pub fn getTextureSizeAlign(device: *Device, desc: api.TextureDesc) Error!api.SizeAlign {
    var texture: PreparedTexture = .{};
    texture.prepare(device, desc);
    if (imageFormatProperties(device, &texture.image_info) == null) {
        report(false, "The GPU cannot create the requested texture format, usage, or extent.", .{});
        return error.Unsupported;
    }
    const requirements = imageMemoryRequirements(device, &texture.image_info);
    return .{ .size = requirements.size, .alignment = requirements.alignment };
}

pub fn createTexture(commands: *CommandBuffer, desc: api.TextureDesc, heap: api.TextureHeap, offset: u64) Error!*Texture {
    const device = commands.device;
    var prepared: PreparedTexture = .{};
    prepared.prepare(device, desc);
    var image: c.VkImage = null;
    try check(native.vkCreateImage(device.device, &prepared.image_info, null, &image), "vkCreateImage", 0);
    errdefer native.vkDestroyImage(device.device, image, null);
    try check(native.vkBindImageMemory(device.device, image, heap.owner.memory, offset), "vkBindImageMemory", 0);
    const texture = create(Texture, .{
        .device = device,
        .image = image,
        .width = desc.extent.x,
        .height = desc.extent.y,
        .depth = desc.extent.z,
        .layer_count = desc.layer_count,
        .type = desc.type,
        .format = desc.format,
    });
    if (desc.aliasable) return texture;

    const image_barrier: c.VkImageMemoryBarrier2 = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = c.VK_PIPELINE_STAGE_2_NONE,
        .dstStageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = c.VK_ACCESS_2_MEMORY_READ_BIT | c.VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = c.VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = .{ .aspectMask = imageAspects(desc.format), .levelCount = desc.mip_levels, .layerCount = desc.layer_count },
    };
    recordImageBarrier(commands.command_buffer, &image_barrier);
    return texture;
}

pub fn destroyTexture(texture: *Texture) void {
    native.vkDestroyImage(texture.device.device, texture.image, null);
    gpa.destroy(texture);
}

pub fn activateTextureAlias(commands: *CommandBuffer, texture: *Texture) void {
    assert(commands.device == texture.device);
    // The outgoing image can have a different format and size, so its writes need a global memory dependency.
    const memory_barrier: c.VkMemoryBarrier2 = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = c.VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = c.VK_ACCESS_2_MEMORY_READ_BIT | c.VK_ACCESS_2_MEMORY_WRITE_BIT,
    };
    recordMemoryBarrier(commands.command_buffer, &memory_barrier);
    const image_barrier: c.VkImageMemoryBarrier2 = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = c.VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = c.VK_ACCESS_2_MEMORY_READ_BIT | c.VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = c.VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
        .image = texture.image,
        .subresourceRange = .{
            .aspectMask = imageAspects(texture.format),
            .levelCount = c.VK_REMAINING_MIP_LEVELS,
            .layerCount = c.VK_REMAINING_ARRAY_LAYERS,
        },
    };
    recordImageBarrier(commands.command_buffer, &image_barrier);
}

pub fn createRenderView(texture: *Texture, desc: api.RenderViewDesc) Error!*RenderView {
    const view_info: c.VkImageViewCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = texture.image,
        .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
        .format = toVkFormat(texture.format),
        .subresourceRange = .{
            .aspectMask = imageAspects(texture.format),
            .baseMipLevel = desc.mip_level,
            .levelCount = 1,
            .baseArrayLayer = desc.slice,
            .layerCount = 1,
        },
    };
    var view: c.VkImageView = null;
    try check(native.vkCreateImageView(texture.device.device, &view_info, null, &view), "vkCreateImageView", 0);
    return create(RenderView, .{
        .device = texture.device,
        .view = view,
        .width = @max(texture.width >> @intCast(desc.mip_level), 1),
        .height = @max(texture.height >> @intCast(desc.mip_level), 1),
    });
}

pub fn destroyRenderView(render_view: *RenderView) void {
    assert(!render_view.swapchain_view);
    native.vkDestroyImageView(render_view.device.?.device, render_view.view, null);
    gpa.destroy(render_view);
}

pub fn createTextureDescriptorHeap(device: *Device, capacity: u32) Error!*TextureDescriptorHeap {
    assert(capacity != 0);
    const storage = try device.allocateDescriptorHeap(@as(u64, capacity) * device.heap_properties.imageDescriptorSize, true);
    return create(TextureDescriptorHeap, .{ .device = device, .storage = storage, .capacity = capacity });
}

pub fn destroyTextureDescriptorHeap(heap: *TextureDescriptorHeap) void {
    destroyGpuHeap(heap.storage);
    gpa.destroy(heap);
}

pub fn createSamplerDescriptorHeap(device: *Device, capacity: u32) Error!*SamplerDescriptorHeap {
    assert(capacity != 0);
    const storage = try device.allocateDescriptorHeap(@as(u64, capacity) * device.heap_properties.samplerDescriptorSize, false);
    return create(SamplerDescriptorHeap, .{ .device = device, .storage = storage, .capacity = capacity });
}

pub fn destroySamplerDescriptorHeap(heap: *SamplerDescriptorHeap) void {
    destroyGpuHeap(heap.storage);
    gpa.destroy(heap);
}

fn copyDescriptors(source: anytype, source_index: u32, destination: anytype, destination_index: u32, count: u32, stride: u64) void {
    assert(source.device == destination.device);
    assert(source_index <= source.capacity and count <= source.capacity - source_index);
    assert(destination_index <= destination.capacity and count <= destination.capacity - destination_index);
    const size = count * stride;
    const from = source.storage.range.cpu.? + source_index * stride;
    const to = destination.storage.range.cpu.? + destination_index * stride;
    @memmove(to[0..size], from[0..size]);
}

pub fn copyTextureDescriptors(
    source: *const TextureDescriptorHeap,
    source_index: u32,
    destination: *TextureDescriptorHeap,
    destination_index: u32,
    count: u32,
) void {
    copyDescriptors(source, source_index, destination, destination_index, count, source.device.heap_properties.imageDescriptorSize);
}

pub fn copySamplerDescriptors(
    source: *const SamplerDescriptorHeap,
    source_index: u32,
    destination: *SamplerDescriptorHeap,
    destination_index: u32,
    count: u32,
) void {
    copyDescriptors(source, source_index, destination, destination_index, count, source.device.heap_properties.samplerDescriptorSize);
}

pub fn writeTextureDescriptor(
    heap: *TextureDescriptorHeap,
    index: u32,
    texture: *const Texture,
    descriptor_type: api.TextureDescriptorType,
    desc: api.TextureDescriptorDesc,
) void {
    assert(index < heap.capacity and heap.device == texture.device);
    const device = heap.device;
    const info = api.getTextureFormatInfo(texture.format);
    const aspect: c.VkImageAspectFlags = switch (desc.aspect) {
        .automatic => if (info.depth) c.VK_IMAGE_ASPECT_DEPTH_BIT else if (info.stencil) c.VK_IMAGE_ASPECT_STENCIL_BIT else c.VK_IMAGE_ASPECT_COLOR_BIT,
        .color => c.VK_IMAGE_ASPECT_COLOR_BIT,
        .depth => c.VK_IMAGE_ASPECT_DEPTH_BIT,
        .stencil => c.VK_IMAGE_ASPECT_STENCIL_BIT,
    };
    const sampled = descriptor_type == .sampled;
    const view_usage: c.VkImageViewUsageCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .usage = if (sampled) c.VK_IMAGE_USAGE_SAMPLED_BIT else c.VK_IMAGE_USAGE_STORAGE_BIT,
    };
    const view_info: c.VkImageViewCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &view_usage,
        .image = texture.image,
        .viewType = toVkViewType(texture.type),
        .format = toVkFormat(if (desc.format == .undefined) texture.format else desc.format),
        .subresourceRange = .{
            .aspectMask = aspect,
            .baseMipLevel = desc.base_mip,
            .levelCount = if (desc.mip_count == 0) c.VK_REMAINING_MIP_LEVELS else desc.mip_count,
            .baseArrayLayer = desc.base_layer,
            .layerCount = if (desc.layer_count == 0) c.VK_REMAINING_ARRAY_LAYERS else desc.layer_count,
        },
    };
    const image_descriptor: c.VkImageDescriptorInfoEXT = .{
        .sType = c.VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
        .pView = &view_info,
        .layout = c.VK_IMAGE_LAYOUT_GENERAL,
    };
    const descriptor_info: c.VkResourceDescriptorInfoEXT = .{
        .sType = c.VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
        .type = if (sampled) c.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE else c.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .data = .{ .pImage = &image_descriptor },
    };
    const stride = device.heap_properties.imageDescriptorSize;
    const destination: c.VkHostAddressRangeEXT = .{ .address = heap.storage.range.cpu.? + index * stride, .size = stride };
    assert(device.fns.vkWriteResourceDescriptorsEXT.?(device.device, 1, &descriptor_info, &destination) == c.VK_SUCCESS);
}

pub fn writeSamplerDescriptor(heap: *SamplerDescriptorHeap, index: u32, desc: api.SamplerDesc) void {
    assert(index < heap.capacity);
    const device = heap.device;
    const sampler_info: c.VkSamplerCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = @backingInt(desc.mag_filter),
        .minFilter = @backingInt(desc.min_filter),
        .mipmapMode = @backingInt(desc.mip_filter),
        .addressModeU = @backingInt(desc.address_u),
        .addressModeV = @backingInt(desc.address_v),
        .addressModeW = @backingInt(desc.address_w),
        .anisotropyEnable = @intFromBool(desc.anisotropic),
        .maxAnisotropy = if (desc.anisotropic) 4.0 else 1.0,
        .compareEnable = @intFromBool(desc.compare_enabled),
        .compareOp = @backingInt(desc.compare),
        .maxLod = c.VK_LOD_CLAMP_NONE,
    };
    const stride = device.heap_properties.samplerDescriptorSize;
    const destination: c.VkHostAddressRangeEXT = .{ .address = heap.storage.range.cpu.? + index * stride, .size = stride };
    assert(device.fns.vkWriteSamplerDescriptorsEXT.?(device.device, 1, &sampler_info, &destination) == c.VK_SUCCESS);
}

fn assertSpirv(code: []const u8) void {
    assert(code.len >= 5 * @sizeOf(u32) and code.len % @sizeOf(u32) == 0 and @intFromPtr(code.ptr) % @alignOf(u32) == 0);
    assert(std.mem.readInt(u32, code[0..4], builtin.cpu.arch.endian()) == 0x07230203);
}

/// Maps every shader uniform and read-only storage buffer binding to the root pointer in push data.
const root_mapping: c.VkDescriptorSetAndBindingMappingEXT = .{
    .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT,
    .bindingCount = 1,
    .resourceMask = c.VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT | c.VK_SPIRV_RESOURCE_TYPE_READ_ONLY_STORAGE_BUFFER_BIT_EXT,
    .source = c.VK_DESCRIPTOR_MAPPING_SOURCE_PUSH_ADDRESS_EXT,
    .sourceData = .{ .pushAddressOffset = 0 },
};

fn shaderModuleInfo(stage: api.ShaderStage) c.VkShaderModuleCreateInfo {
    // Unused stages have an empty, unaligned default slice.
    const code: ?[*]const u32 = if (stage.code.len == 0) null else @ptrCast(@alignCast(stage.code.ptr));
    return .{ .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = stage.code.len, .pCode = code };
}

fn rootMappingInfo(module_info: *const c.VkShaderModuleCreateInfo) c.VkShaderDescriptorSetAndBindingMappingInfoEXT {
    return .{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT,
        .pNext = module_info,
        .mappingCount = 1,
        .pMappings = &root_mapping,
    };
}

fn createRasterPso(
    device: *Device,
    first_stage: api.ShaderStage,
    fragment: api.ShaderStage,
    color_targets: []const api.ColorTargetDesc,
    depth_format: Format,
    stencil_format: Format,
    rasterization_state: api.RasterizationState,
    mesh: bool,
    task: api.ShaderStage,
) Error!*Pso {
    assertSpirv(first_stage.code);
    if (task.code.len != 0) assertSpirv(task.code);
    if (fragment.code.len != 0) assertSpirv(fragment.code);
    assert(color_targets.len <= max_color_attachments);

    const module_infos = [3]c.VkShaderModuleCreateInfo{ shaderModuleInfo(task), shaderModuleInfo(first_stage), shaderModuleInfo(fragment) };
    const mappings = [3]c.VkShaderDescriptorSetAndBindingMappingInfoEXT{
        rootMappingInfo(&module_infos[0]),
        rootMappingInfo(&module_infos[1]),
        rootMappingInfo(&module_infos[2]),
    };
    const stages = [3]c.VkPipelineShaderStageCreateInfo{
        .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &mappings[0],
            .stage = c.VK_SHADER_STAGE_TASK_BIT_EXT,
            .pName = task.entry_point,
        },
        .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &mappings[1],
            .stage = if (mesh) c.VK_SHADER_STAGE_MESH_BIT_EXT else c.VK_SHADER_STAGE_VERTEX_BIT,
            .pName = first_stage.entry_point,
        },
        .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &mappings[2],
            .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
            .pName = fragment.entry_point,
        },
    };
    const vertex_input: c.VkPipelineVertexInputStateCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    const input_assembly: c.VkPipelineInputAssemblyStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const viewport_state: c.VkPipelineViewportStateCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    const rasterization: c.VkPipelineRasterizationStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = c.VK_POLYGON_MODE_FILL,
        .cullMode = if (rasterization_state.cull == .none) c.VK_CULL_MODE_NONE else c.VK_CULL_MODE_BACK_BIT,
        .frontFace = if (rasterization_state.cull == .counter_clockwise) c.VK_FRONT_FACE_CLOCKWISE else c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = @intFromBool(rasterization_state.depth_bias_constant != 0 or rasterization_state.depth_bias_clamp != 0 or
            rasterization_state.depth_bias_slope != 0),
        .depthBiasConstantFactor = rasterization_state.depth_bias_constant,
        .depthBiasClamp = rasterization_state.depth_bias_clamp,
        .depthBiasSlopeFactor = rasterization_state.depth_bias_slope,
        .lineWidth = 1.0,
    };
    const multisample: c.VkPipelineMultisampleStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
    };
    const depth_stencil: c.VkPipelineDepthStencilStateCreateInfo = .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    var color_attachments: [max_color_attachments]c.VkPipelineColorBlendAttachmentState = undefined;
    var color_formats: [max_color_attachments]c.VkFormat = undefined;
    for (color_targets, color_attachments[0..color_targets.len], color_formats[0..color_targets.len]) |target, *attachment, *format| {
        attachment.* = .{
            .blendEnable = @intFromBool(target.blend.enabled),
            .srcColorBlendFactor = toVkBlendFactor(target.blend.color.source),
            .dstColorBlendFactor = toVkBlendFactor(target.blend.color.destination),
            .colorBlendOp = @backingInt(target.blend.color.operation),
            .srcAlphaBlendFactor = toVkBlendFactor(target.blend.alpha.source),
            .dstAlphaBlendFactor = toVkBlendFactor(target.blend.alpha.destination),
            .alphaBlendOp = @backingInt(target.blend.alpha.operation),
            .colorWriteMask = target.write_mask,
        };
        format.* = toVkFormat(target.format);
    }
    const color_count: u32 = @intCast(color_targets.len);
    const color_blend: c.VkPipelineColorBlendStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = color_count,
        .pAttachments = if (color_count != 0) &color_attachments else null,
    };
    const dynamic_states = [_]c.VkDynamicState{
        c.VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
        c.VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
        c.VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        c.VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        c.VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        c.VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        c.VK_DYNAMIC_STATE_STENCIL_OP,
        c.VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        c.VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        c.VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    };
    const dynamic_state: c.VkPipelineDynamicStateCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = dynamic_states.len,
        .pDynamicStates = &dynamic_states,
    };
    const rendering_info: c.VkPipelineRenderingCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = color_count,
        .pColorAttachmentFormats = if (color_count != 0) &color_formats else null,
        .depthAttachmentFormat = toVkFormat(depth_format),
        .stencilAttachmentFormat = toVkFormat(stencil_format),
    };
    const flags_info: c.VkPipelineCreateFlags2CreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .pNext = &rendering_info,
        .flags = c.VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
    };
    const has_task = task.code.len != 0;
    const pso_info: c.VkGraphicsPipelineCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &flags_info,
        .stageCount = 1 + @as(u32, @intFromBool(has_task)) + @intFromBool(fragment.code.len != 0),
        .pStages = if (has_task) &stages else stages[1..].ptr,
        .pVertexInputState = if (mesh) null else &vertex_input,
        .pInputAssemblyState = if (mesh) null else &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_state,
        .basePipelineIndex = -1,
    };
    var pipeline: c.VkPipeline = null;
    try check(native.vkCreateGraphicsPipelines(device.device, null, 1, &pso_info, null, &pipeline), "vkCreateGraphicsPipelines", 0);
    return create(Pso, .{ .device = device, .pso = pipeline, .bind_point = c.VK_PIPELINE_BIND_POINT_GRAPHICS });
}

pub fn createGraphicsPso(device: *Device, desc: api.GraphicsPsoDesc) Error!*Pso {
    return createRasterPso(device, desc.vertex, desc.fragment, desc.color_targets, desc.depth_format, desc.stencil_format, desc.rasterization, false, .{});
}

pub fn createMeshPso(device: *Device, desc: api.MeshPsoDesc) Error!*Pso {
    return createRasterPso(device, desc.mesh, desc.fragment, desc.color_targets, desc.depth_format, desc.stencil_format, desc.rasterization, true, desc.task);
}

pub fn createComputePso(device: *Device, compute: api.ShaderStage) Error!*Pso {
    assertSpirv(compute.code);
    const module_info = shaderModuleInfo(compute);
    const mapping = rootMappingInfo(&module_info);
    const flags_info: c.VkPipelineCreateFlags2CreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
        .flags = c.VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT,
    };
    const pso_info: c.VkComputePipelineCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = &flags_info,
        .stage = .{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &mapping,
            .stage = c.VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = compute.entry_point,
        },
        .basePipelineIndex = -1,
    };
    var pipeline: c.VkPipeline = null;
    try check(native.vkCreateComputePipelines(device.device, null, 1, &pso_info, null, &pipeline), "vkCreateComputePipelines", 0);
    return create(Pso, .{ .device = device, .pso = pipeline, .bind_point = c.VK_PIPELINE_BIND_POINT_COMPUTE });
}

pub fn destroyPso(pso: *Pso) void {
    native.vkDestroyPipeline(pso.device.device, pso.pso, null);
    gpa.destroy(pso);
}

pub fn createCommandPool(device: *Device, queue_index: u32) Error!*CommandPool {
    assert(queue_index < device.queues.len);
    const pool_info: c.VkCommandPoolCreateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = c.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = device.queues[queue_index].family_index,
    };
    var command_pool: c.VkCommandPool = null;
    try check(native.vkCreateCommandPool(device.device, &pool_info, null, &command_pool), "vkCreateCommandPool", 0);
    return create(CommandPool, .{ .device = device, .command_pool = command_pool, .timestamps = device.queues[queue_index].timestamps });
}

pub fn destroyCommandPool(pool: *CommandPool) void {
    while (pool.first) |commands| {
        assert(commands.swapchain == null);
        pool.first = commands.next;
        if (commands.timestamp_pool != null) native.vkDestroyQueryPool(pool.device.device, commands.timestamp_pool, null);
        gpa.free(commands.timestamp_destinations);
        gpa.free(commands.timestamp_results);
        gpa.destroy(commands);
    }
    native.vkDestroyCommandPool(pool.device.device, pool.command_pool, null);
    gpa.destroy(pool);
}

pub fn resetCommandPool(pool: *CommandPool) void {
    var next = pool.first;
    while (next) |commands| : (next = commands.next) {
        assert(commands.swapchain == null);
        commands.submitted = false;
        commands.timestamp_count = 0;
    }
    assert(c.vkResetCommandPool(pool.device.device, pool.command_pool, 0) == c.VK_SUCCESS);
    pool.next_buffer = pool.first;
}

fn allocateCommandBuffer(device: *const Device, pool: c.VkCommandPool, output: *c.VkCommandBuffer) Error!void {
    const allocate_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    try check(native.vkAllocateCommandBuffers(device.device, &allocate_info, output), "vkAllocateCommandBuffers", 0);
}

fn newCommandBuffer(pool: *CommandPool) Error!*CommandBuffer {
    const device = pool.device;
    var command_buffer: c.VkCommandBuffer = null;
    try allocateCommandBuffer(device, pool.command_pool, &command_buffer);
    errdefer native.vkFreeCommandBuffers(device.device, pool.command_pool, 1, &command_buffer);
    var timestamp_pool: c.VkQueryPool = null;
    if (pool.timestamps) {
        const query_info: c.VkQueryPoolCreateInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = c.VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = device.timestamp_query_count,
        };
        try check(native.vkCreateQueryPool(device.device, &query_info, null, &timestamp_pool), "vkCreateQueryPool", 0);
    }
    const commands = create(
        CommandBuffer,
        .{ .device = device, .command_pool = pool.command_pool, .command_buffer = command_buffer, .timestamp_pool = timestamp_pool },
    );
    if (pool.timestamps) {
        commands.timestamp_destinations = gpa.alloc(*u64, device.timestamp_query_count) catch @panic("out of memory");
        commands.timestamp_results = gpa.alloc(u64, device.timestamp_query_count) catch @panic("out of memory");
    }
    if (pool.last) |last| last.next = commands else pool.first = commands;
    pool.last = commands;
    return commands;
}

pub fn beginCommands(pool: *CommandPool) Error!*CommandBuffer {
    const commands = if (pool.next_buffer) |reused| reused: {
        pool.next_buffer = reused.next;
        break :reused reused;
    } else try newCommandBuffer(pool);
    assert(commands.swapchain == null);
    const begin_info: c.VkCommandBufferBeginInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    assert(c.vkBeginCommandBuffer(commands.command_buffer, &begin_info) == c.VK_SUCCESS);
    commands.timestamp_count = 0;
    commands.submitted = false;
    commands.has_epilogue = false;
    if (commands.timestamp_pool != null) c.vkResetQueryPool(pool.device.device, commands.timestamp_pool, 0, pool.device.timestamp_query_count);
    return commands;
}

pub fn endCommands(commands: *CommandBuffer) void {
    var command_buffer = commands.command_buffer;
    if (commands.swapchain) |swapchain| {
        // Presentation transitions go into an epilogue buffer that submits after every suspended/resumed segment.
        assert(c.vkEndCommandBuffer(command_buffer) == c.VK_SUCCESS);
        if (commands.epilogue == null) requireOk(allocateCommandBuffer(commands.device, commands.command_pool, &commands.epilogue), "Epilogue allocation");
        const begin_info: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        command_buffer = commands.epilogue;
        assert(c.vkBeginCommandBuffer(command_buffer, &begin_info) == c.VK_SUCCESS);
        commands.has_epilogue = true;
        assert(swapchain.acquired and swapchain.transition_commands == commands);
        const image_barrier: c.VkImageMemoryBarrier2 = .{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = c.VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = c.VK_PIPELINE_STAGE_2_NONE,
            .oldLayout = c.VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain.images[swapchain.image_index],
            .subresourceRange = .{ .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
        };
        recordImageBarrier(command_buffer, &image_barrier);
    }
    assert(c.vkEndCommandBuffer(command_buffer) == c.VK_SUCCESS);
}

/// Grows per-queue submission scratch storage; it is reused across submissions and never shrinks.
fn reserve(comptime T: type, storage: *[]T, count: usize) void {
    if (count <= storage.len) return;
    storage.* = gpa.realloc(storage.*, @max(count, storage.len * 2, 4)) catch @panic("out of memory");
}

fn submitCommands(device: *Device, desc: api.SubmitDesc, queue_index: u32, wait_semaphore: c.VkSemaphore, signal_semaphore: c.VkSemaphore) void {
    const queue = &device.queues[queue_index];
    var command_count = desc.commands.len;
    for (desc.commands) |commands| command_count += @intFromBool(commands.has_epilogue);
    reserve(c.VkCommandBufferSubmitInfo, &queue.command_submit_infos, command_count);
    for (desc.commands, queue.command_submit_infos[0..desc.commands.len]) |commands, *info|
        info.* = .{ .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = commands.command_buffer, .deviceMask = 1 };
    // Presentation transitions must follow every suspended/resumed segment.
    var epilogue_index = desc.commands.len;
    for (desc.commands) |commands| {
        if (!commands.has_epilogue) continue;
        queue.command_submit_infos[epilogue_index] = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = commands.epilogue,
            .deviceMask = 1,
        };
        epilogue_index += 1;
    }

    const wait_count = desc.waits.len + @intFromBool(wait_semaphore != null);
    reserve(c.VkSemaphoreSubmitInfo, &queue.wait_submit_infos, wait_count);
    for (desc.waits, queue.wait_submit_infos[0..desc.waits.len]) |point, *info| {
        info.* = .{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = point.semaphore.semaphore,
            .value = point.value,
            .stageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
    }
    if (wait_semaphore != null) queue.wait_submit_infos[desc.waits.len] = .{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = wait_semaphore,
        .stageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };

    var signal_infos: [3]c.VkSemaphoreSubmitInfo = undefined;
    signal_infos[0] = .{
        .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = desc.completion.semaphore.semaphore,
        .value = desc.completion.value,
        .stageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    var signal_count: u32 = 1;
    if (signal_semaphore != null) {
        signal_infos[1] = .{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = signal_semaphore,
            .stageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        signal_infos[2] = .{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = device.presentation_retirement,
            .value = device.nextPresentationRetirement(),
            .stageMask = c.VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        signal_count = 3;
    }
    const submit_info: c.VkSubmitInfo2 = .{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = @intCast(wait_count),
        .pWaitSemaphoreInfos = queue.wait_submit_infos.ptr,
        .commandBufferInfoCount = @intCast(command_count),
        .pCommandBufferInfos = queue.command_submit_infos.ptr,
        .signalSemaphoreInfoCount = signal_count,
        .pSignalSemaphoreInfos = &signal_infos,
    };
    assert(c.vkQueueSubmit2(queue.queue, 1, &submit_info, null) == c.VK_SUCCESS);
    for (desc.commands) |commands| commands.submitted = true;
}

pub fn submit(device: *Device, desc: api.SubmitDesc, queue_index: u32) void {
    assert(queue_index < device.queues.len);
    for (desc.commands) |commands| assert(commands.swapchain == null);
    submitCommands(device, desc, queue_index, null, null);
}

pub fn submitAndPresent(device: *Device, desc: api.SubmitDesc) void {
    const swapchain = device.swapchain.?;
    assert(swapchain.acquired and swapchain.transition_commands != null and swapchain.present_context != null);
    var contains_swapchain = false;
    for (desc.commands) |commands| {
        assert(commands.swapchain == null or commands == swapchain.transition_commands);
        if (commands == swapchain.transition_commands) contains_swapchain = true;
    }
    assert(contains_swapchain);
    const present_context = swapchain.present_context.?;
    assert(!present_context.present_pending and present_context.swapchain == null);
    assert(c.vkResetFences(device.device, 1, &present_context.presented) == c.VK_SUCCESS);
    submitCommands(device, desc, 0, present_context.acquired, present_context.rendered);
    swapchain.transition_commands.?.swapchain = null;
    swapchain.transition_commands = null;
    swapchain.initialized[swapchain.image_index] = true;

    const fence_info: c.VkSwapchainPresentFenceInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_KHR,
        .swapchainCount = 1,
        .pFences = &present_context.presented,
    };
    const present_info: c.VkPresentInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = &fence_info,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &present_context.rendered,
        .swapchainCount = 1,
        .pSwapchains = &swapchain.handle,
        .pImageIndices = &swapchain.image_index,
    };
    const result = c.vkQueuePresentKHR(device.queues[0].queue, &present_info);
    present_context.present_pending = true;
    present_context.swapchain = swapchain.handle;
    swapchain.present_context = null;
    swapchain.acquired = false;
    device.acquired_swapchain = null;
    device.next_present_context = (device.next_present_context + 1) % device.present_context_count;
    if (result == c.VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain.recreate_required = true;
    } else if (result == c.VK_SUBOPTIMAL_KHR) {
        swapchain.recreate_required = swapchain.recreate_required or swapchainSurfaceConfigurationChanged(swapchain);
    } else if (result != c.VK_SUCCESS) {
        abortVkFailure(result, "vkQueuePresentKHR");
    }
}

pub fn bindPso(commands: *CommandBuffer, pso: *const Pso) void {
    c.vkCmdBindPipeline(commands.command_buffer, pso.bind_point, pso.pso);
}

fn textureCopyRegion(texture: *const Texture, copy: api.TextureCopyDesc, memory: api.GpuRange) c.VkDeviceMemoryImageCopyKHR {
    const mip: u5 = @intCast(copy.mip_level);
    const mip_width = @max(texture.width >> mip, 1);
    const mip_height = @max(texture.height >> mip, 1);
    const mip_depth = @max(texture.depth >> mip, 1);
    const width = if (copy.extent.x == 0) mip_width - copy.offset.x else copy.extent.x;
    const info = api.getTextureFormatInfo(texture.format);
    const blocks_per_row = (@as(u64, width) + info.block_extent.x - 1) / info.block_extent.x;
    const row_pitch = if (copy.row_pitch_bytes == 0) blocks_per_row * info.bytes_per_block else copy.row_pitch_bytes;
    assert(copy.slice_pitch_bytes == 0 or row_pitch != 0);
    return .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
        .addressRange = .{ .address = address(memory.gpu), .size = memory.size },
        .addressFlags = address_flags,
        .addressRowLength = @intCast(copy.row_pitch_bytes / info.bytes_per_block * info.block_extent.x),
        .addressImageHeight = if (copy.slice_pitch_bytes == 0) 0 else @intCast(copy.slice_pitch_bytes / row_pitch * info.block_extent.y),
        .imageSubresource = .{
            .aspectMask = imageAspects(texture.format),
            .mipLevel = copy.mip_level,
            .baseArrayLayer = copy.base_slice,
            .layerCount = if (copy.slice_count == 0) texture.layer_count - copy.base_slice else copy.slice_count,
        },
        .imageLayout = c.VK_IMAGE_LAYOUT_GENERAL,
        .imageOffset = .{ .x = @intCast(copy.offset.x), .y = @intCast(copy.offset.y), .z = @intCast(copy.offset.z) },
        .imageExtent = .{
            .width = width,
            .height = if (copy.extent.y == 0) mip_height - copy.offset.y else copy.extent.y,
            .depth = if (copy.extent.z == 0) mip_depth - copy.offset.z else copy.extent.z,
        },
    };
}

fn emitRootPointer(commands: *CommandBuffer, root: ?*const anyopaque) void {
    if (root == null) return;
    assert(@intFromPtr(root) % 16 == 0);
    const info: c.VkPushDataInfoEXT = .{
        .sType = c.VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
        .data = .{ .address = @ptrCast(&root), .size = @sizeOf(?*const anyopaque) },
    };
    commands.device.fns.vkCmdPushDataEXT.?(commands.command_buffer, &info);
}

pub fn setTextureDescriptorHeap(commands: *CommandBuffer, heap: *TextureDescriptorHeap) void {
    assert(commands.device == heap.device);
    const properties = &commands.device.heap_properties;
    const bind_info = heapBindInfo(
        heap.storage.gpuRange(),
        @max(properties.imageDescriptorAlignment, properties.bufferDescriptorAlignment),
        properties.minResourceHeapReservedRange,
    );
    commands.device.fns.vkCmdBindResourceHeapEXT.?(commands.command_buffer, &bind_info);
}

pub fn setSamplerDescriptorHeap(commands: *CommandBuffer, heap: *SamplerDescriptorHeap) void {
    assert(commands.device == heap.device);
    const properties = &commands.device.heap_properties;
    const bind_info = heapBindInfo(heap.storage.gpuRange(), properties.samplerDescriptorAlignment, properties.minSamplerHeapReservedRange);
    commands.device.fns.vkCmdBindSamplerHeapEXT.?(commands.command_buffer, &bind_info);
}

pub fn setViewport(commands: *CommandBuffer, viewport: api.Viewport) void {
    const vk_viewport: c.VkViewport = .{
        .x = viewport.x,
        .y = viewport.y,
        .width = viewport.width,
        .height = viewport.height,
        .minDepth = viewport.min_depth,
        .maxDepth = viewport.max_depth,
    };
    c.vkCmdSetViewportWithCount(commands.command_buffer, 1, &vk_viewport);
}

pub fn setScissor(commands: *CommandBuffer, scissor: api.Scissor) void {
    const rect: c.VkRect2D = .{ .offset = .{ .x = scissor.x, .y = scissor.y }, .extent = .{ .width = scissor.width, .height = scissor.height } };
    c.vkCmdSetScissorWithCount(commands.command_buffer, 1, &rect);
}

pub fn setDepthStencil(commands: *CommandBuffer, state: api.DepthStencilState) void {
    const command_buffer = commands.command_buffer;
    c.vkCmdSetDepthTestEnable(command_buffer, @intFromBool(state.depth_test));
    if (state.depth_test) {
        c.vkCmdSetDepthWriteEnable(command_buffer, @intFromBool(state.depth_write));
        c.vkCmdSetDepthCompareOp(command_buffer, @backingInt(state.depth_compare));
    }
    c.vkCmdSetStencilTestEnable(command_buffer, @intFromBool(state.stencil_test));
    if (!state.stencil_test) return;
    const faces = [_]struct { c.VkStencilFaceFlags, api.StencilFaceState }{
        .{ c.VK_STENCIL_FACE_FRONT_BIT, state.front },
        .{ c.VK_STENCIL_FACE_BACK_BIT, state.back },
    };
    for (faces) |face| {
        c.vkCmdSetStencilOp(
            command_buffer,
            face[0],
            @backingInt(face[1].fail),
            @backingInt(face[1].pass),
            @backingInt(face[1].depth_fail),
            @backingInt(face[1].compare),
        );
        c.vkCmdSetStencilReference(command_buffer, face[0], face[1].reference);
    }
    c.vkCmdSetStencilCompareMask(command_buffer, c.VK_STENCIL_FACE_FRONT_AND_BACK, state.stencil_read_mask);
    c.vkCmdSetStencilWriteMask(command_buffer, c.VK_STENCIL_FACE_FRONT_AND_BACK, state.stencil_write_mask);
}

fn attachmentInfo(render_view: ?*const RenderView, load: api.LoadOp, store: api.StoreOp, clear: c.VkClearValue) c.VkRenderingAttachmentInfo {
    return .{
        .sType = c.VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = if (render_view) |view| view.view else null,
        .imageLayout = c.VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = @backingInt(load),
        .storeOp = @backingInt(store),
        .clearValue = clear,
    };
}

pub fn beginRenderPass(commands: *CommandBuffer, desc: api.RenderingDesc, flags: api.RenderingFlags) void {
    assert(desc.colors.len <= max_color_attachments);
    const area_view: *const RenderView = if (desc.colors.len != 0) desc.colors[0].render_view else desc.depth.render_view orelse desc.stencil.render_view.?;
    var color_attachments: [max_color_attachments]c.VkRenderingAttachmentInfo = undefined;
    for (desc.colors, color_attachments[0..desc.colors.len]) |attachment, *info| {
        const clear = attachment.clear;
        info.* = attachmentInfo(
            attachment.render_view,
            attachment.load,
            attachment.store,
            .{ .color = .{ .float32 = .{ clear.x, clear.y, clear.z, clear.w } } },
        );
    }
    const depth = attachmentInfo(desc.depth.render_view, desc.depth.load, desc.depth.store, .{ .depthStencil = .{ .depth = desc.depth.clear } });
    const stencil = attachmentInfo(desc.stencil.render_view, desc.stencil.load, desc.stencil.store, .{ .depthStencil = .{ .stencil = desc.stencil.clear } });
    const rendering_info: c.VkRenderingInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_RENDERING_INFO,
        .flags = @bitCast(flags),
        .renderArea = .{ .extent = .{ .width = area_view.width, .height = area_view.height } },
        .layerCount = 1,
        .colorAttachmentCount = @intCast(desc.colors.len),
        .pColorAttachments = &color_attachments,
        .pDepthAttachment = if (desc.depth.render_view != null) &depth else null,
        .pStencilAttachment = if (desc.stencil.render_view != null) &stencil else null,
    };
    c.vkCmdBeginRendering(commands.command_buffer, &rendering_info);
    setViewport(commands, .{ .width = @floatFromInt(area_view.width), .height = @floatFromInt(area_view.height) });
    setScissor(commands, .{ .width = area_view.width, .height = area_view.height });
    setDepthStencil(commands, .{});
}

pub fn endRenderPass(commands: *CommandBuffer) void {
    c.vkCmdEndRendering(commands.command_buffer);
}

pub fn draw(commands: *CommandBuffer, root: ?*const anyopaque, args: api.Draw) void {
    emitRootPointer(commands, root);
    c.vkCmdDraw(commands.command_buffer, args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
}

fn bindIndices(commands: *CommandBuffer, indices: api.GpuRange, index_type: api.IndexType) void {
    const bind_info: c.VkBindIndexBuffer3InfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_BIND_INDEX_BUFFER_3_INFO_KHR,
        .addressRange = .{ .address = address(indices.gpu), .size = indices.size },
        .addressFlags = address_flags,
        .indexType = @backingInt(index_type),
    };
    commands.device.fns.vkCmdBindIndexBuffer3KHR.?(commands.command_buffer, &bind_info);
}

fn indirectInfo(arguments: api.GpuRange, draw_count: u32, stride: u32, comptime Command: type) c.VkDrawIndirect2InfoKHR {
    return .{
        .sType = c.VK_STRUCTURE_TYPE_DRAW_INDIRECT_2_INFO_KHR,
        .addressRange = .{ .address = address(arguments.gpu), .size = arguments.size, .stride = if (stride == 0) @sizeOf(Command) else stride },
        .addressFlags = address_flags,
        .drawCount = draw_count,
    };
}

pub fn drawIndexed(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndexed) void {
    emitRootPointer(commands, root);
    bindIndices(commands, args.indices, args.index_type);
    c.vkCmdDrawIndexed(commands.command_buffer, args.index_count, args.instance_count, args.first_index, args.vertex_offset, args.first_instance);
}

pub fn drawIndirect(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndirect) void {
    emitRootPointer(commands, root);
    const info = indirectInfo(args.arguments, args.draw_count, args.stride, c.VkDrawIndirectCommand);
    commands.device.fns.vkCmdDrawIndirect2KHR.?(commands.command_buffer, &info);
}

pub fn drawIndexedIndirect(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndexedIndirect) void {
    emitRootPointer(commands, root);
    bindIndices(commands, args.indices, args.index_type);
    const info = indirectInfo(args.arguments, args.draw_count, args.stride, c.VkDrawIndexedIndirectCommand);
    commands.device.fns.vkCmdDrawIndexedIndirect2KHR.?(commands.command_buffer, &info);
}

pub fn dispatch(commands: *CommandBuffer, root: ?*const anyopaque, group_count: api.Uint32x3) void {
    emitRootPointer(commands, root);
    c.vkCmdDispatch(commands.command_buffer, group_count.x, group_count.y, group_count.z);
}

pub fn dispatchIndirect(commands: *CommandBuffer, root: ?*const anyopaque, arguments: api.GpuRange) void {
    emitRootPointer(commands, root);
    const info: c.VkDispatchIndirect2InfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_DISPATCH_INDIRECT_2_INFO_KHR,
        .addressRange = .{ .address = address(arguments.gpu), .size = arguments.size },
        .addressFlags = address_flags,
    };
    commands.device.fns.vkCmdDispatchIndirect2KHR.?(commands.command_buffer, &info);
}

pub fn drawMeshlets(commands: *CommandBuffer, root: ?*const anyopaque, group_count: api.Uint32x3) void {
    emitRootPointer(commands, root);
    commands.device.fns.vkCmdDrawMeshTasksEXT.?(commands.command_buffer, group_count.x, group_count.y, group_count.z);
}

pub fn drawMeshletsIndirect(commands: *CommandBuffer, root: ?*const anyopaque, args: api.DrawIndirect) void {
    assert(commands.device.caps.indirect_mesh_draw);
    emitRootPointer(commands, root);
    const info = indirectInfo(args.arguments, args.draw_count, args.stride, c.VkDrawMeshTasksIndirectCommandEXT);
    commands.device.fns.vkCmdDrawMeshTasksIndirect2EXT.?(commands.command_buffer, &info);
}

pub fn copyMemory(commands: *CommandBuffer, source: api.GpuRange, destination: api.GpuRange) void {
    const region: c.VkDeviceMemoryCopyKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_MEMORY_COPY_KHR,
        .srcRange = .{ .address = address(source.gpu), .size = source.size },
        .srcFlags = address_flags,
        .dstRange = .{ .address = address(destination.gpu), .size = destination.size },
        .dstFlags = address_flags,
    };
    const info: c.VkCopyDeviceMemoryInfoKHR = .{ .sType = c.VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_INFO_KHR, .regionCount = 1, .pRegions = &region };
    commands.device.fns.vkCmdCopyMemoryKHR.?(commands.command_buffer, &info);
}

pub fn copyMemoryToTexture(commands: *CommandBuffer, source: api.GpuRange, destination: *Texture, copy: api.TextureCopyDesc) void {
    const region = textureCopyRegion(destination, copy, source);
    const info: c.VkCopyDeviceMemoryImageInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_IMAGE_INFO_KHR,
        .image = destination.image,
        .regionCount = 1,
        .pRegions = &region,
    };
    commands.device.fns.vkCmdCopyMemoryToImageKHR.?(commands.command_buffer, &info);
}

pub fn copyTextureToMemory(commands: *CommandBuffer, source: *Texture, destination: api.GpuRange, copy: api.TextureCopyDesc) void {
    const region = textureCopyRegion(source, copy, destination);
    const info: c.VkCopyDeviceMemoryImageInfoKHR = .{
        .sType = c.VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_IMAGE_INFO_KHR,
        .image = source.image,
        .regionCount = 1,
        .pRegions = &region,
    };
    commands.device.fns.vkCmdCopyImageToMemoryKHR.?(commands.command_buffer, &info);
}

pub fn barrier(commands: *CommandBuffer, before: api.Stage, before_access: api.Access, after: api.Stage, after_access: api.Access) void {
    const memory_barrier: c.VkMemoryBarrier2 = .{
        .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = toVkStages(before),
        .srcAccessMask = toVkAccess(before_access),
        .dstStageMask = toVkStages(after),
        .dstAccessMask = toVkAccess(after_access),
    };
    recordMemoryBarrier(commands.command_buffer, &memory_barrier);
}
