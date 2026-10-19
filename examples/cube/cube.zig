// The texture map is adapted from Vulkan-Tools' vkcube sample. Copyright
// notices and the Apache-2.0 license are in NOTICE.md and LICENSE-Apache-2.0.txt.

const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const utility = @import("NoGraphicsAPIUtility");
const comath = @import("comath");
const support = @import("example_support");
const shared = @import("cube_shared");
const math = utility.math;

const width = 500;
const height = 500;
const texture_width = 256;
const texture_height = 256;
const texture_byte_count = texture_width * texture_height * 4;
const data_heap_size = 1024 * 1024;
const texture_heap_size = 256 * 1024 * 1024;
const radians_per_frame = 4.0 * math.pi / 180.0;

const cube_vertices = [_]shared.CubeVertex{
    .{ .position = .{ .x = -1, .y = -1, .z = -1, .w = 1 }, .uv = .{ .x = 0, .y = 1 } },
    .{ .position = .{ .x = -1, .y = -1, .z = 1, .w = 1 }, .uv = .{ .x = 1, .y = 1 } },
    .{ .position = .{ .x = -1, .y = 1, .z = 1, .w = 1 }, .uv = .{ .x = 1, .y = 0 } },
    .{ .position = .{ .x = -1, .y = 1, .z = -1, .w = 1 }, .uv = .{ .x = 0, .y = 0 } },

    .{ .position = .{ .x = -1, .y = -1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 1 } },
    .{ .position = .{ .x = -1, .y = 1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 0 } },
    .{ .position = .{ .x = 1, .y = 1, .z = -1, .w = 1 }, .uv = .{ .x = 0, .y = 0 } },
    .{ .position = .{ .x = 1, .y = -1, .z = -1, .w = 1 }, .uv = .{ .x = 0, .y = 1 } },

    .{ .position = .{ .x = -1, .y = -1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 0 } },
    .{ .position = .{ .x = 1, .y = -1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 1 } },
    .{ .position = .{ .x = 1, .y = -1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 1 } },
    .{ .position = .{ .x = -1, .y = -1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 0 } },

    .{ .position = .{ .x = -1, .y = 1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 0 } },
    .{ .position = .{ .x = -1, .y = 1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 0 } },
    .{ .position = .{ .x = 1, .y = 1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 1 } },
    .{ .position = .{ .x = 1, .y = 1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 1 } },

    .{ .position = .{ .x = 1, .y = 1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 0 } },
    .{ .position = .{ .x = 1, .y = 1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 0 } },
    .{ .position = .{ .x = 1, .y = -1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 1 } },
    .{ .position = .{ .x = 1, .y = -1, .z = -1, .w = 1 }, .uv = .{ .x = 1, .y = 1 } },

    .{ .position = .{ .x = -1, .y = -1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 1 } },
    .{ .position = .{ .x = 1, .y = -1, .z = 1, .w = 1 }, .uv = .{ .x = 1, .y = 1 } },
    .{ .position = .{ .x = 1, .y = 1, .z = 1, .w = 1 }, .uv = .{ .x = 1, .y = 0 } },
    .{ .position = .{ .x = -1, .y = 1, .z = 1, .w = 1 }, .uv = .{ .x = 0, .y = 0 } },
};

const cube_indices = [_]u16{
    0,  1,  2,  2,  3,  0,
    4,  5,  6,  6,  7,  4,
    8,  9,  10, 10, 11, 8,
    12, 13, 14, 14, 15, 12,
    16, 17, 18, 18, 19, 16,
    20, 21, 22, 22, 23, 20,
};

pub fn main(init: std.process.Init) !void {
    var window = support.Window.open("NoGraphicsAPI spinning textured cube", width, height) orelse return error.WindowCreationFailed;
    defer window.close();
    const device = try gpu.createDevice(.{ .window = window.handle, .display = window.display, .swapchain_format = .bgra8_srgb });
    defer gpu.destroyDevice(device);
    std.debug.print("Using {s}\n", .{gpu.getDeviceCaps(device).device_name});

    // Shaders
    const vertex_code = try support.readShader(init.io, init.gpa, "shaders/cube.vert." ++ gpu.shader_extension);
    const fragment_code = try support.readShader(init.io, init.gpa, "shaders/cube.frag." ++ gpu.shader_extension);
    const cube_pso = try gpu.createGraphicsPso(device, .{
        .vertex = .{ .code = vertex_code, .entry_point = "vertexMain" },
        .fragment = .{ .code = fragment_code, .entry_point = "fragmentMain" },
        .color_targets = &.{.{ .format = .bgra8_srgb }},
        .depth_format = .d32_float,
        .rasterization = .{ .cull = .clockwise },
    });
    defer gpu.destroyPso(cube_pso);
    init.gpa.free(fragment_code);
    init.gpa.free(vertex_code);

    // GPU resources
    const data_heap = try gpu.createGpuHeap(device, data_heap_size, .cpu_visible);
    defer gpu.destroyGpuHeap(data_heap);
    var data_allocator: utility.BumpAllocator = .init(data_heap.range);
    const frame_ranges = [_]gpu.GpuCpuRange(u8){ data_allocator.allocate(u8, 4096).?, data_allocator.allocate(u8, 4096).? };
    const vertex_allocation = data_allocator.allocate(shared.CubeVertex, cube_vertices.len).?;
    const index_allocation = data_allocator.allocate(u16, cube_indices.len).?;
    const upload_allocation = data_allocator.allocate(u8, texture_byte_count).?;
    @memcpy(vertex_allocation.cpu.?[0..cube_vertices.len], &cube_vertices);
    @memcpy(index_allocation.cpu.?[0..cube_indices.len], &cube_indices);
    const texture_data = upload_allocation.cpu.?[0..texture_byte_count];
    try support.readBinaryFile(init.io, "lunarg_logo_256x256.rgba8", texture_data);

    var uploads: utility.UploadQueue = try .init(init.gpa, device, .{ .capacity = texture_byte_count });
    const texture_descriptor_heap = try gpu.createTextureDescriptorHeap(device, 1);
    defer gpu.destroyTextureDescriptorHeap(texture_descriptor_heap);
    const sampler_descriptor_heap = try gpu.createSamplerDescriptorHeap(device, 1);
    defer gpu.destroySamplerDescriptorHeap(sampler_descriptor_heap);
    const texture_heap = try gpu.createTextureHeap(device, texture_heap_size);
    defer gpu.destroyTextureHeap(texture_heap);
    var texture_allocator: utility.TextureAllocator = .init(init.gpa, device, texture_heap, 16);
    defer texture_allocator.deinit(init.gpa);

    var latest_completion: gpu.TimelinePoint = .{ .semaphore = try gpu.createTimelineSemaphore(device, 0), .value = 0 };
    defer gpu.destroyTimelineSemaphore(latest_completion.semaphore);
    const command_pools = [_]*gpu.CommandPool{ try gpu.createCommandPool(device, 0), try gpu.createCommandPool(device, 0) };
    defer for (command_pools) |pool| gpu.destroyCommandPool(pool);
    const texture_commands = try gpu.beginCommands(command_pools[0]);

    // Textures
    const texture = (try texture_allocator.allocate(texture_commands, .{
        .extent = .{ .x = texture_width, .y = texture_height, .z = 1 },
        .format = .rgba8_srgb,
        .usage = .{ .sampled = true, .transfer_destination = true },
    })).?;
    defer texture_allocator.free(texture);

    gpu.writeTextureDescriptor(texture_descriptor_heap, 0, texture.texture, .sampled, .{});
    gpu.writeSamplerDescriptor(sampler_descriptor_heap, 0, .{
        .min_filter = .nearest,
        .mag_filter = .nearest,
        .address_u = .clamp_to_edge,
        .address_v = .clamp_to_edge,
    });
    gpu.endCommands(texture_commands);
    latest_completion.value += 1;
    gpu.submit(device, .{ .commands = &.{texture_commands}, .completion = latest_completion }, 0);
    uploads.uploadTexture(texture.texture, texture_data, .{});
    uploads.deinit(init.gpa);

    var depth: ?utility.TextureAllocator.PlacedTexture = null;
    var depth_render_view: ?*gpu.RenderView = null;
    defer if (depth) |placed| {
        gpu.destroyRenderView(depth_render_view.?);
        texture_allocator.free(placed);
    };
    var depth_extent: gpu.Uint32x2 = .{};
    var frame_index: u64 = 0;
    const view = math.lookAtRh(.{ .x = 0, .y = 3, .z = 5 }, .{ .x = 0, .y = 0, .z = 0 }, .{ .x = 0, .y = 1, .z = 0 });
    defer gpu.waitIdle(device);

    while (window.pump()) {
        if (latest_completion.value >= 2) gpu.waitTimeline(.{ .semaphore = latest_completion.semaphore, .value = latest_completion.value - 1 });
        var frame_data: utility.BumpAllocator = .init(frame_ranges[latest_completion.value % 2]);
        const command_pool = command_pools[latest_completion.value % 2];
        gpu.resetCommandPool(command_pool);
        const commands = try gpu.beginCommands(command_pool);
        const frame = gpu.acquire(commands) orelse continue;

        // Window resize?
        if (frame.extent.x != depth_extent.x or frame.extent.y != depth_extent.y) {
            if (depth) |placed| {
                gpu.waitTimeline(latest_completion);
                gpu.destroyRenderView(depth_render_view.?);
                texture_allocator.free(placed);
            }
            depth = (try texture_allocator.allocate(commands, .{
                .extent = .{ .x = frame.extent.x, .y = frame.extent.y, .z = 1 },
                .format = .d32_float,
                .usage = .{ .depth_stencil_attachment = true },
            })).?;
            depth_render_view = try gpu.createRenderView(depth.?.texture, .{});
            depth_extent = frame.extent;
        }

        // Render
        gpu.setTextureDescriptorHeap(commands, texture_descriptor_heap);
        gpu.setSamplerDescriptorHeap(commands, sampler_descriptor_heap);
        gpu.barrier(commands, .{ .depth_stencil_tests = true }, .{ .depth_stencil_write = true }, .{ .depth_stencil_tests = true }, .{
            .depth_stencil_write = true,
        });
        gpu.beginRenderPass(commands, .{
            .colors = &.{.{ .render_view = frame.render_view, .load = .clear, .clear = .{ .x = 0.2, .y = 0.2, .z = 0.2, .w = 0.2 } }},
            .depth = .{ .render_view = depth_render_view, .load = .clear, .store = .discard },
        }, .{});
        gpu.setDepthStencil(commands, .{ .depth_test = true, .depth_write = true });
        gpu.bindPso(commands, cube_pso);

        const aspect = @as(f32, @floatFromInt(frame.extent.x)) / @as(f32, @floatFromInt(frame.extent.y));
        var projection = math.perspectiveRhZo(45.0 * math.pi / 180.0, aspect, 0.1, 100.0);
        projection.rows[1].y = -projection.rows[1].y;
        const rotation = math.rotationY(radians_per_frame * @as(f32, @floatFromInt(frame_index)));
        frame_index += 1;
        const transform = try comath.eval("projection * view * rotation", math.context, .{ .projection = projection, .view = view, .rotation = rotation });
        const root = frame_data.allocate(shared.CubeRootArguments, 1).?;
        root.cpu.?[0] = .{ .vertices = vertex_allocation.gpu, .transform = transform };
        gpu.drawIndexed(commands, root.gpu, .{ .indices = index_allocation.gpuRange(), .index_type = .uint16, .index_count = cube_indices.len });
        gpu.endRenderPass(commands);

        // Submit
        gpu.endCommands(commands);
        latest_completion.value += 1;
        gpu.submitAndPresent(device, .{ .commands = &.{commands}, .completion = latest_completion });
    }
}
