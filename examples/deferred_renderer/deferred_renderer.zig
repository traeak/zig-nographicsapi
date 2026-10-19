const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const utility = @import("NoGraphicsAPIUtility");
const comath = @import("comath");
const support = @import("example_support");
const shared = @import("deferred_renderer_shared");
const math = utility.math;
const BumpAllocator = utility.BumpAllocator;
const DeleteQueue = utility.DeleteQueue;
const TextureAllocator = utility.TextureAllocator;
const float3 = math.float3;

/// Mirrors GBufferTexture in deferred_lighting_shared.h.
const GBufferTexture = enum(u32) { albedo, normal_roughness, depth, count };

/// Mirrors DeferredLightingRoot in deferred_lighting_shared.h.
const DeferredLightingRoot = extern struct {
    camera_position: math.float4,
    ray_center: math.float4,
    ray_horizontal: math.float4,
    ray_vertical: math.float4,
    depth_linearize: math.float2,
    gbuffer_pixel_scale: math.float2,
    gbuffer_texture_base: u32,
};

comptime {
    std.debug.assert(@sizeOf(DeferredLightingRoot) == 84 and @offsetOf(DeferredLightingRoot, "gbuffer_texture_base") == 80);
}

const initial_width = 1280;
const initial_height = 720;
const object_count = shared.object_grid_width * shared.object_grid_width;
const frames_in_flight = 2;
const gbuffer_texture_count = @backingInt(GBufferTexture.count);
const data_heap_size = 16 * 1024 * 1024;
const texture_heap_size = 256 * 1024 * 1024;
const cube_scale = 0.42;
const cube_rotation_speed = 0.5;
const minimum_orbit_radius = 54.0;
const maximum_orbit_radius = 240.0;
const camera_position: float3 = .{ .x = 260, .y = 210, .z = 300 };

const GBuffer = struct {
    albedo: TextureAllocator.PlacedTexture,
    normal_roughness: TextureAllocator.PlacedTexture,
    depth: TextureAllocator.PlacedTexture,
    albedo_render_view: *gpu.RenderView,
    normal_roughness_render_view: *gpu.RenderView,
    depth_render_view: *gpu.RenderView,
    width: u32,
    height: u32,
};

fn destroyGBuffer(texture_allocator: *TextureAllocator, gbuffer: GBuffer) void {
    gpu.destroyRenderView(gbuffer.depth_render_view);
    gpu.destroyRenderView(gbuffer.normal_roughness_render_view);
    gpu.destroyRenderView(gbuffer.albedo_render_view);
    texture_allocator.free(gbuffer.depth);
    texture_allocator.free(gbuffer.normal_roughness);
    texture_allocator.free(gbuffer.albedo);
}

fn createGBuffer(
    commands: *gpu.CommandBuffer,
    texture_allocator: *TextureAllocator,
    descriptors: *gpu.TextureDescriptorHeap,
    first_descriptor: u32,
    width: u32,
    height: u32,
) gpu.Error!GBuffer {
    const extent: gpu.Uint32x3 = .{ .x = width, .y = height, .z = 1 };
    const albedo = (try texture_allocator.allocate(commands, .{ .extent = extent, .usage = .{ .sampled = true, .color_attachment = true } })).?;
    const normal_roughness = (try texture_allocator.allocate(commands, .{
        .extent = extent,
        .format = .rgba16_float,
        .usage = .{ .sampled = true, .color_attachment = true },
    })).?;
    const depth = (try texture_allocator.allocate(commands, .{
        .extent = extent,
        .format = .d32_float,
        .usage = .{ .sampled = true, .depth_stencil_attachment = true },
    })).?;
    const gbuffer: GBuffer = .{
        .albedo = albedo,
        .normal_roughness = normal_roughness,
        .depth = depth,
        .albedo_render_view = try gpu.createRenderView(albedo.texture, .{}),
        .normal_roughness_render_view = try gpu.createRenderView(normal_roughness.texture, .{}),
        .depth_render_view = try gpu.createRenderView(depth.texture, .{}),
        .width = width,
        .height = height,
    };
    gpu.writeTextureDescriptor(descriptors, first_descriptor + @backingInt(GBufferTexture.albedo), albedo.texture, .sampled, .{});
    gpu.writeTextureDescriptor(descriptors, first_descriptor + @backingInt(GBufferTexture.normal_roughness), normal_roughness.texture, .sampled, .{});
    gpu.writeTextureDescriptor(descriptors, first_descriptor + @backingInt(GBufferTexture.depth), depth.texture, .sampled, .{});
    return gbuffer;
}

fn randomSigned(state: *u32) f32 {
    state.* ^= state.* << 13;
    state.* ^= state.* >> 17;
    state.* ^= state.* << 5;
    return @as(f32, @floatFromInt(state.* >> 8)) * (2.0 / 16777216.0) - 1.0;
}

fn randomFloat3(state: *u32, scale: f32) float3 {
    return .{ .x = randomSigned(state) * scale, .y = randomSigned(state) * scale, .z = randomSigned(state) * scale };
}

fn initializeObjectData(objects: []shared.ObjectData) void {
    const minimum_radius_squared = minimum_orbit_radius * minimum_orbit_radius;
    const maximum_radius_squared = maximum_orbit_radius * maximum_orbit_radius;
    const softening_squared = shared.gravity_softening * shared.gravity_softening;
    var random_state: u32 = 0x12345678;

    for (objects) |*object| {
        var position: float3 = undefined;
        var radius_squared: f32 = undefined;
        while (true) {
            position = randomFloat3(&random_state, maximum_orbit_radius);
            radius_squared = math.dot(position, position);
            if (radius_squared >= minimum_radius_squared and radius_squared <= maximum_radius_squared) break;
        }

        var tangent: float3 = undefined;
        while (true) {
            const random_axis = randomFloat3(&random_state, 1);
            const axis_length_squared = math.dot(random_axis, random_axis);
            tangent = math.cross(position, random_axis);
            if (axis_length_squared >= 0.01 and axis_length_squared <= 1 and math.dot(tangent, tangent) >= 0.01) break;
        }

        const inverse_softened_distance = math.rsqrt(radius_squared + softening_squared);
        const circular_speed = @sqrt(shared.central_gravity * radius_squared *
            inverse_softened_distance * inverse_softened_distance * inverse_softened_distance);
        const speed_scale = 0.985 + randomSigned(&random_state) * 0.045;
        object.* = .{
            .position = position,
            .velocity = math.mul(math.normalize(tangent), circular_speed * speed_scale),
        };
    }
}

pub fn main(init: std.process.Init) !void {
    var window = support.Window.open("NoGraphicsAPI deferred renderer", initial_width, initial_height) orelse return error.WindowCreationFailed;
    defer window.close();
    const device = try gpu.createDevice(.{ .window = window.handle, .display = window.display, .swapchain_format = .bgra8_srgb });
    defer gpu.destroyDevice(device);
    std.debug.print("Using {s}\n", .{gpu.getDeviceCaps(device).device_name});

    // Shaders
    const simulation_code = try support.readShader(init.io, init.gpa, "shaders/simulation.comp." ++ gpu.shader_extension);
    const simulation_pso = try gpu.createComputePso(device, .{
        .code = simulation_code,
        .entry_point = "computeMain",
        .threadgroup_size = .{ .x = shared.simulation_thread_count, .y = 1, .z = 1 },
    });
    defer gpu.destroyPso(simulation_pso);
    init.gpa.free(simulation_code);

    const gbuffer_mesh_code = try support.readShader(init.io, init.gpa, "shaders/gbuffer.mesh." ++ gpu.shader_extension);
    const gbuffer_fragment_code = try support.readShader(init.io, init.gpa, "shaders/gbuffer.frag." ++ gpu.shader_extension);
    const gbuffer_pso = try gpu.createMeshPso(device, .{
        .mesh = .{ .code = gbuffer_mesh_code, .entry_point = "meshMain", .threadgroup_size = .{ .x = shared.gbuffer_thread_count, .y = 1, .z = 1 } },
        .fragment = .{ .code = gbuffer_fragment_code, .entry_point = "fragmentMain" },
        .color_targets = &.{ .{ .format = .rgba8_unorm }, .{ .format = .rgba16_float } },
        .depth_format = .d32_float,
        .rasterization = .{ .cull = .clockwise },
    });
    defer gpu.destroyPso(gbuffer_pso);
    init.gpa.free(gbuffer_fragment_code);
    init.gpa.free(gbuffer_mesh_code);

    const deferred_vertex_code = try support.readShader(init.io, init.gpa, "shaders/deferred.vert." ++ gpu.shader_extension);
    const deferred_fragment_code = try support.readShader(init.io, init.gpa, "shaders/deferred.frag." ++ gpu.shader_extension);
    const deferred_lighting_pso = try gpu.createGraphicsPso(device, .{
        .vertex = .{ .code = deferred_vertex_code, .entry_point = "vertexMain" },
        .fragment = .{ .code = deferred_fragment_code, .entry_point = "fragmentMain" },
        .color_targets = &.{.{ .format = .bgra8_srgb }},
    });
    defer gpu.destroyPso(deferred_lighting_pso);
    init.gpa.free(deferred_fragment_code);
    init.gpa.free(deferred_vertex_code);

    // GPU resources
    const texture_descriptor_heap = try gpu.createTextureDescriptorHeap(device, gbuffer_texture_count * frames_in_flight);
    defer gpu.destroyTextureDescriptorHeap(texture_descriptor_heap);
    const data_heap = try gpu.createGpuHeap(device, data_heap_size, .cpu_visible);
    defer gpu.destroyGpuHeap(data_heap);
    var data_allocator: BumpAllocator = .init(data_heap.range);
    const frame_ranges = [frames_in_flight]gpu.GpuCpuRange(u8){ data_allocator.allocate(u8, 4096).?, data_allocator.allocate(u8, 4096).? };
    const object_allocation = data_allocator.allocate(shared.ObjectData, object_count).?;
    initializeObjectData(object_allocation.cpu.?[0..object_count]);
    const texture_heap = try gpu.createTextureHeap(device, texture_heap_size);
    defer gpu.destroyTextureHeap(texture_heap);
    var texture_allocator: TextureAllocator = .init(init.gpa, device, texture_heap, 64);
    defer texture_allocator.deinit(init.gpa);

    // Camera and simulation
    const view = math.lookAtRh(camera_position, .{}, .{ .y = 1 });
    const view_right = math.convert(float3, view.rows[0]);
    const view_up = math.convert(float3, view.rows[1]);
    const view_forward = math.neg(math.convert(float3, view.rows[2]));
    var gbuffer: ?GBuffer = null;
    defer if (gbuffer) |current| destroyGBuffer(&texture_allocator, current);
    var descriptor_row: u32 = 0;
    var previous_time = support.timeSeconds();
    var rotation_angle: f32 = 0;

    var latest_completion: gpu.TimelinePoint = .{ .semaphore = try gpu.createTimelineSemaphore(device, 0), .value = 0 };
    defer gpu.destroyTimelineSemaphore(latest_completion.semaphore);
    var delete_queue: DeleteQueue = .init(init.gpa, latest_completion.semaphore, frames_in_flight);
    defer delete_queue.deinit(init.gpa);
    const command_pools = [frames_in_flight]*gpu.CommandPool{ try gpu.createCommandPool(device, 0), try gpu.createCommandPool(device, 0) };
    defer for (command_pools) |pool| gpu.destroyCommandPool(pool);
    defer {
        gpu.waitIdle(device);
        delete_queue.drain();
    }

    while (window.pump()) {
        // Two frames in flight keep the double-buffered descriptors and frame data safe to reuse.
        if (latest_completion.value >= frames_in_flight)
            gpu.waitTimeline(.{ .semaphore = latest_completion.semaphore, .value = latest_completion.value - frames_in_flight + 1 });
        delete_queue.tick();

        var frame_data: BumpAllocator = .init(frame_ranges[latest_completion.value % frames_in_flight]);
        const command_pool = command_pools[latest_completion.value % frames_in_flight];
        gpu.resetCommandPool(command_pool);
        const commands = try gpu.beginCommands(command_pool);
        const frame = gpu.acquire(commands) orelse continue;
        const extent = frame.extent;

        // Window resize
        if (gbuffer == null or extent.x != gbuffer.?.width or extent.y != gbuffer.?.height) {
            if (gbuffer) |old| {
                delete_queue.enqueue(latest_completion.value, destroyGBuffer, .{ &texture_allocator, old });
                descriptor_row = (descriptor_row + 1) % frames_in_flight;
            }
            gbuffer = try createGBuffer(commands, &texture_allocator, texture_descriptor_heap, descriptor_row * gbuffer_texture_count, extent.x, extent.y);
        }
        const targets = gbuffer.?;

        gpu.setTextureDescriptorHeap(commands, texture_descriptor_heap);

        // Simulation
        const current_time = support.timeSeconds();
        const delta_seconds: f32 = @floatCast(current_time - previous_time);
        previous_time = current_time;

        gpu.barrier(commands, .{ .compute = true, .mesh = true }, .{ .shader_write = true, .shader_read = true }, .{ .compute = true }, .{
            .shader_read = true,
            .shader_write = true,
        });

        gpu.bindPso(commands, simulation_pso);
        const simulation_root = frame_data.allocate(shared.SimulationRoot, 1).?;
        simulation_root.cpu.?[0] = .{ .objects = object_allocation.gpu, .delta_seconds = delta_seconds };
        gpu.dispatch(commands, simulation_root.gpu, .{ .x = object_count / shared.simulation_thread_count, .y = 1, .z = 1 });

        // G-buffer
        gpu.barrier(commands, .{ .compute = true }, .{ .shader_write = true }, .{ .mesh = true }, .{ .shader_read = true });
        gpu.barrier(
            commands,
            .{ .fragment = true, .color_output = true, .depth_stencil_tests = true },
            .{ .shader_read = true, .color_write = true, .depth_stencil_write = true },
            .{ .color_output = true, .depth_stencil_tests = true },
            .{ .color_write = true, .depth_stencil_write = true },
        );

        gpu.beginRenderPass(commands, .{
            .colors = &.{
                .{ .render_view = targets.albedo_render_view, .load = .clear },
                .{ .render_view = targets.normal_roughness_render_view, .load = .clear },
            },
            .depth = .{ .render_view = targets.depth_render_view, .load = .clear },
        }, .{});
        gpu.setDepthStencil(commands, .{ .depth_test = true, .depth_write = true });
        gpu.bindPso(commands, gbuffer_pso);

        const aspect = @as(f32, @floatFromInt(extent.x)) / @as(f32, @floatFromInt(extent.y));
        var projection = math.perspectiveRhZo(math.pi / 3.0, aspect, 0.3, 1500);
        projection.rows[1].y = -projection.rows[1].y;
        rotation_angle += cube_rotation_speed * delta_seconds;

        const orientation = try comath.eval("rotation_y * rotation_x * scale", math.context, .{
            .rotation_y = math.rotationY(rotation_angle),
            .rotation_x = math.rotationX(rotation_angle * 0.5),
            .scale = math.scale(math.splat(float3, cube_scale)),
        });
        const gbuffer_root = frame_data.allocate(shared.GBufferRoot, 1).?;
        gbuffer_root.cpu.?[0] = .{
            .objects = object_allocation.gpu,
            .view_projection = math.mul(projection, view),
            .orientation = math.toFloat3x4(orientation),
        };
        gpu.drawMeshlets(commands, gbuffer_root.gpu, .{ .x = shared.object_grid_width, .y = shared.object_grid_width, .z = 1 });
        gpu.endRenderPass(commands);

        // Lighting
        gpu.barrier(
            commands,
            .{ .color_output = true, .depth_stencil_tests = true },
            .{ .color_write = true, .depth_stencil_write = true },
            .{ .fragment = true },
            .{ .shader_read = true },
        );
        gpu.beginRenderPass(commands, .{ .colors = &.{.{ .render_view = frame.render_view, .load = .discard }} }, .{});
        gpu.bindPso(commands, deferred_lighting_pso);

        const lighting_root = frame_data.allocate(DeferredLightingRoot, 1).?;
        lighting_root.cpu.?[0] = .{
            .camera_position = math.extend(camera_position, 1),
            .ray_center = math.extend(view_forward, 0),
            .ray_horizontal = math.extend(math.div(view_right, projection.rows[0].x), 0),
            .ray_vertical = math.extend(math.div(view_up, projection.rows[1].y), 0),
            .depth_linearize = .{ .x = -projection.rows[2].w, .y = -projection.rows[2].z },
            .gbuffer_pixel_scale = .{
                .x = @as(f32, @floatFromInt(targets.width)) / @as(f32, @floatFromInt(extent.x)),
                .y = @as(f32, @floatFromInt(targets.height)) / @as(f32, @floatFromInt(extent.y)),
            },
            .gbuffer_texture_base = descriptor_row * gbuffer_texture_count,
        };
        gpu.draw(commands, lighting_root.gpu, .{ .vertex_count = 3 });
        gpu.endRenderPass(commands);

        // Submit
        gpu.endCommands(commands);
        latest_completion.value += 1;
        gpu.submitAndPresent(device, .{ .commands = &.{commands}, .completion = latest_completion });
    }
}
