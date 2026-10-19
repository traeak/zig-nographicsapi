const std = @import("std");
const gpu = @import("NoGraphicsAPI");
const support = @import("example_support");

pub fn main(init: std.process.Init) !void {
    const width = 512;
    const height = 512;

    var window = support.Window.open("NoGraphicsAPI triangle", width, height) orelse return error.WindowCreationFailed;
    defer window.close();
    const device = try gpu.createDevice(.{ .window = window.handle, .display = window.display, .swapchain_format = .bgra8_srgb });
    defer gpu.destroyDevice(device);
    std.debug.print("Using {s}\n", .{gpu.getDeviceCaps(device).device_name});

    const vertex_code = try support.readShader(init.io, init.gpa, "shaders/triangle.vert." ++ gpu.shader_extension);
    const fragment_code = try support.readShader(init.io, init.gpa, "shaders/triangle.frag." ++ gpu.shader_extension);
    const triangle_pso = try gpu.createGraphicsPso(device, .{
        .vertex = .{ .code = vertex_code, .entry_point = "vertexMain" },
        .fragment = .{ .code = fragment_code, .entry_point = "fragmentMain" },
        .color_targets = &.{.{ .format = .bgra8_srgb }},
    });
    defer gpu.destroyPso(triangle_pso);
    init.gpa.free(fragment_code);
    init.gpa.free(vertex_code);

    var latest_completion: gpu.TimelinePoint = .{ .semaphore = try gpu.createTimelineSemaphore(device, 0), .value = 0 };
    defer gpu.destroyTimelineSemaphore(latest_completion.semaphore);
    const command_pools = [_]*gpu.CommandPool{ try gpu.createCommandPool(device, 0), try gpu.createCommandPool(device, 0) };
    defer for (command_pools) |pool| gpu.destroyCommandPool(pool);
    defer gpu.waitIdle(device);

    while (window.pump()) {
        if (latest_completion.value >= 2) gpu.waitTimeline(.{ .semaphore = latest_completion.semaphore, .value = latest_completion.value - 1 });
        const command_pool = command_pools[latest_completion.value % 2];
        gpu.resetCommandPool(command_pool);
        const commands = try gpu.beginCommands(command_pool);
        const frame = gpu.acquire(commands) orelse continue;
        gpu.beginRenderPass(commands, .{ .colors = &.{.{ .render_view = frame.render_view, .load = .clear }} }, .{});
        gpu.bindPso(commands, triangle_pso);
        gpu.draw(commands, null, .{ .vertex_count = 3 });
        gpu.endRenderPass(commands);
        gpu.endCommands(commands);
        latest_completion.value += 1;
        gpu.submitAndPresent(device, .{ .commands = &.{commands}, .completion = latest_completion });
    }
}
