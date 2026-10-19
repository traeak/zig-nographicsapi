const std = @import("std");

const Backend = enum { vulkan, metal };

const Shader = struct {
    name: []const u8,
    source: []const u8,
    entry: []const u8 = "",
    stage: []const u8 = "",
    defines: []const []const u8 = &.{},
    deps: []const []const u8 = &.{},
};

const test_shaders = [_]Shader{
    .{
        .name = "upload_queue.comp",
        .source = "tests/shaders/upload_queue.slang",
        .entry = "computeMain",
        .stage = "compute",
        .deps = &.{"tests/shaders/upload_queue_shared.h"},
    },
    .{
        .name = "queue_family.comp",
        .source = "tests/queue_family.slang",
        .entry = "computeMain",
        .stage = "compute",
        .deps = &.{"tests/queue_family_shared.h"},
    },
    .{ .name = "render_continuation.vert", .source = "tests/render_continuation.slang", .entry = "vertexMain", .stage = "vertex" },
    .{ .name = "render_continuation.frag", .source = "tests/render_continuation.slang", .entry = "fragmentMain", .stage = "fragment" },
    .{
        .name = "task_shader.task",
        .source = "tests/task_shader.slang",
        .entry = "taskMain",
        .stage = "amplification",
        .deps = &.{ "tests/task_shader_shared.h", "tests/task_shader_common.h" },
    },
    .{
        .name = "task_mesh.mesh",
        .source = "tests/task_mesh.slang",
        .entry = "meshMain",
        .stage = "mesh",
        .deps = &.{ "tests/task_mesh_shared.h", "tests/task_shader_common.h" },
    },
    .{
        .name = "task_mesh.frag",
        .source = "tests/task_mesh.slang",
        .entry = "fragmentMain",
        .stage = "fragment",
        .deps = &.{ "tests/task_mesh_shared.h", "tests/task_shader_common.h" },
    },
    .{ .name = "root_data.comp", .source = "tests/root_data.slang", .entry = "computeMain", .stage = "compute", .deps = &.{"tests/root_data_shared.h"} },
    .{
        .name = "root_data_aligned.comp",
        .source = "tests/root_data.slang",
        .entry = "computeMain",
        .stage = "compute",
        .defines = &.{"ROOT_DATA_ALIGNMENT=16"},
        .deps = &.{"tests/root_data_shared.h"},
    },
    .{
        .name = "root_data_generate.comp",
        .source = "tests/root_data_generate.slang",
        .entry = "computeMain",
        .stage = "compute",
        .deps = &.{ "tests/root_data_generate_shared.h", "tests/root_data_shared.h" },
    },
    .{ .name = "shader_abi.comp", .source = "tests/shader_abi.slang", .entry = "computeMain", .stage = "compute", .deps = &.{"tests/shader_abi_shared.h"} },
};

const example_shaders = [_]Shader{
    .{ .name = "triangle.vert", .source = "examples/triangle/triangle.slang", .entry = "vertexMain", .stage = "vertex" },
    .{ .name = "triangle.frag", .source = "examples/triangle/triangle.slang", .entry = "fragmentMain", .stage = "fragment" },
    .{ .name = "cube.vert", .source = "examples/cube/cube.slang", .entry = "vertexMain", .stage = "vertex", .deps = &.{"examples/cube/cube_shared.h"} },
    .{ .name = "cube.frag", .source = "examples/cube/cube.slang", .entry = "fragmentMain", .stage = "fragment", .deps = &.{"examples/cube/cube_shared.h"} },
    .{
        .name = "simulation.comp",
        .source = "examples/deferred_renderer/simulation.slang",
        .entry = "computeMain",
        .stage = "compute",
        .deps = &.{ "examples/deferred_renderer/simulation_shared.h", "examples/deferred_renderer/object_data.h" },
    },
    .{
        .name = "gbuffer.mesh",
        .source = "examples/deferred_renderer/gbuffer.slang",
        .entry = "meshMain",
        .stage = "mesh",
        .deps = &.{ "examples/deferred_renderer/gbuffer_shared.h", "examples/deferred_renderer/object_data.h" },
    },
    .{
        .name = "gbuffer.frag",
        .source = "examples/deferred_renderer/gbuffer.slang",
        .entry = "fragmentMain",
        .stage = "fragment",
        .deps = &.{ "examples/deferred_renderer/gbuffer_shared.h", "examples/deferred_renderer/object_data.h" },
    },
    .{
        .name = "deferred.vert",
        .source = "examples/deferred_renderer/deferred_lighting.slang",
        .entry = "vertexMain",
        .stage = "vertex",
        .deps = &.{"examples/deferred_renderer/deferred_lighting_shared.h"},
    },
    .{
        .name = "deferred.frag",
        .source = "examples/deferred_renderer/deferred_lighting.slang",
        .entry = "fragmentMain",
        .stage = "fragment",
        .deps = &.{"examples/deferred_renderer/deferred_lighting_shared.h"},
    },
};

const ZigTest = struct {
    name: []const u8,
    source: []const u8,
    /// Runs from the compiled shader directory.
    shaders: bool = false,
    /// Links against a backend whose native Vulkan calls go through `tests/startup_failures_vulkan.zig`.
    native_hooks: bool = false,
};

const zig_tests = [_]ZigTest{
    .{ .name = "test_math", .source = "tests/math_test.zig" },
    .{ .name = "test_allocators", .source = "utility/tests/allocator_test.zig" },
    .{ .name = "test_device", .source = "tests/device_test.zig" },
    .{ .name = "test_command_context", .source = "tests/command_context_test.zig" },
    .{ .name = "test_multithreading", .source = "tests/multithreading_test.zig" },
    .{ .name = "test_texture_copy", .source = "tests/texture_copy_test.zig" },
    .{ .name = "test_texture_alias", .source = "tests/texture_alias_test.zig" },
    .{ .name = "test_placed_texture", .source = "tests/placed_texture_test.zig" },
    .{ .name = "test_upload_queue", .source = "utility/tests/upload_queue_test.zig" },
    .{ .name = "test_upload_queue_compute", .source = "utility/tests/upload_queue_compute_test.zig", .shaders = true },
    .{ .name = "test_root_data", .source = "tests/root_data_test.zig", .shaders = true },
    .{ .name = "test_shader_abi", .source = "tests/shader_abi_test.zig", .shaders = true },
    .{ .name = "test_task_shader", .source = "tests/task_shader_test.zig", .shaders = true },
    .{ .name = "test_render_continuation", .source = "tests/render_continuation_test.zig", .shaders = true },
    .{ .name = "test_queue_families", .source = "tests/queue_family_test.zig", .shaders = true },
    .{ .name = "test_startup_failures", .source = "tests/startup_failures_test.zig", .shaders = true, .native_hooks = true },
};

/// Shared CPU/GPU headers imported into Zig through translate-c. Slang names structures without `struct`, so a
/// generated wrapper typedefs `structs` before including `headers` from `directory`. The translated vector and matrix
/// types are replaced with the `shader_types` module's, so `math` results assign directly.
const SharedHeaders = struct {
    name: []const u8,
    directory: []const u8,
    headers: []const []const u8,
    structs: []const []const u8,
};

const zig_test_headers = [_]SharedHeaders{
    .{ .name = "queue_family_shared", .directory = "tests", .headers = &.{"queue_family_shared.h"}, .structs = &.{"QueueFamilyRoot"} },
    .{ .name = "root_data_shared", .directory = "tests", .headers = &.{"root_data_shared.h"}, .structs = &.{"RootData"} },
    .{
        .name = "root_data_generate_shared",
        .directory = "tests",
        .headers = &.{"root_data_generate_shared.h"},
        .structs = &.{ "RootData", "RootDataGenerate" },
    },
    .{ .name = "shader_abi_shared", .directory = "tests", .headers = &.{"shader_abi_shared.h"}, .structs = &.{ "ShaderAbiData", "ShaderAbiRoot" } },
    .{ .name = "upload_queue_shared", .directory = "tests/shaders", .headers = &.{"upload_queue_shared.h"}, .structs = &.{"UploadQueueRoot"} },
};

/// `deferred_lighting_shared.h` uses a C++ enum class, so the deferred renderer mirrors it by hand.
const zig_example_headers = [_]SharedHeaders{
    .{ .name = "cube_shared", .directory = "examples/cube", .headers = &.{"cube_shared.h"}, .structs = &.{ "CubeVertex", "CubeRootArguments" } },
    .{
        .name = "deferred_renderer_shared",
        .directory = "examples/deferred_renderer",
        .headers = &.{ "gbuffer_shared.h", "simulation_shared.h" },
        .structs = &.{ "ObjectData", "GBufferRoot", "SimulationRoot" },
    },
};

fn translateHeaders(ctx: Context, comptime list: []const SharedHeaders) [list.len]std.Build.Module.Import {
    const b = ctx.b;
    const share = b.addExecutable(.{
        .name = "share_shader_types",
        .root_module = b.createModule(.{ .root_source_file = b.path("tools/share_shader_types.zig"), .target = b.graph.host, .optimize = .debug }),
    });
    var imports: [list.len]std.Build.Module.Import = undefined;
    for (&imports, list) |*import, shared| {
        var wrapper: std.ArrayList(u8) = .empty;
        for (shared.structs) |name| wrapper.print(b.allocator, "typedef struct {0s} {0s};\n", .{name}) catch @panic("OOM");
        for (shared.headers) |header| wrapper.print(b.allocator, "#include \"{s}\"\n", .{header}) catch @panic("OOM");
        const file = b.addWriteFiles().add(b.fmt("{s}_c.h", .{shared.name}), wrapper.items);
        const translated = b.addTranslateC(.{ .root_source_file = file, .target = ctx.target, .optimize = ctx.optimize });
        for ([_][]const u8{ "include", "utility/include", shared.directory }) |path| translated.addIncludePath(b.path(path));
        const run = b.addRunArtifact(share);
        run.addFileArg(ctx.shader_types.getOutput());
        run.addFileArg(translated.getOutput());
        const module = b.createModule(.{ .root_source_file = run.addOutputFileArg(b.fmt("{s}.zig", .{shared.name})), .target = ctx.target });
        module.addImport("shader_types", ctx.shader_types_module);
        import.* = .{ .name = shared.name, .module = module };
    }
    return imports;
}

const Context = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    backend: Backend,
    shader_types: *std.Build.Step.TranslateC,
    shader_types_module: *std.Build.Module,
    objc: *std.Build.Module,

    fn shaderPath(ctx: Context, name: []const u8) []const u8 {
        return ctx.b.fmt("shaders/{s}.{s}", .{ name, if (ctx.backend == .metal) "metallib" else "spv" });
    }
};

pub fn build(b: *std.Build) void {
    const target = resolveTarget(b);
    const optimize = b.standardOptimizeOption(.{});
    const backend: Backend = if (target.result.os.tag.isDarwin()) .metal else .vulkan;

    const shader_types = b.addTranslateC(.{
        .root_source_file = b.path("utility/include/NoGraphicsAPIUtility/shader_types.h"),
        .target = target,
        .optimize = optimize,
    });
    shader_types.addIncludePath(b.path("include"));
    const comath = b.dependency("comath", .{}).module("comath");
    const shader_types_module = shader_types.addModule("shader_types");
    const core = b.addModule("NoGraphicsAPI", .{
        .root_source_file = b.path("src/NoGraphicsAPI.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const utility = b.addModule("NoGraphicsAPIUtility", .{
        .root_source_file = b.path("utility/src/NoGraphicsAPIUtility.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "NoGraphicsAPI", .module = core },
            .{ .name = "shader_types", .module = shader_types_module },
            .{ .name = "comath", .module = comath },
        },
    });
    // The Metal backend and the macOS examples reach Apple frameworks through the Objective-C runtime.
    const objc = b.createModule(.{ .root_source_file = b.path("src/objc.zig"), .target = target, .optimize = optimize });
    if (backend == .metal) {
        core.addImport("objc", objc);
        for ([_][]const u8{ "Metal", "Foundation", "QuartzCore" }) |framework| core.linkFramework(framework, .{});
    }
    const ctx: Context = .{
        .b = b,
        .target = target,
        .optimize = optimize,
        .backend = backend,
        .shader_types = shader_types,
        .shader_types_module = shader_types_module,
        .objc = objc,
    };
    const imports: []const std.Build.Module.Import = &.{
        .{ .name = "NoGraphicsAPI", .module = core },
        .{ .name = "NoGraphicsAPIUtility", .module = utility },
        .{ .name = "comath", .module = comath },
        .{ .name = "shader_types", .module = shader_types_module },
    };
    var hooked_imports: ?[]const std.Build.Module.Import = null;
    if (backend == .vulkan) {
        const translated = b.addTranslateC(.{ .root_source_file = b.path("src/vulkan.h"), .target = target, .optimize = optimize });
        if (target.result.os.tag == .windows) translated.addIncludePath(vulkanSdkPath(b, "Include"));
        const vulkan = translated.createModule();
        // `vulkan_native` carries the calls that create and destroy native objects; tests can replace it to inject failures.
        core.addImport("vulkan", vulkan);
        core.addImport("vulkan_native", vulkan);
        linkVulkan(b, core, target);

        const hooks = b.createModule(.{
            .root_source_file = b.path("tests/startup_failures_vulkan.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "vulkan", .module = vulkan }},
        });
        const hooked_core = b.createModule(.{
            .root_source_file = b.path("src/NoGraphicsAPI.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "vulkan", .module = vulkan }, .{ .name = "vulkan_native", .module = hooks } },
        });
        linkVulkan(b, hooked_core, target);
        const hooked_utility = b.createModule(.{
            .root_source_file = b.path("utility/src/NoGraphicsAPIUtility.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "NoGraphicsAPI", .module = hooked_core },
                .{ .name = "shader_types", .module = shader_types_module },
                .{ .name = "comath", .module = comath },
            },
        });
        hooked_imports = b.allocator.dupe(std.Build.Module.Import, &.{
            .{ .name = "NoGraphicsAPI", .module = hooked_core },
            .{ .name = "NoGraphicsAPIUtility", .module = hooked_utility },
            .{ .name = "vulkan_hooks", .module = hooks },
            .{ .name = "vulkan", .module = vulkan },
        }) catch @panic("OOM");
    }
    addTests(ctx, imports, hooked_imports);
    addExamples(ctx, imports);
}

fn resolveTarget(b: *std.Build) std.Build.ResolvedTarget {
    var query = b.standardTargetOptionsQueryOnly(.{});
    const resolved = b.resolveTargetQuery(query).result;
    const os = resolved.os.tag;
    // The math utilities require AVX2 and FMA on x86-64, MSVC is the only supported Windows ABI, and Metal 4 requires macOS 26.
    if (resolved.cpu.arch == .x86_64 and !std.Target.x86.featureSetHasAll(resolved.cpu.features, .{ .avx2, .fma }))
        query.cpu_features_add.addFeatureSet(std.Target.x86.featureSet(&.{ .avx2, .fma }));
    if (os == .windows) query.abi = .msvc;
    if (os == .macos) query.os_version_min = .{ .semver = .{ .major = 26, .minor = 0, .patch = 0 } };
    return b.resolveTargetQuery(query);
}

fn linkVulkan(b: *std.Build, module: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    if (target.result.os.tag != .windows) {
        module.linkSystemLibrary("vulkan", .{ .use_pkg_config = .no });
        return;
    }
    module.addIncludePath(vulkanSdkPath(b, "Include"));
    module.addLibraryPath(vulkanSdkPath(b, "Lib"));
    module.linkSystemLibrary("vulkan-1", .{ .use_pkg_config = .no });
}

fn vulkanSdkPath(b: *std.Build, sub_path: []const u8) std.Build.LazyPath {
    const sdk = b.graph.environ_map.get("VULKAN_SDK") orelse std.debug.panic("VULKAN_SDK must point to a Vulkan SDK 1.4.357 or newer", .{});
    return .{ .cwd_relative = b.pathJoin(&.{ sdk, sub_path }) };
}

/// Compiles shaders into a `shaders/` directory. Tests and examples load them relative to their working directory.
fn compileShaders(ctx: Context, shaders: []const Shader) *std.Build.Step.WriteFile {
    const files = ctx.b.addWriteFiles();
    for (shaders) |shader| _ = files.addCopyFile(compileSlang(ctx, files, shader), ctx.shaderPath(shader.name));
    return files;
}

fn compileSlang(ctx: Context, files: *std.Build.Step.WriteFile, shader: Shader) std.Build.LazyPath {
    const b = ctx.b;
    const slangc = b.addSystemCommand(&.{"slangc"});
    slangc.setName(b.fmt("slangc {s}", .{shader.name}));
    slangc.addFileArg(b.path(shader.source));
    slangc.addArgs(&.{ "-entry", shader.entry, "-stage", shader.stage, "-fvk-use-c-layout", "-matrix-layout-row-major", "-I" });
    slangc.addDirectoryArg(b.path(std.fs.path.dirname(shader.source).?));
    slangc.addArg("-I");
    slangc.addDirectoryArg(b.path("include"));
    slangc.addArg("-I");
    slangc.addDirectoryArg(b.path("utility/include"));
    for (shader.defines) |define| slangc.addArg(b.fmt("-D{s}", .{define}));
    for (shader.deps) |dep| slangc.addFileInput(b.path(dep));
    for ([_][]const u8{ "include/NoGraphicsAPI/types.h", "include/NoGraphicsAPI/shader.slang", "utility/include/NoGraphicsAPIUtility/shader_types.h" }) |dep|
        slangc.addFileInput(b.path(dep));

    if (ctx.backend == .metal) {
        slangc.addArgs(&.{ "-target", "metal", "-DNOGRAPHICSAPI_METAL", "-o" });
        return compileMetal(b, slangc.addOutputFileArg(b.fmt("{s}.metal", .{shader.name})), shader.name);
    }
    slangc.addArgs(&.{ "-target", "spirv", "-profile", "spirv_1_5", "-emit-spirv-directly", "-fvk-use-entrypoint-name" });
    slangc.addArgs(&.{ "-capability", "spvDescriptorHeapEXT" });
    if (std.mem.eql(u8, shader.stage, "mesh") or std.mem.eql(u8, shader.stage, "amplification")) slangc.addArgs(&.{ "-capability", "spvMeshShadingEXT" });
    slangc.addArg("-o");
    const output = slangc.addOutputFileArg(b.fmt("{s}.spv", .{shader.name}));
    const validate = b.addSystemCommand(&.{ "spirv-val", "--target-env", "vulkan1.4", "--scalar-block-layout" });
    validate.setName(b.fmt("spirv-val {s}", .{shader.name}));
    validate.addFileArg(output);
    files.step.dependOn(&validate.step);
    return output;
}

fn compileMetal(b: *std.Build, source: std.Build.LazyPath, name: []const u8) std.Build.LazyPath {
    const metal = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c" });
    metal.addFileArg(source);
    metal.addArg("-o");
    const air = metal.addOutputFileArg(b.fmt("{s}.air", .{name}));
    const metallib = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib" });
    metallib.addFileArg(air);
    metallib.addArg("-o");
    return metallib.addOutputFileArg(b.fmt("{s}.metallib", .{name}));
}

fn addTests(ctx: Context, imports: []const std.Build.Module.Import, hooked_imports: ?[]const std.Build.Module.Import) void {
    const b = ctx.b;
    const test_step = b.step("test", "Build and run all tests");
    const shaders = compileShaders(ctx, &test_shaders);
    const headers = translateHeaders(ctx, &zig_test_headers);
    for (zig_tests) |zig_test| {
        const exe = b.addTest(.{
            .name = zig_test.name,
            .root_module = b.createModule(.{ .root_source_file = b.path(zig_test.source), .target = ctx.target, .optimize = ctx.optimize }),
        });
        for (if (zig_test.native_hooks) hooked_imports orelse continue else imports) |import| exe.root_module.addImport(import.name, import.module);
        for (headers) |import| exe.root_module.addImport(import.name, import.module);
        const run = b.addRunArtifact(exe);
        if (zig_test.shaders) run.setCwd(shaders.getDirectory());
        run.has_side_effects = true;
        b.step(zig_test.name, b.fmt("Run {s}", .{zig_test.name})).dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
}

/// Example name, root source, and an optional asset installed next to it.
const examples = [_]struct { []const u8, []const u8, ?[]const u8 }{
    .{ "triangle", "examples/triangle/triangle.zig", null },
    .{ "cube", "examples/cube/cube.zig", "examples/cube/lunarg_logo_256x256.rgba8" },
    .{ "deferred_renderer", "examples/deferred_renderer/deferred_renderer.zig", null },
};

fn addExamples(ctx: Context, imports: []const std.Build.Module.Import) void {
    const b = ctx.b;
    const os = ctx.target.result.os.tag;
    if (os != .windows and os != .linux and os != .macos) return;
    const step = b.step("examples", "Build and install the examples");
    const check = b.step("check", "Compile the examples without linking them");
    const shaders = compileShaders(ctx, &example_shaders);
    step.dependOn(&b.addInstallDirectory(.{ .source_dir = shaders.getDirectory(), .install_dir = .bin, .install_subdir = "" }).step);

    const headers = translateHeaders(ctx, &zig_example_headers);
    const support = b.createModule(.{ .root_source_file = b.path("examples/common/example_support.zig"), .target = ctx.target, .optimize = ctx.optimize });
    if (os == .macos) support.addImport("objc", ctx.objc);
    for (examples) |example| {
        const name, const source, const asset = example;
        if (asset) |path| step.dependOn(&b.addInstallBinFile(b.path(path), std.fs.path.basename(path)).step);
        const module = b.createModule(.{ .root_source_file = b.path(source), .target = ctx.target, .optimize = ctx.optimize, .imports = imports });
        module.addImport("example_support", support);
        for (headers) |import| module.addImport(import.name, import.module);
        switch (os) {
            .windows => module.linkSystemLibrary("user32", .{ .use_pkg_config = .no }),
            .linux => module.linkSystemLibrary("X11", .{ .use_pkg_config = .no }),
            else => module.linkFramework("AppKit", .{}),
        }
        const exe = b.addExecutable(.{ .name = name, .root_module = module });
        if (os == .windows and ctx.optimize != .debug) {
            exe.subsystem = .windows;
            exe.entry = .{ .symbol_name = "mainCRTStartup" };
        }
        check.dependOn(&b.addExecutable(.{ .name = name, .root_module = module }).step);
        step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        const run = b.addRunArtifact(exe);
        run.setCwd(.{ .relative = .{ .base = .install_bin } });
        run.step.dependOn(step);
        b.step(b.fmt("run_{s}", .{name}), b.fmt("Run {s}", .{name})).dependOn(&run.step);
    }
}
