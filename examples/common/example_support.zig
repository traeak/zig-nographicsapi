//! Window, timer and file helpers shared by the Zig examples: Win32 on Windows, X11 on Linux, AppKit on macOS.

const std = @import("std");
const builtin = @import("builtin");

const platform = switch (builtin.os.tag) {
    .windows => @import("window_win32.zig"),
    .linux => @import("window_x11.zig"),
    .macos => @import("window_cocoa.zig"),
    else => @compileError("The Zig examples support Win32, X11 and AppKit windows."),
};

/// `handle` and `display` go to `DeviceDesc.window` and `DeviceDesc.display`.
pub const Window = platform.Window;
pub const timeSeconds = platform.timeSeconds;

/// SPIR-V words must be 4-byte aligned. Free the code after creating the PSO that uses it.
pub fn readShader(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]align(4) u8 {
    return std.Io.Dir.cwd().readFileAllocOptions(io, path, allocator, .unlimited, .@"4", null);
}

/// Fills `data` from the start of the file.
pub fn readBinaryFile(io: std.Io, path: []const u8, data: []u8) !void {
    const read = try std.Io.Dir.cwd().readFile(io, path, data);
    if (read.len != data.len) return error.EndOfStream;
}
