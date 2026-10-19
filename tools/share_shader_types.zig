//! Makes a translate-c module reuse the `shader_types` module's structures:
//!     share_shader_types <translated shader_types.zig> <translated header.zig> <output.zig>
//! Every `pub const struct_X = extern struct {...};` that shader_types also defines becomes a reference to it, so math
//! results assign directly into shared roots.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4) return error.Usage;
    const cwd = std.Io.Dir.cwd();
    const shared = try cwd.readFileAlloc(init.io, args[1], arena, .unlimited);
    const header = try cwd.readFileAlloc(init.io, args[2], arena, .unlimited);

    const prefix = "pub const struct_";
    const suffix = " = extern struct {";
    var output: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, header, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, prefix) and std.mem.endsWith(u8, line, suffix)) {
            const declaration = line[0 .. line.len - suffix.len];
            if (std.mem.indexOf(u8, shared, line) != null) {
                try output.print(arena, "{s} = @import(\"shader_types\").{s};\n", .{ declaration, declaration["pub const ".len..] });
                while (lines.next()) |body| if (std.mem.eql(u8, body, "};")) break;
                continue;
            }
        }
        try output.print(arena, "{s}\n", .{line});
    }
    try cwd.writeFile(init.io, .{ .sub_path = args[3], .data = output.items });
}
