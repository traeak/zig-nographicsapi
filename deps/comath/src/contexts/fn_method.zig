const std = @import("std");
const cm = @import("../comath.zig");
const util = @import("util");

pub inline fn context(
    sub_ctx: anytype,
    /// must be a struct literal wherein each field name is an operator,
    /// with a string literal value corresponding to the method name, or a list of possible method names.
    /// ie `.{ .@"+" = "add", .@"-" = &.{ "sub", "neg" } }`
    comptime method_names: anytype,
) Context(@TypeOf(sub_ctx), method_names) {
    const SubCtx = @TypeOf(sub_ctx);
    if (util.NamespaceOf(SubCtx) == null) @compileError(
        "Expected struct/union/enum, or pointer to struct/union/enum/opaque, got '" ++ @typeName(SubCtx) ++ "'" ++
            if (SubCtx != type) "" else " (" ++ @typeName(sub_ctx) ++ ")",
    );
    return .{ .sub_ctx = sub_ctx };
}

pub fn Context(
    comptime SubCtx: type,
    comptime method_names: anytype,
) type {
    {
        const Deduped = DedupedMethodNames(method_names);
        if (@TypeOf(method_names) != Deduped) {
            return Context(SubCtx, Deduped{});
        }
    }
    return struct {
        sub_ctx: SubCtx,
        const Self = @This();
        const Ns = util.NamespaceOf(SubCtx) orelse cm.ctx.Null;

        pub inline fn matchUnOp(comptime str: []const u8) bool {
            return @TypeOf(Ns.matchUnOp) != void and Ns.matchUnOp(str);
        }

        pub inline fn matchBinOp(comptime str: []const u8) bool {
            return @TypeOf(Ns.matchBinOp) != void and Ns.matchBinOp(str);
        }

        pub inline fn orderBinOp(comptime lhs: []const u8, comptime rhs: []const u8) ?cm.Order {
            return Ns.orderBinOp(lhs, rhs);
        }

        pub fn EvalNumberLiteral(comptime src: []const u8) type {
            if (@TypeOf(Ns.EvalNumberLiteral) != void) {
                return Ns.EvalNumberLiteral(src);
            }
            return cm.ctx.DefaultEvalNumberLiteral(src);
        }
        pub inline fn evalNumberLiteral(comptime src: []const u8) EvalNumberLiteral(src) {
            if (@TypeOf(Ns.evalNumberLiteral) != void) {
                return Ns.evalNumberLiteral(src);
            }
            return cm.ctx.defaultEvalNumberLiteral(src);
        }

        pub fn EvalIdent(comptime ident: []const u8) type {
            if (@TypeOf(Ns.EvalIdent) != void) {
                return Ns.EvalIdent(ident);
            }
            return noreturn;
        }
        pub fn evalIdent(ctx: Self, comptime ident: []const u8) !EvalIdent(ident) {
            if (@TypeOf(Ns.evalIdent) != void) {
                return ctx.sub_ctx.evalIdent(ident);
            }
            comptime unreachable;
        }

        pub fn EvalProperty(comptime T: type, comptime field: []const u8) type {
            if (getOpMapping(T, "evalProperty", method_names, 2)) |name| {
                const Method = util.ImplicitDeref(@TypeOf(@field(T, name)));
                const method_info = @typeInfo(Method).@"fn";
                if (method_info.return_type) |Ret| return util.GetPayloadIfErrorUnion(Ret);

                const val: (method_info.param_types[0] orelse T) = undefined;
                const field_arg = if (method_info.params[1].type) |FieldArg| castStringTo(FieldArg, field) else field;
                return util.GetPayloadIfErrorUnion(@TypeOf(@field(T, name)(val, field_arg)));
            }
            return Ns.EvalProperty(T, field);
        }
        pub inline fn evalProperty(ctx: Self, val: anytype, comptime field: []const u8) !EvalProperty(@TypeOf(val), field) {
            const T = @TypeOf(val);
            if (getOpMapping(@TypeOf(val), "evalProperty", method_names, 2)) |name| {
                const Method = util.ImplicitDeref(@TypeOf(@field(T, name)));
                const method_info = @typeInfo(Method).@"fn";

                const field_arg = if (method_info.params[1].type) |FieldArg| castStringTo(FieldArg, field) else field;
                return @field(T, name)(val, field_arg);
            }
            return ctx.sub_ctx.evalProperty(val, field);
        }

        pub fn EvalIndexAccess(comptime Lhs: type, comptime Rhs: type) type {
            if (getOpMapping(Lhs, "evalIndexAccess", method_names, 2)) |name| {
                const Method = util.ImplicitDeref(@TypeOf(@field(Rhs, name)));
                const method_info = @typeInfo(Method).@"fn";
                if (method_info.return_type) |Ret| return util.GetPayloadIfErrorUnion(Ret);

                const lhs: (method_info.param_types[0] orelse Lhs) = undefined;
                const rhs: (method_info.param_types[0] orelse Rhs) = undefined;
                return util.GetPayloadIfErrorUnion(@TypeOf(@field(Lhs, name)(lhs, rhs)));
            }
            return Ns.EvalIndexAccess(Lhs, Rhs);
        }
        pub inline fn evalIndexAccess(ctx: Self, lhs: anytype, rhs: anytype) !EvalIndexAccess(@TypeOf(lhs), @TypeOf(rhs)) {
            const Lhs = @TypeOf(lhs);
            if (getOpMapping(Lhs, "evalIndexAccess", method_names, 2)) |name| {
                return @field(Lhs, name)(lhs, rhs);
            }
            return ctx.sub_ctx.evalIndexAccess(lhs, rhs);
        }

        pub fn EvalFuncCall(comptime Callee: type, comptime Args: type) type {
            const args_info = @typeInfo(Args).@"struct";
            if (getOpMapping(Callee, "evalFuncCall", method_names, 1 + args_info.field_names.len)) |name| {
                const Method = util.ImplicitDeref(@TypeOf(@field(Callee, name)));
                const method_info = @typeInfo(Method).@"fn";
                if (method_info.return_type) |Ret| return util.GetPayloadIfErrorUnion(Ret);

                const lhs: (method_info.param_types[0] orelse Callee) = undefined;
                const rhs: (method_info.param_types[0] orelse Args) = undefined;
                return util.GetPayloadIfErrorUnion(@TypeOf(@call(.auto, @field(Callee, name), .{lhs} ++ rhs)));
            }
            return Ns.EvalFuncCall(Callee, Args);
        }
        pub inline fn evalFuncCall(ctx: Self, callee: anytype, args: anytype) !EvalFuncCall(@TypeOf(callee), @TypeOf(args)) {
            const Callee = @TypeOf(callee);
            const Args = @TypeOf(args);
            const args_info = @typeInfo(Args).@"struct";
            if (getOpMapping(Callee, "evalFuncCall", method_names, 1 + args_info.field_names.len)) |name| {
                const func = @field(Callee, name);
                return switch (comptime args_info.field_names.len) {
                    0 => @call(.auto, func, .{callee}),
                    1 => @call(.auto, func, .{ callee, args[0] }),
                    2 => @call(.auto, func, .{ callee, args[0], args[1] }),
                    3 => @call(.auto, func, .{ callee, args[0], args[1], args[2] }),
                    4 => @call(.auto, func, .{ callee, args[0], args[1], args[2], args[3] }),
                    5 => @call(.auto, func, .{ callee, args[0], args[1], args[2], args[3], args[4] }),
                    6 => @call(.auto, func, .{ callee, args[0], args[1], args[2], args[3], args[4], args[5] }),
                    7 => @call(.auto, func, .{ callee, args[0], args[1], args[2], args[3], args[4], args[5], args[6] }),
                    8 => @call(.auto, func, .{ callee, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7] }),
                    else => @compileError("evalFuncCall: too many arguments (max 8)"),
                };
            }
            return ctx.sub_ctx.evalFuncCall(callee, args);
        }

        pub fn EvalMethodCall(comptime lhs: type, comptime method: []const u8, comptime Args: type) type {
            if (@TypeOf(Ns.EvalMethodCall) != void) {
                if (Ns.EvalMethodCall(lhs, method, Args) != noreturn) {
                    return Ns.EvalMethodCall(lhs, method, Args);
                }
            }
            return noreturn;
        }
        pub fn evalMethodCall(ctx: @This(), self_param: anytype, comptime method: []const u8, args: anytype) !EvalMethodCall(@TypeOf(self_param), method, @TypeOf(args)) {
            const func = @field(util.ImplicitDeref(@TypeOf(self_param)), method);
            return ctx.evalFuncCall(func, .{self_param} ++ args);
        }

        pub fn EvalUnOp(comptime op: []const u8, comptime T: type) type {
            if (getOpMapping(T, op, method_names, 1)) |name| {
                const Method = util.ImplicitDeref(@TypeOf(@field(T, name)));
                const method_info = @typeInfo(Method).@"fn";
                if (method_info.return_type) |Ret| return util.GetPayloadIfErrorUnion(Ret);

                const val: (method_info.param_types[0] orelse T) = undefined;
                return util.GetPayloadIfErrorUnion(@TypeOf(@field(T, name)(val)));
            }
            return Ns.EvalUnOp(op, T);
        }
        pub inline fn evalUnOp(ctx: Self, comptime op: []const u8, val: anytype) !EvalUnOp(op, @TypeOf(val)) {
            const Val = @TypeOf(val);
            if (comptime getOpMapping(Val, op, method_names, 1)) |name| {
                return @field(Val, name)(val);
            }
            return ctx.sub_ctx.evalUnOp(op, val);
        }

        pub fn EvalBinOp(comptime Lhs: type, comptime op: []const u8, comptime Rhs: type) type {
            if (comptime getOpMapping(Lhs, op, method_names, 2)) |name| {
                const Method = util.ImplicitDeref(@TypeOf(@field(Lhs, name)));
                const method_info = @typeInfo(Method).@"fn";
                if (method_info.return_type) |Ret| return util.GetPayloadIfErrorUnion(Ret);

                const lhs: (method_info.param_types[0] orelse Lhs) = undefined;
                const rhs: (method_info.params[1].type orelse Rhs) = undefined;
                return util.GetPayloadIfErrorUnion(@TypeOf(@field(Lhs, name)(lhs, rhs)));
            }
            return Ns.EvalBinOp(Lhs, op, Rhs);
        }
        pub inline fn evalBinOp(ctx: Self, lhs: anytype, comptime op: []const u8, rhs: anytype) !EvalBinOp(@TypeOf(lhs), op, @TypeOf(rhs)) {
            const Lhs = @TypeOf(lhs);
            if (comptime getOpMapping(Lhs, op, method_names, 2)) |name| {
                return @field(Lhs, name)(lhs, rhs);
            }
            return ctx.sub_ctx.evalBinOp(lhs, op, rhs);
        }
    };
}

inline fn castStringTo(comptime T: type, comptime str: []const u8) T {
    comptime switch (@typeInfo(T)) {
        .@"enum" => return @field(T, str),
        .array => return str[0..].*,
        .pointer => |pointer| ptr: {
            var sentinel: ?u8 = null;
            switch (pointer.size) {
                .Slice, .Many => {
                    if (pointer.child != u8) break :ptr;
                    if (pointer.sentinel) |maybe_ptr| blk: {
                        const ptr = maybe_ptr orelse break :blk;
                        sentinel = @as(*align(1) const pointer.child, @ptrCast(ptr)).*;
                    }
                },
                .One => switch (@typeInfo(pointer.child)) {
                    .@"enum" => return &@field(pointer.child, str),
                    .array => |array| {
                        if (array.child != u8) break :ptr;
                        if (array.sentinel) |maybe_ptr| blk: {
                            const ptr = maybe_ptr orelse break :blk;
                            sentinel = @as(*align(1) const pointer.child, @ptrCast(ptr)).*;
                        }
                    },
                },
                else => break :ptr,
            }
            return if (sentinel) |s|
                str[0..] ++ &[_:s]u8{}
            else
                str[0..];
        },
        else => {},
    };
    @compileError("Can't cast to type " ++ @typeName(T));
}

inline fn getOpMapping(
    comptime Operand: type,
    comptime op: []const u8,
    comptime mapping: anytype,
    comptime arity: comptime_int,
) ?[]const u8 {
    comptime {
        if (!@hasField(@TypeOf(mapping), op)) return null;
        const entry = @field(mapping, op);

        const T = util.ImplicitDeref(Operand);
        switch (@typeInfo(T)) {
            .@"struct", .@"union", .@"enum", .@"opaque" => {},
            else => return null,
        }

        const is_single = @TypeOf(entry) == []const u8;
        const entry_list: []const []const u8 = if (is_single) &.{entry} else entry;

        var idx: comptime_int = entry_list.len;

        const result = for (entry_list, 0..) |name, i| {
            if (!@hasDecl(T, name)) continue;
            const Fn = util.ImplicitDeref(@TypeOf(@field(T, name)));
            if (@typeInfo(Fn).@"fn".param_types.len != arity) continue;
            idx = i + 1;
            break name;
        } else null;

        for (entry_list[idx..]) |next_name| {
            if (!@hasDecl(T, next_name)) continue;
            const NextFn = util.ImplicitDeref(@TypeOf(@field(T, next_name)));
            if (@typeInfo(NextFn).@"fn".param_types.len != arity) continue;
            @compileError("Ambiguous resolution between method '" ++ result.? ++ "' and '" ++ next_name ++ "'");
        }

        return result;
    }
}

fn DedupedMethodNames(comptime method_names: anytype) type {
    const info = @typeInfo(@TypeOf(method_names)).@"struct";

    const FieldAttrs = std.lang.Type.Struct.FieldAttributes;

    @setEvalBranchQuota(1000 + info.field_names.len * 10);
    var names: [info.field_names.len][]const u8 = undefined;
    var types: [info.field_names.len]type = undefined;
    var attrs: [info.field_names.len]FieldAttrs = undefined;

    for (
        info.field_names,
        info.field_types,
        info.field_attrs,
        &names,
        &types,
        &attrs,
    ) |old_name, OldT, old_attr, *name, *T, *attr| {
        const NewT = switch (@typeInfo(OldT)) {
            .pointer => |pointer| switch (@typeInfo(pointer.child)) {
                .int, .comptime_int => []const u8,
                .array => |array| switch (@typeInfo(array.child)) {
                    .int, .comptime_int => []const u8,
                    else => []const []const u8,
                },
                else => []const []const u8,
            },
            else => @compileError("Unexpected type " ++ @typeName(OldT)),
        };
        const old_value: NewT = old_attr.defaultValue(OldT).?;
        name.* = old_name;
        T.* = NewT;
        attr.* = .{
            .@"comptime" = true,
            .@"align" = null,
            .default_value_ptr = @ptrCast(&old_value),
        };
    }

    return @Struct(.auto, null, &names, &types, &attrs);
}

test context {
    const CustomNum = enum(i32) {
        _,

        pub inline fn from(value: anytype) @This() {
            return @fromBackingInt(@intCast(value));
        }

        pub inline fn add(self: @This(), other: @This()) @This() {
            return @fromBackingInt(@intCast(@backingInt(self) + @backingInt(other)));
        }
        pub inline fn sub(self: @This(), other: @This()) @This() {
            return @fromBackingInt(@intCast(@backingInt(self) - @backingInt(other)));
        }
        pub inline fn neg(self: @This()) @This() {
            return @fromBackingInt(@intCast(-@backingInt(self)));
        }
        pub inline fn mul(self: @This(), other: anytype) @This() {
            return @fromBackingInt(@intCast(@backingInt(self) * switch (@TypeOf(other)) {
                @This() => @backingInt(other),
                comptime_int => other,
                else => |Other| @compileError("Unexpected type " ++ @typeName(Other)),
            }));
        }
    };

    const fm_ctx = context(cm.ctx.simple.context({}), .{
        .@"+" = "add",
        .@"-" = &.{ "sub", "neg" },
        .@"*" = "mul",
        .evalFuncCall = "mul",
    });
    try std.testing.expectEqual(CustomNum.from(2), cm.eval("a + -b - c", fm_ctx, .{
        .a = @as(CustomNum, @fromBackingInt(@intCast(22))),
        .b = @as(CustomNum, @fromBackingInt(@intCast(9))),
        .c = @as(CustomNum, @fromBackingInt(@intCast(11))),
    }));
    try std.testing.expectEqual(CustomNum.from(77), cm.eval("a(b) * 1", fm_ctx, .{
        .a = @as(CustomNum, @fromBackingInt(@intCast(11))),
        .b = @as(CustomNum, @fromBackingInt(@intCast(7))),
    }));
    try std.testing.expectEqual(CustomNum.from(62), cm.eval("num(31)(2)", fm_ctx, .{
        .num = CustomNum.from,
    }));
}

test "fn call with mixed comptime literal and runtime args" {
    const FakeVector = struct {
        x: f32,
        y: f32,
        z: f32,

        pub fn init(x_: f32, y_: f32, z_: f32) @This() {
            return .{ .x = x_, .y = y_, .z = z_ };
        }

        pub fn add(self: @This(), rhs: f32) @This() {
            return .{
                .x = self.x + rhs,
                .y = self.y + rhs,
                .z = self.z + rhs,
            };
        }
    };

    const fake_ctx = context(
        cm.ctx.simple.context(
            cm.ctx.namespace.context(.{
                .v3 = FakeVector.init,
            }),
        ),
        .{ .@"+" = "add" },
    );

    {
        const x: f32 = 12.0;
        const result = try cm.eval("v3(0,0,x) + x", fake_ctx, .{ .x = x });
        try std.testing.expectEqualDeep(FakeVector.init(12, 12, 24), result);
    }

    {
        var x: f32 = 12.0;
        x += 3;
        const result = try cm.eval("v3(0,0,x) + x", fake_ctx, .{ .x = x });
        try std.testing.expectEqualDeep(FakeVector.init(15, 15, 30), result);
    }
}
