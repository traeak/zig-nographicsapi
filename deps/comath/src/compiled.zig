const std = @import("std");

const cm = @import("comath.zig");
const parse = @import("parse.zig");
const Tokenizer = @import("Tokenizer.zig");

const util = @import("util");

/// Compiles `expr_src` as an expression, wherein the operations are defined by `ctx`.
pub fn expr(
    /// Must be a value of a type with a namespace with the list of methods that will be described following this paragraph.
    /// Each declaration must be present as described, or must be set to void, i.e. `pub const <name> = {};`
    /// + `matchUnOp: fn (comptime str: []const u8) callconv(.Inline) bool`
    ///     Function receiving a string of symbols, which should return true for any string of
    ///     symbols matching a recognized unary operator.
    ///
    /// + `matchBinOp: fn (comptime str: []const u8) callconv(.Inline) bool`
    ///     Function receiving a string of symbols, which should return true for any string of
    ///     symbols matching a recognized binary operator.
    ///
    /// + `orderBinOp: fn (comptime lhs: []const u8, comptime rhs: []const u8) callconv(.Inline) ?operator.Order`
    ///     Function receiving a pair of strings representing two chained binary operators,
    ///     wherein for both `matchBinOp(s) = true`, which should return the ordering relation
    ///     between the two.
    ///     For example, given an expression `a $ b @ c`:
    ///         - With `orderBinOp("$", "@") = .lt`           => the expression will be interpreted as `a $ (b @ c)`.
    ///         - With `orderBinOp("$", "@") = .gt`           => the expression will be interpreted as `(a $ b) @ c`.
    ///         - With `orderBinOp("$", "@") = .incompatible` => the expression will be considered invalid, and issue an error about unchainable operators.
    ///         - With `orderBinOp("$", "@") = null`          => the expression will be considered invalid, and may issue an error about an undefined operator relationship.
    ///     NOTE: returning null may be used as a hint to a "wrapper context" that it should use a fallback if possible.
    ///
    /// + `EvalNumberLiteral: fn (comptime src: []const u8) type`
    /// + `evalNumberLiteral: fn (comptime src: []const u8) EvalNumberLiteral(src)`
    ///     Returns the value that should be used to represent a number literal.
    ///
    /// + `EvalIdent: fn (comptime ident: []const u8) type`
    /// + `evalIdent: fn (ctx: @This(), comptime ident: []const u8) !EvalIdent(ident)`
    ///     Returns the value of the identifier. `EvalIdent(ident) = noreturn` will make the
    ///     function get the value from the field of the `inputs` with the matching name.
    ///
    /// + `EvalProperty: fn (comptime Lhs: type, comptime field: []const u8) type`
    /// + `evalProperty: fn (ctx: @This(), lhs: anytype, comptime field: []const u8) !EvalProperty(@TypeOf(lhs), field)`
    ///     Returns the value that should result from an expression `lhs.field`.
    ///
    /// + `EvalIndexAccess: fn (comptime Lhs: type, comptime Rhs: type) type`
    /// + `evalIndexAccess: fn (ctx: @This(), lhs: anytype, rhs: anytype) !EvalIndexAccess(@TypeOf(lhs), @TypeOf(rhs))`
    ///     Returns the value that should result from an expression `lhs[rhs...]`, where each of the elements of the tuple `rhs`
    ///     are the indices that should be used to access `lhs`.
    ///
    /// + `EvalFuncCall: fn (comptime Callee: type, comptime Args: type) type`
    /// + `evalFuncCall: fn (ctx: @This(), callee: anytype, args: anytype) !EvalFuncCall(@TypeOf(callee), @TypeOf(args))`
    ///     Returns the value that should result from an expression `callee(args...)`, where each of the elements of the tuple `args`
    ///     are the arguments that should be passed to `callee`.
    ///
    /// + `EvalMethodCall: fn (comptime SelfParam: type, comptime method: []const u8, comptime Args: type) type`
    /// + `evalMethodCall: fn (ctx: @This(), self_param: anytype, comptime method: []const u8, args: anytype) !EvalMethodCall(@TypeOf(self_param), method, @TypeOf(args))`
    ///     Returns the value that should result from an expression `self_param.method(args...)`, where each of the elements of the tuple `args`
    ///     are the arguments that should be passed to `method`, possibly in addition to the `self_param`.
    ///     This is a specialisation of `expr(args...)` which specifically represents `expr.access(args...)` which would otherwise
    ///     be interpreted as `(expr.access)(args...)`. This allows for the context to define method-like calls more easily.
    ///     If `EvalMethodCall(S, m, A) = noreturn`, it will be interpreted as `(self_param.method)(args...)`,
    ///     falling back to `evalFuncCall(evalProperty(self_param, method), args)`.
    ///
    /// + `EvalUnOp: fn (comptime op: []const u8, comptime T: type) type`
    /// + `evalUnOp: fn (ctx: @This(), comptime op: []const u8, value: anytype) !EvalUnOp(op, @TypeOf(value))`
    ///     Returns the value that should result from an expression `op value`.
    ///
    /// + `EvalBinOp: fn (comptime Lhs: type, comptime op: []const u8, comptime Rhs: type) type`
    /// + `evalBinOp: fn (ctx: @This(), lhs: anytype, comptime op: []const u8, rhs: anytype) !EvalBinOp(@TypeOf(lhs), op, @TypeOf(rhs))`
    ///     Returns the value that should result from an expression `lhs op rhs`.
    /// NOTE: type methods may return `noreturn` as a hint to a "wrapper context" that it should use a fallback if possible.
    ctx: anytype,
    /// Source code representing an expression generally comparable in structure to
    /// mathematical and programming notation in most C-like programming languages.
    /// A simple representation of the accepted grammar is:
    /// ```
    ///   Expr = $Ident | $Integer | $Char | $Float | $Group | $FieldAccess | $IndexAccess | $FuncCall | $UnOp | $BinOp
    ///
    ///   Ident = $IdentStartSymbol $IdentSymbol*
    ///   Integer # String of characters tokenized as a zig integer literal
    ///   Float   # String of characters tokenized as a zig float literal
    ///
    ///   FieldAccess = $Expr '.' ($IdentSymbol|$OperatorSymbol)+
    ///   Group = '(' $Expr ')'
    ///   UnOp = $Operator $Expr
    ///   BinOp = $Expr $Operator $Expr
    ///   IndexAccess = $Expr '[' $ExprList ']'
    ///   FuncCall    = $Expr '(' $ExprList ')'
    ///
    ///   ExprList = ($Expr ','?)*
    ///
    ///   Operator = $OperatorSymbol+
    ///   OperatorSymbol = ['!' '#' '$' '%' '&' '*' '+' '-' '/' '<' '=' '>' '?' '@' '~' '^' '|' ':']
    ///
    ///   IdentStartSymbol = ['A'-'Z' 'a'-'z' '_']
    ///   IdentSymbol = $IdentStartSymbol | ['0'-'9']
    /// ```
    comptime expr_src: []const u8,
) Expr(@TypeOf(ctx), expr_src) {
    return .with(ctx);
}

pub fn Expr(
    comptime Ctx: type,
    comptime expr_src: []const u8,
) type {
    const Ns = util.NamespaceOf(Ctx) orelse @compileError(std.fmt.comptimePrint(
        "Expected struct/union/enum, or pointer to struct/union/enum/opaque, instead got `{s}`, which has no associated namespace",
        .{@typeName(Ctx)},
    ));

    const root = parse.parseExpr(
        expr_src,
        if (@TypeOf(Ns.matchUnOp) != void) Ns.matchUnOp else null,
        if (@TypeOf(Ns.matchBinOp) != void) Ns.matchBinOp else null,
        if (@TypeOf(Ns.orderBinOp) != void) Ns.orderBinOp else null,
    );

    return struct {
        ctx: Ctx,
        const Self = @This();

        pub fn with(ctx: Ctx) Self {
            return .{ .ctx = ctx };
        }

        /// Returns the type used to represent the result of evaluating the `expr` with the given context and inputs.
        pub fn Eval(comptime Inputs: type) type {
            return Parameterized(Inputs).EvalImpl(root);
        }

        pub inline fn eval(
            self: Self,
            /// a non-tuple struct literal whose field names correspond to identifiers in the `expr_src` code.
            inputs: anytype,
        ) !Eval(@TypeOf(inputs)) {
            return Parameterized(@TypeOf(inputs)).evalUnusedMode(.no_unused, self.ctx, inputs);
        }

        pub inline fn evalWithUnused(
            self: Self,
            /// a non-tuple struct literal whose field names correspond to identifiers in the `expr_src` code.
            inputs: anytype,
        ) !Eval(@TypeOf(inputs)) {
            return Parameterized(@TypeOf(inputs)).evalUnusedMode(.allow_unused, self.ctx, inputs);
        }

        fn Parameterized(comptime Inputs: type) type {
            const InputTag = input_tag: {
                const inputs_s_info = @typeInfo(Inputs).@"struct";

                var field_names: [inputs_s_info.field_names.len][]const u8 = undefined;
                var field_values: [inputs_s_info.field_names.len]u64 = undefined;
                for (
                    inputs_s_info.field_names,
                    &field_names,
                    &field_values,
                    0..,
                ) |old_name, *name, *value, i| {
                    name.* = old_name;
                    value.* = i;
                }

                break :input_tag @Enum(u64, .exhaustive, &field_names, &field_values);
            };

            return struct {
                inline fn evalUnusedMode(
                    comptime unused_mode: enum { no_unused, allow_unused },
                    ctx: Ctx,
                    inputs: Inputs,
                ) !Eval(@TypeOf(inputs)) {
                    comptime switch (unused_mode) {
                        .allow_unused => {},
                        .no_unused => if (@typeInfo(Inputs).@"struct".field_names.len != 0) {
                            const EvalIdent = switch (@TypeOf(Ns.EvalIdent)) {
                                void => struct {
                                    fn EvalIdentAlwaysNoreturn(comptime _: []const u8) type {
                                        return noreturn;
                                    }
                                }.EvalIdentAlwaysNoreturn,
                                else => Ns.EvalIdent,
                            };
                            if (analyzeInputs(expr_src, InputTag, EvalIdent)) |err| {
                                @compileError(@errorName(err));
                            }
                        },
                    };

                    return evalImpl(root, ctx, inputs);
                }

                fn EvalImpl(comptime expr_node: parse.ExprNode) type {
                    return switch (expr_node) {
                        .null => noreturn,
                        .err => noreturn,
                        .ident => |ident| blk: {
                            if (@TypeOf(Ns.EvalIdent) != void) {
                                const IdentType = Ns.EvalIdent(ident);
                                if (IdentType != noreturn) break :blk IdentType;
                            }
                            if (!@hasField(Inputs, ident)) {
                                break :blk noreturn;
                            }
                            break :blk @FieldType(Inputs, ident);
                        },
                        .number => |number| blk: {
                            const EvalNumberLiteral = Ns.EvalNumberLiteral;
                            if (@TypeOf(EvalNumberLiteral) == void) {
                                break :blk noreturn;
                            }
                            break :blk EvalNumberLiteral(number);
                        },
                        .group => |group| EvalImpl(group.*),
                        .field_access => |fa| blk: {
                            const Lhs = EvalImpl(fa.accessed);
                            break :blk Ns.EvalProperty(Lhs, util.dedupeScalarSlice(u8, fa.accessor[0..].*));
                        },
                        .index_access => |ia| blk: {
                            const Lhs = EvalImpl(ia.accessed);
                            const Rhs = EvalExprTupleImpl(ia.accessor);
                            break :blk Ns.EvalIndexAccess(Lhs, Rhs);
                        },
                        .func_call => |fc| blk: {
                            switch (fc.callee) {
                                .field_access => |fa| method: {
                                    const SelfParam = EvalImpl(fa.accessed);
                                    const Args = EvalExprTupleImpl(fc.args);
                                    const MethodResult = Ns.EvalMethodCall(SelfParam, fa.accessor, Args);
                                    if (MethodResult == noreturn) break :method;
                                    break :blk MethodResult;
                                },
                                else => {},
                            }
                            const Callee = EvalImpl(fc.callee);
                            const Args = EvalExprTupleImpl(fc.args);
                            break :blk Ns.EvalFuncCall(Callee, Args);
                        },
                        .un_op => |un| blk: {
                            const Val = EvalImpl(un.val);
                            break :blk Ns.EvalUnOp(un.op, Val);
                        },
                        .bin_op => |bin| blk: {
                            const Lhs = EvalImpl(bin.lhs);
                            const Rhs = EvalImpl(bin.rhs);
                            break :blk Ns.EvalBinOp(Lhs, bin.op, Rhs);
                        },
                    };
                }

                inline fn evalImpl(
                    comptime expr_node: parse.ExprNode,
                    ctx: Ctx,
                    inputs: Inputs,
                ) !EvalImpl(expr_node) {
                    return switch (comptime expr_node) {
                        .null => @compileError("Incomplete AST (encountered null expression)"),
                        .err => |err| @compileError(err),

                        .ident => |ident| blk: {
                            const EvalIdent = Ns.EvalIdent;
                            const evalIdent = Ns.evalIdent;
                            if (@TypeOf(EvalIdent) != void) {
                                if (@TypeOf(evalIdent) == void) @compileError(
                                    "Can't have a type & value method pair be mismatched",
                                );
                                if (EvalIdent(ident) != noreturn) {
                                    break :blk ctx.evalIdent(ident);
                                }
                            }
                            break :blk @field(inputs, ident);
                        },
                        .number => |number| blk: {
                            const EvalNumberLiteral = Ns.EvalNumberLiteral;
                            const evalNumberLiteral = Ns.evalNumberLiteral;
                            if (@TypeOf(EvalNumberLiteral) != void) {
                                if (@TypeOf(evalNumberLiteral) == void) @compileError(
                                    "Can't have a type & value method pair be mismatched",
                                );
                                if (EvalNumberLiteral(number) != noreturn) {
                                    break :blk evalNumberLiteral(number);
                                }
                            }
                            @compileError("Lacking implementation for evalNumberLiteral");
                        },
                        .group => |group| evalImpl(group.*, ctx, inputs),
                        .field_access => |fa| blk: {
                            const Lhs = EvalImpl(fa.accessed);
                            const lhs: Lhs = try evalImpl(fa.accessed, ctx, inputs);
                            break :blk ctx.evalProperty(lhs, util.dedupeScalarSlice(u8, fa.accessor[0..].*));
                        },
                        .index_access => |ia| blk: {
                            const Lhs = EvalImpl(ia.accessed);
                            const Args = EvalExprTupleImpl(ia.accessor);
                            const lhs: Lhs = try evalImpl(ia.accessed, ctx, inputs);
                            break :blk evalArgsAndCall(ia.accessor, ctx, inputs, "evalIndexAccess", .{ ctx, lhs }, Ns.EvalIndexAccess(Lhs, Args));
                        },
                        .func_call => |fc| blk: {
                            const Args = EvalExprTupleImpl(fc.args);
                            switch (fc.callee) {
                                .field_access => |fa| method: {
                                    const SelfParam = EvalImpl(fa.accessed);
                                    const MethodRet = Ns.EvalMethodCall(SelfParam, fa.accessor, Args);
                                    if (MethodRet == noreturn) break :method;
                                    const self_param: SelfParam = try evalImpl(fa.accessed, ctx, inputs);
                                    break :blk evalArgsAndCall(fc.args, ctx, inputs, "evalMethodCall", .{ ctx, self_param, fa.accessor }, MethodRet);
                                },
                                else => {},
                            }

                            const Callee = EvalImpl(fc.callee);
                            const callee: Callee = try evalImpl(fc.callee, ctx, inputs);
                            break :blk evalArgsAndCall(fc.args, ctx, inputs, "evalFuncCall", .{ ctx, callee }, Ns.EvalFuncCall(Callee, Args));
                        },
                        .un_op => |un| blk: {
                            const Val = EvalImpl(un.val);
                            const val: Val = try evalImpl(un.val, ctx, inputs);
                            break :blk ctx.evalUnOp(un.op, val);
                        },
                        .bin_op => |bin| blk: {
                            const Lhs = EvalImpl(bin.lhs);
                            const lhs: Lhs = try evalImpl(bin.lhs, ctx, inputs);

                            const Rhs = EvalImpl(bin.rhs);
                            const rhs: Rhs = try evalImpl(bin.rhs, ctx, inputs);

                            break :blk ctx.evalBinOp(lhs, bin.op, rhs);
                        },
                    };
                }

                fn EvalExprTupleImpl(comptime list: []const parse.ExprNode) type {
                    var field_types: [list.len]type = undefined;
                    for (list, &field_types) |arg, *T| {
                        T.* = EvalImpl(arg);
                    }
                    return @Tuple(&field_types);
                }

                /// Evaluates `list` element-wise and forwards the result tuple to `Ctx.<method>`,
                /// prefixed by `prefix`. Building the tuple via an anonymous literal lets Zig
                /// auto-promote comptime-known elements to comptime fields, so mixed
                /// comptime/runtime arg lists work without forcing comptime evaluation of the
                /// whole tuple.
                inline fn evalArgsAndCall(
                    comptime list: []const parse.ExprNode,
                    ctx: Ctx,
                    inputs: Inputs,
                    comptime method: []const u8,
                    prefix: anytype,
                    comptime Ret: type,
                ) !Ret {
                    const f = @field(Ctx, method);
                    return switch (comptime list.len) {
                        0 => try @call(.auto, f, prefix ++ .{.{}}),
                        1 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                        }}),
                        2 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                            try evalImpl(list[1], ctx, inputs),
                        }}),
                        3 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                            try evalImpl(list[1], ctx, inputs),
                            try evalImpl(list[2], ctx, inputs),
                        }}),
                        4 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                            try evalImpl(list[1], ctx, inputs),
                            try evalImpl(list[2], ctx, inputs),
                            try evalImpl(list[3], ctx, inputs),
                        }}),
                        5 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                            try evalImpl(list[1], ctx, inputs),
                            try evalImpl(list[2], ctx, inputs),
                            try evalImpl(list[3], ctx, inputs),
                            try evalImpl(list[4], ctx, inputs),
                        }}),
                        6 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                            try evalImpl(list[1], ctx, inputs),
                            try evalImpl(list[2], ctx, inputs),
                            try evalImpl(list[3], ctx, inputs),
                            try evalImpl(list[4], ctx, inputs),
                            try evalImpl(list[5], ctx, inputs),
                        }}),
                        7 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                            try evalImpl(list[1], ctx, inputs),
                            try evalImpl(list[2], ctx, inputs),
                            try evalImpl(list[3], ctx, inputs),
                            try evalImpl(list[4], ctx, inputs),
                            try evalImpl(list[5], ctx, inputs),
                            try evalImpl(list[6], ctx, inputs),
                        }}),
                        8 => try @call(.auto, f, prefix ++ .{.{
                            try evalImpl(list[0], ctx, inputs),
                            try evalImpl(list[1], ctx, inputs),
                            try evalImpl(list[2], ctx, inputs),
                            try evalImpl(list[3], ctx, inputs),
                            try evalImpl(list[4], ctx, inputs),
                            try evalImpl(list[5], ctx, inputs),
                            try evalImpl(list[6], ctx, inputs),
                            try evalImpl(list[7], ctx, inputs),
                        }}),
                        else => @compileError("Too many arguments (max 8): " ++ method),
                    };
                }
            };
        }
    };
}

/// Returns null if everything is fine, and an error message otherwise.
inline fn analyzeInputs(
    comptime expr_src: []const u8,
    comptime InputTag: type,
    comptime EvalIdent: fn (comptime []const u8) type,
) ?[]const u8 {
    comptime {
        var unused_set: std.EnumSet(InputTag) = .full;
        var shadow_set: std.EnumSet(InputTag) = .empty;
        var tokenizer: Tokenizer = .init;
        @setEvalBranchQuota(expr_src.len * 100);
        while (true) switch (tokenizer.next(expr_src)) {
            .eof => break,
            .ident => |ident| {
                const global_ident = EvalIdent(ident) != noreturn;
                const input_ident = @hasField(InputTag, ident);

                if (input_ident) {
                    const tag = @field(InputTag, ident);
                    if (global_ident) shadow_set.insert(tag);
                    unused_set.remove(tag);
                }
            },
            else => {},
        };

        var err_str: []const u8 = "";

        for (.{ unused_set, shadow_set }, .{
            if (unused_set.count() > 1) "Unused inputs: " else "Unused input: ",
            if (shadow_set.count() > 1) "Inputs shadowing context identifiers: " else "Input shadowing context identifier: ",
        }) |set, err_msg_prefix| {
            if (set.count() == 0) continue;
            err_str = err_str ++ err_msg_prefix;

            var iter = set.iterator();
            @setEvalBranchQuota(set.count() + 1);
            var need_comma = false;
            while (iter.next()) |unused| {
                const comma = if (need_comma) ", " else "";
                need_comma = true;
                err_str = err_str ++ comma ++ @tagName(unused);
            }

            err_str = err_str ++ "\n";
        }

        if (err_str.len == 0) return null;
        return err_str;
    }
}
