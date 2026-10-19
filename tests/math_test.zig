const std = @import("std");
const comath = @import("comath");
const math = @import("NoGraphicsAPIUtility").math;
const testing = std.testing;

const float2 = math.float2;
const float3 = math.float3;
const float4 = math.float4;
const int3 = math.int3;
const uint3 = math.uint3;
const float3x3 = math.float3x3;
const float3x4 = math.float3x4;
const float4x4 = math.float4x4;
const quaternion = math.quaternion;

comptime {
    const sizes = .{
        .{ float2, 8 },          .{ float3, 12 },         .{ float4, 16 },        .{ math.int2, 8 },      .{ int3, 12 },
        .{ math.int4, 16 },      .{ math.uint2, 8 },      .{ uint3, 12 },         .{ math.uint4, 16 },    .{ float3x3, 36 },
        .{ float3x4, 48 },       .{ float4x4, 64 },       .{ quaternion, 16 },    .{ math.float16_t, 2 }, .{ math.float16_t2, 4 },
        .{ math.float16_t3, 6 }, .{ math.float16_t4, 8 }, .{ math.int16_t2, 4 },  .{ math.int16_t3, 6 },  .{ math.int16_t4, 8 },
        .{ math.uint16_t2, 4 },  .{ math.uint16_t3, 6 },  .{ math.uint16_t4, 8 },
    };
    for (sizes) |entry| {
        if (@sizeOf(entry[0]) != entry[1]) @compileError(@typeName(entry[0]) ++ " has the wrong size");
        if (@alignOf(entry[0]) > 4) @compileError(@typeName(entry[0]) ++ " has the wrong alignment");
    }
}

const matrix_a: float4x4 = .{ .rows = .{
    .{ .x = 1, .y = 2, .z = 3, .w = 4 },
    .{ .x = 5, .y = 6, .z = 7, .w = 8 },
    .{ .x = 9, .y = 10, .z = 11, .w = 12 },
    .{ .x = 13, .y = 14, .z = 15, .w = 16 },
} };
const matrix_b: float4x4 = .{ .rows = .{
    .{ .x = 17, .y = 18, .z = 19, .w = 20 },
    .{ .x = 21, .y = 22, .z = 23, .w = 24 },
    .{ .x = 25, .y = 26, .z = 27, .w = 28 },
    .{ .x = 29, .y = 30, .z = 31, .w = 32 },
} };
const matrix_product: float4x4 = .{ .rows = .{
    .{ .x = 250, .y = 260, .z = 270, .w = 280 },
    .{ .x = 618, .y = 644, .z = 670, .w = 696 },
    .{ .x = 986, .y = 1028, .z = 1070, .w = 1112 },
    .{ .x = 1354, .y = 1412, .z = 1470, .w = 1528 },
} };
const matrix_vector: float4 = .{ .x = 2, .y = -1, .z = 3, .w = 4 };

fn expectEql(expected: anytype, actual: @TypeOf(expected)) !void {
    if (!math.eql(expected, actual)) {
        std.debug.print("expected {any}, found {any}\n", .{ expected, actual });
        return error.TestExpectedEqual;
    }
}

fn expectNear(expected: anytype, actual: anytype) !void {
    if (@TypeOf(actual) == f32) return testing.expectApproxEqAbs(@as(f32, expected), actual, 1e-5);
    const diff = math.abs(math.sub(@as(@TypeOf(actual), expected), actual));
    if (!math.eql(math.min(diff, math.splat(@TypeOf(diff), 1e-5)), diff)) {
        std.debug.print("expected {any}, found {any}\n", .{ expected, actual });
        return error.TestExpectedApproxEqAbs;
    }
}

fn projectedDepth(projection: float4x4, distance: f32) f32 {
    const clip = math.mul(projection, float4{ .z = -distance, .w = 1 });
    return clip.z / clip.w;
}

test "constants and conversions" {
    try expectEql(float3{ .x = 2, .y = 2, .z = 2 }, math.splat(float3, 2));
    try expectEql(float4{ .x = 1, .y = 2, .z = 3, .w = 4 }, math.extend(float3{ .x = 1, .y = 2, .z = 3 }, 4));
    try expectEql(float2{ .x = 1, .y = 2 }, math.convert(float2, float4{ .x = 1, .y = 2, .z = 3, .w = 4 }));
    try expectEql(int3{ .x = 8, .y = 12, .z = 16 }, math.convert(int3, uint3{ .x = 8, .y = 12, .z = 16 }));
    try expectEql(int3{ .x = 1, .y = -2, .z = 3 }, math.convert(int3, float3{ .x = 1.75, .y = -2.5, .z = 3 }));
    try expectEql(float3{ .x = 8, .y = 12, .z = 16 }, math.convert(float3, uint3{ .x = 8, .y = 12, .z = 16 }));
    try expectEql(quaternion{ .x = 1, .y = 2, .z = 3, .w = 4 }, math.convert(quaternion, float4{ .x = 1, .y = 2, .z = 3, .w = 4 }));
    try testing.expectEqual(1.0, math.identity3x3.rows[1].y);
    try testing.expectEqual(0.0, math.identity3x4.rows[2].w);
    try testing.expectEqual(1.0, math.identity4x4.rows[3].w);
    try testing.expectEqual(0xc0003c00, math.packHalf2x16(.{ .x = 1, .y = -2 }));
}

test "vectors" {
    var value: float3 = .{ .x = 1, .y = -2, .z = 3 };
    value = math.add(value, float3{ .x = 2, .y = 4, .z = 6 });
    value = math.mul(value, 0.5);
    try expectEql(float3{ .x = 1.5, .y = 1, .z = 4.5 }, value);
    try expectEql(float3{ .x = 5, .y = 7, .z = 9 }, math.add(float3{ .x = 1, .y = 2, .z = 3 }, float3{ .x = 4, .y = 5, .z = 6 }));
    try testing.expectEqual(32.0, math.dot(float3{ .x = 1, .y = 2, .z = 3 }, float3{ .x = 4, .y = 5, .z = 6 }));
    try expectEql(float3{ .z = 1 }, math.cross(.{ .x = 1 }, .{ .y = 1 }));
    try expectEql(float3{ .x = 1.5, .y = 0, .z = 4.5 }, math.min(value, float3{ .x = 2, .y = 0, .z = 5 }));
    try expectEql(float3{ .x = 2, .y = 1, .z = 5 }, math.max(value, float3{ .x = 2, .y = 0, .z = 5 }));
    try expectEql(float3{ .x = 1.5, .y = 1.25, .z = 4 }, math.clamp(value, 1.25, 4.0));
    try expectEql(float3{ .x = 1, .y = 2, .z = 3 }, math.abs(float3{ .x = -1, .y = 2, .z = -3 }));
    try expectEql(int3{ .x = 1, .y = 2, .z = 3 }, math.abs(int3{ .x = -1, .y = 2, .z = -3 }));

    var negative: float2 = .{ .x = -0.0, .y = @bitCast(@as(u32, 0xffc12345)) };
    _ = &negative;
    const positive = math.abs(negative);
    try testing.expectEqual(0, @as(u32, @bitCast(positive.x)));
    try testing.expectEqual(0x7fc12345, @as(u32, @bitCast(positive.y)));

    try expectEql(float3{ .x = 1, .y = -2, .z = 2 }, math.floor(float3{ .x = 1.75, .y = -1.25, .z = 2 }));
    try expectEql(float3{ .x = 2, .y = -1, .z = 2 }, math.ceil(float3{ .x = 1.25, .y = -1.75, .z = 2 }));
    try expectEql(float3{ .x = 1, .y = -2, .z = 3 }, math.round(float3{ .x = 1.25, .y = -1.75, .z = 2.5 }));
    try expectNear(0.5, math.rsqrt(4));
    try expectNear(7.0, math.length(float3{ .x = 2, .y = 3, .z = 6 }));
    try expectNear(float3{ .x = 0, .y = 0.6, .z = 0.8 }, math.normalize(float3{ .x = 0, .y = 3, .z = 4 }));
    try expectNear(float3{ .x = 0.5, .y = 2.5, .z = 4.5 }, math.lerp(float3{ .x = 0, .y = 2, .z = 4 }, float3{ .x = 2, .y = 4, .z = 6 }, 0.25));
    try expectNear(0.5, math.smoothstep(0, 1, @as(f32, 0.5)));
    try expectNear(float2{ .x = 0, .y = 0.5 }, math.smoothstep(0, 1, float2{ .x = -1, .y = 0.5 }));

    try expectEql(uint3{ .x = 2, .y = 3, .z = 4 }, math.div(uint3{ .x = 8, .y = 12, .z = 16 }, 4));
    try expectEql(int3{ .x = -2, .y = 3, .z = -4 }, math.div(int3{ .x = -8, .y = 12, .z = -17 }, 4));

    var wide: float4 = .{ .x = 1, .y = -2, .z = 4, .w = 8 };
    const other: float4 = .{ .x = 2, .y = 4, .z = -1, .w = 0.5 };
    try expectEql(float4{ .x = 3, .y = 2, .z = 3, .w = 8.5 }, math.add(wide, other));
    try expectEql(float4{ .x = -1, .y = -6, .z = 5, .w = 7.5 }, math.sub(wide, other));
    try expectEql(float4{ .x = 2, .y = -8, .z = -4, .w = 4 }, math.mul(wide, other));
    try expectEql(float4{ .x = 0.5, .y = -0.5, .z = -4, .w = 16 }, math.div(wide, other));
    try testing.expectEqual(-6.0, math.dot(wide, other));
    wide = math.add(math.mul(2, wide), float4{ .x = 1, .y = 2, .z = 3, .w = 4 });
    try expectEql(float4{ .x = 3, .y = -2, .z = 11, .w = 20 }, wide);
    try expectEql(float4{ .x = -3, .y = 2, .z = -11, .w = -20 }, math.neg(wide));
}

test "matrices" {
    var a = matrix_a;
    _ = &a;
    try expectEql(matrix_product, math.mul(a, matrix_b));
    try expectEql(float4{ .x = 25, .y = 57, .z = 89, .w = 121 }, math.mul(a, matrix_vector));
    try expectEql(float4{ .x = 76, .y = 84, .z = 92, .w = 100 }, math.mul(matrix_vector, a));
    try expectEql(a, math.mul(math.identity4x4, a));
    try expectEql(a, math.mul(a, math.identity4x4));
    try expectEql(math.mul(math.mul(a, matrix_b), matrix_vector), math.mul(a, math.mul(matrix_b, matrix_vector)));
    try expectEql(math.mul(matrix_vector, math.mul(a, matrix_b)), math.mul(math.mul(matrix_vector, a), matrix_b));
    try expectEql(a, math.transpose(math.transpose(a)));
    try testing.expectEqual(5.0, math.transpose(a).rows[0].y);
    try expectEql(float4{ .x = 5, .y = 7, .z = 9, .w = 1 }, math.mul(math.translation(.{ .x = 4, .y = 5, .z = 6 }), float4{ .x = 1, .y = 2, .z = 3, .w = 1 }));
    try expectEql(float4{ .x = 1, .y = 2, .z = 3, .w = 0 }, math.mul(math.translation(.{ .x = 4, .y = 5, .z = 6 }), float4{ .x = 1, .y = 2, .z = 3, .w = 0 }));
    try expectEql(a.rows[2], math.toFloat3x4(a).rows[2]);

    const translated = math.setTranslation(math.identity3x4, .{ .x = 4, .y = 5, .z = 6 });
    try expectEql(float3x4{ .rows = .{ .{ .x = 1, .w = 4 }, .{ .y = 1, .w = 5 }, .{ .z = 1, .w = 6 } } }, translated);
    try expectEql(float3{ .x = 5, .y = 7, .z = 9 }, math.transformPoint(translated, .{ .x = 1, .y = 2, .z = 3 }));
    try expectEql(float3{ .x = 1, .y = 2, .z = 3 }, math.transformVector(translated, .{ .x = 1, .y = 2, .z = 3 }));

    const transform = math.mul(math.mul(math.translation(.{ .x = 4, .y = 5, .z = 6 }), math.rotationY(math.half_pi)), math.scale(.{ .x = 2, .y = 3, .z = 4 }));
    const transformed = math.mul(transform, float4{ .x = 1, .w = 1 });
    try expectNear(float4{ .x = 4, .y = 5, .z = 4, .w = 1 }, transformed);
    try expectNear(float4{ .x = 1, .w = 1 }, math.mul(math.inverse(transform), transformed));

    const affine: float3x4 = .{ .rows = .{ .{ .x = 2, .w = 4 }, .{ .y = 3, .w = 5 }, .{ .z = 4, .w = 6 } } };
    const affine_point = math.transformPoint(affine, .{ .x = 1, .y = 2, .z = 3 });
    try expectEql(float3{ .x = 6, .y = 11, .z = 18 }, affine_point);
    try expectNear(float3{ .x = 1, .y = 2, .z = 3 }, math.transformPoint(math.inverse(affine), affine_point));
    const affine_rhs: float3x4 = .{ .rows = .{
        .{ .x = 1, .y = 2, .z = 0, .w = 7 },
        .{ .x = 0, .y = 1, .z = 3, .w = 8 },
        .{ .x = 4, .y = 0, .z = 1, .w = 9 },
    } };
    try expectEql(float3x4{ .rows = .{
        .{ .x = 2, .y = 4, .z = 0, .w = 18 },
        .{ .x = 0, .y = 3, .z = 9, .w = 29 },
        .{ .x = 16, .y = 0, .z = 4, .w = 42 },
    } }, math.mul(affine, affine_rhs));

    const linear: float3x3 = .{ .rows = .{ .{ .x = 2, .y = 1 }, .{ .y = 3, .z = 1 }, .{ .x = 1, .z = 4 } } };
    try expectNear(25.0, math.determinant(linear));
    const round_trip = math.mul(linear, math.inverse(linear));
    for (round_trip.rows, math.identity3x3.rows) |row, expected| try expectNear(expected, row);
    try expectEql(math.mul(math.transpose(linear), float3{ .x = 1, .y = 2, .z = 3 }), math.mul(float3{ .x = 1, .y = 2, .z = 3 }, linear));

    const view = math.lookAtRh(.{ .z = 5 }, .{}, .{ .y = 1 });
    try expectNear(float4{ .w = 1 }, math.mul(view, float4{ .z = 5, .w = 1 }));
    try expectNear(float4{ .z = -5, .w = 1 }, math.mul(view, float4{ .w = 1 }));

    const forward = math.perspectiveRhZo(math.half_pi, 1, 0.25, 100);
    try expectNear(0.0, projectedDepth(forward, 0.25));
    try expectNear(1.0, projectedDepth(forward, 100));
    const reverse = math.perspectiveRhZo(math.half_pi, 1, 100, 0.25);
    try expectNear(1.0, projectedDepth(reverse, 0.25));
    try expectNear(0.0, projectedDepth(reverse, 100));

    const ortho = math.orthoRhZo(-2, 2, -1, 1, 1, 11);
    try expectNear(float4{ .x = 1, .y = -1, .z = 0, .w = 1 }, math.mul(ortho, float4{ .x = 2, .y = -1, .z = -1, .w = 1 }));
    try expectNear(float4{ .z = 1, .w = 1 }, math.mul(ortho, float4{ .z = -11, .w = 1 }));
}

test "quaternions" {
    const quarter_turn = math.axisAngle(math.half_pi, .{ .y = 1 });
    try expectNear(float3{ .z = -1 }, math.mul(quarter_turn, float3{ .x = 1 }));
    try expectNear(float3{ .z = -1 }, math.mul(math.quaternionToMatrix(quarter_turn), float3{ .x = 1 }));
    try expectNear(float4{ .z = -1, .w = 1 }, math.mul(math.rotation(quarter_turn), float4{ .x = 1, .w = 1 }));
    const halfway = math.slerp(math.identity_quaternion, quarter_turn, 0.5);
    const diagonal = @sqrt(@as(f32, 0.5));
    try expectNear(float3{ .x = diagonal, .z = -diagonal }, math.mul(halfway, float3{ .x = 1 }));
    try expectNear(float3{ .x = 1, .y = 2, .z = 3 }, math.mul(math.inverse(quarter_turn), math.mul(quarter_turn, float3{ .x = 1, .y = 2, .z = 3 })));
    const composed = math.mul(quarter_turn, quarter_turn);
    try expectNear(float3{ .x = -1 }, math.mul(composed, float3{ .x = 1 }));
}

test "half packing" {
    try testing.expectEqual(0x80000000, math.packHalf2x16(.{ .x = 0, .y = -0.0 }));
    try testing.expectEqual(0x7bff, math.floatToHalfBits(65504));
    try testing.expectEqual(0x0400, math.floatToHalfBits(0x1p-14));
    try testing.expectEqual(0x0001, math.floatToHalfBits(0x1p-24));
    try testing.expectEqual(0x3c00, math.floatToHalfBits(1 + 0x1p-11));
    try testing.expectEqual(0x3c02, math.floatToHalfBits(1 + 3 * 0x1p-11));
    try testing.expectEqual(0x7c00, math.floatToHalfBits(std.math.inf(f32)));
    try testing.expectEqual(0xfc00, math.floatToHalfBits(-std.math.inf(f32)));
    const nan = math.floatToHalfBits(std.math.nan(f32));
    try testing.expect(nan & 0x7c00 == 0x7c00 and nan & 0x03ff != 0);
    try testing.expectEqual(1.0, math.halfBitsToFloat(0x3c00));
    try testing.expectEqual(0x00000000, @as(u32, @bitCast(math.halfBitsToFloat(0x0000))));
    try testing.expectEqual(0x80000000, @as(u32, @bitCast(math.halfBitsToFloat(0x8000))));
    try testing.expectEqual(0x1p-24, math.halfBitsToFloat(0x0001));
    try testing.expectEqual(1023 * 0x1p-24, math.halfBitsToFloat(0x03ff));
    try testing.expectEqual(0x1p-14, math.halfBitsToFloat(0x0400));
    try testing.expectEqual(65504, math.halfBitsToFloat(0x7bff));
    try testing.expectEqual(-2, math.halfBitsToFloat(0xc000));
    try testing.expectEqual(std.math.inf(f32), math.halfBitsToFloat(0x7c00));
    try testing.expectEqual(-std.math.inf(f32), math.halfBitsToFloat(0xfc00));
    try testing.expect(std.math.isNan(math.halfBitsToFloat(0x7e00)));
    try testing.expect(std.math.isNan(math.halfBitsToFloat(0xfc01)));
    for (0..0x10000) |index| {
        const bits: u16 = @intCast(index);
        if (bits & 0x7c00 == 0x7c00 and bits & 0x03ff != 0) continue;
        try testing.expectEqual(bits, math.floatToHalfBits(math.halfBitsToFloat(bits)));
    }
}

test "comath operators" {
    const model = math.translation(.{ .x = 1, .y = 2, .z = 3 });
    const position: float4 = .{ .x = 1, .y = 1, .z = 1, .w = 1 };
    const world = try comath.eval("model * position", math.context, .{ .model = model, .position = position });
    try expectEql(float4{ .x = 2, .y = 3, .z = 4, .w = 1 }, world);
    const product = try comath.eval("a * b * v", math.context, .{ .a = matrix_a, .b = matrix_b, .v = matrix_vector });
    try expectEql(math.mul(math.mul(matrix_a, matrix_b), matrix_vector), product);
    const blended = try comath.eval("-(a + b) * 0.5 - c / 2", math.context, .{
        .a = float3{ .x = 1, .y = 2, .z = 3 },
        .b = float3{ .x = 3, .y = 2, .z = 1 },
        .c = float3{ .x = 2, .y = 4, .z = 6 },
    });
    try expectEql(float3{ .x = -3, .y = -4, .z = -5 }, blended);
    const rotated = try comath.eval("q * v", math.context, .{ .q = math.axisAngle(math.half_pi, .{ .y = 1 }), .v = float3{ .x = 1 } });
    try expectNear(float3{ .z = -1 }, rotated);
}
