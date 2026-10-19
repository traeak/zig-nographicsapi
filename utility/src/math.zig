//! CPU math for the shared shader types in `shader_types.h`. Vectors and quaternions are HLSL-style; matrices are
//! row-major and multiply column vectors, matching Slang's `-matrix-layout-row-major` shaders.
//!
//! `mul` follows HLSL `mul`: component-wise for vectors and scalars, matrix products for
//! matrices, the Hamilton product for quaternions, and rotation for `quaternion * float3`. `context` maps comath
//! operators onto `add`, `sub`, `mul`, `div` and `neg`:
//!
//!     const clip = try comath.eval("projection * view * position", math.context, .{ .projection = p, .view = v, .position = x });

const std = @import("std");
const builtin = @import("builtin");
const comath = @import("comath");
const assert = std.debug.assert;

pub const types = @import("shader_types");

comptime {
    const arch = builtin.cpu.arch;
    if (!(arch == .x86_64 or (arch == .aarch64 and builtin.os.tag.isDarwin())))
        @compileError("NoGraphicsAPI math requires x86-64 or Apple ARM64");
    if (arch == .x86_64 and !std.Target.x86.featureSetHasAll(builtin.cpu.features, .{ .avx2, .fma }))
        @compileError("NoGraphicsAPI math requires AVX2 and FMA");
}

pub const float16_t = types.float16_t;
pub const float2 = types.float2;
pub const float3 = types.float3;
pub const float4 = types.float4;
pub const int2 = types.int2;
pub const int3 = types.int3;
pub const int4 = types.int4;
pub const uint2 = types.uint2;
pub const uint3 = types.uint3;
pub const uint4 = types.uint4;
pub const float3x3 = types.float3x3;
pub const float3x4 = types.float3x4;
pub const float4x4 = types.float4x4;
pub const quaternion = types.quaternion;
pub const float16_t2 = types.float16_t2;
pub const float16_t3 = types.float16_t3;
pub const float16_t4 = types.float16_t4;
pub const int16_t2 = types.int16_t2;
pub const int16_t3 = types.int16_t3;
pub const int16_t4 = types.int16_t4;
pub const uint16_t2 = types.uint16_t2;
pub const uint16_t3 = types.uint16_t3;
pub const uint16_t4 = types.uint16_t4;

pub const pi: f32 = std.math.pi;
pub const half_pi: f32 = pi * 0.5;
pub const two_pi: f32 = pi * 2.0;

pub const identity3x3: float3x3 = .{ .rows = .{ .{ .x = 1 }, .{ .y = 1 }, .{ .z = 1 } } };
pub const identity3x4: float3x4 = .{ .rows = .{ .{ .x = 1 }, .{ .y = 1 }, .{ .z = 1 } } };
pub const identity4x4: float4x4 = .{ .rows = .{ .{ .x = 1 }, .{ .y = 1 }, .{ .z = 1 }, .{ .w = 1 } } };
pub const identity_quaternion: quaternion = .{ .w = 1 };

// Vectors and quaternions are viewed as SIMD vectors of their `x` field type.

fn Scalar(comptime T: type) type {
    return @FieldType(T, "x");
}

fn count(comptime T: type) comptime_int {
    return @sizeOf(T) / @sizeOf(Scalar(T));
}

fn Simd(comptime T: type) type {
    return @Vector(count(T), Scalar(T));
}

inline fn simd(value: anytype) Simd(@TypeOf(value)) {
    const T = @TypeOf(value);
    return @as(*const [count(T)]Scalar(T), @ptrCast(&value)).*;
}

inline fn from(comptime T: type, value: Simd(T)) T {
    const array: [count(T)]Scalar(T) = value;
    return @as(*const T, @ptrCast(&array)).*;
}

fn isScalar(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .float, .comptime_int, .comptime_float => true,
        else => false,
    };
}

fn Elementwise(comptime A: type, comptime B: type) type {
    return if (isScalar(A)) B else A;
}

inline fn lanes(comptime T: type, value: anytype) Simd(T) {
    return if (@TypeOf(value) == T) simd(value) else @splat(value);
}

fn Product(comptime A: type, comptime B: type) type {
    if (A == float4x4 and B == float4) return float4;
    if (A == float3x4 and B == float4) return float3;
    if (A == float3x3 and B == float3) return float3;
    if (A == float4 and B == float4x4) return float4;
    if (A == float3 and B == float3x3) return float3;
    if (A == quaternion and B == float3) return float3;
    return Elementwise(A, B);
}

const V4 = @Vector(4, f32);

// Arithmetic. Either operand of `add`, `sub`, `mul` and `div` may be a scalar that is applied to every component.

pub inline fn add(a: anytype, b: anytype) Elementwise(@TypeOf(a), @TypeOf(b)) {
    const T = Elementwise(@TypeOf(a), @TypeOf(b));
    return from(T, lanes(T, a) + lanes(T, b));
}

pub inline fn sub(a: anytype, b: anytype) Elementwise(@TypeOf(a), @TypeOf(b)) {
    const T = Elementwise(@TypeOf(a), @TypeOf(b));
    return from(T, lanes(T, a) - lanes(T, b));
}

pub inline fn div(a: anytype, b: anytype) Elementwise(@TypeOf(a), @TypeOf(b)) {
    const T = Elementwise(@TypeOf(a), @TypeOf(b));
    if (@typeInfo(Scalar(T)) == .float) return from(T, lanes(T, a) / lanes(T, b));
    return from(T, @divTrunc(lanes(T, a), lanes(T, b)));
}

pub inline fn neg(value: anytype) @TypeOf(value) {
    return from(@TypeOf(value), -simd(value));
}

pub fn mul(a: anytype, b: anytype) Product(@TypeOf(a), @TypeOf(b)) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    if (A == float4x4 and B == float4x4) return mul4x4(a, b);
    if (A == float3x4 and B == float3x4) return mul3x4(a, b);
    if (A == float3x3 and B == float3x3) return mul3x3(a, b);
    if (A == float4x4 and B == float4) return from(float4, combine(simd(b), transpose4(a.rows)));
    if (A == float3x4 and B == float4) return convert(float3, from(float4, combine(simd(b), transpose4(.{ a.rows[0], a.rows[1], a.rows[2], .{} }))));
    if (A == float3x3 and B == float3) return .{ .x = dot(a.rows[0], b), .y = dot(a.rows[1], b), .z = dot(a.rows[2], b) };
    if (A == float4 and B == float4x4) return from(float4, combine(simd(a), .{ simd(b.rows[0]), simd(b.rows[1]), simd(b.rows[2]), simd(b.rows[3]) }));
    if (A == float3 and B == float3x3) return add(add(mul(b.rows[0], a.x), mul(b.rows[1], a.y)), mul(b.rows[2], a.z));
    if (A == quaternion and B == quaternion) return hamilton(a, b);
    if (A == quaternion and B == float3) return rotate(a, b);
    const T = Elementwise(A, B);
    return from(T, lanes(T, a) * lanes(T, b));
}

inline fn combine(coefficients: V4, rows: [4]V4) V4 {
    var result = @as(V4, @splat(coefficients[0])) * rows[0];
    result = @mulAdd(V4, @splat(coefficients[1]), rows[1], result);
    result = @mulAdd(V4, @splat(coefficients[2]), rows[2], result);
    return @mulAdd(V4, @splat(coefficients[3]), rows[3], result);
}

inline fn transpose4(rows: [4]float4) [4]V4 {
    const even01 = @shuffle(f32, simd(rows[0]), simd(rows[1]), [4]i32{ 0, -1, 2, -3 });
    const odd01 = @shuffle(f32, simd(rows[0]), simd(rows[1]), [4]i32{ 1, -2, 3, -4 });
    const even23 = @shuffle(f32, simd(rows[2]), simd(rows[3]), [4]i32{ 0, -1, 2, -3 });
    const odd23 = @shuffle(f32, simd(rows[2]), simd(rows[3]), [4]i32{ 1, -2, 3, -4 });
    return .{
        @shuffle(f32, even01, even23, [4]i32{ 0, 1, -1, -2 }),
        @shuffle(f32, odd01, odd23, [4]i32{ 0, 1, -1, -2 }),
        @shuffle(f32, even01, even23, [4]i32{ 2, 3, -3, -4 }),
        @shuffle(f32, odd01, odd23, [4]i32{ 2, 3, -3, -4 }),
    };
}

fn mul4x4(lhs: float4x4, rhs: float4x4) float4x4 {
    const rows = [4]V4{ simd(rhs.rows[0]), simd(rhs.rows[1]), simd(rhs.rows[2]), simd(rhs.rows[3]) };
    var result: float4x4 = undefined;
    for (&result.rows, lhs.rows) |*out, row| out.* = from(float4, combine(simd(row), rows));
    return result;
}

fn mul3x4(lhs: float3x4, rhs: float3x4) float3x4 {
    var result: float3x4 = undefined;
    for (&result.rows, lhs.rows) |*out, row| {
        var value = V4{ 0, 0, 0, row.w };
        value = @mulAdd(V4, @splat(row.x), simd(rhs.rows[0]), value);
        value = @mulAdd(V4, @splat(row.y), simd(rhs.rows[1]), value);
        out.* = from(float4, @mulAdd(V4, @splat(row.z), simd(rhs.rows[2]), value));
    }
    return result;
}

fn mul3x3(lhs: float3x3, rhs: float3x3) float3x3 {
    var result: float3x3 = undefined;
    for (&result.rows, lhs.rows) |*out, row| out.* = mul(row, rhs);
    return result;
}

fn hamilton(lhs: quaternion, rhs: quaternion) quaternion {
    return .{
        .x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
        .y = lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
        .z = lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
        .w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
    };
}

/// Compares vectors, quaternions or matrices component-wise with `==`.
pub fn eql(a: anytype, b: @TypeOf(a)) bool {
    if (@hasField(@TypeOf(a), "rows")) {
        for (a.rows, b.rows) |lhs, rhs| if (!eql(lhs, rhs)) return false;
        return true;
    }
    return @reduce(.And, simd(a) == simd(b));
}

// Construction and conversion.

pub inline fn splat(comptime T: type, value: Scalar(T)) T {
    return from(T, @splat(value));
}

/// Converts the component type, keeping the leading components of `value`. Integer conversions keep the bit pattern;
/// float-to-integer conversions truncate toward zero. Also converts between `float4` and `quaternion`.
pub fn convert(comptime T: type, value: anytype) T {
    const S = @TypeOf(value);
    comptime assert(count(T) <= count(S));
    const array: [count(S)]Scalar(S) = simd(value);
    const source: @Vector(count(T), Scalar(S)) = array[0..count(T)].*;
    const From = @typeInfo(Scalar(S));
    const To = @typeInfo(Scalar(T));
    if (Scalar(S) == Scalar(T)) return from(T, source);
    if (From == .int and To == .int) return from(T, @bitCast(source));
    if (From == .int) return from(T, @floatFromInt(source));
    return from(T, @intFromFloat(source));
}

fn Extended(comptime T: type) type {
    if (T == float2) return float3;
    if (T == float3) return float4;
    if (T == int2) return int3;
    if (T == int3) return int4;
    if (T == uint2) return uint3;
    if (T == uint3) return uint4;
    @compileError("cannot extend " ++ @typeName(T));
}

/// Appends one component, for example `float3` to `float4`.
pub fn extend(value: anytype, last: Scalar(@TypeOf(value))) Extended(@TypeOf(value)) {
    const T = @TypeOf(value);
    const array: [count(T)]Scalar(T) = simd(value);
    return from(Extended(T), array ++ [1]Scalar(T){last});
}

// Component-wise functions. `clamp` bounds may be vectors or scalars.

pub inline fn min(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return from(@TypeOf(a), @min(simd(a), simd(b)));
}

pub inline fn max(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return from(@TypeOf(a), @max(simd(a), simd(b)));
}

pub inline fn clamp(value: anytype, low: anytype, high: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    return from(T, @min(@max(simd(value), lanes(T, low)), lanes(T, high)));
}

pub inline fn abs(value: anytype) @TypeOf(value) {
    return from(@TypeOf(value), @bitCast(@abs(simd(value))));
}

pub inline fn floor(value: anytype) @TypeOf(value) {
    return from(@TypeOf(value), @floor(simd(value)));
}

pub inline fn ceil(value: anytype) @TypeOf(value) {
    return from(@TypeOf(value), @ceil(simd(value)));
}

pub inline fn round(value: anytype) @TypeOf(value) {
    return from(@TypeOf(value), @round(simd(value)));
}

pub inline fn lerp(a: anytype, b: @TypeOf(a), t: f32) @TypeOf(a) {
    return from(@TypeOf(a), @mulAdd(Simd(@TypeOf(a)), simd(b) - simd(a), @splat(t), simd(a)));
}

/// Accepts a scalar or a float vector `value`.
pub fn smoothstep(edge0: f32, edge1: f32, value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (T == f32) {
        const t = std.math.clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
    const V = Simd(T);
    const t = @min(@max((simd(value) - @as(V, @splat(edge0))) / @as(V, @splat(edge1 - edge0)), @as(V, @splat(0.0))), @as(V, @splat(1.0)));
    return from(T, t * t * (@as(V, @splat(3.0)) - @as(V, @splat(2.0)) * t));
}

// Geometry.

pub inline fn dot(a: anytype, b: @TypeOf(a)) Scalar(@TypeOf(a)) {
    return @reduce(.Add, simd(a) * simd(b));
}

pub fn cross(a: float3, b: float3) float3 {
    return .{ .x = a.y * b.z - a.z * b.y, .y = a.z * b.x - a.x * b.z, .z = a.x * b.y - a.y * b.x };
}

pub inline fn rsqrt(value: f32) f32 {
    return 1.0 / @sqrt(value);
}

pub inline fn length(value: anytype) f32 {
    return @sqrt(dot(value, value));
}

pub inline fn normalize(value: anytype) @TypeOf(value) {
    return mul(value, rsqrt(dot(value, value)));
}

// Matrices.

pub fn toFloat3x4(matrix: float4x4) float3x4 {
    return .{ .rows = matrix.rows[0..3].* };
}

pub fn setTranslation(transform: float3x4, value: float3) float3x4 {
    var result = transform;
    result.rows[0].w = value.x;
    result.rows[1].w = value.y;
    result.rows[2].w = value.z;
    return result;
}

pub fn transformVector(transform: float3x4, vector: float3) float3 {
    return mul(transform, extend(vector, 0.0));
}

pub fn transformPoint(transform: float3x4, point: float3) float3 {
    return mul(transform, extend(point, 1.0));
}

/// Transposes a `float3x3` or `float4x4`.
pub fn transpose(matrix: anytype) @TypeOf(matrix) {
    const T = @TypeOf(matrix);
    if (T == float4x4) {
        const rows = transpose4(matrix.rows);
        return .{ .rows = .{ from(float4, rows[0]), from(float4, rows[1]), from(float4, rows[2]), from(float4, rows[3]) } };
    }
    comptime assert(T == float3x3);
    const r = matrix.rows;
    return .{ .rows = .{
        .{ .x = r[0].x, .y = r[1].x, .z = r[2].x },
        .{ .x = r[0].y, .y = r[1].y, .z = r[2].y },
        .{ .x = r[0].z, .y = r[1].z, .z = r[2].z },
    } };
}

pub fn determinant(matrix: float3x3) f32 {
    const a = matrix.rows[0];
    const b = matrix.rows[1];
    const c = matrix.rows[2];
    return a.x * (b.y * c.z - b.z * c.y) - a.y * (b.x * c.z - b.z * c.x) + a.z * (b.x * c.y - b.y * c.x);
}

/// Inverts a `float3x3`, an affine `float3x4`, a `float4x4` or a `quaternion`. The value must be invertible.
pub fn inverse(value: anytype) @TypeOf(value) {
    const T = @TypeOf(value);
    if (T == float3x3) return inverse3x3(value);
    if (T == float3x4) return inverseAffine(value);
    if (T == float4x4) return inverse4x4(value);
    comptime assert(T == quaternion);
    const length_squared = dot(value, value);
    assert(length_squared != 0.0);
    return div(conjugate(value), length_squared);
}

fn inverse3x3(matrix: float3x3) float3x3 {
    const a = matrix.rows[0];
    const b = matrix.rows[1];
    const c = matrix.rows[2];
    const det = determinant(matrix);
    assert(det != 0.0);
    const r = 1.0 / det;
    return .{ .rows = .{
        .{ .x = (b.y * c.z - b.z * c.y) * r, .y = (a.z * c.y - a.y * c.z) * r, .z = (a.y * b.z - a.z * b.y) * r },
        .{ .x = (b.z * c.x - b.x * c.z) * r, .y = (a.x * c.z - a.z * c.x) * r, .z = (a.z * b.x - a.x * b.z) * r },
        .{ .x = (b.x * c.y - b.y * c.x) * r, .y = (a.y * c.x - a.x * c.y) * r, .z = (a.x * b.y - a.y * b.x) * r },
    } };
}

fn inverseAffine(transform: float3x4) float3x4 {
    const r = transform.rows;
    const linear = inverse3x3(.{ .rows = .{ convert(float3, r[0]), convert(float3, r[1]), convert(float3, r[2]) } });
    const translation_value = neg(mul(linear, float3{ .x = r[0].w, .y = r[1].w, .z = r[2].w }));
    return .{ .rows = .{
        extend(linear.rows[0], translation_value.x),
        extend(linear.rows[1], translation_value.y),
        extend(linear.rows[2], translation_value.z),
    } };
}

fn inverse4x4(matrix: float4x4) float4x4 {
    var augmented: [4][8]f32 = undefined;
    for (&augmented, matrix.rows, 0..) |*row, source, index| {
        row.* = .{ source.x, source.y, source.z, source.w, 0, 0, 0, 0 };
        row[index + 4] = 1.0;
    }
    for (0..4) |pivot_column| {
        var pivot_row = pivot_column;
        var pivot_magnitude = @abs(augmented[pivot_row][pivot_column]);
        for (pivot_column + 1..4) |row| {
            const magnitude = @abs(augmented[row][pivot_column]);
            if (magnitude > pivot_magnitude) {
                pivot_row = row;
                pivot_magnitude = magnitude;
            }
        }
        assert(pivot_magnitude != 0.0);
        std.mem.swap([8]f32, &augmented[pivot_column], &augmented[pivot_row]);
        const reciprocal_pivot = 1.0 / augmented[pivot_column][pivot_column];
        for (&augmented[pivot_column]) |*value| value.* *= reciprocal_pivot;
        for (&augmented, 0..) |*row, index| {
            if (index == pivot_column) continue;
            const factor = row[pivot_column];
            for (row, augmented[pivot_column]) |*value, pivot_value| value.* -= factor * pivot_value;
        }
    }
    var result: float4x4 = undefined;
    for (&result.rows, augmented) |*row, source| row.* = .{ .x = source[4], .y = source[5], .z = source[6], .w = source[7] };
    return result;
}

pub fn translation(value: float3) float4x4 {
    return .{ .rows = .{ .{ .x = 1, .w = value.x }, .{ .y = 1, .w = value.y }, .{ .z = 1, .w = value.z }, .{ .w = 1 } } };
}

pub fn scale(value: float3) float4x4 {
    return .{ .rows = .{ .{ .x = value.x }, .{ .y = value.y }, .{ .z = value.z }, .{ .w = 1 } } };
}

pub fn rotationX(angle: f32) float4x4 {
    const s = @sin(angle);
    const c = @cos(angle);
    return .{ .rows = .{ .{ .x = 1 }, .{ .y = c, .z = -s }, .{ .y = s, .z = c }, .{ .w = 1 } } };
}

pub fn rotationY(angle: f32) float4x4 {
    const s = @sin(angle);
    const c = @cos(angle);
    return .{ .rows = .{ .{ .x = c, .z = s }, .{ .y = 1 }, .{ .x = -s, .z = c }, .{ .w = 1 } } };
}

pub fn rotationZ(angle: f32) float4x4 {
    const s = @sin(angle);
    const c = @cos(angle);
    return .{ .rows = .{ .{ .x = c, .y = -s }, .{ .x = s, .y = c }, .{ .z = 1 }, .{ .w = 1 } } };
}

pub fn quaternionToMatrix(value: quaternion) float3x3 {
    const xx = value.x * value.x;
    const yy = value.y * value.y;
    const zz = value.z * value.z;
    const xy = value.x * value.y;
    const xz = value.x * value.z;
    const yz = value.y * value.z;
    const wx = value.w * value.x;
    const wy = value.w * value.y;
    const wz = value.w * value.z;
    return .{ .rows = .{
        .{ .x = 1.0 - 2.0 * (yy + zz), .y = 2.0 * (xy - wz), .z = 2.0 * (xz + wy) },
        .{ .x = 2.0 * (xy + wz), .y = 1.0 - 2.0 * (xx + zz), .z = 2.0 * (yz - wx) },
        .{ .x = 2.0 * (xz - wy), .y = 2.0 * (yz + wx), .z = 1.0 - 2.0 * (xx + yy) },
    } };
}

pub fn rotation(value: quaternion) float4x4 {
    const m = quaternionToMatrix(value);
    return .{ .rows = .{ extend(m.rows[0], 0.0), extend(m.rows[1], 0.0), extend(m.rows[2], 0.0), .{ .w = 1 } } };
}

/// Right-handed perspective projection with a [0, 1] depth range. Swap the planes for reversed depth.
pub fn perspectiveRhZo(vertical_field_of_view: f32, aspect: f32, near_plane: f32, far_plane: f32) float4x4 {
    assert(aspect != 0.0);
    assert(near_plane > 0.0 and far_plane > 0.0 and far_plane != near_plane);
    const focal_length = 1.0 / @tan(vertical_field_of_view * 0.5);
    const depth = 1.0 / (near_plane - far_plane);
    return .{ .rows = .{
        .{ .x = focal_length / aspect },
        .{ .y = focal_length },
        .{ .z = far_plane * depth, .w = near_plane * far_plane * depth },
        .{ .z = -1 },
    } };
}

pub fn orthoRhZo(left: f32, right: f32, bottom: f32, top: f32, near_plane: f32, far_plane: f32) float4x4 {
    assert(right != left and top != bottom and far_plane != near_plane);
    return .{ .rows = .{
        .{ .x = 2.0 / (right - left), .w = -(right + left) / (right - left) },
        .{ .y = 2.0 / (top - bottom), .w = -(top + bottom) / (top - bottom) },
        .{ .z = -1.0 / (far_plane - near_plane), .w = -near_plane / (far_plane - near_plane) },
        .{ .w = 1 },
    } };
}

pub fn lookAtRh(eye: float3, center: float3, up: float3) float4x4 {
    const forward = normalize(sub(center, eye));
    const side = normalize(cross(forward, up));
    const camera_up = cross(side, forward);
    return .{ .rows = .{
        extend(side, -dot(side, eye)),
        extend(camera_up, -dot(camera_up, eye)),
        extend(neg(forward), dot(forward, eye)),
        .{ .w = 1 },
    } };
}

// Quaternions.

pub fn conjugate(value: quaternion) quaternion {
    return .{ .x = -value.x, .y = -value.y, .z = -value.z, .w = value.w };
}

pub fn axisAngle(angle: f32, axis: float3) quaternion {
    const half_angle = angle * 0.5;
    const unit_axis = mul(normalize(axis), @sin(half_angle));
    return .{ .x = unit_axis.x, .y = unit_axis.y, .z = unit_axis.z, .w = @cos(half_angle) };
}

pub fn rotate(rotation_value: quaternion, vector: float3) float3 {
    const imaginary = convert(float3, rotation_value);
    const uv = cross(imaginary, vector);
    const uuv = cross(imaginary, uv);
    return add(vector, mul(2.0, add(mul(rotation_value.w, uv), uuv)));
}

pub fn slerp(a: quaternion, b: quaternion, t: f32) quaternion {
    var target = b;
    var cosine = dot(a, b);
    if (cosine < 0.0) {
        target = neg(b);
        cosine = -cosine;
    }
    if (cosine > 0.9995) return normalize(add(a, mul(sub(target, a), t)));
    const angle = std.math.acos(@min(cosine, 1.0));
    const reciprocal_sine = 1.0 / @sin(angle);
    return add(mul(a, @sin((1.0 - t) * angle) * reciprocal_sine), mul(target, @sin(t * angle) * reciprocal_sine));
}

// Binary16 packing. Conversion rounds to nearest even and keeps NaNs quiet and non-zero.

pub fn floatToHalfBits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    const sign = (bits >> 16) & 0x8000;
    const exponent = (bits >> 23) & 0xff;
    const mantissa = bits & 0x7fffff;
    if (exponent == 0xff) {
        if (mantissa == 0) return @intCast(sign | 0x7c00);
        return @intCast(sign | 0x7c00 | @max(mantissa >> 13, 1));
    }
    const half_exponent = @as(i32, @intCast(exponent)) - 127 + 15;
    if (half_exponent >= 31) return @intCast(sign | 0x7c00);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return @intCast(sign);
        const significand = mantissa | 0x800000;
        const shift: u5 = @intCast(14 - half_exponent);
        var rounded = significand >> shift;
        const remainder = significand & ((@as(u32, 1) << shift) - 1);
        const halfway = @as(u32, 1) << (shift - 1);
        if (remainder > halfway or (remainder == halfway and (rounded & 1) != 0)) rounded += 1;
        return @intCast(sign | rounded);
    }
    var rounded = (@as(u32, @intCast(half_exponent)) << 10) | (mantissa >> 13);
    const remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 or (remainder == 0x1000 and (rounded & 1) != 0)) rounded += 1;
    return @intCast(sign | rounded);
}

pub fn halfBitsToFloat(value: u16) f32 {
    const sign = @as(u32, value & 0x8000) << 16;
    const exponent: u32 = (value >> 10) & 0x1f;
    const mantissa: u32 = value & 0x03ff;
    if (exponent == 0) {
        const magnitude = @as(f32, @floatFromInt(mantissa)) * 0x1p-24;
        return if (sign != 0) -magnitude else magnitude;
    }
    return @bitCast(sign | ((if (exponent == 31) 255 else exponent + 112) << 23) | (mantissa << 13));
}

pub fn packHalf2x16(value: float2) u32 {
    return @as(u32, floatToHalfBits(value.x)) | (@as(u32, floatToHalfBits(value.y)) << 16);
}

// comath operators.

pub const context = comath.ctx.simple.context(Operators{});

const Operators = struct {
    pub const matchUnOp = {};
    pub const matchBinOp = {};
    pub const orderBinOp = {};
    pub const EvalNumberLiteral = {};
    pub const evalNumberLiteral = {};
    pub const EvalIdent = {};
    pub const evalIdent = {};
    pub const EvalProperty = {};
    pub const evalProperty = {};
    pub const EvalIndexAccess = {};
    pub const evalIndexAccess = {};
    pub const EvalFuncCall = {};
    pub const evalFuncCall = {};
    pub const EvalMethodCall = {};
    pub const evalMethodCall = {};

    fn isMath(comptime T: type) bool {
        return @typeInfo(T) == .@"struct" and (@hasField(T, "rows") or @hasField(T, "x"));
    }

    pub fn EvalBinOp(comptime Lhs: type, comptime op: []const u8, comptime Rhs: type) type {
        if (!isMath(Lhs) and !isMath(Rhs)) return noreturn;
        if (std.mem.eql(u8, op, "*")) return Product(Lhs, Rhs);
        if (std.mem.eql(u8, op, "+") or std.mem.eql(u8, op, "-") or std.mem.eql(u8, op, "/")) return Elementwise(Lhs, Rhs);
        return noreturn;
    }

    pub fn evalBinOp(_: Operators, lhs: anytype, comptime op: []const u8, rhs: anytype) EvalBinOp(@TypeOf(lhs), op, @TypeOf(rhs)) {
        if (comptime std.mem.eql(u8, op, "*")) return mul(lhs, rhs);
        if (comptime std.mem.eql(u8, op, "+")) return add(lhs, rhs);
        if (comptime std.mem.eql(u8, op, "-")) return sub(lhs, rhs);
        return div(lhs, rhs);
    }

    pub fn EvalUnOp(comptime op: []const u8, comptime T: type) type {
        return if (isMath(T) and std.mem.eql(u8, op, "-")) T else noreturn;
    }

    pub fn evalUnOp(_: Operators, comptime op: []const u8, value: anytype) EvalUnOp(op, @TypeOf(value)) {
        return neg(value);
    }
};
