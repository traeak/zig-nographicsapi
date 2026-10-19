//! Minimal Objective-C runtime access for the Metal backend and the macOS examples. Messages go through a typed
//! `objc_msgSend`, which arm64 uses for every return type; argument types must match the method's C types.

pub const Id = *anyopaque;
pub const Sel = *opaque {};

extern "objc" fn sel_registerName(name: [*:0]const u8) Sel;
extern "objc" fn objc_lookUpClass(name: [*:0]const u8) ?Id;
extern "objc" fn objc_msgSend() void;
pub extern "objc" fn objc_retain(object: ?Id) ?Id;
pub extern "objc" fn objc_release(object: ?Id) void;
pub extern "objc" fn objc_autoreleasePoolPush() *anyopaque;
pub extern "objc" fn objc_autoreleasePoolPop(pool: *anyopaque) void;

/// Registers each selector once per process. Each cache captures its name, which gives every selector its own storage.
pub fn sel(comptime name: [:0]const u8) Sel {
    const cache = struct {
        const key = name;
        var value: ?Sel = null;
    };
    if (@atomicLoad(?Sel, &cache.value, .monotonic)) |value| return value;
    const value = sel_registerName(name);
    @atomicStore(?Sel, &cache.value, value, .monotonic);
    return value;
}

/// Looks up each class once per process; the class's framework must be linked.
pub fn class(comptime name: [:0]const u8) Id {
    const cache = struct {
        const key = name;
        var value: ?Id = null;
    };
    if (@atomicLoad(?Id, &cache.value, .monotonic)) |value| return value;
    const value = objc_lookUpClass(name).?;
    @atomicStore(?Id, &cache.value, value, .monotonic);
    return value;
}

/// Sends `selector` with `args`, a tuple of exactly typed C arguments. A null receiver returns zero.
pub fn send(comptime Return: type, receiver: anytype, comptime selector: [:0]const u8, args: anytype) Return {
    const fields = @typeInfo(@TypeOf(args)).@"struct".field_types;
    const parameters = [_]type{ @TypeOf(receiver), Sel } ++ fields[0..fields.len].*;
    const Function = @Fn(&parameters, &@splat(.{}), Return, .{ .@"callconv" = .c });
    const function: *const Function = @ptrCast(&objc_msgSend);
    return @call(.auto, function, .{ receiver, sel(selector) } ++ args);
}

/// `[[name alloc] init]`; the caller owns the result.
pub fn new(comptime name: [:0]const u8) Id {
    return send(Id, class(name), "new", .{});
}
