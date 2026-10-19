//! AppKit window with a CAMetalLayer, and a timer, for the Zig examples.

const std = @import("std");
const objc = @import("objc");

const Id = objc.Id;
const send = objc.send;

pub fn timeSeconds() f64 {
    var now: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &now);
    return @as(f64, @floatFromInt(now.sec)) + @as(f64, @floatFromInt(now.nsec)) * 1e-9;
}

pub const Window = struct {
    /// The CAMetalLayer, owned by the window's content view.
    handle: *anyopaque,
    display: ?*anyopaque = null,
    window: Id,

    pub fn open(title: [*:0]const u8, width: u32, height: u32) ?Window {
        std.debug.assert(width != 0 and height != 0);
        const pool = objc.objc_autoreleasePoolPush();
        defer objc.objc_autoreleasePoolPop(pool);
        const application = send(Id, objc.class("NSApplication"), "sharedApplication", .{});
        _ = send(bool, application, "setActivationPolicy:", .{activation_policy_regular});
        send(void, application, "finishLaunching", .{});
        const frame: Rect = .{ .x = 0, .y = 0, .width = @floatFromInt(width), .height = @floatFromInt(height) };
        const style = window_style_titled | window_style_closable | window_style_resizable;
        const allocated = send(Id, objc.class("NSWindow"), "alloc", .{});
        const selector = "initWithContentRect:styleMask:backing:defer:";
        const window = send(?Id, allocated, selector, .{ frame, style, backing_store_buffered, false }) orelse return null;
        send(void, window, "setReleasedWhenClosed:", .{false});
        send(void, window, "setTitle:", .{send(Id, objc.class("NSString"), "stringWithUTF8String:", .{title})});
        const view = send(Id, window, "contentView", .{});
        send(void, view, "setWantsLayer:", .{true});
        const layer = send(Id, objc.class("CAMetalLayer"), "layer", .{});
        const scale = send(f64, window, "backingScaleFactor", .{});
        send(void, layer, "setContentsScale:", .{scale});
        send(void, layer, "setDrawableSize:", .{Size{ .width = frame.width * scale, .height = frame.height * scale }});
        send(void, view, "setLayer:", .{layer});
        send(void, window, "center", .{});
        send(void, window, "makeKeyAndOrderFront:", .{@as(?Id, null)});
        send(void, application, "activateIgnoringOtherApps:", .{true});
        return .{ .handle = layer, .window = window };
    }

    /// Returns false once the window is closed or Escape is pressed. The drawable size is zero while minimized.
    pub fn pump(window: *Window) bool {
        const pool = objc.objc_autoreleasePoolPush();
        defer objc.objc_autoreleasePoolPop(pool);
        const application = send(Id, objc.class("NSApplication"), "sharedApplication", .{});
        const past = send(Id, objc.class("NSDate"), "distantPast", .{});
        const selector = "nextEventMatchingMask:untilDate:inMode:dequeue:";
        while (send(?Id, application, selector, .{ event_mask_any, past, NSDefaultRunLoopMode, true })) |event| {
            if (send(usize, event, "type", .{}) == event_type_key_down and send(u16, event, "keyCode", .{}) == key_code_escape) return false;
            send(void, application, "sendEvent:", .{event});
        }
        if (!send(bool, window.window, "isVisible", .{})) return false;
        const scale = send(f64, window.window, "backingScaleFactor", .{});
        send(void, window.handle, "setContentsScale:", .{scale});
        const bounds = send(Rect, send(Id, window.window, "contentView", .{}), "bounds", .{});
        const minimized = send(bool, window.window, "isMiniaturized", .{});
        const size: Size = if (minimized) .{ .width = 0, .height = 0 } else .{ .width = bounds.width * scale, .height = bounds.height * scale };
        send(void, window.handle, "setDrawableSize:", .{size});
        return true;
    }

    pub fn close(window: Window) void {
        send(void, window.window, "close", .{});
        objc.objc_release(window.window);
    }
};

const Size = extern struct { width: f64, height: f64 };
const Rect = extern struct { x: f64, y: f64, width: f64, height: f64 };

const activation_policy_regular: isize = 0;
const window_style_titled: usize = 1 << 0;
const window_style_closable: usize = 1 << 1;
const window_style_resizable: usize = 1 << 3;
const backing_store_buffered: usize = 2;
const event_mask_any: u64 = std.math.maxInt(u64);
const event_type_key_down: usize = 10;
const key_code_escape: u16 = 53;

extern const NSDefaultRunLoopMode: Id;
