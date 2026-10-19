//! X11 window and timer for the Zig examples.

const std = @import("std");

pub fn timeSeconds() f64 {
    var now: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &now);
    return @as(f64, @floatFromInt(now.sec)) + @as(f64, @floatFromInt(now.nsec)) * 1e-9;
}

pub const Window = struct {
    handle: *anyopaque,
    display: ?*anyopaque,
    delete_atom: c_ulong,
    mapped: bool = false,

    pub fn open(title: [*:0]const u8, width: u32, height: u32) ?Window {
        std.debug.assert(width != 0 and height != 0);
        const display = XOpenDisplay(null) orelse return null;
        const window = XCreateSimpleWindow(display, XDefaultRootWindow(display), 0, 0, width, height, 0, 0, 0);
        if (window == 0) {
            _ = XCloseDisplay(display);
            return null;
        }
        _ = XStoreName(display, window, title);
        _ = XSelectInput(display, window, KeyPressMask | StructureNotifyMask);
        var delete_atom = XInternAtom(display, "WM_DELETE_WINDOW", 0);
        _ = XSetWMProtocols(display, window, &delete_atom, 1);
        _ = XMapWindow(display, window);
        return .{ .handle = @ptrFromInt(window), .display = display, .delete_atom = delete_atom };
    }

    /// Returns false once the window is closed; blocks while it is unmapped or minimized.
    pub fn pump(window: *Window) bool {
        const display: *Display = @ptrCast(window.display);
        while (true) {
            while (XPending(display) != 0) {
                var event: XEvent = undefined;
                _ = XNextEvent(display, &event);
                switch (event.type) {
                    KeyPress => if (XLookupKeysym(&event, 0) == XK_Escape) return false,
                    ClientMessage => if (@as(c_ulong, @bitCast(event.xclient.data[0])) == window.delete_atom) return false,
                    MapNotify => window.mapped = true,
                    UnmapNotify => window.mapped = false,
                    else => {},
                }
            }
            if (window.mapped) return true;
            var event: XEvent = undefined;
            _ = XPeekEvent(display, &event);
        }
    }

    pub fn close(window: Window) void {
        const display: *Display = @ptrCast(window.display);
        _ = XDestroyWindow(display, @intFromPtr(window.handle));
        _ = XCloseDisplay(display);
    }
};

const Display = opaque {};

const XEvent = extern union {
    type: c_int,
    xclient: extern struct {
        type: c_int,
        serial: c_ulong,
        send_event: c_int,
        display: ?*Display,
        window: c_ulong,
        message_type: c_ulong,
        format: c_int,
        data: [5]c_long,
    },
    pad: [24]c_long,
};

const KeyPressMask = 1 << 0;
const StructureNotifyMask = 1 << 17;
const KeyPress = 2;
const UnmapNotify = 18;
const MapNotify = 19;
const ClientMessage = 33;
const XK_Escape = 0xff1b;

extern "X11" fn XOpenDisplay(name: ?[*:0]const u8) ?*Display;
extern "X11" fn XCloseDisplay(display: *Display) c_int;
extern "X11" fn XDefaultRootWindow(display: *Display) c_ulong;
extern "X11" fn XCreateSimpleWindow(
    display: *Display,
    parent: c_ulong,
    x: c_int,
    y: c_int,
    width: c_uint,
    height: c_uint,
    border_width: c_uint,
    border: c_ulong,
    background: c_ulong,
) c_ulong;
extern "X11" fn XDestroyWindow(display: *Display, window: c_ulong) c_int;
extern "X11" fn XStoreName(display: *Display, window: c_ulong, name: [*:0]const u8) c_int;
extern "X11" fn XSelectInput(display: *Display, window: c_ulong, mask: c_long) c_int;
extern "X11" fn XInternAtom(display: *Display, name: [*:0]const u8, only_if_exists: c_int) c_ulong;
extern "X11" fn XSetWMProtocols(display: *Display, window: c_ulong, protocols: *c_ulong, count: c_int) c_int;
extern "X11" fn XMapWindow(display: *Display, window: c_ulong) c_int;
extern "X11" fn XPending(display: *Display) c_int;
extern "X11" fn XNextEvent(display: *Display, event: *XEvent) c_int;
extern "X11" fn XPeekEvent(display: *Display, event: *XEvent) c_int;
extern "X11" fn XLookupKeysym(event: *XEvent, index: c_int) c_ulong;
