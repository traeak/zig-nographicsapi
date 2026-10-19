//! Win32 window and timer for the Zig examples.

const std = @import("std");

pub fn timeSeconds() f64 {
    var frequency: i64 = 0;
    var counter: i64 = 0;
    _ = QueryPerformanceFrequency(&frequency);
    _ = QueryPerformanceCounter(&counter);
    return @as(f64, @floatFromInt(counter)) / @as(f64, @floatFromInt(frequency));
}

pub const Window = struct {
    handle: *anyopaque,
    display: ?*anyopaque = null,

    pub fn open(title: [*:0]const u8, width: u32, height: u32) ?Window {
        std.debug.assert(width != 0 and height != 0);
        const instance = GetModuleHandleA(null);
        const window_class: WNDCLASSEXA = .{
            .style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
            .lpfnWndProc = windowProc,
            .hInstance = instance,
            .hCursor = LoadCursorA(null, @ptrFromInt(IDC_ARROW)),
            .lpszClassName = class_name,
        };
        if (RegisterClassExA(&window_class) == 0 and GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return null;
        var rectangle: RECT = .{ .right = @intCast(width), .bottom = @intCast(height) };
        if (AdjustWindowRectEx(&rectangle, window_style, 0, 0) == 0) return null;
        const hwnd = CreateWindowExA(
            0,
            class_name,
            title,
            window_style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rectangle.right - rectangle.left,
            rectangle.bottom - rectangle.top,
            null,
            null,
            instance,
            null,
        ) orelse return null;
        _ = ShowWindow(hwnd, SW_SHOWDEFAULT);
        _ = UpdateWindow(hwnd);
        return .{ .handle = hwnd };
    }

    /// Returns false once the window is closed; blocks while it is minimized.
    pub fn pump(window: *Window) bool {
        while (true) {
            var message: MSG = undefined;
            while (PeekMessageA(&message, null, 0, 0, PM_REMOVE) != 0) {
                if (message.message == WM_QUIT) return false;
                _ = TranslateMessage(&message);
                _ = DispatchMessageA(&message);
            }
            if (IsIconic(@ptrCast(window.handle)) == 0) return true;
            _ = WaitMessage();
        }
    }

    pub fn close(window: Window) void {
        _ = DestroyWindow(@ptrCast(window.handle));
    }
};

const class_name = "NoGraphicsAPI_example_window";
const window_style: u32 = 0x00CF0000; // WS_OVERLAPPEDWINDOW

fn windowProc(hwnd: Handle, message: u32, wparam: usize, lparam: isize) callconv(.winapi) isize {
    switch (message) {
        WM_ERASEBKGND => return 1,
        WM_CLOSE => {
            _ = ShowWindow(hwnd, SW_HIDE);
            PostQuitMessage(0);
            return 0;
        },
        WM_KEYDOWN => if (wparam == VK_ESCAPE) {
            _ = ShowWindow(hwnd, SW_HIDE);
            PostQuitMessage(0);
            return 0;
        },
        WM_DESTROY => {
            PostQuitMessage(0);
            return 0;
        },
        else => {},
    }
    return DefWindowProcA(hwnd, message, wparam, lparam);
}

const Handle = *opaque {};

const WNDCLASSEXA = extern struct {
    cbSize: u32 = @sizeOf(WNDCLASSEXA),
    style: u32 = 0,
    lpfnWndProc: *const fn (Handle, u32, usize, isize) callconv(.winapi) isize,
    cbClsExtra: i32 = 0,
    cbWndExtra: i32 = 0,
    hInstance: ?Handle = null,
    hIcon: ?Handle = null,
    hCursor: ?Handle = null,
    hbrBackground: ?Handle = null,
    lpszMenuName: ?[*:0]const u8 = null,
    lpszClassName: [*:0]const u8,
    hIconSm: ?Handle = null,
};

const RECT = extern struct { left: i32 = 0, top: i32 = 0, right: i32 = 0, bottom: i32 = 0 };

const MSG = extern struct {
    hwnd: ?Handle,
    message: u32,
    wParam: usize,
    lParam: isize,
    time: u32,
    pt: extern struct { x: i32, y: i32 },
    lPrivate: u32,
};

const CS_VREDRAW = 0x1;
const CS_HREDRAW = 0x2;
const CS_OWNDC = 0x20;
const CW_USEDEFAULT: i32 = @bitCast(@as(u32, 0x80000000));
const IDC_ARROW = 32512;
const ERROR_CLASS_ALREADY_EXISTS = 1410;
const SW_HIDE = 0;
const SW_SHOWDEFAULT = 10;
const PM_REMOVE = 1;
const WM_DESTROY = 0x2;
const WM_CLOSE = 0x10;
const WM_QUIT = 0x12;
const WM_ERASEBKGND = 0x14;
const WM_KEYDOWN = 0x100;
const VK_ESCAPE = 0x1B;

extern "kernel32" fn GetModuleHandleA(name: ?[*:0]const u8) callconv(.winapi) ?Handle;
extern "kernel32" fn GetLastError() callconv(.winapi) u32;
extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(frequency: *i64) callconv(.winapi) i32;
extern "user32" fn RegisterClassExA(class: *const WNDCLASSEXA) callconv(.winapi) u16;
extern "user32" fn AdjustWindowRectEx(rect: *RECT, style: u32, menu: i32, ex_style: u32) callconv(.winapi) i32;
extern "user32" fn CreateWindowExA(
    ex_style: u32,
    class: [*:0]const u8,
    title: [*:0]const u8,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: ?Handle,
    menu: ?Handle,
    instance: ?Handle,
    param: ?*anyopaque,
) callconv(.winapi) ?Handle;
extern "user32" fn DestroyWindow(hwnd: Handle) callconv(.winapi) i32;
extern "user32" fn ShowWindow(hwnd: Handle, command: i32) callconv(.winapi) i32;
extern "user32" fn UpdateWindow(hwnd: Handle) callconv(.winapi) i32;
extern "user32" fn IsIconic(hwnd: Handle) callconv(.winapi) i32;
extern "user32" fn LoadCursorA(instance: ?Handle, name: *const anyopaque) callconv(.winapi) ?Handle;
extern "user32" fn PeekMessageA(message: *MSG, hwnd: ?Handle, min: u32, max: u32, remove: u32) callconv(.winapi) i32;
extern "user32" fn TranslateMessage(message: *const MSG) callconv(.winapi) i32;
extern "user32" fn DispatchMessageA(message: *const MSG) callconv(.winapi) isize;
extern "user32" fn WaitMessage() callconv(.winapi) i32;
extern "user32" fn PostQuitMessage(exit_code: i32) callconv(.winapi) void;
extern "user32" fn DefWindowProcA(hwnd: Handle, message: u32, wparam: usize, lparam: isize) callconv(.winapi) isize;
