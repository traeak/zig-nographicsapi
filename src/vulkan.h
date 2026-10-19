// Translated by Zig's translate-c into the `vulkan` module used by src/vulkan.zig.
#pragma once

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include <vulkan/vulkan.h>

#if defined(__linux__)
// X11 presentation needs only these Xlib types, so the library builds without X11 headers.
typedef struct _XDisplay Display;
typedef unsigned long Window;
typedef unsigned long VisualID;
#include <vulkan/vulkan_xlib.h>
#endif
