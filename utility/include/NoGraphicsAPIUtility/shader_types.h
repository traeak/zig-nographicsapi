#pragma once

#if defined(__SLANG__)

// Slang already provides float[2-4], int[2-4], uint[2-4], bool[2-4],
// float16_t[2-4], int16_t[2-4], uint16_t[2-4], and native matrix types.
// Redeclaring those types would hide the built-ins and can change shader
// semantics. Only add the short scalar aliases used by shared structures.
typedef uint8_t uint8;
typedef int8_t int8;
typedef uint16_t uint16;
typedef int16_t int16;
typedef uint32_t uint32;
typedef int32_t int32;
typedef uint64_t uint64;
typedef int64_t int64;

#else

#	include <NoGraphicsAPI/types.h>

// CPU-side binary16 values are deliberately opaque storage. Conversion to or
// from float is kept out of the shared ABI types so every bit pattern,
// including NaNs, crosses the CPU/GPU boundary unchanged.
typedef struct float16_t {
	uint16 bits;
} float16_t;

// Slang bool vectors use 32-bit elements and are not ABI-compatible with C
// bool aggregates. Deliberately omit CPU-side bool vector declarations.

typedef struct float2 {
	float x;
	float y;
#	if defined(__cplusplus)
	constexpr float& operator[](size_t index) noexcept { return index == 0 ? x : y; }
	constexpr const float& operator[](size_t index) const noexcept { return index == 0 ? x : y; }
#	endif
} float2;

typedef struct float3 {
	float x;
	float y;
	float z;
#	if defined(__cplusplus)
	constexpr float& operator[](size_t index) noexcept { return index == 0 ? x : (index == 1 ? y : z); }
	constexpr const float& operator[](size_t index) const noexcept { return index == 0 ? x : (index == 1 ? y : z); }
#	endif
} float3;

typedef struct float4 {
	float x;
	float y;
	float z;
	float w;
#	if defined(__cplusplus)
	constexpr float& operator[](size_t index) noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
	constexpr const float& operator[](size_t index) const noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
#	endif
} float4;

typedef struct int2 {
	int32 x;
	int32 y;
#	if defined(__cplusplus)
	constexpr int32& operator[](size_t index) noexcept { return index == 0 ? x : y; }
	constexpr const int32& operator[](size_t index) const noexcept { return index == 0 ? x : y; }
#	endif
} int2;

typedef struct int3 {
	int32 x;
	int32 y;
	int32 z;
#	if defined(__cplusplus)
	constexpr int32& operator[](size_t index) noexcept { return index == 0 ? x : (index == 1 ? y : z); }
	constexpr const int32& operator[](size_t index) const noexcept { return index == 0 ? x : (index == 1 ? y : z); }
#	endif
} int3;

typedef struct int4 {
	int32 x;
	int32 y;
	int32 z;
	int32 w;
#	if defined(__cplusplus)
	constexpr int32& operator[](size_t index) noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
	constexpr const int32& operator[](size_t index) const noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
#	endif
} int4;

typedef struct uint2 {
	uint32 x;
	uint32 y;
#	if defined(__cplusplus)
	constexpr uint32& operator[](size_t index) noexcept { return index == 0 ? x : y; }
	constexpr const uint32& operator[](size_t index) const noexcept { return index == 0 ? x : y; }
#	endif
} uint2;

typedef struct uint3 {
	uint32 x;
	uint32 y;
	uint32 z;
#	if defined(__cplusplus)
	constexpr uint32& operator[](size_t index) noexcept { return index == 0 ? x : (index == 1 ? y : z); }
	constexpr const uint32& operator[](size_t index) const noexcept { return index == 0 ? x : (index == 1 ? y : z); }
#	endif
} uint3;

typedef struct uint4 {
	uint32 x;
	uint32 y;
	uint32 z;
	uint32 w;
#	if defined(__cplusplus)
	constexpr uint32& operator[](size_t index) noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
	constexpr const uint32& operator[](size_t index) const noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
#	endif
} uint4;

// Slang names matrix dimensions as row-count x column-count. Matrices use
// row-major storage and multiply column vectors.
typedef struct float3x3 {
	float3 rows[3];
#	if defined(__cplusplus)
	constexpr float3& operator[](size_t index) noexcept { return rows[index]; }
	constexpr const float3& operator[](size_t index) const noexcept { return rows[index]; }
#	endif
} float3x3;

typedef struct float3x4 {
	float4 rows[3];
#	if defined(__cplusplus)
	constexpr float4& operator[](size_t index) noexcept { return rows[index]; }
	constexpr const float4& operator[](size_t index) const noexcept { return rows[index]; }
#	endif
} float3x4;

typedef struct float4x4 {
	float4 rows[4];
#	if defined(__cplusplus)
	constexpr float4& operator[](size_t index) noexcept { return rows[index]; }
	constexpr const float4& operator[](size_t index) const noexcept { return rows[index]; }
#	endif
} float4x4;

typedef struct quaternion {
	float x;
	float y;
	float z;
	float w;
#	if defined(__cplusplus)
	constexpr float& operator[](size_t index) noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
	constexpr const float& operator[](size_t index) const noexcept { return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w)); }
#	endif
} quaternion;

typedef struct float16_t2 {
	float16_t x;
	float16_t y;
} float16_t2;
typedef struct float16_t3 {
	float16_t x;
	float16_t y;
	float16_t z;
} float16_t3;
typedef struct float16_t4 {
	float16_t x;
	float16_t y;
	float16_t z;
	float16_t w;
} float16_t4;
typedef struct int16_t2 {
	int16 x;
	int16 y;
} int16_t2;
typedef struct int16_t3 {
	int16 x;
	int16 y;
	int16 z;
} int16_t3;
typedef struct int16_t4 {
	int16 x;
	int16 y;
	int16 z;
	int16 w;
} int16_t4;
typedef struct uint16_t2 {
	uint16 x;
	uint16 y;
} uint16_t2;
typedef struct uint16_t3 {
	uint16 x;
	uint16 y;
	uint16 z;
} uint16_t3;
typedef struct uint16_t4 {
	uint16 x;
	uint16 y;
	uint16 z;
	uint16 w;
} uint16_t4;

#endif
