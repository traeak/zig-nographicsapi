# Coding Guidelines

- Write simple, efficient, minimal Zig code. Format it with `zig fmt`.
- Target one fixed feature set on the latest GPUs and drivers. Avoid optional feature flags, feature queries, and fallback paths.
  Require features universally supported across the target hardware; otherwise ask the user before adding them.
  Preserve the supported baseline and any documented API capability exceptions. Do not silently emulate unsupported operations.
- Do not add obvious comments such as "arguments must be live objects". Using destroyed objects is self-evidently invalid.
- Do not add asserts or comments describing 32-bit overflow cases. The existing 32-bit ranges (about 4 billion elements or 4 GB) are sufficient.
- Use Zig's sized integer types (`u8`, `u32`, `i64`, ...). Shared CPU/GPU C headers use the `int8`..`uint64` typedefs from `include/NoGraphicsAPI/types.h`.
- Avoid adding named variables for trivial expressions, especially when the value is used only once.
- Perform error checks as early as possible. Check application initialization, resource loading, and Vulkan/Metal object creation immediately.
  Avoid error checking after initialization; normal code should not fail.
- Treat the library as a low-level, thin wrapper, not as a validation layer. Validate only inputs and state whose misuse could make the wrapper itself crash.
  Document those preconditions and enforce programmer errors with `std.debug.assert`.
- Data and parameters passed directly from the user to Vulkan or Metal are the user's responsibility. Do not duplicate native validation or maintain shadow
  state solely to validate them; users should enable the Vulkan validation layer or Metal API validation during development.
- NoGraphicsAPI resource destruction is immediate. In applications, examples, and tests, destroy resources only after no recorded or executing GPU frame
  uses them. Wait for the submission timeline value covering the final use, or use the optional NoGraphicsAPIUtility `DeleteQueue` to defer destruction.
- At shutdown, call `waitIdle`, drain every NoGraphicsAPIUtility `DeleteQueue`, and then destroy resources and the device.
- Wait for every submitted frame to drain before calling `destroyDevice`.
- Group APIs logically in the public API file; for example, keep all command buffer APIs together.
- Keep each public resource creation function next to its matching destruction function. Keep the shared lifetime policy in one place rather than repeating it
  for individual resource types.
- Do not abort for programming errors. Enforce their documented preconditions with asserts, which release builds omit.
- Return error unions only for invalid external input data and initialization failures, and report details through the error callback.
- Do not check for memory allocation failures. We cannot recover from running out of memory; managing memory usage properly is the user's responsibility.
  Where an allocator returns an error, use `catch @panic("out of memory")` rather than propagating it.
- Avoid memory allocations, including short-lived local lists; prefer fixed-capacity arrays sized by documented limits.
- Avoid `anytype`-heavy generic code and complex comptime metaprogramming.
- Avoid trivial single-line wrapper functions.
- Avoid trivial single-field wrapper structs.
- Do not use hash maps or ordered maps.
- Avoid generic library algorithms where a straightforward loop is clearer.
- Queues and command pools are externally synchronized; independent pools, queues, resource creation, and timeline waits can run concurrently.
  Keep ordinary recording, lookup and submission free of internal mutexes. Restrict synchronization to shared bookkeeping and native API requirements;
  use atomics where appropriate, and keep diagnostic workarounds isolated from production paths. Document backend-specific exceptions with the implementation.
- Avoid copying large user data structures. Pass small structs, slices and `GpuRange` by value so their fields travel in registers; pass large
  structures by const pointer. Use slices for array data in structures and function parameters. Do not retain a slice of a temporary past the call.
- Give public API structure fields useful default values. At call sites, initialize only the non-default fields, using decl literals and
  anonymous struct literals (`.{ .vertex_count = 3 }`).
- Vertex and pixel shaders and their variants share a shader source file. Put entirely different shaders in separate files. Do not combine unrelated shaders
  behind preprocessor conditionals.
- Keep each shader source file's CPU-shared declarations in its own matching shared header. Keep shader-specific root data and constants in that header;
  put types used by multiple shader files in a neutral common header.
- Use shared (CPU/GPU) C headers for structs, enums, and constants, read by Zig through translate-c. Do not put arbitrary code in them behind
  `#if !defined(__SLANG__)` blocks. Mirror a header by hand only when translate-c cannot parse it, and check the mirrored layout.
- Always review code for performance issues before considering work complete.
- Line length is 160 characters. Please don't chop expressions to multiple lines if not needed.

# Documentation Guidelines

- Keep contributor rules generic. Put implementation details, limits, build commands and platform exceptions in their relevant documentation.
- Describe the current design and remaining limitations. Remove obsolete workarounds and narratives about fixed intermediate versions.
- Keep Markdown documentation short, precise, and focused on user-facing behavior and fidelity to the *No Graphics API* design. Emphasize GPU pointers,
  the root ABI, and relevant Vulkan extensions. Avoid obvious Zig conventions, internal plumbing, exhaustive `Desc` or API catalogs, and sample-specific
  asset or format details better left in source files. Keep example descriptions brief.
