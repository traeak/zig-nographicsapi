# Metal 4 validation

The results below come from the original C++ NoGraphicsAPI Metal backend, which shares this backend's design. They were
obtained on Apple M3 Max with macOS 26.6.2, Xcode 27 and stock Slang 2026.18.2; M1/M2 execution was not verified.
See [Metal implementation](metal-support.md) for the status of this backend.

## Running with validation

Metal API validation and shader validation are runtime switches of Apple's tools, so they apply to the tests and examples
without rebuilding:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 zig build test
```

## Observed behavior

Shared root layouts, GPU pointers, descriptor indexing/copies, direct and indirect commands, texture transfers, viewport
orientation, presentation and command-pool reuse passed with API validation. Concurrency coverage included independent
recording, cross-queue waits, heap creation/destruction, concurrent updates to disjoint descriptor slots, and reaching the
64-heap address-index limit while repeatedly wrapping its snapshot storage.

Resource-free producer barriers allowed later vertex/mesh work to overlap earlier fragment work, both within a submission and
across submissions with timestamps enabled; broad barriers waited as expected. A three-drawable presentation probe also
observed overlap between frames. This establishes a scheduling opportunity on M3 Max, not a guaranteed speedup.
Drawable availability can still stall presentation.

Native render timestamps interleaved with draws can leave counter entries unwritten; see
[known driver issues](known-driver-issues.md#metal-4--macos-2662-render-timestamps-leave-counter-entries-unwritten).

## MetalTools limitations

- Concurrent sampler creation/destruction can crash API validation in `MTLSamplerDescriptorHashMap`.
  A native Metal-only stress test reproduces it; uninstrumented execution passes.
- Shader validation cannot reliably enumerate placed resources through heap residency. When `MTL_SHADER_VALIDATION` is set,
  the backend registers individual resources and serializes MetalTools residency enumeration.
- Shader instrumentation of native indirect mesh commands can crash in `resolvedSharedPacketData`
  or leave readbacks unwritten. Direct commands pass with instrumentation; indirect commands pass
  with API validation alone.

These diagnostic accommodations do not change production commands.
