# Vendored Dependencies

## comath

- **Upstream**: https://codeberg.org/InKryption/comath
- **Commit**: `d5fa4cca8e498ed617e0cae45af1893ce2016ff2`
  ("Allow up to 8 function args & tuple elements", 2026-04-28)
- **License**: MIT, Copyright (c) 2023 Trevor Berrange Sanchez — `deps/comath/LICENSE` retained verbatim.
- **Why vendored, not fetched**: upstream targets Zig 0.15. It does **not** compile on 0.17.
  A `zig fetch` pin would point at source we cannot build, so the patched tree lives in-tree.

### Local patches

Apply in order from a pristine upstream checkout:

```bash
git apply deps/comath-patches/0001-zig-0.17-port.patch
```

`0001-zig-0.17-port.patch` covers the 0.15 → 0.17 migration:

| Change | Reason (0.17) |
|---|---|
| `[_]u1{x} ** len` → `@splat(...)` | array multiplication `**` removed |
| `std.meta.Int(.unsigned, n)` → `@Int(.unsigned, n)` | `std.meta.Int` removed |
| `@intFromEnum`/`@enumFromInt` → `@backingInt`/`@fromBackingInt` | auto-migrated by `zig fmt` |
| `.fields` → `field_names` / `field_types` / `field_attrs` | type reflection rewritten SoA |
| `std.builtin.Type.StructField` → `@Struct(layout, null, names, types, attrs)` + `std.lang.Type.Struct.FieldAttributes` | AoS struct-metadata type removed |
| `.params` → `.param_types` | `lang.Type.Fn` is SoA |
| `EnumSet.initFull()` / `.initEmpty()` → `.full` / `.empty` | initializer syntax changed |

### Re-syncing upstream later

1. `git clone` upstream at the new commit into a scratch dir.
2. `zig fmt` it (handles the enum builtins).
3. `diff -ruN <upstream>/src deps/comath/src > deps/comath-patches/0002-resync.patch`
4. Re-apply the SoA reflection fixes by hand — they are the parts that will conflict.
5. Re-run the verification below.

### Verification

comath's own suite was verified green on Zig 0.17 **in a scratch tree**, not from this
package:

```
$ cd <scratch> && zig build test
All 8 tests passed.
```

It is **not** wired into `zig build test` here, because comath's `build.zig` returns early
when it is not the root package (`if (!is_root) return;`), so its `unit-test` step is
unreachable as a dependency. Our own tests exercise the comath paths we actually use.
