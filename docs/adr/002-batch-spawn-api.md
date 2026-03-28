# ADR-002: Batch Spawn API (`spawn_with`)

**Status:** Accepted
**Date:** 2026-03-27
**Context:** ADR-001 identified spawn-with-components as the #1 optimization target (42 ns/entity, 6.8x slower than empty spawn)

## Problem

Spawning an entity with N components requires N+1 operations:
1. `spawn_empty()` — place in empty archetype
2. `add_component(A)` — migrate to `{A}` archetype
3. `add_component(B)` — migrate to `{A, B}` archetype

Each migration physically moves all entity data between archetype column storage.
For 2 components: 2 data moves for what should be 1 direct insert.

## Solution

### `SpawnBundle` trait + tuple macro

A compile-time trait that knows all component types, enabling direct insertion:

```rust
pub trait SpawnBundle: 'static + Send + Sync {
    fn component_set() -> ComponentSet;
    fn register(world: &mut World);
    fn push_components(self, archetype: &mut Archetype);
    fn for_each_type_id(f: impl FnMut(TypeId));
}
```

Macro-generated for tuples of 1-8 components. Usage:

```rust
let entity = world.spawn_with((
    Position { x: 0.0, y: 0.0 },
    Velocity { dx: 1.0, dy: 2.0 },
));
```

### Archetype cache

Bundle type → archetype mapping is cached in `bundle_archetype_cache: Vec<(TypeId, ArchetypeId)>`.
First call builds the ComponentSet and creates/finds the archetype. Subsequent calls
do a cheap linear scan (typically 1-3 entries).

### Observer firing order

On-add observers fire after ALL bundle components are placed, so observers for
component A can read sibling component B. This matches Bevy's semantics and is
the correct behavior for most use cases.

## Initial Attempt: No Cache (FAILED)

Without the archetype cache, `spawn_with` was **1.66x slower** than `add_component` chain:
- `add_component` chain: 980 us (10.2 Melem/s)
- `spawn_with` (no cache): 1.63 ms (6.1 Melem/s)

**Root cause:** Rebuilding `ComponentSet` (Vec allocation + TypeId sorting) and
HashMap lookup on every single spawn call. The `add_component` path had transition
caches that avoided this.

## Final Results

| Benchmark (10K entities) | Time | Throughput | Speedup |
|--------------------------|------|------------|---------|
| `add_component` chain (baseline) | 424 us | 23.6 Melem/s | 1.0x |
| **`spawn_with` (2 components)** | **202 us** | **49.5 Melem/s** | **2.1x** |
| **`spawn_with` (3 components)** | **241 us** | **41.5 Melem/s** | **1.8x** |

**2.1x speedup** for the 2-component case. The 3-component case is only 19% more
expensive, showing the per-component overhead (column downcast + push) is small.

## Files Changed

- `src/bundle.rs` — New: SpawnBundle trait + tuple macro impls
- `src/archetype.rs` — `track_entity` made pub(crate), added `push_typed_component`
- `src/world.rs` — `spawn_with` method, `fire_on_add_by_type_id`, `bundle_archetype_cache`
- `src/lib.rs` — Module registration + export
- `benches/micro.rs` — Expanded benchmark suite

## Lessons Learned

1. **Cache all the things on hot paths.** The naive implementation was slower than
   the existing code because it did redundant work (ComponentSet allocation, HashMap
   lookup) that the `add_component` path had already cached.

2. **Zero-allocation iteration matters.** Switching from `Vec<TypeId>` return to
   `for_each_type_id(FnMut)` callback eliminated per-entity allocation in the
   change-detection/observer loop.

3. **Measure the optimization, not just the theory.** The first version was worse.
   The archetype cache made it 8x faster than the uncached version.
