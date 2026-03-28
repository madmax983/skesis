# skesis

A high-performance Entity Component System for Rust.

```toml
[dependencies]
skesis = "0.1"
```

## Quick Start

```rust
use skesis::World;

struct Position { x: f32, y: f32 }
struct Velocity { dx: f32, dy: f32 }

let mut world = World::new();

let entity = world.spawn_empty();
world.add_component(entity, Position { x: 0.0, y: 0.0 });
world.add_component(entity, Velocity { dx: 1.0, dy: 2.0 });

// Iterate matching entities
world.for_each_pair::<Position, Velocity>(|entity, pos, vel| {
    println!("{:?}: ({}, {})", entity, pos.x, pos.y);
});

// Tuple queries for 1-8 component types
for (entity, (pos, vel)) in world.query_tuple::<(&Position, &Velocity)>() {
    // ...
}
```

## Features

### Hybrid Storage
- **Archetypes** — contiguous typed columns (`Vec<T>`) grouped by component shape
- **Sparse sets** — `register_sparse::<T>()` for rare components (no archetype fragmentation)
- **Entity generations** — 64-bit IDs with index + generation for safe reuse

### Queries
- **Tuple queries** — `query_tuple::<(&A, &B, &C)>()` for 1-8 component types
- **Specialized queries** — `query_pair`, `for_each_pair`, `for_each_mut` (optimized hot path)
- **Query plans** — precompiled archetype matches with prebound chunk pointers
- **Without filter** — `query_pair_without::<A, B, Exclude>()`
- **Optional** — `query_with_optional::<T, Opt>()`
- **Toggle-aware** — `query_toggled`, `for_each_pair_toggled` respect disabled components

### Component Lifecycle
- `add_component` / `remove_component` — archetype transitions with cached edges
- `set_component` — write value + notify observers + bump change detection
- `get_component_mut` — silent hot-path mutation (zero overhead)
- `disable_component` / `enable_component` — toggle without archetype change

### Observers
- `on_add::<T>()` — fires when a component is first added
- `on_remove::<T>()` — fires when a component is removed
- `on_set::<T>()` — fires on explicit `set_component` calls (not `get_mut`)

### Relationships & Hierarchies
- **Typed directed edges** — `add_relation::<ChildOf>(child, parent)`
- **Hierarchy sugar** — `add_child`, `remove_child`, `children`, `parent`, `is_root`, `is_leaf`
- **Recursive despawn** — `despawn_recursive::<ChildOf>(root)` kills entire subtree
- **Up traversal** — `find_up` (component inheritance), `ancestors`, `depth`

### Change Detection
- **Anchor+delta model** — zero write-path overhead
- Column-level version counter for fast "did anything change?" exit
- Per-entity byte-level diffing with AVX2 SIMD acceleration
- Per-reader snapshots for independent system change tracking
- `ChangeHistory` separated from World (serializable for networking/replay)

### Systems & Scheduling
- Exclusive and parallel system functions
- `.before(other_fn)` / `.after(other_fn)` ordering with topological sort
- Declared access metadata for automatic conflict detection
- Deterministic parallel batching via Rayon

### Serialization
- `save_state()` / `load_state()` — raw byte world snapshots
- Zero-copy, zero-serde overhead
- Perfect for save games, networking, and replay

## Performance

Benchmarked against Bevy ECS 0.17 on identical workloads (1k entities):

| Benchmark | Skesis | vs Bevy |
|-----------|--------|---------|
| `has` | 1.7 us | **1.6x faster** |
| `get_not_found` | 230 ns | **8.2x faster** |
| `get_mut` | 3.9 us | **2.0x faster** |
| `create_delete_empty` | 14.9 us | **3.8x faster** |
| `remove_component` | 77.8 us | **1.7x faster** |
| `query_pair_uncached` | 1.1 us | **1.7x faster** |
| `query_optional` | 911 ns | **1.4x faster** |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
