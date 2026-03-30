# System Parameter Extraction for Skesis

## Problem

Skesis systems are `fn(&mut World)`. The single `&mut self` borrow forces users into
workaround patterns (`resource_scope`, `split_resource_mut`, collect-then-apply) when
a system needs simultaneous access to resources and components. These patterns add
overhead and complexity that bevy avoids via compile-time parameter extraction.

## Solution

A `#[system]` proc macro that transforms typed function parameters into pre-split
borrows from World internals. Systems declare what they access through their signature;
the framework validates non-aliasing and provides direct field access with zero overhead.

## Architecture

Four components:

### 1. UnsafeWorldCell (`src/unsafe_cell.rs`)

`Copy` wrapper around `*mut World`. Provides unchecked access to World's internal fields.
All unsafe is centralized here. The scheduler validates non-aliasing via `SystemAccess`
before any system runs.

Methods: `get_resource`, `get_resource_mut`, `archetypes`, `archetypes_mut`,
`get_component`, `get_component_mut`, `world_mut`, `events`, `events_mut`.

### 2. SystemParam trait (`src/param.rs`)

```rust
pub trait SystemParam: Sized {
    type State: Send + Sync + 'static;
    fn access(access: &mut SystemAccess);
    fn init(world: &mut World) -> Self::State;
    unsafe fn get<'w>(cell: UnsafeWorldCell<'w>, state: &'w mut Self::State) -> Self;
}
```

Parameter types:

| Type | Access | State |
|---|---|---|
| `Res<T>` | reads_resource | () |
| `ResMut<T>` | writes_resource | () |
| `Query<Q>` | per-component reads/writes | Cached archetype matches |
| `Commands` | deferred | CommandRecorder |
| `Local<T>` | private | T |
| `EventReader<E>` | reads events | Cursor |
| `EventWriter<E>` | writes events | () |

### 3. WorldQuery trait (`src/query.rs`)

```rust
pub unsafe trait WorldQuery {
    type Item<'a>;
    const MUTABLE: bool;
    fn access(access: &mut SystemAccess);
    fn matches(archetype: &Archetype) -> bool;
    unsafe fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a>;
}
```

Implemented for `&T`, `&mut T`, `Option<&T>`, and tuples up to 8 elements.
Query iteration is cache-friendly archetype-column access.

### 4. #[system] proc macro (`skesis-macros/`)

Transforms:
```rust
#[system]
fn my_system(scene: ResMut<Scene>, query: Query<(&A, &mut B)>) { ... }
```

Into:
- Original function body (unchanged)
- `SystemAccess` declaration (auto-generated from params)
- State init function
- Runner that creates `UnsafeWorldCell`, extracts each param, calls body

## Crate structure

```
skesis/
  src/unsafe_cell.rs   -- UnsafeWorldCell
  src/param.rs         -- SystemParam + Res/ResMut/Local/Commands/Events
  src/query.rs         -- WorldQuery + tuple impls
  skesis-macros/       -- proc-macro crate
    src/lib.rs         -- #[system] macro
```

## Safety argument

1. UnsafeWorldCell methods access disjoint World fields (resources vs archetypes vs events)
2. SystemParam::access declares what each parameter touches
3. The scheduler checks SystemAccess for conflicts before parallel execution
4. Within a single system, the proc macro generates accesses to non-overlapping fields
5. Query's WorldQuery trait ensures component column borrows are disjoint (same
   principle as existing `components_ref_and_mut`)
