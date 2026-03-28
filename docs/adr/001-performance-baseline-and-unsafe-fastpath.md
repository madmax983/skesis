# ADR-001: Performance Baseline & `unsafe_fastpath` Regression

**Status:** Accepted
**Date:** 2026-03-27
**Context:** First comprehensive performance audit of skesis ECS using the Abrash Method

## Baseline Measurements

**Hardware:** Windows 11, criterion 0.5, release profile
**Test setup:** 10,000 entities with Position (8B) + Velocity (8B), single archetype unless noted

### Query Iteration

| Benchmark | Time (10K entities) | Throughput | ns/entity |
|-----------|---------------------|------------|-----------|
| `for_each_pair` (uncached) | 5.67 us | 1.76 Gelem/s | 0.57 |
| `for_each_pair_with_plan` (cached) | 5.51 us | 1.81 Gelem/s | 0.55 |
| `for_each_mut` (single component) | 951 ns | 10.5 Gelem/s | 0.095 |
| `for_each_mut_with_plan` | 969 ns | 10.3 Gelem/s | 0.097 |

### Query Scaling (cached plan, single archetype)

| Entity count | Time | Throughput | ns/entity |
|-------------|------|------------|-----------|
| 100 | 60 ns | 1.65 Gelem/s | 0.60 |
| 1,000 | 569 ns | 1.76 Gelem/s | 0.57 |
| 10,000 | 5.5 us | 1.81 Gelem/s | 0.55 |
| 100,000 | 58.7 us | 1.70 Gelem/s | 0.59 |

Linear scaling confirmed. Slight throughput dip at 100K from L2/L3 cache pressure
(100K entities x 24B = 2.4MB, exceeds typical L2).

### Archetype Fragmentation (10K entities, 5 archetypes)

| Benchmark | Time | Throughput |
|-----------|------|------------|
| Uncached (5 archetypes) | 5.89 us | 1.70 Gelem/s |
| Cached plan (5 archetypes) | 5.59 us | 1.79 Gelem/s |

Caching provides ~5% improvement with 5 archetypes. Gap should widen significantly
with 50-100+ archetypes (common in real games).

### Entity Creation

| Benchmark | Time (10K entities) | Throughput | ns/entity |
|-----------|---------------------|------------|-----------|
| `spawn_empty` | 62 us | 159 Melem/s | 6.2 |
| `spawn_empty` + 2x `add_component` | 424 us | 23.5 Melem/s | 42.4 |

**spawn_with_2_components is 6.8x slower than spawn_empty.** Each `add_component` call
triggers an archetype migration (entity moves from archetype A to archetype A+B).
The chain is: empty -> {Position} -> {Position, Velocity}. This is the biggest
optimization opportunity.

## Critical Finding: `unsafe_fastpath` is a Performance Regression

### Hypothesis

The `unsafe_fastpath` feature flag replaces safe slice indexing with raw pointer
arithmetic in query loops. The assumption was that eliminating bounds checks would
improve throughput.

### Measurement

| Benchmark | Safe (default) | `unsafe_fastpath` | Change |
|-----------|---------------|-------------------|--------|
| cached plan | 5.67 us | 7.84 us | **+48% slower** |
| uncached | 5.61 us | 6.52 us | **+11% slower** |

### Analysis

The safe version `for index in 0..entities.len() { entities[index] }` enables the
compiler to:
1. **Prove bounds-check elision** from the loop invariant `index < len`
2. **Leverage aliasing information** from `&[T]` (slices guarantee no overlap)
3. **Auto-vectorize** the inner loop more aggressively

The unsafe version uses `*entities_ptr.add(index)` which:
1. Loses slice aliasing guarantees (raw pointers may alias)
2. Prevents auto-vectorization patterns the optimizer recognizes
3. Adds no benefit since LLVM already elides the bounds checks

This is a textbook case from Abrash's methodology: **measure everything, assume nothing.**

### Decision

**Removed the `unsafe_fastpath` feature flag.** It added code complexity,
maintenance burden, and safety risk while making performance strictly worse.

## Optimization Roadmap (Priority Order)

### P0: Remove `unsafe_fastpath` — DONE
- Removed feature from Cargo.toml, all 6 cfg-gated blocks in world.rs, README entry
- ~120 lines of unsafe code deleted with zero functional change
- Binary output is identical (feature was opt-in, never default)

### P1: Batch spawn API (`world.spawn_with(...)`) — DONE (see ADR-002)
- 2.1x faster than `add_component` chain for 2-component bundles
- `SpawnBundle` trait + tuple macro impls for 1-8 components

### P2: Expand fragmentation benchmarks — DONE
- Tested 66 archetypes (6 independent tag bits): cached is 13% faster than uncached
- Per-archetype scan cost: ~29 ns (two binary searches on ComponentSet)
- Conclusion: cached plans are worth using but uncached doesn't fall off a cliff until 100+

### P3: Investigate `tuple_query::fetch` Vec allocation — DEFERRED
- The Vec allocation is in the convenience API (`query_tuple`), not the hot path
- Performance-critical code should use `for_each_pair` / `for_each_mut` (zero allocation)
- Not worth the API churn for a path that isn't in any real hot loop
