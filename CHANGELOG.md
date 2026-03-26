# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-26

Initial release of Skesis ECS.

### Features

- Hybrid archetype + sparse set storage engine
- Anchor+delta change detection (O(stride) targeted byte comparison)
- Typed relationships with forward/reverse adjacency indices
- Observer system with direct pointer cast (no `dyn Any` downcast)
- Query system with cached query plans
- System scheduler with topological sort and automatic parallelism
- World snapshots via raw byte serialization
- Plugin system for modular composition
- Event storage and dispatch
- Resource store for singleton data

### Performance

- Archetype columns: sorted `Vec` + binary search replacing `HashMap` ([9c8fea8])
- RelationshipStore: O(degree) lookups via forward/reverse adjacency indices ([6a0cb54])
- SparseSet: 4x memory reduction via `NonZeroU32` niche optimization ([0ef74a7])
- Observers: direct pointer cast replacing `dyn Any` downcast ([4d57fe2])
- Change detection: O(stride) targeted byte comparison instead of O(N) full column scan ([7b7a41a])

[0.1.0]: https://github.com/madmax983/skesis/releases/tag/v0.1.0

[9c8fea8]: https://github.com/madmax983/skesis/commit/9c8fea8
[6a0cb54]: https://github.com/madmax983/skesis/commit/6a0cb54
[0ef74a7]: https://github.com/madmax983/skesis/commit/0ef74a7
[4d57fe2]: https://github.com/madmax983/skesis/commit/4d57fe2
[7b7a41a]: https://github.com/madmax983/skesis/commit/7b7a41a
