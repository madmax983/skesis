# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-26

### Archetype

- Replace HashMap columns with sorted Vec + binary search ([9c8fea8](https://github.com/madmax983/skesis/commit/9c8fea8165a3f97c453b1e21e7c59a2c24d2a779))

### CI/CD

- Auto-commit generated CHANGELOG.md back to trunk on release ([4fc9c7f](https://github.com/madmax983/skesis/commit/4fc9c7f5c9167217d3f8f1af992d4fb4a63cd001))
- Allow dirty working dir for cargo publish in release workflow ([8327e73](https://github.com/madmax983/skesis/commit/8327e73d9caf45e4ef73d246a6ce901073bd5331))
- Add codecov, dependabot, release workflow, and changelog for v0.1.0 ([62dc8ef](https://github.com/madmax983/skesis/commit/62dc8ef2c43103a0b8f2f13874b751b4498daeac))

### Observers

- Replace dyn Any downcast with direct pointer cast ([4d57fe2](https://github.com/madmax983/skesis/commit/4d57fe277856be3fe0b033dd0837558a005dc8a3))

### RelationshipStore

- O(degree) lookups via forward/reverse adjacency indices ([6a0cb54](https://github.com/madmax983/skesis/commit/6a0cb546c062480fd8b955f8b230ec6b0060e050))

### SparseSet

- 4x memory reduction via NonZeroU32 niche optimization ([0ef74a7](https://github.com/madmax983/skesis/commit/0ef74a7643a670f996a275788a170dd41a22e4cc))

### Is_changed

- O(stride) targeted byte comparison instead of O(N) full column ([7b7a41a](https://github.com/madmax983/skesis/commit/7b7a41afd5270ad794fa20cd64aeca6b7da054e2))
