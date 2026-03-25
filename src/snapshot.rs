//! World serialization via raw byte snapshots.
//!
//! Captures the full ECS state (entities, components, archetypes) as raw bytes
//! for save/load, networking, and replay. Zero serde overhead — just memcpy.
//!
//! **Limitations:**
//! - Snapshots are only valid for the same binary (same struct layouts).
//! - Components must not contain pointers, references, or non-`repr(C)` padding.
//! - Observers, caches, and transient state are not captured.

use std::any::TypeId;

/// A frozen snapshot of the entire World state as raw bytes.
///
/// Contains entity metadata, archetype structure, and component column data.
/// Can be stored, transmitted, or restored with `World::load_state()`.
#[derive(Clone)]
pub struct WorldSnapshot {
    /// Entity generation counters.
    pub(crate) entity_generations: Vec<u32>,
    /// Free entity index pool.
    pub(crate) free_indices: Vec<u32>,
    /// Next entity index to allocate.
    pub(crate) next_entity_index: u32,
    /// Per-archetype snapshot.
    pub(crate) archetypes: Vec<ArchetypeSnapshot>,
}

/// Snapshot of a single archetype's data.
#[derive(Clone)]
pub(crate) struct ArchetypeSnapshot {
    /// The TypeIds that define this archetype's component set.
    pub component_type_ids: Vec<TypeId>,
    /// Entity list (as raw bytes — each Entity is two u32s).
    pub entity_bytes: Vec<u8>,
    /// Entity indices (as raw bytes).
    pub entity_index_bytes: Vec<u8>,
    /// Number of entities in this archetype.
    pub entity_count: usize,
    /// Per-column data: TypeId → (element_stride, raw_bytes).
    pub columns: Vec<(TypeId, usize, Vec<u8>)>,
}

impl WorldSnapshot {
    /// Approximate byte size of this snapshot (for diagnostics).
    pub fn byte_size(&self) -> usize {
        let meta = self.entity_generations.len() * 4 + self.free_indices.len() * 4 + 4; // next_entity_index
        let arch: usize = self
            .archetypes
            .iter()
            .map(|a| {
                a.entity_bytes.len()
                    + a.entity_index_bytes.len()
                    + a.columns
                        .iter()
                        .map(|(_, _, bytes)| bytes.len())
                        .sum::<usize>()
            })
            .sum();
        meta + arch
    }

    /// Number of entities in the snapshot (across all archetypes).
    pub fn entity_count(&self) -> usize {
        self.archetypes.iter().map(|a| a.entity_count).sum()
    }

    /// Number of archetypes in the snapshot.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }
}
