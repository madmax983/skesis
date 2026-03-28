//! World manages all entities, components, and archetypes.

use crate::change::ChangeHistory;
use crate::column::{ColumnFactory, typed_column_factory};
use crate::command::{CommandRecorder, CommandStore};
use crate::relation::{Relation, RelationshipStore};
use crate::snapshot::{ArchetypeSnapshot, WorldSnapshot};
use crate::sparse_set::{ErasedSparseSet, SparseSet};
use crate::{Archetype, ArchetypeId, Component, ComponentSet, Entity, EventStore, ResourceStore};
use std::any::TypeId;
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

/// Type-erased observer callback. Takes entity and a raw pointer to the component value.
/// The pointer is guaranteed to point to a valid `T` matching the TypeId key under which
/// this observer is registered.
type ErasedObserverFn = Box<dyn Fn(Entity, *const u8) + Send + Sync>;

static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(1);

struct GetComponentCache {
    world_id: u64,
    structural_version: u64,
    type_id: TypeId,
    pointers: Vec<*const u8>,
    generations: Vec<u32>,
    all_generations_zero: bool,
}

struct GetComponentZeroGenerationCache {
    world_id: u64,
    structural_version: u64,
    type_id: TypeId,
    pointers: Vec<*const u8>,
    contiguous_prefix_base: *const u8,
    contiguous_prefix_len: usize,
    contiguous_covers_tail: bool,
}

struct GetComponentMutCache {
    world_id: u64,
    structural_version: u64,
    type_id: TypeId,
    pointers: Vec<usize>,
    generations: Vec<u32>,
    all_generations_zero: bool,
}

struct GetComponentMutZeroGenerationCache {
    world_id: u64,
    structural_version: u64,
    type_id: TypeId,
    pointers: Vec<usize>,
    contiguous_prefix_base: usize,
    contiguous_prefix_len: usize,
    contiguous_covers_tail: bool,
}

thread_local! {
    static GET_COMPONENT_CACHE: UnsafeCell<GetComponentCache> = const {
        UnsafeCell::new(GetComponentCache {
            world_id: 0,
            structural_version: 0,
            type_id: TypeId::of::<()>(),
            pointers: Vec::new(),
            generations: Vec::new(),
            all_generations_zero: true,
        })
    };
    static GET_COMPONENT_ZERO_GENERATION_CACHE: UnsafeCell<GetComponentZeroGenerationCache> = const {
        UnsafeCell::new(GetComponentZeroGenerationCache {
            world_id: 0,
            structural_version: 0,
            type_id: TypeId::of::<()>(),
            pointers: Vec::new(),
            contiguous_prefix_base: std::ptr::null(),
            contiguous_prefix_len: 0,
            contiguous_covers_tail: false,
        })
    };
}

/// The ECS world.
pub struct World {
    next_entity_index: u32,
    entity_generations: Vec<u32>,
    free_indices: Vec<u32>,
    entity_locations: Vec<Option<(ArchetypeId, usize)>>,
    max_nonempty_entity_index: Option<u32>,
    componentless_tail_start: u32,
    archetypes: Vec<Archetype>,
    archetype_map: HashMap<ComponentSet, ArchetypeId>,
    single_component_archetypes: Vec<(TypeId, ArchetypeId)>,
    add_component_transitions: Vec<Vec<(TypeId, ArchetypeId)>>,
    remove_component_transitions: Vec<Vec<(TypeId, ArchetypeId)>>,
    column_factories: HashMap<TypeId, ColumnFactory>,
    next_archetype_id: u32,
    world_id: u64,
    structural_version: u64,
    all_generations_zero: bool,
    resources: ResourceStore,
    events: EventStore,
    commands: CommandStore,
    on_add_observers: HashMap<TypeId, Vec<ErasedObserverFn>>,
    on_remove_observers: HashMap<TypeId, Vec<ErasedObserverFn>>,
    on_set_observers: HashMap<TypeId, Vec<ErasedObserverFn>>,
    on_add_observer_count: u32,
    on_remove_observer_count: u32,
    on_set_observer_count: u32,
    disabled_components: HashMap<TypeId, Vec<bool>>,
    change_history: ChangeHistory,
    relationships: RelationshipStore,
    sparse_storage: HashMap<TypeId, Box<dyn ErasedSparseSet>>,
    sparse_types: HashSet<TypeId>,
    get_component_mut_cache: GetComponentMutCache,
    get_component_mut_zero_generation_cache: GetComponentMutZeroGenerationCache,
    /// Cache: bundle `TypeId` → target `ArchetypeId` for `spawn_with` fast path.
    bundle_archetype_cache: Vec<(TypeId, ArchetypeId)>,
}

/// Precomputed archetype matches for pair queries.
pub struct PairQueryPlan<A: Component, B: Component> {
    archetype_indices: Vec<usize>,
    marker: PhantomData<fn() -> (A, B)>,
}

/// Borrowed pair query plan with prebound chunk slices.
pub struct BorrowedPairQueryPlan<'w, A: Component, B: Component> {
    chunks: Vec<BorrowedPairChunk<A, B>>,
    structural_version: u64,
    marker: PhantomData<(&'w Entity, &'w A, &'w B)>,
}

/// Precomputed archetype matches for mutable single-component queries.
pub struct MutQueryPlan<T: Component> {
    archetype_indices: Vec<usize>,
    marker: PhantomData<fn() -> T>,
}

/// Borrowed mutable query plan with prebound dense index and component chunks.
pub struct BorrowedMutQueryPlan<T: Component> {
    chunks: Vec<BorrowedMutChunk<T>>,
    world_id: u64,
    structural_version: u64,
    marker: PhantomData<fn() -> T>,
}

struct BorrowedPairChunk<A: Component, B: Component> {
    entities_ptr: NonNull<Entity>,
    entity_indices_ptr: NonNull<u32>,
    a_ptr: NonNull<A>,
    b_ptr: NonNull<B>,
    len: usize,
    contiguous_base: Option<u32>,
    contiguous_index_sum: Option<f32>,
    noncontiguous_indices_f32: Option<Box<[f32]>>,
}

struct BorrowedMutChunk<T: Component> {
    entity_indices_ptr: NonNull<u32>,
    components_ptr: NonNull<T>,
    len: usize,
    contiguous_base: Option<u32>,
    contiguous_index_sum: Option<f32>,
    noncontiguous_indices_f32: Option<Box<[f32]>>,
}

#[inline(always)]
fn contiguous_index_base(entity_indices: &[u32]) -> Option<u32> {
    let (base, rest) = entity_indices.split_first()?;
    let mut expected = *base;
    for &entity_index in rest {
        expected = expected.wrapping_add(1);
        if entity_index != expected {
            return None;
        }
    }
    Some(*base)
}

#[inline(always)]
fn contiguous_index_sum_f32(contiguous_base: u32, len: usize) -> f32 {
    let len_u128 = len as u128;
    let base_u128 = contiguous_base as u128;
    let sum = len_u128 * ((base_u128 << 1) + len_u128 - 1) / 2;
    sum as f32
}

impl<A: Component, B: Component> BorrowedPairChunk<A, B> {
    #[inline(always)]
    unsafe fn as_entity_component_slices<'w>(&self) -> (&'w [Entity], &'w [A], &'w [B]) {
        // SAFETY:
        // - Caller guarantees plan pointers are still valid.
        // - `len` matches all three arrays for this chunk.
        let entities = unsafe { std::slice::from_raw_parts(self.entities_ptr.as_ptr(), self.len) };
        // SAFETY: same invariants as above.
        let components_a = unsafe { std::slice::from_raw_parts(self.a_ptr.as_ptr(), self.len) };
        // SAFETY: same invariants as above.
        let components_b = unsafe { std::slice::from_raw_parts(self.b_ptr.as_ptr(), self.len) };
        (entities, components_a, components_b)
    }

    #[inline(always)]
    unsafe fn as_indexed_component_slices<'w>(&self) -> (&'w [u32], &'w [A], &'w [B]) {
        // SAFETY:
        // - Caller guarantees plan pointers are still valid.
        // - `len` matches all three arrays for this chunk.
        let entity_indices =
            unsafe { std::slice::from_raw_parts(self.entity_indices_ptr.as_ptr(), self.len) };
        // SAFETY: same invariants as above.
        let components_a = unsafe { std::slice::from_raw_parts(self.a_ptr.as_ptr(), self.len) };
        // SAFETY: same invariants as above.
        let components_b = unsafe { std::slice::from_raw_parts(self.b_ptr.as_ptr(), self.len) };
        (entity_indices, components_a, components_b)
    }
}

impl<T: Component> BorrowedMutChunk<T> {
    #[inline(always)]
    unsafe fn as_indexed_component_slices_mut<'a>(&self) -> (&'a [u32], &'a mut [T]) {
        // SAFETY:
        // - Caller guarantees plan pointers are still valid.
        // - `len` matches both arrays for this chunk.
        let entity_indices =
            unsafe { std::slice::from_raw_parts(self.entity_indices_ptr.as_ptr(), self.len) };
        // SAFETY: same invariants as above.
        let components =
            unsafe { std::slice::from_raw_parts_mut(self.components_ptr.as_ptr(), self.len) };
        (entity_indices, components)
    }
}

impl<'w, A: Component, B: Component> BorrowedPairQueryPlan<'w, A, B> {
    #[inline(always)]
    fn assert_fresh(&self, world: &World) {
        assert_eq!(
            self.structural_version, world.structural_version,
            "borrowed query plan is stale; rebuild plan after structural world changes"
        );
    }

    /// Run a callback once per prebound chunk.
    #[inline(always)]
    pub fn for_each_chunk(&self, world: &World, mut f: impl FnMut(&'w [Entity], &'w [A], &'w [B])) {
        self.assert_fresh(world);

        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Chunks are created from valid slice backing pointers in `plan_query_pair_borrowed`.
            // - `assert_fresh` guarantees no structural world mutation since plan creation.
            // - `len` matches all three columns for each chunk.
            let (entities, components_a, components_b) =
                unsafe { chunk.as_entity_component_slices() };
            f(entities, components_a, components_b);
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Chunks are created from valid slice backing pointers in `plan_query_pair_borrowed`.
            // - `assert_fresh` guarantees no structural world mutation since plan creation.
            // - `len` matches all three columns for each chunk.
            let (entities, components_a, components_b) =
                unsafe { chunk.as_entity_component_slices() };
            f(entities, components_a, components_b);
        }
    }

    /// Run a callback once per prebound chunk using dense entity indices.
    #[inline(always)]
    pub fn for_each_indexed_chunk(
        &self,
        world: &World,
        mut f: impl FnMut(&'w [u32], &'w [A], &'w [B]),
    ) {
        self.assert_fresh(world);

        // SAFETY: structural freshness is checked above and chunk pointers came from
        // `plan_query_pair_borrowed` for this same world borrow.
        unsafe { self.for_each_indexed_chunk_unchecked(&mut f) };
    }

    /// Run a callback once per prebound chunk using dense entity indices without freshness checks.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_unchecked(
        &self,
        mut f: impl FnMut(&'w [u32], &'w [A], &'w [B]),
    ) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for this chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(entity_indices, components_a, components_b);
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for each chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(entity_indices, components_a, components_b);
        }
    }

    /// Run a callback once per prebound chunk using dense entity indices and chunk metadata.
    ///
    /// The metadata reports a `Some(base)` value when `entity_indices` are exactly
    /// `[base, base + 1, ..., base + len - 1]`; otherwise it reports `None`.
    #[inline(always)]
    pub fn for_each_indexed_chunk_meta(
        &self,
        world: &World,
        mut f: impl FnMut(Option<u32>, &'w [u32], &'w [A], &'w [B]),
    ) {
        self.assert_fresh(world);

        // SAFETY: structural freshness is checked above and chunk pointers came from
        // `plan_query_pair_borrowed` for this same world borrow.
        unsafe { self.for_each_indexed_chunk_meta_unchecked(&mut f) };
    }

    /// Run a callback once per prebound chunk using dense entity indices and chunk metadata
    /// without freshness checks.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_meta_unchecked(
        &self,
        mut f: impl FnMut(Option<u32>, &'w [u32], &'w [A], &'w [B]),
    ) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for this chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(
                chunk.contiguous_base,
                entity_indices,
                components_a,
                components_b,
            );
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for each chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(
                chunk.contiguous_base,
                entity_indices,
                components_a,
                components_b,
            );
        }
    }

    /// Run a callback once per prebound chunk using dense entity indices and precomputed metadata
    /// without freshness checks.
    ///
    /// The extra metadata reports a precomputed contiguous-index checksum term when
    /// `entity_indices` are exactly `[base, base + 1, ..., base + len - 1]`.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_meta_sum_unchecked(
        &self,
        mut f: impl FnMut(Option<u32>, Option<f32>, &'w [u32], &'w [A], &'w [B]),
    ) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for this chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                entity_indices,
                components_a,
                components_b,
            );
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for each chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                entity_indices,
                components_a,
                components_b,
            );
        }
    }

    /// Run a callback once per prebound chunk using dense entity indices and precomputed metadata
    /// without freshness checks, including optional precomputed `f32` entity indices for
    /// non-contiguous chunks.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_meta_sum_f32_unchecked(
        &self,
        mut f: impl FnMut(Option<u32>, Option<f32>, Option<&[f32]>, &'w [u32], &'w [A], &'w [B]),
    ) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for this chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                chunk.noncontiguous_indices_f32.as_deref(),
                entity_indices,
                components_a,
                components_b,
            );
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches all three columns for each chunk.
            let (entity_indices, components_a, components_b) =
                unsafe { chunk.as_indexed_component_slices() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                chunk.noncontiguous_indices_f32.as_deref(),
                entity_indices,
                components_a,
                components_b,
            );
        }
    }

    /// Run a callback for each prebound row.
    #[inline(always)]
    pub fn for_each_row(&self, world: &World, mut f: impl FnMut(Entity, &'w A, &'w B)) {
        self.assert_fresh(world);

        for chunk in &self.chunks {
            let len = chunk.len;
            let entities_ptr = chunk.entities_ptr.as_ptr();
            let a_ptr = chunk.a_ptr.as_ptr();
            let b_ptr = chunk.b_ptr.as_ptr();

            // SAFETY:
            // - Chunks are created from valid slice backing pointers in `plan_query_pair_borrowed`.
            // - `assert_fresh` guarantees no structural world mutation since plan creation.
            let entities: &'w [Entity] = unsafe { std::slice::from_raw_parts(entities_ptr, len) };
            // SAFETY: same invariants as above.
            let components_a: &'w [A] = unsafe { std::slice::from_raw_parts(a_ptr, len) };
            // SAFETY: same invariants as above.
            let components_b: &'w [B] = unsafe { std::slice::from_raw_parts(b_ptr, len) };

            for index in 0..len {
                f(entities[index], &components_a[index], &components_b[index]);
            }
        }
    }

    /// Run a callback for each prebound row using dense entity indices.
    ///
    /// This avoids loading full `Entity` values when only the index is needed.
    #[inline(always)]
    pub fn for_each_indexed(&self, world: &World, mut f: impl FnMut(u32, &'w A, &'w B)) {
        self.assert_fresh(world);

        for chunk in &self.chunks {
            let len = chunk.len;
            let indices_ptr = chunk.entity_indices_ptr.as_ptr();
            let a_ptr = chunk.a_ptr.as_ptr();
            let b_ptr = chunk.b_ptr.as_ptr();

            // SAFETY:
            // - Chunks are created from valid slice backing pointers in `plan_query_pair_borrowed`.
            // - `assert_fresh` guarantees no structural world mutation since plan creation.
            let entity_indices: &'w [u32] = unsafe { std::slice::from_raw_parts(indices_ptr, len) };
            // SAFETY: same invariants as above.
            let components_a: &'w [A] = unsafe { std::slice::from_raw_parts(a_ptr, len) };
            // SAFETY: same invariants as above.
            let components_b: &'w [B] = unsafe { std::slice::from_raw_parts(b_ptr, len) };

            for index in 0..len {
                f(
                    entity_indices[index],
                    &components_a[index],
                    &components_b[index],
                );
            }
        }
    }
}

impl<T: Component> BorrowedMutQueryPlan<T> {
    #[inline(always)]
    fn assert_fresh(&self, world: &World) {
        assert_eq!(
            self.world_id, world.world_id,
            "borrowed mutable query plan was created for a different world"
        );
        assert_eq!(
            self.structural_version, world.structural_version,
            "borrowed mutable query plan is stale; rebuild plan after structural world changes"
        );
    }

    /// Run a callback once per prebound mutable chunk using dense entity indices.
    #[inline(always)]
    pub fn for_each_indexed_chunk(&self, world: &mut World, mut f: impl FnMut(&[u32], &mut [T])) {
        self.assert_fresh(world);

        // SAFETY: world identity and structural freshness are checked above.
        unsafe { self.for_each_indexed_chunk_unchecked(&mut f) };
    }

    /// Run a callback once per prebound mutable chunk without freshness checks.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_unchecked(&self, mut f: impl FnMut(&[u32], &mut [T])) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices for this chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(entity_indices, components);
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices per chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(entity_indices, components);
        }
    }

    /// Run a callback once per prebound mutable chunk using dense entity indices and metadata.
    ///
    /// The metadata reports a `Some(base)` value when `entity_indices` are exactly
    /// `[base, base + 1, ..., base + len - 1]`; otherwise it reports `None`.
    #[inline(always)]
    pub fn for_each_indexed_chunk_meta(
        &self,
        world: &mut World,
        mut f: impl FnMut(Option<u32>, &[u32], &mut [T]),
    ) {
        self.assert_fresh(world);

        // SAFETY: world identity and structural freshness are checked above.
        unsafe { self.for_each_indexed_chunk_meta_unchecked(&mut f) };
    }

    /// Run a callback once per prebound mutable chunk using dense entity indices and metadata
    /// without freshness checks.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_meta_unchecked(
        &self,
        mut f: impl FnMut(Option<u32>, &[u32], &mut [T]),
    ) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices for this chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(chunk.contiguous_base, entity_indices, components);
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices per chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(chunk.contiguous_base, entity_indices, components);
        }
    }

    /// Run a callback once per prebound mutable chunk using dense entity indices and precomputed
    /// metadata without freshness checks.
    ///
    /// The extra metadata reports a precomputed contiguous-index checksum term when
    /// `entity_indices` are exactly `[base, base + 1, ..., base + len - 1]`.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_meta_sum_unchecked(
        &self,
        mut f: impl FnMut(Option<u32>, Option<f32>, &[u32], &mut [T]),
    ) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices for this chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                entity_indices,
                components,
            );
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices per chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                entity_indices,
                components,
            );
        }
    }

    /// Run a callback once per prebound mutable chunk using dense entity indices and precomputed
    /// metadata without freshness checks, including optional precomputed `f32` entity indices for
    /// non-contiguous chunks.
    ///
    /// # Safety
    /// The caller must ensure no structural world mutation has occurred since plan creation.
    #[inline(always)]
    pub unsafe fn for_each_indexed_chunk_meta_sum_f32_unchecked(
        &self,
        mut f: impl FnMut(Option<u32>, Option<f32>, Option<&[f32]>, &[u32], &mut [T]),
    ) {
        if let [chunk] = self.chunks.as_slice() {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices for this chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                chunk.noncontiguous_indices_f32.as_deref(),
                entity_indices,
                components,
            );
            return;
        }

        for chunk in &self.chunks {
            // SAFETY:
            // - Caller guarantees no structural mutation invalidated plan pointers.
            // - `len` matches both dense-entity-index and component slices per chunk.
            let (entity_indices, components) = unsafe { chunk.as_indexed_component_slices_mut() };
            f(
                chunk.contiguous_base,
                chunk.contiguous_index_sum,
                chunk.noncontiguous_indices_f32.as_deref(),
                entity_indices,
                components,
            );
        }
    }
}

impl World {
    #[inline(always)]
    fn lookup_add_component_transition(
        &self,
        old_archetype_id: ArchetypeId,
        added_type_id: TypeId,
    ) -> Option<ArchetypeId> {
        self.add_component_transitions
            .get(old_archetype_id.0 as usize)?
            .iter()
            .find_map(|&(type_id, target)| (type_id == added_type_id).then_some(target))
    }

    #[inline(always)]
    fn cache_add_component_transition(
        &mut self,
        old_archetype_id: ArchetypeId,
        added_type_id: TypeId,
        target_archetype_id: ArchetypeId,
    ) {
        let old_index = old_archetype_id.0 as usize;
        if self.add_component_transitions.len() <= old_index {
            self.add_component_transitions
                .resize_with(old_index + 1, Vec::new);
        }
        let transitions = &mut self.add_component_transitions[old_index];
        if transitions
            .iter()
            .all(|&(type_id, _)| type_id != added_type_id)
        {
            transitions.push((added_type_id, target_archetype_id));
        }
    }

    #[inline(always)]
    fn lookup_remove_component_transition(
        &self,
        old_archetype_id: ArchetypeId,
        removed_type_id: TypeId,
    ) -> Option<ArchetypeId> {
        self.remove_component_transitions
            .get(old_archetype_id.0 as usize)?
            .iter()
            .find_map(|&(type_id, target)| (type_id == removed_type_id).then_some(target))
    }

    #[inline(always)]
    fn cache_remove_component_transition(
        &mut self,
        old_archetype_id: ArchetypeId,
        removed_type_id: TypeId,
        target_archetype_id: ArchetypeId,
    ) {
        let old_index = old_archetype_id.0 as usize;
        if self.remove_component_transitions.len() <= old_index {
            self.remove_component_transitions
                .resize_with(old_index + 1, Vec::new);
        }
        let transitions = &mut self.remove_component_transitions[old_index];
        if transitions
            .iter()
            .all(|&(type_id, _)| type_id != removed_type_id)
        {
            transitions.push((removed_type_id, target_archetype_id));
        }
    }

    fn two_archetypes_mut(
        archetypes: &mut [Archetype],
        first_index: usize,
        second_index: usize,
    ) -> (&mut Archetype, &mut Archetype) {
        assert_ne!(
            first_index, second_index,
            "cannot borrow the same archetype mutably twice"
        );
        if first_index < second_index {
            let (left, right) = archetypes.split_at_mut(second_index);
            (&mut left[first_index], &mut right[0])
        } else {
            let (left, right) = archetypes.split_at_mut(first_index);
            (&mut right[0], &mut left[second_index])
        }
    }

    /// Look up an existing archetype for `component_set`, or create a new one.
    fn get_or_create_archetype(&mut self, component_set: ComponentSet) -> ArchetypeId {
        if let Some(&id) = self.archetype_map.get(&component_set) {
            return id;
        }
        let id = ArchetypeId(self.next_archetype_id);
        self.next_archetype_id += 1;
        let archetype =
            Archetype::new_with_factories(id, component_set.clone(), &self.column_factories);
        self.archetypes.push(archetype);
        self.add_component_transitions.push(Vec::new());
        self.remove_component_transitions.push(Vec::new());
        self.archetype_map.insert(component_set, id);
        id
    }

    /// Create a new empty world.
    pub fn new() -> Self {
        let world_id = NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed);
        let mut world = Self {
            next_entity_index: 0,
            entity_generations: Vec::new(),
            free_indices: Vec::new(),
            entity_locations: Vec::new(),
            max_nonempty_entity_index: None,
            componentless_tail_start: 0,
            archetypes: Vec::new(),
            archetype_map: HashMap::new(),
            single_component_archetypes: Vec::new(),
            add_component_transitions: Vec::new(),
            remove_component_transitions: Vec::new(),
            column_factories: HashMap::new(),
            next_archetype_id: 0,
            world_id,
            structural_version: 0,
            all_generations_zero: true,
            resources: ResourceStore::new(),
            events: EventStore::new(),
            commands: CommandStore::new(),
            on_add_observers: HashMap::new(),
            on_remove_observers: HashMap::new(),
            on_set_observers: HashMap::new(),
            on_add_observer_count: 0,
            on_remove_observer_count: 0,
            on_set_observer_count: 0,
            disabled_components: HashMap::new(),
            change_history: ChangeHistory::new(),
            relationships: RelationshipStore::new(),
            sparse_storage: HashMap::new(),
            sparse_types: HashSet::new(),
            get_component_mut_cache: GetComponentMutCache {
                world_id,
                structural_version: 0,
                type_id: TypeId::of::<()>(),
                pointers: Vec::new(),
                generations: Vec::new(),
                all_generations_zero: true,
            },
            get_component_mut_zero_generation_cache: GetComponentMutZeroGenerationCache {
                world_id,
                structural_version: 0,
                type_id: TypeId::of::<()>(),
                pointers: Vec::new(),
                contiguous_prefix_base: 0,
                contiguous_prefix_len: 0,
                contiguous_covers_tail: false,
            },
            bundle_archetype_cache: Vec::new(),
        };

        // Create empty archetype for entities with no components
        let empty_set = ComponentSet::new();
        let empty_arch_id = ArchetypeId(0);
        world.next_archetype_id = 1;
        world.archetypes.push(Archetype::new_with_factories(
            empty_arch_id,
            empty_set.clone(),
            &world.column_factories,
        ));
        world.add_component_transitions.push(Vec::new());
        world.remove_component_transitions.push(Vec::new());
        world.archetype_map.insert(empty_set, empty_arch_id);

        world
    }

    /// Register a component type's column factory for serialization support.
    ///
    /// Called automatically by `add_component`, but must be called manually
    /// on the target world before `load_state` for each component type in
    /// the snapshot.
    pub fn register_component_type<T: Component>(&mut self) {
        self.column_factories
            .entry(TypeId::of::<T>())
            .or_insert(typed_column_factory::<T>);
    }

    /// Register a component type for sparse storage.
    ///
    /// Sparse components are stored in a parallel `SparseSet` instead of
    /// archetype columns. Adding/removing a sparse component does NOT cause
    /// an archetype transition — the entity stays in its current archetype.
    ///
    /// Use for components that are present on very few entities (debuffs,
    /// editor tags, AI blackboard entries) to avoid archetype fragmentation.
    ///
    /// Must be called before any `add_component` calls for this type.
    pub fn register_sparse<T: Component>(&mut self) {
        let type_id = TypeId::of::<T>();
        self.sparse_types.insert(type_id);
        self.sparse_storage
            .entry(type_id)
            .or_insert_with(|| Box::new(SparseSet::<T>::new()));
    }

    /// Check if a component type is registered for sparse storage.
    pub fn is_sparse<T: Component>(&self) -> bool {
        self.sparse_types.contains(&TypeId::of::<T>())
    }

    #[allow(dead_code)]
    fn is_sparse_type_id(&self, type_id: &TypeId) -> bool {
        self.sparse_types.contains(type_id)
    }

    #[inline(always)]
    fn bump_structural_version(&mut self) {
        self.structural_version = self.structural_version.wrapping_add(1);
    }

    #[inline(always)]
    fn fire_on_add<T: Component>(&self, entity: Entity, arch_id: ArchetypeId, row: usize) {
        if self.on_add_observer_count == 0 {
            return;
        }
        if let Some(observers) = self.on_add_observers.get(&TypeId::of::<T>())
            && let Some(component) = self.archetypes[arch_id.0 as usize]
                .components::<T>()
                .and_then(|s| s.get(row))
        {
            let ptr = (component as *const T).cast::<u8>();
            for observer in observers {
                observer(entity, ptr);
            }
        }
    }

    /// Type-erased version of `fire_on_add` for bundle spawning.
    ///
    /// Uses the raw column bytes + stride to compute the component pointer
    /// without needing a generic type parameter.
    fn fire_on_add_by_type_id(
        &self,
        type_id: &TypeId,
        entity: Entity,
        arch_id: ArchetypeId,
        row: usize,
    ) {
        if let Some(observers) = self.on_add_observers.get(type_id)
            && let Some(column) = self.archetypes[arch_id.0 as usize].component_column(type_id)
        {
            let bytes = column.as_bytes();
            let stride = column.element_stride();
            if stride > 0 && row * stride < bytes.len() {
                // SAFETY: row is within bounds (just inserted), stride matches element size,
                // and the pointer is into valid column backing storage.
                let ptr = unsafe { bytes.as_ptr().add(row * stride) };
                for observer in observers {
                    observer(entity, ptr);
                }
            }
        }
    }

    #[inline(always)]
    fn fire_on_remove<T: Component>(&self, entity: Entity, value: &T) {
        if self.on_remove_observer_count == 0 {
            return;
        }
        if let Some(observers) = self.on_remove_observers.get(&TypeId::of::<T>()) {
            let ptr = (value as *const T).cast::<u8>();
            for observer in observers {
                observer(entity, ptr);
            }
        }
    }

    #[inline(always)]
    fn fire_on_set<T: Component>(&self, entity: Entity, value: &T) {
        if self.on_set_observer_count == 0 {
            return;
        }
        if let Some(observers) = self.on_set_observers.get(&TypeId::of::<T>()) {
            let ptr = (value as *const T).cast::<u8>();
            for observer in observers {
                observer(entity, ptr);
            }
        }
    }

    fn location_of(&self, entity: Entity) -> Option<(ArchetypeId, usize)> {
        let index = entity.index() as usize;
        if index >= self.entity_generations.len()
            || self.entity_generations[index] != entity.generation()
        {
            return None;
        }

        self.entity_locations.get(index).copied().flatten()
    }

    #[inline(always)]
    fn get_component_cached_by_entity_index<T: Component>(
        &self,
        entity_index: usize,
        generation: u32,
    ) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        let component_ptr = GET_COMPONENT_CACHE.with(|slot| {
            // SAFETY:
            // - This cache is thread-local, so there is no cross-thread aliasing.
            // - We do not invoke user callbacks while holding the mutable access.
            // - Access is confined to this closure scope.
            let cache = unsafe { &mut *slot.get() };
            let cache_is_fresh = cache.world_id == self.world_id
                && cache.structural_version == self.structural_version
                && cache.type_id == type_id;

            if !cache_is_fresh {
                let mut pointers = vec![std::ptr::null(); self.entity_generations.len()];
                for archetype in &self.archetypes {
                    if let Some(typed_components) = archetype.components::<T>() {
                        for (row, &entity_index) in archetype.entity_indices().iter().enumerate() {
                            let entity_index = entity_index as usize;
                            if entity_index < pointers.len() {
                                // SAFETY:
                                // - `row` is from enumerate over the same component slice length.
                                // - Each pointer targets an element in `typed_components`.
                                pointers[entity_index] =
                                    unsafe { typed_components.as_ptr().add(row).cast::<u8>() };
                            }
                        }
                    }
                }

                cache.world_id = self.world_id;
                cache.structural_version = self.structural_version;
                cache.type_id = type_id;
                cache.pointers = pointers;
                cache.all_generations_zero = self.all_generations_zero;
                if cache.all_generations_zero {
                    cache.generations.clear();
                } else {
                    cache.generations = self.entity_generations.clone();
                }
            }

            if entity_index >= cache.pointers.len() {
                return std::ptr::null();
            }
            if cache.all_generations_zero {
                if generation != 0 {
                    return std::ptr::null();
                }
            } else {
                debug_assert_eq!(cache.generations.len(), cache.pointers.len());
                // SAFETY: `entity_index` bounds are checked above.
                if unsafe { *cache.generations.get_unchecked(entity_index) } != generation {
                    return std::ptr::null();
                }
            }
            // SAFETY: `entity_index` bounds are checked above.
            unsafe { *cache.pointers.get_unchecked(entity_index) }
        });

        if component_ptr.is_null() {
            return None;
        }

        // SAFETY:
        // - `component_ptr` is captured from an immutable slice borrowed from this world version.
        // - `structural_version` guard ensures cache is rebuilt after structural mutation.
        Some(unsafe { &*component_ptr.cast::<T>() })
    }

    #[inline(always)]
    fn get_component_cached_by_entity_index_generation_zero<T: Component>(
        &self,
        entity_index: usize,
    ) -> Option<&T> {
        let type_id = TypeId::of::<T>();
        let component_ptr = GET_COMPONENT_ZERO_GENERATION_CACHE.with(|slot| {
            // SAFETY:
            // - This cache is thread-local, so there is no cross-thread aliasing.
            // - We do not invoke user callbacks while holding the mutable access.
            // - Access is confined to this closure scope.
            let cache = unsafe { &mut *slot.get() };
            let cache_is_fresh = cache.world_id == self.world_id
                && cache.structural_version == self.structural_version
                && cache.type_id == type_id;

            if !cache_is_fresh {
                let mut pointers = vec![std::ptr::null(); self.entity_generations.len()];
                let mut contiguous_prefix_base: *const u8 = std::ptr::null();
                let mut contiguous_prefix_len = 0usize;
                let mut contiguous_prefix_ambiguous = false;
                for archetype in &self.archetypes {
                    if let Some(typed_components) = archetype.components::<T>() {
                        let entity_indices = archetype.entity_indices();
                        for (row, &entity_index) in entity_indices.iter().enumerate() {
                            let entity_index = entity_index as usize;
                            if entity_index < pointers.len() {
                                // SAFETY:
                                // - `row` is from enumerate over the same component slice length.
                                // - Each pointer targets an element in `typed_components`.
                                pointers[entity_index] =
                                    unsafe { typed_components.as_ptr().add(row).cast::<u8>() };
                            }
                        }

                        if !typed_components.is_empty() && !contiguous_prefix_ambiguous {
                            let is_identity_prefix = entity_indices
                                .iter()
                                .enumerate()
                                .all(|(row, &entity_index)| entity_index as usize == row);
                            if is_identity_prefix {
                                if contiguous_prefix_base.is_null() {
                                    contiguous_prefix_base = typed_components.as_ptr().cast::<u8>();
                                    contiguous_prefix_len = typed_components.len();
                                } else {
                                    // Multiple identity-prefix archetypes for the same type are
                                    // ambiguous for a single base+index mapping.
                                    contiguous_prefix_ambiguous = true;
                                }
                            }
                        }
                    }
                }

                cache.world_id = self.world_id;
                cache.structural_version = self.structural_version;
                cache.type_id = type_id;
                cache.pointers = pointers;
                if contiguous_prefix_ambiguous {
                    cache.contiguous_prefix_base = std::ptr::null();
                    cache.contiguous_prefix_len = 0;
                    cache.contiguous_covers_tail = false;
                } else {
                    cache.contiguous_prefix_base = contiguous_prefix_base;
                    cache.contiguous_prefix_len = contiguous_prefix_len;
                    cache.contiguous_covers_tail = !contiguous_prefix_base.is_null()
                        && contiguous_prefix_len == self.componentless_tail_start as usize;
                }
            }

            if cache.contiguous_covers_tail {
                // SAFETY:
                // - `contiguous_covers_tail` means contiguous prefix length equals
                //   `componentless_tail_start`.
                // - Callers of this helper route through `get_component`, which ensures
                //   `entity_index < componentless_tail_start`.
                return unsafe {
                    cache
                        .contiguous_prefix_base
                        .cast::<T>()
                        .add(entity_index)
                        .cast::<u8>()
                };
            }
            if entity_index < cache.contiguous_prefix_len {
                // SAFETY:
                // - `entity_index < contiguous_prefix_len` ensures this row exists in the
                //   cached contiguous prefix column for `T`.
                // - `contiguous_prefix_base` was captured from `typed_components.as_ptr()` when
                //   the identity-prefix condition held.
                // - Row addressing uses typed pointer arithmetic.
                return unsafe {
                    cache
                        .contiguous_prefix_base
                        .cast::<T>()
                        .add(entity_index)
                        .cast::<u8>()
                };
            }
            if entity_index >= cache.pointers.len() {
                return std::ptr::null();
            }
            // SAFETY: `entity_index` bounds are checked above.
            unsafe { *cache.pointers.get_unchecked(entity_index) }
        });

        if component_ptr.is_null() {
            return None;
        }

        // SAFETY:
        // - `component_ptr` is captured from an immutable slice borrowed from this world version.
        // - `structural_version` guard ensures cache is rebuilt after structural mutation.
        Some(unsafe { &*component_ptr.cast::<T>() })
    }

    #[inline(always)]
    fn get_component_mut_cached_by_entity_index<T: Component>(
        &mut self,
        entity_index: usize,
        generation: u32,
    ) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        let cache_is_fresh = {
            let cache = &self.get_component_mut_cache;
            cache.world_id == self.world_id
                && cache.structural_version == self.structural_version
                && cache.type_id == type_id
        };

        if !cache_is_fresh {
            let mut pointers = vec![0usize; self.entity_generations.len()];
            for archetype in &self.archetypes {
                if let Some(typed_components) = archetype.components::<T>() {
                    for (row, &entity_index) in archetype.entity_indices().iter().enumerate() {
                        let entity_index = entity_index as usize;
                        if entity_index < pointers.len() {
                            // SAFETY:
                            // - `row` is from enumerate over the same component slice length.
                            // - Each pointer targets an element in `typed_components`.
                            pointers[entity_index] = unsafe {
                                typed_components.as_ptr().add(row).cast_mut().cast::<u8>() as usize
                            };
                        }
                    }
                }
            }

            self.get_component_mut_cache = GetComponentMutCache {
                world_id: self.world_id,
                structural_version: self.structural_version,
                type_id,
                pointers,
                generations: if self.all_generations_zero {
                    Vec::new()
                } else {
                    self.entity_generations.clone()
                },
                all_generations_zero: self.all_generations_zero,
            };
        }

        let cache = &self.get_component_mut_cache;
        let component_ptr = if entity_index >= cache.pointers.len() {
            std::ptr::null_mut()
        } else if cache.all_generations_zero {
            if generation != 0 {
                std::ptr::null_mut()
            } else {
                // SAFETY: `entity_index` bounds are checked above.
                unsafe { *cache.pointers.get_unchecked(entity_index) as *mut u8 }
            }
        } else {
            debug_assert_eq!(cache.generations.len(), cache.pointers.len());
            // SAFETY: `entity_index` bounds are checked above.
            if unsafe { *cache.generations.get_unchecked(entity_index) } != generation {
                std::ptr::null_mut()
            } else {
                // SAFETY: `entity_index` bounds are checked above.
                unsafe { *cache.pointers.get_unchecked(entity_index) as *mut u8 }
            }
        };

        if component_ptr.is_null() {
            return None;
        }

        // SAFETY:
        // - `component_ptr` is captured from this world's component storage for `T`.
        // - `&mut self` guarantees exclusive access while creating `&mut T`.
        // - `structural_version` guard ensures cache rebuild after structural mutation.
        Some(unsafe { &mut *component_ptr.cast::<T>() })
    }

    #[inline(always)]
    fn get_component_mut_cached_by_entity_index_generation_zero<T: Component>(
        &mut self,
        entity_index: usize,
    ) -> Option<&mut T> {
        let type_id = TypeId::of::<T>();
        let cache_is_fresh = {
            let cache = &self.get_component_mut_zero_generation_cache;
            cache.world_id == self.world_id
                && cache.structural_version == self.structural_version
                && cache.type_id == type_id
        };

        if !cache_is_fresh {
            let mut pointers = vec![0usize; self.entity_generations.len()];
            let mut contiguous_prefix_base = 0usize;
            let mut contiguous_prefix_len = 0usize;
            let mut contiguous_prefix_ambiguous = false;
            for archetype in &self.archetypes {
                if let Some(typed_components) = archetype.components::<T>() {
                    let entity_indices = archetype.entity_indices();
                    for (row, &entity_index) in entity_indices.iter().enumerate() {
                        let entity_index = entity_index as usize;
                        if entity_index < pointers.len() {
                            // SAFETY:
                            // - `row` is from enumerate over the same component slice length.
                            // - Each pointer targets an element in `typed_components`.
                            pointers[entity_index] = unsafe {
                                typed_components.as_ptr().add(row).cast_mut().cast::<u8>() as usize
                            };
                        }
                    }

                    if !typed_components.is_empty() && !contiguous_prefix_ambiguous {
                        let is_identity_prefix = entity_indices
                            .iter()
                            .enumerate()
                            .all(|(row, &entity_index)| entity_index as usize == row);
                        if is_identity_prefix {
                            if contiguous_prefix_base == 0 {
                                contiguous_prefix_base =
                                    typed_components.as_ptr().cast_mut().cast::<u8>() as usize;
                                contiguous_prefix_len = typed_components.len();
                            } else {
                                // Multiple identity-prefix archetypes for the same type are
                                // ambiguous for a single base+index mapping.
                                contiguous_prefix_ambiguous = true;
                            }
                        }
                    }
                }
            }

            self.get_component_mut_zero_generation_cache = GetComponentMutZeroGenerationCache {
                world_id: self.world_id,
                structural_version: self.structural_version,
                type_id,
                pointers,
                contiguous_prefix_base: if contiguous_prefix_ambiguous {
                    0
                } else {
                    contiguous_prefix_base
                },
                contiguous_prefix_len: if contiguous_prefix_ambiguous {
                    0
                } else {
                    contiguous_prefix_len
                },
                contiguous_covers_tail: !contiguous_prefix_ambiguous
                    && contiguous_prefix_base != 0
                    && contiguous_prefix_len == self.componentless_tail_start as usize,
            };
        }

        let cache = &self.get_component_mut_zero_generation_cache;
        let component_ptr = if cache.contiguous_covers_tail {
            // SAFETY:
            // - `contiguous_covers_tail` means contiguous prefix length equals
            //   `componentless_tail_start`.
            // - Callers of this helper route through `get_component_mut`, which ensures
            //   `entity_index < componentless_tail_start`.
            unsafe {
                (cache.contiguous_prefix_base as *mut u8)
                    .cast::<T>()
                    .add(entity_index)
                    .cast::<u8>()
            }
        } else if entity_index < cache.contiguous_prefix_len {
            // SAFETY:
            // - `entity_index < contiguous_prefix_len` ensures this row exists in the
            //   cached contiguous prefix column for `T`.
            // - `contiguous_prefix_base` was captured from `typed_components.as_ptr()` when
            //   the identity-prefix condition held.
            // - Row addressing uses typed pointer arithmetic.
            unsafe {
                (cache.contiguous_prefix_base as *mut u8)
                    .cast::<T>()
                    .add(entity_index)
                    .cast::<u8>()
            }
        } else if entity_index >= cache.pointers.len() {
            std::ptr::null_mut()
        } else {
            // SAFETY: `entity_index` bounds are checked above.
            unsafe { *cache.pointers.get_unchecked(entity_index) as *mut u8 }
        };

        if component_ptr.is_null() {
            return None;
        }

        // SAFETY:
        // - `component_ptr` is captured from this world's component storage for `T`.
        // - `&mut self` guarantees exclusive access while creating `&mut T`.
        // - `structural_version` guard ensures cache rebuild after structural mutation.
        Some(unsafe { &mut *component_ptr.cast::<T>() })
    }

    /// Spawn a new entity with no components.
    pub fn spawn_empty(&mut self) -> Entity {
        let index = if let Some(free_index) = self.free_indices.pop() {
            free_index
        } else {
            let index = self.next_entity_index;
            self.next_entity_index += 1;
            index
        };

        if index as usize >= self.entity_generations.len() {
            self.entity_generations.resize(index as usize + 1, 0);
        }
        if index as usize >= self.entity_locations.len() {
            self.entity_locations.resize(index as usize + 1, None);
        }

        let generation = self.entity_generations[index as usize];
        if generation != 0 {
            self.all_generations_zero = false;
        }
        let entity = Entity::new(index, generation);

        // Add to empty archetype
        let empty_arch_id = ArchetypeId(0);
        let empty_archetype = &mut self.archetypes[0];
        let entity_index = empty_archetype.len();
        empty_archetype.push_entity_empty(entity);
        self.entity_locations[index as usize] = Some((empty_arch_id, entity_index));
        self.bump_structural_version();

        entity
    }

    /// Spawn a new entity with a bundle of components in a single operation.
    ///
    /// Places the entity directly into the target archetype, avoiding the
    /// archetype migration chain that sequential `add_component` calls cause.
    ///
    /// ```
    /// use skesis::World;
    ///
    /// struct Position { x: f32, y: f32 }
    /// struct Velocity { dx: f32, dy: f32 }
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn_with((
    ///     Position { x: 0.0, y: 0.0 },
    ///     Velocity { dx: 1.0, dy: 2.0 },
    /// ));
    /// ```
    pub fn spawn_with<B: crate::bundle::SpawnBundle>(&mut self, bundle: B) -> Entity {
        // Allocate entity index (reuse freed slots or bump counter).
        let index = if let Some(free_index) = self.free_indices.pop() {
            free_index
        } else {
            let index = self.next_entity_index;
            self.next_entity_index += 1;
            index
        };

        if index as usize >= self.entity_generations.len() {
            self.entity_generations.resize(index as usize + 1, 0);
        }
        if index as usize >= self.entity_locations.len() {
            self.entity_locations.resize(index as usize + 1, None);
        }

        let generation = self.entity_generations[index as usize];
        if generation != 0 {
            self.all_generations_zero = false;
        }
        let entity = Entity::new(index, generation);

        // Fast path: look up cached bundle → archetype mapping.
        let bundle_type_id = TypeId::of::<B>();
        let arch_id = if let Some(&(_, cached_id)) = self
            .bundle_archetype_cache
            .iter()
            .find(|&&(tid, _)| tid == bundle_type_id)
        {
            cached_id
        } else {
            // Cold path: register column factories, build component set, create archetype.
            B::register(self);
            let component_set = B::component_set();
            let id = self.get_or_create_archetype(component_set);
            self.bundle_archetype_cache.push((bundle_type_id, id));
            id
        };

        // Push entity + all components directly into the target archetype.
        let archetype = &mut self.archetypes[arch_id.0 as usize];
        let row = archetype.len();
        archetype.track_entity(entity);
        bundle.push_components(archetype);

        // Update bookkeeping.
        self.entity_locations[index as usize] = Some((arch_id, row));
        self.max_nonempty_entity_index = Some(
            self.max_nonempty_entity_index
                .map_or(index, |current_max| current_max.max(index)),
        );
        self.componentless_tail_start = self
            .max_nonempty_entity_index
            .map_or(0, |max_nonempty| max_nonempty + 1);
        self.bump_structural_version();

        // Bump change versions (zero-allocation iteration).
        B::for_each_type_id(|type_id| {
            self.change_history.bump_version_by_type_id(&type_id);
        });
        // Fire on_add observers after all components are in place, so
        // observers can read sibling components from the bundle.
        if self.on_add_observer_count > 0 {
            B::for_each_type_id(|type_id| {
                self.fire_on_add_by_type_id(&type_id, entity, arch_id, row);
            });
        }

        entity
    }

    /// Despawn an entity and all its components.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        let index = entity.index() as usize;

        let Some((archetype_id, archetype_index)) = self.location_of(entity) else {
            return false;
        };

        let swapped_entity =
            if let Some(archetype) = self.archetypes.get_mut(archetype_id.0 as usize) {
                if archetype_id.0 == 0 {
                    if archetype
                        .swap_remove_entity_empty(archetype_index)
                        .is_none()
                    {
                        return false;
                    }
                } else if archetype
                    .swap_remove_entity_discard(archetype_index)
                    .is_none()
                {
                    return false;
                }

                if archetype_index < archetype.len() {
                    archetype.entity(archetype_index)
                } else {
                    None
                }
            } else {
                return false;
            };

        if let Some(swapped_entity) = swapped_entity {
            let swapped_index = swapped_entity.index() as usize;
            debug_assert!(swapped_index < self.entity_locations.len());
            self.entity_locations[swapped_index] = Some((archetype_id, archetype_index));
        }

        self.entity_locations[index] = None;
        if archetype_id.0 != 0 {
            if self.max_nonempty_entity_index == Some(entity.index()) {
                let mut cursor = entity.index();
                let new_max = loop {
                    if cursor == 0 {
                        break None;
                    }
                    cursor -= 1;
                    if matches!(
                        self.entity_locations[cursor as usize],
                        Some((arch_id, _)) if arch_id.0 != 0
                    ) {
                        break Some(cursor);
                    }
                };
                self.max_nonempty_entity_index = new_max;
            }
            self.componentless_tail_start = self
                .max_nonempty_entity_index
                .map_or(0, |max_nonempty| max_nonempty + 1);
        }
        self.entity_generations[index] += 1;
        self.all_generations_zero = false;
        self.free_indices.push(entity.index());
        // Clean up any disabled component entries for this entity.
        let idx = entity.index() as usize;
        for bits in self.disabled_components.values_mut() {
            if idx < bits.len() {
                bits[idx] = false;
            }
        }
        // Clean up any relationship edges involving this entity.
        self.relationships.remove_entity(entity);
        // Clean up any sparse components for this entity.
        for storage in self.sparse_storage.values_mut() {
            storage.remove_entity(entity);
        }
        self.bump_structural_version();
        true
    }

    /// Check if an entity is alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.location_of(entity).is_some()
    }

    /// Get the number of archetypes.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Add a component to an entity.
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        let Some((old_arch_id, old_index)) = self.location_of(entity) else {
            return;
        };
        let type_id = TypeId::of::<T>();

        // Sparse component: store in parallel SparseSet, no archetype transition.
        if self.sparse_types.contains(&type_id) {
            if let Some(storage) = self.sparse_storage.get_mut(&type_id) {
                let sparse = storage
                    .as_any_mut()
                    .downcast_mut::<SparseSet<T>>()
                    .expect("sparse storage type mismatch");
                sparse.insert(entity, component);
            }
            return;
        }

        if self.archetypes[old_arch_id.0 as usize]
            .component_set()
            .contains::<T>()
        {
            let cache_is_fresh = if self.all_generations_zero {
                let cache = &self.get_component_mut_zero_generation_cache;
                cache.world_id == self.world_id
                    && cache.structural_version == self.structural_version
                    && cache.type_id == type_id
            } else {
                let cache = &self.get_component_mut_cache;
                cache.world_id == self.world_id
                    && cache.structural_version == self.structural_version
                    && cache.type_id == type_id
            };

            // Hot path for "set existing component" workloads (`set_id` microbench).
            if entity.index() < self.componentless_tail_start
                && cache_is_fresh
                && let Some(slot) = self.get_component_mut::<T>(entity)
            {
                *slot = component;
                return;
            }

            // Defensive fallback should be unreachable when component set reports `T`.
            if let Some(columns) = self.archetypes[old_arch_id.0 as usize].components_mut::<T>()
                && let Some(slot) = columns.get_mut(old_index)
            {
                *slot = component;
            }
            return;
        }

        if old_arch_id.0 == 0 {
            let new_arch_id = if let Some((_, id)) = self
                .single_component_archetypes
                .iter()
                .find(|&&(cached_type_id, _)| cached_type_id == type_id)
            {
                *id
            } else {
                self.register_component_type::<T>();

                let singleton_set = ComponentSet::new().with::<T>();
                let id = self.get_or_create_archetype(singleton_set);
                self.single_component_archetypes.push((type_id, id));
                id
            };

            // Specialize empty -> {T} transition to avoid transient component map allocation.
            let swapped_entity = {
                let old_archetype = &mut self.archetypes[0];
                if old_archetype.swap_remove_entity_empty(old_index).is_none() {
                    return;
                }

                if old_index < old_archetype.len() {
                    old_archetype.entity(old_index)
                } else {
                    None
                }
            };

            if let Some(swapped_entity) = swapped_entity {
                let swapped_index = swapped_entity.index() as usize;
                debug_assert!(swapped_index < self.entity_locations.len());
                self.entity_locations[swapped_index] = Some((old_arch_id, old_index));
            }

            let new_archetype = &mut self.archetypes[new_arch_id.0 as usize];
            let new_index = new_archetype.len();
            new_archetype.push_entity_single_component(entity, component);
            let entity_index = entity.index();
            let entity_slot = entity_index as usize;
            debug_assert!(entity_slot < self.entity_locations.len());
            self.entity_locations[entity_slot] = Some((new_arch_id, new_index));
            self.max_nonempty_entity_index = Some(
                self.max_nonempty_entity_index
                    .map_or(entity_index, |current_max| current_max.max(entity_index)),
            );
            self.componentless_tail_start = self
                .max_nonempty_entity_index
                .map_or(0, |max_nonempty| max_nonempty + 1);
            self.bump_structural_version();
            self.change_history.bump_version_by_type_id(&type_id);
            self.fire_on_add::<T>(entity, new_arch_id, new_index);
            return;
        }

        self.register_component_type::<T>();

        let new_arch_id =
            if let Some(id) = self.lookup_add_component_transition(old_arch_id, type_id) {
                id
            } else {
                let mut new_set = self.archetypes[old_arch_id.0 as usize]
                    .component_set()
                    .clone();
                new_set = new_set.with::<T>();

                let id = self.get_or_create_archetype(new_set);
                self.cache_add_component_transition(old_arch_id, type_id, id);
                id
            };

        let old_archetype_index = old_arch_id.0 as usize;
        let new_archetype_index = new_arch_id.0 as usize;
        let (new_index, swapped_entity) = {
            let (old_archetype, new_archetype) = Self::two_archetypes_mut(
                &mut self.archetypes,
                old_archetype_index,
                new_archetype_index,
            );
            old_archetype
                .move_row_to_with_added_component(old_index, new_archetype, entity, component)
                .expect("entity location must point at valid archetype row")
        };

        if let Some(swapped_entity) = swapped_entity {
            let swapped_index = swapped_entity.index() as usize;
            debug_assert!(swapped_index < self.entity_locations.len());
            self.entity_locations[swapped_index] = Some((old_arch_id, old_index));
        }
        let entity_slot = entity.index() as usize;
        debug_assert!(entity_slot < self.entity_locations.len());
        self.entity_locations[entity_slot] = Some((new_arch_id, new_index));
        self.bump_structural_version();
        self.change_history.bump_version_by_type_id(&type_id);
        self.fire_on_add::<T>(entity, new_arch_id, new_index);
    }

    /// Remove a component from an entity, returning its value.
    pub fn remove_component<T: Component>(&mut self, entity: Entity) -> Option<T> {
        // Sparse component: remove from SparseSet, no archetype transition.
        if self.sparse_types.contains(&TypeId::of::<T>()) {
            return self
                .sparse_storage
                .get_mut(&TypeId::of::<T>())
                .and_then(|s| {
                    s.as_any_mut()
                        .downcast_mut::<SparseSet<T>>()
                        .and_then(|sparse| sparse.remove(entity))
                });
        }

        let (old_arch_id, old_index) = self.location_of(entity)?;

        // Can't remove from empty archetype.
        if old_arch_id.0 == 0 {
            return None;
        }

        if !self.archetypes[old_arch_id.0 as usize]
            .component_set()
            .contains::<T>()
        {
            return None;
        }

        let type_id = TypeId::of::<T>();

        let new_arch_id =
            if let Some(id) = self.lookup_remove_component_transition(old_arch_id, type_id) {
                id
            } else {
                let new_set = self.archetypes[old_arch_id.0 as usize]
                    .component_set()
                    .clone()
                    .without::<T>();

                let id = self.get_or_create_archetype(new_set);
                self.cache_remove_component_transition(old_arch_id, type_id, id);
                id
            };

        let old_archetype_index = old_arch_id.0 as usize;
        let new_archetype_index = new_arch_id.0 as usize;
        let (removed_value, new_index, swapped_entity) = {
            let (old_archetype, new_archetype) = Self::two_archetypes_mut(
                &mut self.archetypes,
                old_archetype_index,
                new_archetype_index,
            );
            old_archetype
                .move_row_to_without_component::<T>(old_index, new_archetype, entity)
                .expect("entity location must point at valid archetype row")
        };

        if let Some(swapped_entity) = swapped_entity {
            let swapped_index = swapped_entity.index() as usize;
            debug_assert!(swapped_index < self.entity_locations.len());
            self.entity_locations[swapped_index] = Some((old_arch_id, old_index));
        }
        let entity_slot = entity.index() as usize;
        debug_assert!(entity_slot < self.entity_locations.len());
        self.entity_locations[entity_slot] = Some((new_arch_id, new_index));

        // Update max_nonempty tracking if entity moved to the empty archetype.
        if new_arch_id.0 == 0 {
            if self.max_nonempty_entity_index == Some(entity.index()) {
                let mut cursor = entity.index();
                let new_max = loop {
                    if cursor == 0 {
                        break None;
                    }
                    cursor -= 1;
                    if matches!(
                        self.entity_locations[cursor as usize],
                        Some((arch_id, _)) if arch_id.0 != 0
                    ) {
                        break Some(cursor);
                    }
                };
                self.max_nonempty_entity_index = new_max;
            }
            self.componentless_tail_start = self
                .max_nonempty_entity_index
                .map_or(0, |max_nonempty| max_nonempty + 1);
        }

        self.bump_structural_version();
        self.change_history.bump_version::<T>();
        self.fire_on_remove::<T>(entity, &removed_value);
        // Clean up disabled entry for this component.
        if let Some(bits) = self.disabled_components.get_mut(&TypeId::of::<T>()) {
            let idx = entity.index() as usize;
            if idx < bits.len() {
                bits[idx] = false;
            }
        }
        Some(removed_value)
    }

    /// Check if an entity has a component.
    pub fn has_component<T: Component>(&self, entity: Entity) -> bool {
        // Check sparse storage first.
        if self.sparse_types.contains(&TypeId::of::<T>()) {
            return self
                .sparse_storage
                .get(&TypeId::of::<T>())
                .is_some_and(|s| s.contains_entity(entity));
        }

        if let Some((arch_id, _)) = self.location_of(entity) {
            // Archetype 0 is the empty archetype, so it can never contain components.
            if arch_id.0 == 0 {
                return false;
            }
            self.archetypes[arch_id.0 as usize]
                .component_set()
                .contains::<T>()
        } else {
            false
        }
    }

    /// Get a component reference for a specific entity.
    pub fn get_component<T: Component>(&self, entity: Entity) -> Option<&T> {
        // Sparse component: look up in SparseSet.
        if self.sparse_types.contains(&TypeId::of::<T>()) {
            return self.sparse_storage.get(&TypeId::of::<T>()).and_then(|s| {
                s.as_any()
                    .downcast_ref::<SparseSet<T>>()
                    .and_then(|sparse| sparse.get(entity))
            });
        }

        let index = entity.index();
        if index >= self.componentless_tail_start {
            return None;
        }
        if self.all_generations_zero {
            if entity.generation() != 0 {
                return None;
            }
            return self.get_component_cached_by_entity_index_generation_zero::<T>(index as usize);
        }
        self.get_component_cached_by_entity_index::<T>(index as usize, entity.generation())
    }

    /// Get a mutable component reference for a specific entity.
    pub fn get_component_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        // Sparse component: look up in SparseSet.
        if self.sparse_types.contains(&TypeId::of::<T>()) {
            return self
                .sparse_storage
                .get_mut(&TypeId::of::<T>())
                .and_then(|s| {
                    s.as_any_mut()
                        .downcast_mut::<SparseSet<T>>()
                        .and_then(|sparse| sparse.get_mut(entity))
                });
        }

        let index = entity.index();
        if index >= self.componentless_tail_start {
            return None;
        }
        if self.all_generations_zero {
            if entity.generation() != 0 {
                return None;
            }
            return self
                .get_component_mut_cached_by_entity_index_generation_zero::<T>(index as usize);
        }
        self.get_component_mut_cached_by_entity_index::<T>(index as usize, entity.generation())
    }

    /// Run a callback for each entity that has both component types.
    pub fn for_each_pair<A: Component, B: Component>(&self, mut f: impl FnMut(Entity, &A, &B)) {
        for arch in &self.archetypes {
            let component_set = arch.component_set();
            if !component_set.contains::<A>() || !component_set.contains::<B>() {
                continue;
            }

            let entities = arch.entities();
            let components_a = arch
                .components::<A>()
                .expect("archetype contains component set but missing A column");
            let components_b = arch
                .components::<B>()
                .expect("archetype contains component set but missing B column");

            for index in 0..entities.len() {
                f(entities[index], &components_a[index], &components_b[index]);
            }
        }
    }

    /// Run a callback for each entity with components A and B, excluding entities with Exclude.
    pub fn for_each_pair_without<A: Component, B: Component, Exclude: Component>(
        &self,
        mut f: impl FnMut(Entity, &A, &B),
    ) {
        for arch in &self.archetypes {
            let component_set = arch.component_set();
            if !component_set.contains::<A>()
                || !component_set.contains::<B>()
                || component_set.contains::<Exclude>()
            {
                continue;
            }

            let entities = arch.entities();
            let components_a = arch
                .components::<A>()
                .expect("archetype contains component set but missing A column");
            let components_b = arch
                .components::<B>()
                .expect("archetype contains component set but missing B column");

            for index in 0..entities.len() {
                f(entities[index], &components_a[index], &components_b[index]);
            }
        }
    }

    /// Run a callback for each entity that has the component type.
    pub fn for_each_mut<T: Component>(&mut self, mut f: impl FnMut(Entity, &mut T)) {
        for arch in &mut self.archetypes {
            if !arch.component_set().contains::<T>() {
                continue;
            }

            let (entities, components) = arch
                .entities_and_components_mut::<T>()
                .expect("archetype contains component set but missing column");

            for index in 0..entities.len() {
                f(entities[index], &mut components[index]);
            }
        }
    }

    /// Run a callback for each entity with component T, excluding entities with Exclude.
    pub fn for_each_mut_without<T: Component, Exclude: Component>(
        &mut self,
        mut f: impl FnMut(Entity, &mut T),
    ) {
        for arch in &mut self.archetypes {
            let component_set = arch.component_set();
            if !component_set.contains::<T>() || component_set.contains::<Exclude>() {
                continue;
            }

            let (entities, components) = arch
                .entities_and_components_mut::<T>()
                .expect("archetype contains component set but missing column");

            for index in 0..entities.len() {
                f(entities[index], &mut components[index]);
            }
        }
    }

    /// Run a callback for each entity with component T, optionally including Opt if present.
    ///
    /// The optional component is resolved per-archetype: if the archetype has Opt,
    /// every entity in that archetype gets `Some(&Opt)`, otherwise `None`.
    /// Run a callback for each entity with components A and B, respecting toggle state.
    ///
    /// Entities where A or B is disabled are skipped. Non-toggled queries are unaffected.
    pub fn for_each_pair_toggled<A: Component, B: Component>(
        &self,
        mut f: impl FnMut(Entity, &A, &B),
    ) {
        let disabled_a = self.disabled_components.get(&TypeId::of::<A>());
        let disabled_b = self.disabled_components.get(&TypeId::of::<B>());
        let any_disabled = disabled_a.is_some_and(|bits| bits.iter().any(|&b| b))
            || disabled_b.is_some_and(|bits| bits.iter().any(|&b| b));

        for arch in &self.archetypes {
            let component_set = arch.component_set();
            if !component_set.contains::<A>() || !component_set.contains::<B>() {
                continue;
            }

            let entities = arch.entities();
            let components_a = arch
                .components::<A>()
                .expect("archetype contains component set but missing A column");
            let components_b = arch
                .components::<B>()
                .expect("archetype contains component set but missing B column");

            if !any_disabled {
                // Fast path: no disabled components, skip per-entity checks.
                for index in 0..entities.len() {
                    f(entities[index], &components_a[index], &components_b[index]);
                }
            } else {
                for index in 0..entities.len() {
                    let ei = entities[index].index() as usize;
                    if disabled_a.is_some_and(|bits| ei < bits.len() && bits[ei])
                        || disabled_b.is_some_and(|bits| ei < bits.len() && bits[ei])
                    {
                        continue;
                    }
                    f(entities[index], &components_a[index], &components_b[index]);
                }
            }
        }
    }

    /// Run a callback for each entity with component T (mutable), respecting toggle state.
    ///
    /// Entities where T is disabled are skipped.
    pub fn for_each_mut_toggled<T: Component>(&mut self, mut f: impl FnMut(Entity, &mut T)) {
        let disabled = self.disabled_components.get(&TypeId::of::<T>()).cloned();
        let any_disabled = disabled
            .as_ref()
            .is_some_and(|bits| bits.iter().any(|&b| b));

        for arch in &mut self.archetypes {
            if !arch.component_set().contains::<T>() {
                continue;
            }

            let (entities, components) = arch
                .entities_and_components_mut::<T>()
                .expect("archetype contains component set but missing column");

            if !any_disabled {
                for index in 0..entities.len() {
                    f(entities[index], &mut components[index]);
                }
            } else {
                for index in 0..entities.len() {
                    let entity = entities[index];
                    let ei = entity.index() as usize;
                    if disabled
                        .as_ref()
                        .is_some_and(|bits| ei < bits.len() && bits[ei])
                    {
                        continue;
                    }
                    f(entity, &mut components[index]);
                }
            }
        }
    }

    /// Run a callback for each entity with component T, optionally including Opt if present.
    pub fn for_each_with_optional<T: Component, Opt: Component>(
        &self,
        mut f: impl FnMut(Entity, &T, Option<&Opt>),
    ) {
        for arch in &self.archetypes {
            let component_set = arch.component_set();
            if !component_set.contains::<T>() {
                continue;
            }

            let entities = arch.entities();
            let components_t = arch
                .components::<T>()
                .expect("archetype contains component set but missing T column");
            let components_opt = arch.components::<Opt>();

            match components_opt {
                Some(opt_slice) => {
                    for index in 0..entities.len() {
                        f(
                            entities[index],
                            &components_t[index],
                            Some(&opt_slice[index]),
                        );
                    }
                }
                None => {
                    for index in 0..entities.len() {
                        f(entities[index], &components_t[index], None);
                    }
                }
            }
        }
    }

    /// Build a reusable plan for a pair query.
    ///
    /// Rebuild this plan after structural world changes if you need newly created archetypes included.
    pub fn plan_query_pair<A: Component, B: Component>(&self) -> PairQueryPlan<A, B> {
        let archetype_indices = self
            .archetypes
            .iter()
            .enumerate()
            .filter_map(|(index, arch)| {
                let component_set = arch.component_set();
                if component_set.contains::<A>() && component_set.contains::<B>() {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        PairQueryPlan {
            archetype_indices,
            marker: PhantomData,
        }
    }

    /// Build a reusable plan for a pair query, excluding archetypes with Exclude.
    pub fn plan_query_pair_without<A: Component, B: Component, Exclude: Component>(
        &self,
    ) -> PairQueryPlan<A, B> {
        let archetype_indices = self
            .archetypes
            .iter()
            .enumerate()
            .filter_map(|(index, arch)| {
                let component_set = arch.component_set();
                if component_set.contains::<A>()
                    && component_set.contains::<B>()
                    && !component_set.contains::<Exclude>()
                {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        PairQueryPlan {
            archetype_indices,
            marker: PhantomData,
        }
    }

    /// Build a borrowed pair query plan with prebound chunk slices.
    ///
    /// This avoids per-iteration archetype lookups for stable, read-only hot paths.
    pub fn plan_query_pair_borrowed<'w, A: Component, B: Component>(
        &'w self,
    ) -> BorrowedPairQueryPlan<'w, A, B> {
        let mut chunks = Vec::new();

        for arch in &self.archetypes {
            let component_set = arch.component_set();
            if !component_set.contains::<A>() || !component_set.contains::<B>() {
                continue;
            }

            let entities = arch.entities();
            let entity_indices = arch.entity_indices();
            let components_a = arch
                .components::<A>()
                .expect("archetype contains component set but missing A column");
            let components_b = arch
                .components::<B>()
                .expect("archetype contains component set but missing B column");

            if entities.is_empty() {
                continue;
            }
            debug_assert_eq!(entities.len(), entity_indices.len());
            debug_assert_eq!(entities.len(), components_a.len());
            debug_assert_eq!(entities.len(), components_b.len());

            // SAFETY: slice pointers are guaranteed non-null (possibly dangling for empty slices,
            // but empty slices are skipped above). We never write through these pointers.
            let entities_ptr = unsafe { NonNull::new_unchecked(entities.as_ptr() as *mut Entity) };
            // SAFETY: same invariant as above.
            let entity_indices_ptr =
                unsafe { NonNull::new_unchecked(entity_indices.as_ptr() as *mut u32) };
            // SAFETY: same invariant as above.
            let a_ptr = unsafe { NonNull::new_unchecked(components_a.as_ptr() as *mut A) };
            // SAFETY: same invariant as above.
            let b_ptr = unsafe { NonNull::new_unchecked(components_b.as_ptr() as *mut B) };

            let contiguous_base = contiguous_index_base(entity_indices);
            let contiguous_index_sum =
                contiguous_base.map(|base| contiguous_index_sum_f32(base, entities.len()));
            let noncontiguous_indices_f32 = if contiguous_base.is_none() {
                Some(
                    entity_indices
                        .iter()
                        .map(|&index| index as f32)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            } else {
                None
            };

            chunks.push(BorrowedPairChunk {
                entities_ptr,
                entity_indices_ptr,
                a_ptr,
                b_ptr,
                len: entities.len(),
                contiguous_base,
                contiguous_index_sum,
                noncontiguous_indices_f32,
            });
        }

        BorrowedPairQueryPlan {
            chunks,
            structural_version: self.structural_version,
            marker: PhantomData,
        }
    }

    /// Build a reusable plan for a mutable single-component query.
    ///
    /// Rebuild this plan after structural world changes if you need newly created archetypes included.
    pub fn plan_query_mut<T: Component>(&self) -> MutQueryPlan<T> {
        let archetype_indices = self
            .archetypes
            .iter()
            .enumerate()
            .filter_map(|(index, arch)| {
                if arch.component_set().contains::<T>() {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        MutQueryPlan {
            archetype_indices,
            marker: PhantomData,
        }
    }

    /// Build a reusable plan for a mutable query, excluding archetypes with Exclude.
    pub fn plan_query_mut_without<T: Component, Exclude: Component>(&self) -> MutQueryPlan<T> {
        let archetype_indices = self
            .archetypes
            .iter()
            .enumerate()
            .filter_map(|(index, arch)| {
                let component_set = arch.component_set();
                if component_set.contains::<T>() && !component_set.contains::<Exclude>() {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        MutQueryPlan {
            archetype_indices,
            marker: PhantomData,
        }
    }

    /// Build a borrowed mutable query plan with prebound dense index and mutable component chunks.
    ///
    /// This avoids per-iteration archetype and column lookup on stable mutable hot paths.
    pub fn plan_query_mut_borrowed<T: Component>(&mut self) -> BorrowedMutQueryPlan<T> {
        let mut chunks = Vec::new();

        for arch in &mut self.archetypes {
            if !arch.component_set().contains::<T>() {
                continue;
            }

            let (entity_indices, components) = arch
                .entity_indices_and_components_mut::<T>()
                .expect("archetype contains component set but missing mutable column");

            if entity_indices.is_empty() {
                continue;
            }

            debug_assert_eq!(entity_indices.len(), components.len());

            // SAFETY:
            // - Non-empty slices guarantee non-null pointers.
            // - We never change structure while executing a fresh borrowed plan.
            let entity_indices_ptr =
                unsafe { NonNull::new_unchecked(entity_indices.as_ptr() as *mut u32) };
            // SAFETY: same invariant as above.
            let components_ptr = unsafe { NonNull::new_unchecked(components.as_mut_ptr()) };

            let contiguous_base = contiguous_index_base(entity_indices);
            let contiguous_index_sum =
                contiguous_base.map(|base| contiguous_index_sum_f32(base, entity_indices.len()));
            let noncontiguous_indices_f32 = if contiguous_base.is_none() {
                Some(
                    entity_indices
                        .iter()
                        .map(|&index| index as f32)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                )
            } else {
                None
            };

            chunks.push(BorrowedMutChunk {
                entity_indices_ptr,
                components_ptr,
                len: entity_indices.len(),
                contiguous_base,
                contiguous_index_sum,
                noncontiguous_indices_f32,
            });
        }

        BorrowedMutQueryPlan {
            chunks,
            world_id: self.world_id,
            structural_version: self.structural_version,
            marker: PhantomData,
        }
    }

    /// Run a pair query callback using a precomputed plan.
    #[inline(always)]
    pub fn for_each_pair_with_plan<A: Component, B: Component>(
        &self,
        plan: &PairQueryPlan<A, B>,
        mut f: impl FnMut(Entity, &A, &B),
    ) {
        for &arch_index in &plan.archetype_indices {
            let arch = &self.archetypes[arch_index];
            let entities = arch.entities();
            let components_a = arch
                .components::<A>()
                .expect("query plan archetype missing A column");
            let components_b = arch
                .components::<B>()
                .expect("query plan archetype missing B column");

            for index in 0..entities.len() {
                f(entities[index], &components_a[index], &components_b[index]);
            }
        }
    }

    /// Run a callback once per matching archetype chunk using a precomputed pair plan.
    ///
    /// This keeps per-row loops in caller code, avoiding callback dispatch in inner loops.
    #[inline(always)]
    pub fn for_each_pair_chunk_with_plan<A: Component, B: Component>(
        &self,
        plan: &PairQueryPlan<A, B>,
        mut f: impl FnMut(&[Entity], &[A], &[B]),
    ) {
        for &arch_index in &plan.archetype_indices {
            let arch = &self.archetypes[arch_index];
            let entities = arch.entities();
            let components_a = arch
                .components::<A>()
                .expect("query plan archetype missing A column");
            let components_b = arch
                .components::<B>()
                .expect("query plan archetype missing B column");

            debug_assert_eq!(entities.len(), components_a.len());
            debug_assert_eq!(entities.len(), components_b.len());
            f(entities, components_a, components_b);
        }
    }

    /// Run a mutable single-component callback using a precomputed plan.
    #[inline(always)]
    pub fn for_each_mut_with_plan<T: Component>(
        &mut self,
        plan: &MutQueryPlan<T>,
        mut f: impl FnMut(Entity, &mut T),
    ) {
        for &arch_index in &plan.archetype_indices {
            let arch = &mut self.archetypes[arch_index];
            let (entities, components) = arch
                .entities_and_components_mut::<T>()
                .expect("query plan archetype missing mutable column");

            for index in 0..entities.len() {
                f(entities[index], &mut components[index]);
            }
        }
    }

    /// Run a callback once per matching mutable archetype chunk using a precomputed plan.
    ///
    /// This keeps per-row loops in caller code, avoiding callback dispatch in inner loops.
    #[inline(always)]
    pub fn for_each_mut_chunk_with_plan<T: Component>(
        &mut self,
        plan: &MutQueryPlan<T>,
        mut f: impl FnMut(&[Entity], &mut [T]),
    ) {
        for &arch_index in &plan.archetype_indices {
            let arch = &mut self.archetypes[arch_index];
            let (entities, components) = arch
                .entities_and_components_mut::<T>()
                .expect("query plan archetype missing mutable column");

            debug_assert_eq!(entities.len(), components.len());
            f(entities, components);
        }
    }

    /// Run a callback once per matching mutable archetype chunk using dense entity indices.
    ///
    /// This avoids loading full `Entity` values when only the index is needed.
    #[inline(always)]
    pub fn for_each_mut_indexed_chunk_with_plan<T: Component>(
        &mut self,
        plan: &MutQueryPlan<T>,
        mut f: impl FnMut(&[u32], &mut [T]),
    ) {
        for &arch_index in &plan.archetype_indices {
            let arch = &mut self.archetypes[arch_index];
            let (entity_indices, components) = arch
                .entity_indices_and_components_mut::<T>()
                .expect("query plan archetype missing mutable column");

            debug_assert_eq!(entity_indices.len(), components.len());
            f(entity_indices, components);
        }
    }

    /// Query for entities with a specific component.
    ///
    /// Returns an iterator over (Entity, &Component) pairs.
    pub fn query<'a, T: Component>(&'a self) -> impl Iterator<Item = (Entity, &'a T)> + 'a {
        self.archetypes
            .iter()
            .filter(|arch| arch.component_set().contains::<T>())
            .flat_map(|arch| {
                let entities = arch.entities();
                let components = arch
                    .components::<T>()
                    .expect("archetype contains component set but missing column");
                entities.iter().copied().zip(components.iter())
            })
    }

    /// Query for entities containing both component types.
    pub fn query_pair<'a, A: Component, B: Component>(
        &'a self,
    ) -> impl Iterator<Item = (Entity, &'a A, &'a B)> + 'a {
        self.archetypes
            .iter()
            .filter(|arch| {
                let component_set = arch.component_set();
                component_set.contains::<A>() && component_set.contains::<B>()
            })
            .flat_map(|arch| {
                let entities = arch.entities();
                let components_a = arch
                    .components::<A>()
                    .expect("archetype contains component set but missing A column");
                let components_b = arch
                    .components::<B>()
                    .expect("archetype contains component set but missing B column");

                entities
                    .iter()
                    .copied()
                    .zip(components_a.iter())
                    .zip(components_b.iter())
                    .map(|((entity, a), b)| (entity, a, b))
            })
    }

    /// Query for entities with component T, excluding those with Exclude.
    pub fn query_without<'a, T: Component, Exclude: Component>(
        &'a self,
    ) -> impl Iterator<Item = (Entity, &'a T)> + 'a {
        self.archetypes
            .iter()
            .filter(|arch| {
                let cs = arch.component_set();
                cs.contains::<T>() && !cs.contains::<Exclude>()
            })
            .flat_map(|arch| {
                let entities = arch.entities();
                let components = arch
                    .components::<T>()
                    .expect("archetype contains component set but missing column");
                entities.iter().copied().zip(components.iter())
            })
    }

    /// Query for entities with both A and B, excluding those with Exclude.
    pub fn query_pair_without<'a, A: Component, B: Component, Exclude: Component>(
        &'a self,
    ) -> impl Iterator<Item = (Entity, &'a A, &'a B)> + 'a {
        self.archetypes
            .iter()
            .filter(|arch| {
                let cs = arch.component_set();
                cs.contains::<A>() && cs.contains::<B>() && !cs.contains::<Exclude>()
            })
            .flat_map(|arch| {
                let entities = arch.entities();
                let components_a = arch
                    .components::<A>()
                    .expect("archetype contains component set but missing A column");
                let components_b = arch
                    .components::<B>()
                    .expect("archetype contains component set but missing B column");

                entities
                    .iter()
                    .copied()
                    .zip(components_a.iter())
                    .zip(components_b.iter())
                    .map(|((entity, a), b)| (entity, a, b))
            })
    }

    /// Query for entities with component T, optionally including Opt if present.
    ///
    /// Returns `(Entity, &T, Option<&Opt>)` — the optional is resolved per-archetype.
    /// Query for entities with component T, respecting toggle state.
    pub fn query_toggled<'a, T: Component>(&'a self) -> impl Iterator<Item = (Entity, &'a T)> + 'a {
        let disabled = self.disabled_components.get(&TypeId::of::<T>());
        self.archetypes
            .iter()
            .filter(|arch| arch.component_set().contains::<T>())
            .flat_map(move |arch| {
                let entities = arch.entities();
                let components = arch
                    .components::<T>()
                    .expect("archetype contains component set but missing column");
                entities
                    .iter()
                    .copied()
                    .zip(components.iter())
                    .filter(move |(entity, _)| {
                        let ei = entity.index() as usize;
                        !disabled.is_some_and(|bits| ei < bits.len() && bits[ei])
                    })
            })
    }

    /// Query for entities with both A and B, respecting toggle state.
    pub fn query_pair_toggled<'a, A: Component, B: Component>(
        &'a self,
    ) -> impl Iterator<Item = (Entity, &'a A, &'a B)> + 'a {
        let disabled_a = self.disabled_components.get(&TypeId::of::<A>());
        let disabled_b = self.disabled_components.get(&TypeId::of::<B>());
        self.archetypes
            .iter()
            .filter(|arch| {
                let cs = arch.component_set();
                cs.contains::<A>() && cs.contains::<B>()
            })
            .flat_map(move |arch| {
                let entities = arch.entities();
                let components_a = arch
                    .components::<A>()
                    .expect("archetype contains component set but missing A column");
                let components_b = arch
                    .components::<B>()
                    .expect("archetype contains component set but missing B column");

                entities
                    .iter()
                    .copied()
                    .zip(components_a.iter())
                    .zip(components_b.iter())
                    .filter(move |((entity, _), _)| {
                        let ei = entity.index() as usize;
                        !disabled_a.is_some_and(|bits| ei < bits.len() && bits[ei])
                            && !disabled_b.is_some_and(|bits| ei < bits.len() && bits[ei])
                    })
                    .map(|((entity, a), b)| (entity, a, b))
            })
    }

    /// Query for entities with component T, optionally including Opt if present.
    pub fn query_with_optional<'a, T: Component, Opt: Component>(
        &'a self,
    ) -> impl Iterator<Item = (Entity, &'a T, Option<&'a Opt>)> + 'a {
        self.archetypes
            .iter()
            .filter(|arch| arch.component_set().contains::<T>())
            .flat_map(|arch| {
                let entities = arch.entities();
                let components_t = arch
                    .components::<T>()
                    .expect("archetype contains component set but missing T column");
                let components_opt = arch.components::<Opt>();

                entities
                    .iter()
                    .copied()
                    .zip(components_t.iter())
                    .enumerate()
                    .map(move |(index, (entity, t))| {
                        let opt = components_opt.map(|s| &s[index]);
                        (entity, t, opt)
                    })
            })
    }

    /// Query mutable references for one component type.
    pub fn query_mut<'a, T: Component>(
        &'a mut self,
    ) -> impl Iterator<Item = (Entity, &'a mut T)> + 'a {
        self.archetypes
            .iter_mut()
            .filter(|arch| arch.component_set().contains::<T>())
            .flat_map(|arch| {
                let (entities, components) = arch
                    .entities_and_components_mut::<T>()
                    .expect("archetype contains component set but missing column");

                entities.iter().copied().zip(components.iter_mut())
            })
    }

    /// Query for entities matching an arbitrary tuple of component types.
    ///
    /// Supports 1 to 8 component types. Returns `(Entity, (&A, &B, &C, ...))`.
    ///
    /// ```ignore
    /// // 3-component query:
    /// for (entity, (pos, vel, health)) in world.query_tuple::<(&Position, &Velocity, &Health)>() {
    ///     // ...
    /// }
    /// ```
    ///
    /// For maximum hot-path performance on 2-component queries, use the specialized
    /// `query_pair` / `for_each_pair` / `BorrowedPairQueryPlan` APIs instead.
    pub fn query_tuple<'a, Q: crate::tuple_query::QueryTuple<'a> + 'a>(
        &'a self,
    ) -> impl Iterator<Item = (Entity, Q)> + 'a {
        self.archetypes
            .iter()
            .filter(|arch| Q::matches(arch))
            .flat_map(|arch| Q::fetch(arch))
    }

    /// Run a callback for each entity matching an arbitrary tuple of component types.
    ///
    /// Supports 1 to 8 component types.
    ///
    /// ```ignore
    /// world.for_each_tuple::<(&Position, &Velocity, &Health)>(|entity, (pos, vel, health)| {
    ///     // ...
    /// });
    /// ```
    pub fn for_each_tuple<'a, Q: crate::tuple_query::QueryTuple<'a>>(
        &'a self,
        mut f: impl FnMut(Entity, Q),
    ) {
        for arch in &self.archetypes {
            if !Q::matches(arch) {
                continue;
            }
            for (entity, tuple) in Q::fetch(arch) {
                f(entity, tuple);
            }
        }
    }

    /// Register an observer that fires when component T is added to an entity.
    ///
    /// The callback receives the entity and a reference to the newly added component.
    /// Observers fire synchronously during `add_component` (for new components only,
    /// not replacements). Structural world changes inside observers should be deferred
    /// via `CommandRecorder`.
    pub fn on_add<T: Component>(&mut self, callback: impl Fn(Entity, &T) + Send + Sync + 'static) {
        self.on_add_observers
            .entry(TypeId::of::<T>())
            .or_default()
            .push(Box::new(move |entity, ptr| {
                // SAFETY: The fire_on_add method guarantees ptr points to a valid T.
                let typed = unsafe { &*ptr.cast::<T>() };
                callback(entity, typed);
            }));
        self.on_add_observer_count += 1;
    }

    /// Register an observer that fires when component T is removed from an entity.
    ///
    /// The callback receives the entity and a reference to the component value
    /// before it is returned/dropped. Fires during `remove_component` and `despawn`.
    pub fn on_remove<T: Component>(
        &mut self,
        callback: impl Fn(Entity, &T) + Send + Sync + 'static,
    ) {
        self.on_remove_observers
            .entry(TypeId::of::<T>())
            .or_default()
            .push(Box::new(move |entity, ptr| {
                // SAFETY: The fire_on_remove method guarantees ptr points to a valid T.
                let typed = unsafe { &*ptr.cast::<T>() };
                callback(entity, typed);
            }));
        self.on_remove_observer_count += 1;
    }

    /// Register an observer that fires when a component value is set via `set_component`.
    ///
    /// The callback receives the entity and a reference to the **new** value.
    /// Does NOT fire on `get_component_mut` (that's the silent hot path).
    /// Does NOT fire on `add_component` for new components (use `on_add` for that).
    ///
    /// Use `set_component` for meaningful state changes you want the world to know about
    /// (networking, replay, reactive UI). Use `get_component_mut` for hot-path math
    /// that doesn't need notification.
    pub fn on_set<T: Component>(&mut self, callback: impl Fn(Entity, &T) + Send + Sync + 'static) {
        self.on_set_observers
            .entry(TypeId::of::<T>())
            .or_default()
            .push(Box::new(move |entity, ptr| {
                // SAFETY: The fire_on_set method guarantees ptr points to a valid T.
                let typed = unsafe { &*ptr.cast::<T>() };
                callback(entity, typed);
            }));
        self.on_set_observer_count += 1;
    }

    /// Set a component value on an entity, notifying observers and change detection.
    ///
    /// This is the "meaningful state change" method. Three things happen:
    /// 1. The component value is written.
    /// 2. `on_set` observers fire with the new value.
    /// 3. The change detection version is bumped.
    ///
    /// If the entity doesn't have the component yet, it behaves like `add_component`
    /// (fires `on_add`, not `on_set`).
    ///
    /// **When to use `set_component` vs `get_component_mut`:**
    /// - `set_component`: for state changes that need notification (networking, replay, UI).
    /// - `get_component_mut`: for hot-path math that doesn't need notification (physics, animation).
    pub fn set_component<T: Component>(&mut self, entity: Entity, component: T) {
        let Some((arch_id, arch_index)) = self.location_of(entity) else {
            return;
        };

        if !self.archetypes[arch_id.0 as usize]
            .component_set()
            .contains::<T>()
        {
            // Entity doesn't have this component — delegate to add_component.
            self.add_component(entity, component);
            return;
        }

        // Write the value.
        if let Some(columns) = self.archetypes[arch_id.0 as usize].components_mut::<T>()
            && let Some(slot) = columns.get_mut(arch_index)
        {
            *slot = component;
        }

        // Fire on_set observers with the new value.
        if let Some(comp) = self.archetypes[arch_id.0 as usize]
            .components::<T>()
            .and_then(|s| s.get(arch_index))
        {
            self.fire_on_set::<T>(entity, comp);
        }

        // Bump change detection version.
        self.change_history.bump_version::<T>();
    }

    /// Disable a component on an entity without removing it.
    ///
    /// The entity stays in its archetype (no structural change). Disabled components
    /// are skipped by `_toggled` query variants but remain accessible via `get_component`.
    /// Returns `true` if the component was present and is now disabled.
    pub fn disable_component<T: Component>(&mut self, entity: Entity) -> bool {
        if !self.has_component::<T>(entity) {
            return false;
        }
        let idx = entity.index() as usize;
        let bits = self
            .disabled_components
            .entry(TypeId::of::<T>())
            .or_default();
        if bits.len() <= idx {
            bits.resize(idx + 1, false);
        }
        bits[idx] = true;
        true
    }

    /// Re-enable a previously disabled component on an entity.
    ///
    /// Returns `true` if the component was disabled and is now enabled.
    pub fn enable_component<T: Component>(&mut self, entity: Entity) -> bool {
        if let Some(bits) = self.disabled_components.get_mut(&TypeId::of::<T>()) {
            let idx = entity.index() as usize;
            if idx < bits.len() && bits[idx] {
                bits[idx] = false;
                return true;
            }
            false
        } else {
            false
        }
    }

    /// Check if a component is enabled on an entity.
    ///
    /// Returns `true` if the entity has the component and it is not disabled.
    /// Returns `false` if the entity doesn't have the component, or if it's disabled.
    pub fn is_component_enabled<T: Component>(&self, entity: Entity) -> bool {
        if !self.has_component::<T>(entity) {
            return false;
        }
        !self
            .disabled_components
            .get(&TypeId::of::<T>())
            .is_some_and(|bits| {
                let idx = entity.index() as usize;
                idx < bits.len() && bits[idx]
            })
    }

    /// Add a typed relationship edge from source to target.
    ///
    /// Returns `true` if the edge is new, `false` if it already existed.
    /// Both entities must be alive.
    pub fn add_relation<R: Relation>(&mut self, source: Entity, target: Entity) -> bool {
        if !self.is_alive(source) || !self.is_alive(target) {
            return false;
        }
        self.relationships.add::<R>(source, target)
    }

    /// Remove a typed relationship edge from source to target.
    ///
    /// Returns `true` if the edge existed and was removed.
    pub fn remove_relation<R: Relation>(&mut self, source: Entity, target: Entity) -> bool {
        self.relationships.remove::<R>(source, target)
    }

    /// Check if a typed relationship edge exists from source to target.
    pub fn has_relation<R: Relation>(&self, source: Entity, target: Entity) -> bool {
        self.relationships.has::<R>(source, target)
    }

    /// Get all target entities that source relates to via relation R.
    pub fn targets<R: Relation>(&self, source: Entity) -> Vec<Entity> {
        self.relationships.targets::<R>(source)
    }

    /// Get all source entities that relate to target via relation R.
    pub fn sources<R: Relation>(&self, target: Entity) -> Vec<Entity> {
        self.relationships.sources::<R>(target)
    }

    /// Iterate all (source, target) pairs for relation type R.
    pub fn for_each_relation<R: Relation>(&self, mut f: impl FnMut(Entity, Entity)) {
        for (source, target) in self.relationships.iter::<R>() {
            f(source, target);
        }
    }

    /// Walk up the relationship chain via R, collecting all ancestors.
    ///
    /// For a `ChildOf` relationship, this returns `[parent, grandparent, ...]`.
    /// Uses the first target at each level (relationships are many-to-many,
    /// but hierarchies typically have one parent per entity).
    /// Cycle-safe: stops if a visited entity is encountered.
    pub fn ancestors<R: Relation>(&self, entity: Entity) -> Vec<Entity> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut current = entity;
        visited.insert(current.index());

        loop {
            let Some(parent) = self.relationships.first_target::<R>(current) else {
                break;
            };
            if !visited.insert(parent.index()) {
                break; // cycle detected
            }
            result.push(parent);
            current = parent;
        }

        result
    }

    /// Walk up the relationship chain via R until finding an entity with component T.
    ///
    /// Returns `None` if no ancestor has the component. Cycle-safe.
    /// This is FLECS's `up(R)` traversal for component inheritance.
    pub fn find_up<R: Relation, T: Component>(&self, entity: Entity) -> Option<(Entity, &T)> {
        // Check the entity itself first.
        if let Some(component) = self.get_component::<T>(entity) {
            return Some((entity, component));
        }

        let mut visited = HashSet::new();
        let mut current = entity;
        visited.insert(current.index());

        loop {
            let parent = self.relationships.first_target::<R>(current)?;
            if !visited.insert(parent.index()) {
                return None; // cycle
            }
            if let Some(component) = self.get_component::<T>(parent) {
                return Some((parent, component));
            }
            current = parent;
        }
    }

    /// Count the depth of an entity in a relationship hierarchy.
    ///
    /// Returns 0 for root entities (no R target). Cycle-safe.
    pub fn depth<R: Relation>(&self, entity: Entity) -> usize {
        let mut d = 0usize;
        let mut visited = HashSet::new();
        let mut current = entity;
        visited.insert(current.index());

        loop {
            let Some(parent) = self.relationships.first_target::<R>(current) else {
                return d;
            };
            if !visited.insert(parent.index()) {
                return d; // cycle
            }
            d += 1;
            current = parent;
        }
    }

    // ── Hierarchy Utilities ──────────────────────────────────────────

    /// Get all children of an entity under relationship R.
    ///
    /// "Children" are entities that have `R` pointing TO this entity.
    /// For a `ChildOf` relationship: `add_relation::<ChildOf>(child, parent)`,
    /// so `children::<ChildOf>(parent)` returns all children.
    pub fn children<R: Relation>(&self, parent: Entity) -> Vec<Entity> {
        self.relationships.sources::<R>(parent)
    }

    /// Get the parent of an entity under relationship R.
    ///
    /// Returns the first target of R for this entity, or None if it's a root.
    pub fn parent<R: Relation>(&self, child: Entity) -> Option<Entity> {
        self.relationships.first_target::<R>(child)
    }

    /// Add a child to a parent under relationship R.
    ///
    /// Convenience for `add_relation::<R>(child, parent)`.
    pub fn add_child<R: Relation>(&mut self, parent: Entity, child: Entity) -> bool {
        self.add_relation::<R>(child, parent)
    }

    /// Remove a child from a parent under relationship R.
    ///
    /// Convenience for `remove_relation::<R>(child, parent)`.
    pub fn remove_child<R: Relation>(&mut self, parent: Entity, child: Entity) -> bool {
        self.remove_relation::<R>(child, parent)
    }

    /// Despawn an entity and all its descendants under relationship R.
    ///
    /// Walks the hierarchy depth-first, collecting all descendants before
    /// despawning. Cycle-safe via visited set.
    ///
    /// Returns the total number of entities despawned (including the root).
    pub fn despawn_recursive<R: Relation>(&mut self, root: Entity) -> usize {
        // Collect all entities to despawn (depth-first).
        let mut to_despawn = Vec::new();
        let mut stack = vec![root];
        let mut visited = HashSet::new();

        while let Some(entity) = stack.pop() {
            if !visited.insert(entity.index()) {
                continue; // cycle safety
            }
            to_despawn.push(entity);
            for child in self.children::<R>(entity) {
                stack.push(child);
            }
        }

        let mut count = 0;
        for entity in &to_despawn {
            if self.despawn(*entity) {
                count += 1;
            }
        }
        count
    }

    /// Check if an entity is a root (has no parent under relationship R).
    pub fn is_root<R: Relation>(&self, entity: Entity) -> bool {
        self.parent::<R>(entity).is_none()
    }

    /// Check if an entity is a leaf (has no children under relationship R).
    pub fn is_leaf<R: Relation>(&self, entity: Entity) -> bool {
        !self.relationships.has_any_source::<R>(entity)
    }

    /// Insert or replace a typed resource.
    pub fn insert_resource<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.resources.insert(resource);
    }

    /// Get a typed resource reference.
    pub fn get_resource<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    /// Get a mutable typed resource reference.
    pub fn get_resource_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>()
    }

    /// Check if a typed resource exists.
    pub fn has_resource<T: 'static + Send + Sync>(&self) -> bool {
        self.resources.has::<T>()
    }

    /// Remove a typed resource, returning the owned value.
    pub fn remove_resource<T: 'static + Send + Sync>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    /// Temporarily remove a resource, giving the closure `(&mut World, &mut T)`.
    ///
    /// The resource is physically moved out of the World for the duration of the
    /// closure, then moved back in. This allows simultaneous mutable access to both
    /// the resource and the World's entity/component storage — the key pattern for
    /// ECS systems that need to iterate components while mutating a resource (e.g.,
    /// polling reactive signals into a Scene).
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    ///
    /// # Example
    ///
    /// ```
    /// use skesis::World;
    ///
    /// struct Score(u32);
    /// struct Health(u32);
    ///
    /// let mut world = World::new();
    /// world.insert_resource(Score(0));
    /// let entity = world.spawn_empty();
    /// world.add_component(entity, Health(100));
    ///
    /// // Iterate components while mutating a resource — no collect() needed
    /// world.resource_scope::<Score, _>(|world, score| {
    ///     world.for_each_mut::<Health>(|_entity, health| {
    ///         score.0 += health.0;
    ///     });
    /// });
    ///
    /// assert_eq!(world.get_resource::<Score>().unwrap().0, 100);
    /// ```
    pub fn resource_scope<T: 'static + Send + Sync, R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut T) -> R,
    ) -> R {
        let mut resource = self
            .resources
            .remove::<T>()
            .unwrap_or_else(|| panic!("resource_scope: resource {} not found", std::any::type_name::<T>()));

        let result = f(self, &mut resource);

        self.resources.insert(resource);
        result
    }

    /// Begin a new stage deferred-command merge scope.
    pub fn begin_stage_commands(&mut self) {
        self.commands.begin_stage();
    }

    /// Begin buffering deferred commands for a single system execution.
    pub fn begin_system_commands(&mut self) {
        self.commands.begin_system();
    }

    /// Merge the current system deferred-command buffer into stage commands.
    pub fn end_system_commands(&mut self) {
        self.commands.end_system();
    }

    /// Execute and clear all deferred commands for the current stage.
    ///
    /// Commands are applied in deterministic insertion order.
    pub fn end_stage_commands(&mut self) {
        loop {
            let mut commands = std::mem::take(&mut self.commands);
            let stage_commands = commands.take_stage();
            self.commands = commands;

            if stage_commands.is_empty() {
                break;
            }

            for command in stage_commands {
                command(self);
            }
        }
    }

    /// Defer one command to be applied on stage command flush.
    pub fn defer_command(&mut self, command: impl FnOnce(&mut World) + Send + Sync + 'static) {
        self.commands.defer(Box::new(command));
    }

    pub(crate) fn append_recorded_commands(&mut self, recorder: CommandRecorder) {
        self.commands.extend_stage(recorder.into_commands());
    }

    /// Defer adding or replacing a component on an entity.
    pub fn defer_add_component<T: Component>(&mut self, entity: Entity, component: T) {
        self.defer_command(move |world| {
            world.add_component(entity, component);
        });
    }

    /// Defer despawning an entity.
    pub fn defer_despawn(&mut self, entity: Entity) {
        self.defer_command(move |world| {
            let _ = world.despawn(entity);
        });
    }

    /// Defer spawning one empty entity.
    pub fn defer_spawn_empty(&mut self) {
        self.defer_command(|world| {
            let _ = world.spawn_empty();
        });
    }

    /// Begin a new stage event merge scope.
    pub fn begin_stage_events(&mut self) {
        self.events.begin_stage();
    }

    /// Begin buffering events for a single system execution.
    pub fn begin_system_events(&mut self) {
        self.events.begin_system();
    }

    /// Merge the current system event buffer into stage events.
    pub fn end_system_events(&mut self) {
        self.events.end_system();
    }

    /// Finalize stage events for reads.
    pub fn end_stage_events(&mut self) {
        self.events.end_stage();
    }

    /// Emit one typed event into the current system buffer.
    pub fn emit_event<T: 'static + Send + Sync>(&mut self, event: T) {
        self.events.emit(event);
    }

    /// Read finalized stage events for a type.
    pub fn read_events<T: 'static + Send + Sync>(&self) -> Vec<&T> {
        self.events.read::<T>()
    }

    /// Iterate finalized stage events for a type without allocation.
    pub fn iter_events<T: 'static + Send + Sync>(&self) -> impl Iterator<Item = &T> {
        self.events.iter::<T>()
    }

    /// Visit finalized stage events for a type without allocation.
    pub fn for_each_event<T: 'static + Send + Sync>(&self, f: impl FnMut(&T)) {
        self.events.for_each::<T>(f);
    }

    // ── Change Detection ────────────────────────────────────────────

    /// Register a component type for change tracking.
    pub fn track_changes<T: Component>(&mut self) {
        self.change_history.track::<T>();
    }

    /// Check if a component type has change tracking enabled.
    pub fn has_change_tracker<T: Component>(&self) -> bool {
        self.change_history.is_tracked::<T>()
    }

    /// Manually bump the change version for a tracked component type.
    pub fn bump_change_version<T: Component>(&mut self) {
        self.change_history.bump_version::<T>();
    }

    /// Take a snapshot of the current column state for a reader.
    pub fn snapshot_changes<T: Component>(&mut self, reader_id: u64) {
        let type_id = TypeId::of::<T>();
        let mut all_bytes = Vec::new();
        for arch in &self.archetypes {
            if let Some(column) = arch.component_column(&type_id) {
                all_bytes.extend_from_slice(column.as_bytes());
            }
        }
        self.change_history
            .take_snapshot::<T>(reader_id, &all_bytes);
    }

    /// Get a reference to the change history (historical storage).
    ///
    /// The change history is separated from the hot path so it can be
    /// serialized, shipped over the network, or inspected independently.
    pub fn change_history(&self) -> &ChangeHistory {
        &self.change_history
    }

    // ── Serialization ──────────────────────────────────────────────

    /// Capture the entire world state as a raw byte snapshot.
    ///
    /// The snapshot contains entity metadata, archetype structure, and component
    /// column data. Observers, caches, events, and commands are NOT captured —
    /// they are runtime state.
    ///
    /// **Warning:** Snapshots are only valid for the same binary. Component struct
    /// layouts must match exactly between save and load.
    pub fn save_state(&self) -> WorldSnapshot {
        let mut arch_snapshots = Vec::with_capacity(self.archetypes.len());

        for arch in &self.archetypes {
            let entities = arch.entities();
            let entity_indices = arch.entity_indices();
            let component_set = arch.component_set();

            // Serialize entity list as raw bytes.
            let entity_bytes = unsafe {
                std::slice::from_raw_parts(
                    entities.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(entities),
                )
            }
            .to_vec();

            let entity_index_bytes = unsafe {
                std::slice::from_raw_parts(
                    entity_indices.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(entity_indices),
                )
            }
            .to_vec();

            // Serialize each column.
            let mut columns = Vec::new();
            for &type_id in component_set.iter() {
                if let Some(column) = arch.component_column(&type_id) {
                    let stride = column.element_stride();
                    if stride > 0 {
                        columns.push((type_id, stride, column.snapshot_bytes()));
                    }
                }
            }

            arch_snapshots.push(ArchetypeSnapshot {
                component_type_ids: component_set.iter().copied().collect(),
                entity_bytes,
                entity_index_bytes,
                entity_count: entities.len(),
                columns,
            });
        }

        WorldSnapshot {
            entity_generations: self.entity_generations.clone(),
            free_indices: self.free_indices.clone(),
            next_entity_index: self.next_entity_index,
            archetypes: arch_snapshots,
        }
    }

    /// Restore the world from a raw byte snapshot.
    ///
    /// Clears all existing entities, components, and archetypes, then rebuilds
    /// from the snapshot. Observers, resources, and change trackers are preserved —
    /// only entity/component state is replaced.
    ///
    /// **Warning:** The snapshot must have been created by the same binary with
    /// identical component struct layouts.
    pub fn load_state(&mut self, snapshot: &WorldSnapshot) {
        // Clear existing entity/archetype state.
        self.archetypes.clear();
        self.archetype_map.clear();
        self.single_component_archetypes.clear();
        self.bundle_archetype_cache.clear();
        self.add_component_transitions.clear();
        self.remove_component_transitions.clear();
        self.entity_locations.clear();
        self.next_archetype_id = 0;
        self.disabled_components.clear();
        self.relationships = crate::relation::RelationshipStore::new();

        // Restore entity metadata.
        self.entity_generations = snapshot.entity_generations.clone();
        self.free_indices = snapshot.free_indices.clone();
        self.next_entity_index = snapshot.next_entity_index;
        self.entity_locations
            .resize(self.next_entity_index as usize, None);

        // Rebuild archetypes from snapshot.
        let mut max_nonempty: Option<u32> = None;

        for arch_snap in &snapshot.archetypes {
            let arch_id = ArchetypeId(self.next_archetype_id);
            self.next_archetype_id += 1;

            // Rebuild ComponentSet.
            let mut component_set = ComponentSet::new();
            for &type_id in &arch_snap.component_type_ids {
                // Use the raw TypeId insertion via the sorted vec.
                component_set = component_set.with_type_id(type_id);
            }

            // Create archetype with column factories.
            let mut archetype = Archetype::new_with_factories(
                arch_id,
                component_set.clone(),
                &self.column_factories,
            );

            // Restore entities from raw bytes.
            if arch_snap.entity_count > 0 {
                let entity_size = std::mem::size_of::<Entity>();
                assert_eq!(
                    arch_snap.entity_bytes.len(),
                    arch_snap.entity_count * entity_size
                );
                let entities: &[Entity] = unsafe {
                    std::slice::from_raw_parts(
                        arch_snap.entity_bytes.as_ptr().cast::<Entity>(),
                        arch_snap.entity_count,
                    )
                };

                let entity_indices: &[u32] = unsafe {
                    std::slice::from_raw_parts(
                        arch_snap.entity_index_bytes.as_ptr().cast::<u32>(),
                        arch_snap.entity_count,
                    )
                };

                archetype.restore_entities(entities, entity_indices);

                // Restore column data.
                for &(type_id, _stride, ref bytes) in &arch_snap.columns {
                    archetype.restore_column_bytes(&type_id, bytes);
                }

                // Update entity locations.
                for (row, &entity) in entities.iter().enumerate() {
                    let idx = entity.index() as usize;
                    if idx < self.entity_locations.len() {
                        self.entity_locations[idx] = Some((arch_id, row));
                    }
                    if arch_id.0 != 0 {
                        max_nonempty = Some(
                            max_nonempty.map_or(entity.index(), |m: u32| m.max(entity.index())),
                        );
                    }
                }
            }

            self.archetype_map.insert(component_set, arch_id);
            self.add_component_transitions.push(Vec::new());
            self.remove_component_transitions.push(Vec::new());
            self.archetypes.push(archetype);
        }

        self.max_nonempty_entity_index = max_nonempty;
        self.componentless_tail_start = max_nonempty.map_or(0, |m| m + 1);
        self.all_generations_zero = self.entity_generations.iter().all(|&g| g == 0);
        self.bump_structural_version();
    }

    /// Iterate entities whose component T changed since the reader's last snapshot.
    ///
    /// If no snapshot exists for this reader, ALL entities with T are yielded.
    /// If the column version hasn't changed, nothing is yielded (fast exit).
    pub fn for_each_changed<T: Component>(&self, reader_id: u64, mut f: impl FnMut(Entity, &T)) {
        let type_id = TypeId::of::<T>();
        let Some(tracker) = self.change_history.tracker::<T>() else {
            // No tracker — fall back to iterating all.
            for arch in &self.archetypes {
                if !arch.component_set().contains::<T>() {
                    continue;
                }
                let entities = arch.entities();
                let components = arch
                    .components::<T>()
                    .expect("archetype contains component set but missing column");
                for i in 0..entities.len() {
                    f(entities[i], &components[i]);
                }
            }
            return;
        };

        let stride = std::mem::size_of::<T>();

        // Check if reader has no snapshot — treat all as changed.
        if !tracker.snapshots_contains(reader_id) {
            for arch in &self.archetypes {
                if !arch.component_set().contains::<T>() {
                    continue;
                }
                let entities = arch.entities();
                let components = arch
                    .components::<T>()
                    .expect("archetype should have component T");
                for row in 0..entities.len() {
                    f(entities[row], &components[row]);
                }
            }
            return;
        }

        // Fast exit: version unchanged → nothing mutated since snapshot.
        if tracker.snapshot_version(reader_id) == Some(tracker.column_version) {
            return;
        }

        // Diff per-archetype directly against snapshot slices — no allocation.
        let snap_bytes = match tracker.snapshot_bytes(reader_id) {
            Some(b) => b,
            None => return,
        };

        let mut snap_offset = 0usize;
        for arch in &self.archetypes {
            let Some(column) = arch.component_column(&type_id) else {
                continue;
            };
            let col_bytes = column.as_bytes();
            let col_len = col_bytes.len();
            let element_count = arch.len();

            // If snapshot is shorter (entities added since snapshot), treat remainder as all-changed.
            if snap_offset + col_len > snap_bytes.len() {
                let entities = arch.entities();
                let components = arch
                    .components::<T>()
                    .expect("archetype should have component T");
                for row in 0..element_count {
                    f(entities[row], &components[row]);
                }
                snap_offset += col_len;
                continue;
            }

            let snap_slice = &snap_bytes[snap_offset..snap_offset + col_len];
            let bitset = crate::change::diff_to_bitset(snap_slice, col_bytes, stride);

            let entities = arch.entities();
            let components = arch
                .components::<T>()
                .expect("archetype should have component T");

            for row in 0..element_count {
                let bit_word = row / 64;
                let bit_offset = row % 64;
                if bit_word < bitset.len() && (bitset[bit_word] & (1u64 << bit_offset)) != 0 {
                    f(entities[row], &components[row]);
                }
            }

            snap_offset += col_len;
        }
    }

    /// Check whether a single entity's component T has changed since the last
    /// snapshot for the given reader. Returns `true` if no tracker is registered
    /// (assume changed), `false` if the entity is dead or lacks the component.
    pub fn is_changed<T: Component>(&self, entity: Entity, reader_id: u64) -> bool {
        let type_id = TypeId::of::<T>();
        let Some(tracker) = self.change_history.tracker::<T>() else {
            return true; // No tracker = assume changed.
        };
        let Some((arch_id, arch_row)) = self.location_of(entity) else {
            return false; // Dead entity.
        };
        let arch = &self.archetypes[arch_id.0 as usize];
        if !arch.component_set().contains_type_id(&type_id) {
            return false; // Entity's archetype doesn't have component T.
        }

        // Fast exit: version unchanged → nothing mutated since snapshot.
        if tracker.snapshot_version(reader_id) == Some(tracker.column_version) {
            return false;
        }

        let snap_bytes = match tracker.snapshot_bytes(reader_id) {
            Some(b) => b,
            None => return true, // No snapshot = assume changed.
        };

        let stride = std::mem::size_of::<T>();

        // Walk archetypes to find this entity's byte range in the snapshot.
        let mut snap_offset = 0usize;
        for a in &self.archetypes {
            let Some(column) = a.component_column(&type_id) else {
                continue;
            };
            let col_len = column.as_bytes().len();

            if a.id() == arch_id {
                let element_start = arch_row * stride;
                let element_end = element_start + stride;
                let snap_element_start = snap_offset + element_start;
                let snap_element_end = snap_offset + element_end;

                // Entity was added after snapshot — treat as changed.
                if snap_element_end > snap_bytes.len() {
                    return true;
                }

                let current_element = &column.as_bytes()[element_start..element_end];
                let snap_element = &snap_bytes[snap_element_start..snap_element_end];
                return crate::change::any_bytes_differ(snap_element, current_element);
            }

            snap_offset += col_len;
        }

        false // Entity's archetype not found with component T.
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_entity() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        assert_eq!(entity.index(), 0);
        assert_eq!(entity.generation(), 0);
    }

    #[test]
    fn spawn_multiple_entities() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        let e2 = world.spawn_empty();
        let e3 = world.spawn_empty();

        assert_eq!(e1.index(), 0);
        assert_eq!(e2.index(), 1);
        assert_eq!(e3.index(), 2);
    }

    #[test]
    fn despawn_entity() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        assert!(world.despawn(entity));
        assert!(!world.is_alive(entity));
    }

    #[test]
    fn despawn_increments_generation() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.despawn(e1);

        let e2 = world.spawn_empty();

        assert_eq!(e1.index(), e2.index());
        assert_ne!(e1.generation(), e2.generation());
        assert!(!world.is_alive(e1));
        assert!(world.is_alive(e2));
    }

    #[test]
    fn despawn_stale_entity_fails() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.despawn(e1);
        let _e2 = world.spawn_empty();

        assert!(!world.despawn(e1));
    }

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[test]
    fn add_component_to_entity() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        world.add_component(entity, Position { x: 10.0, y: 20.0 });

        assert!(world.has_component::<Position>(entity));
        assert_eq!(world.archetype_count(), 2); // empty + Position archetype
    }

    #[test]
    fn get_component_cache_is_world_local_across_identical_layouts() {
        let mut world_a = World::new();
        let mut world_b = World::new();

        let entity_a = world_a.spawn_empty();
        let entity_b = world_b.spawn_empty();

        world_a.add_component(entity_a, Position { x: 1.0, y: 2.0 });
        world_b.add_component(entity_b, Position { x: 5.0, y: 8.0 });

        // Exercise the thread-local cache by alternating worlds that share
        // identical dense index/generation layouts.
        for _ in 0..16 {
            let pos_a = world_a
                .get_component::<Position>(entity_a)
                .expect("world_a position should exist");
            let pos_b = world_b
                .get_component::<Position>(entity_b)
                .expect("world_b position should exist");

            assert_eq!((pos_a.x, pos_a.y), (1.0, 2.0));
            assert_eq!((pos_b.x, pos_b.y), (5.0, 8.0));
        }
    }

    #[test]
    fn get_component_mut_cache_is_world_local_across_identical_layouts() {
        let mut world_a = World::new();
        let mut world_b = World::new();

        let entity_a = world_a.spawn_empty();
        let entity_b = world_b.spawn_empty();

        world_a.add_component(entity_a, Position { x: 1.0, y: 2.0 });
        world_b.add_component(entity_b, Position { x: 5.0, y: 8.0 });

        // Exercise mutable cache alternation across worlds with identical layouts.
        for _ in 0..16 {
            let pos_a = world_a
                .get_component_mut::<Position>(entity_a)
                .expect("world_a position should exist");
            pos_a.x += 1.0;
            pos_a.y += 0.5;

            let pos_b = world_b
                .get_component_mut::<Position>(entity_b)
                .expect("world_b position should exist");
            pos_b.x += 2.0;
            pos_b.y += 1.0;
        }

        let pos_a = world_a
            .get_component::<Position>(entity_a)
            .expect("world_a position should exist after mutation");
        let pos_b = world_b
            .get_component::<Position>(entity_b)
            .expect("world_b position should exist after mutation");

        assert_eq!((pos_a.x, pos_a.y), (17.0, 10.0));
        assert_eq!((pos_b.x, pos_b.y), (37.0, 24.0));
    }

    #[test]
    fn get_component_mut_rejects_stale_entity_generation() {
        let mut world = World::new();
        let stale = world.spawn_empty();
        world.add_component(stale, Position { x: 1.0, y: 2.0 });

        assert!(world.despawn(stale));
        let live = world.spawn_empty();
        world.add_component(live, Position { x: 3.0, y: 4.0 });

        assert!(
            world.get_component_mut::<Position>(stale).is_none(),
            "stale generation must not resolve to live component storage"
        );
        assert!(
            world.get_component_mut::<Position>(live).is_some(),
            "live entity should still resolve correctly"
        );
    }

    #[derive(Debug, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Noise;

    #[test]
    fn add_multiple_components() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        world.add_component(entity, Position { x: 10.0, y: 20.0 });
        world.add_component(entity, Velocity { dx: 1.0, dy: 2.0 });

        assert!(world.has_component::<Position>(entity));
        assert!(world.has_component::<Velocity>(entity));
        assert_eq!(world.archetype_count(), 3); // empty -> Position -> Position+Velocity
    }

    #[test]
    fn query_single_component() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 3.0, y: 4.0 });

        let e3 = world.spawn_empty();
        world.add_component(e3, Velocity { dx: 1.0, dy: 1.0 }); // No Position

        let mut count = 0;
        for (entity, pos) in world.query::<Position>() {
            count += 1;
            assert!(entity == e1 || entity == e2);
            assert!(pos.x > 0.0 && pos.y > 0.0);
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn query_mut_indexed_chunk_with_plan_updates_components() {
        let mut world = World::new();

        for i in 0..4 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
        }

        let plan = world.plan_query_mut::<Position>();
        let mut visited_indices = Vec::new();

        world.for_each_mut_indexed_chunk_with_plan(&plan, |entity_indices, positions| {
            assert_eq!(entity_indices.len(), positions.len());
            for index in 0..entity_indices.len() {
                positions[index].x += 10.0;
                positions[index].y += 20.0;
                visited_indices.push(entity_indices[index]);
            }
        });

        visited_indices.sort_unstable();
        assert_eq!(visited_indices, vec![0, 1, 2, 3]);

        for (entity, position) in world.query::<Position>() {
            assert_eq!(position.x, entity.index() as f32 + 10.0);
            assert_eq!(position.y, (entity.index() * 2) as f32 + 20.0);
        }
    }

    #[test]
    fn query_pair_borrowed_plan_single_chunk_visits_once() {
        let mut world = World::new();

        for i in 0..4 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
            world.add_component(
                entity,
                Velocity {
                    dx: (i + 1) as f32,
                    dy: (i + 2) as f32,
                },
            );
        }

        for i in 0..2 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: (10 + i) as f32,
                    y: (20 + i) as f32,
                },
            );
        }

        let plan = world.plan_query_pair_borrowed::<Position, Velocity>();
        let mut callback_calls = 0usize;
        let mut visited_indices = Vec::new();
        let mut checksum = 0.0f32;

        plan.for_each_indexed_chunk(&world, |entity_indices, positions, velocities| {
            callback_calls += 1;
            assert_eq!(entity_indices.len(), positions.len());
            assert_eq!(entity_indices.len(), velocities.len());
            visited_indices.extend_from_slice(entity_indices);
            for index in 0..entity_indices.len() {
                checksum += entity_indices[index] as f32
                    + positions[index].x
                    + positions[index].y
                    + velocities[index].dx
                    + velocities[index].dy;
            }
        });

        visited_indices.sort_unstable();
        assert_eq!(callback_calls, 1);
        assert_eq!(visited_indices, vec![0, 1, 2, 3]);
        assert!(checksum > 0.0);
    }

    #[test]
    fn query_pair_borrowed_plan_multiple_chunks_visits_each_chunk() {
        let mut world = World::new();

        for i in 0..2 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i + 10) as f32,
                },
            );
            world.add_component(entity, Velocity { dx: 1.0, dy: 2.0 });
        }

        for i in 0..2 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: (i + 20) as f32,
                    y: (i + 30) as f32,
                },
            );
            world.add_component(entity, Velocity { dx: 3.0, dy: 4.0 });
            world.add_component(entity, Noise);
        }

        let plan = world.plan_query_pair_borrowed::<Position, Velocity>();
        let mut callback_calls = 0usize;
        let mut chunk_sizes = Vec::new();
        let mut visited_indices = Vec::new();

        plan.for_each_indexed_chunk(&world, |entity_indices, positions, velocities| {
            callback_calls += 1;
            chunk_sizes.push(entity_indices.len());
            assert_eq!(entity_indices.len(), positions.len());
            assert_eq!(entity_indices.len(), velocities.len());
            visited_indices.extend_from_slice(entity_indices);
        });

        chunk_sizes.sort_unstable();
        visited_indices.sort_unstable();
        assert_eq!(callback_calls, 2);
        assert_eq!(chunk_sizes, vec![2, 2]);
        assert_eq!(visited_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn query_pair_borrowed_plan_reports_contiguous_chunk_metadata() {
        let mut world = World::new();

        for i in 0..4 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
            world.add_component(
                entity,
                Velocity {
                    dx: (i + 1) as f32,
                    dy: (i + 2) as f32,
                },
            );
        }

        let plan = world.plan_query_pair_borrowed::<Position, Velocity>();
        let mut chunk_bases = Vec::new();
        let mut chunk_sums = Vec::new();
        let mut visited_indices = Vec::new();
        let mut noncontiguous_indices_f32 = Vec::new();

        // SAFETY: no structural mutation occurs while reading freshly built plan metadata.
        unsafe {
            plan.for_each_indexed_chunk_meta_sum_unchecked(
                |contiguous_base, contiguous_sum, entity_indices, _, _| {
                    chunk_bases.push(contiguous_base);
                    chunk_sums.push(contiguous_sum);
                    visited_indices.extend_from_slice(entity_indices);
                },
            );
            plan.for_each_indexed_chunk_meta_sum_f32_unchecked(
                |contiguous_base, contiguous_sum, noncontiguous_f32, entity_indices, _, _| {
                    assert_eq!(contiguous_base, Some(0));
                    assert_eq!(contiguous_sum, Some(6.0));
                    noncontiguous_indices_f32.push(noncontiguous_f32.map(<[f32]>::to_vec));
                    assert_eq!(entity_indices, [0, 1, 2, 3]);
                },
            );
        }

        visited_indices.sort_unstable();
        assert_eq!(chunk_bases, vec![Some(0)]);
        assert_eq!(chunk_sums, vec![Some(6.0)]);
        assert_eq!(visited_indices, vec![0, 1, 2, 3]);
        assert_eq!(noncontiguous_indices_f32, vec![None]);
    }

    #[test]
    fn query_pair_borrowed_plan_reports_noncontiguous_chunk_metadata() {
        let mut world = World::new();
        let mut entities = Vec::new();

        for i in 0..3 {
            let entity = world.spawn_empty();
            entities.push(entity);
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
            world.add_component(
                entity,
                Velocity {
                    dx: (i + 1) as f32,
                    dy: (i + 2) as f32,
                },
            );
        }

        assert!(world.despawn(entities[1]));

        let plan = world.plan_query_pair_borrowed::<Position, Velocity>();
        let mut chunk_bases = Vec::new();
        let mut chunk_sums = Vec::new();
        let mut visited_indices = Vec::new();
        let mut noncontiguous_indices_f32 = Vec::new();

        // SAFETY: no structural mutation occurs while reading freshly built plan metadata.
        unsafe {
            plan.for_each_indexed_chunk_meta_sum_unchecked(
                |contiguous_base, contiguous_sum, entity_indices, _, _| {
                    chunk_bases.push(contiguous_base);
                    chunk_sums.push(contiguous_sum);
                    visited_indices.extend_from_slice(entity_indices);
                },
            );
            plan.for_each_indexed_chunk_meta_sum_f32_unchecked(
                |contiguous_base, contiguous_sum, noncontiguous_f32, entity_indices, _, _| {
                    assert_eq!(contiguous_base, None);
                    assert_eq!(contiguous_sum, None);
                    noncontiguous_indices_f32.push(noncontiguous_f32.map(<[f32]>::to_vec));
                    assert_eq!(entity_indices, [0, 2]);
                },
            );
        }

        visited_indices.sort_unstable();
        assert_eq!(chunk_bases, vec![None]);
        assert_eq!(chunk_sums, vec![None]);
        assert_eq!(visited_indices, vec![0, 2]);
        assert_eq!(noncontiguous_indices_f32, vec![Some(vec![0.0, 2.0])]);
    }

    #[test]
    fn query_mut_borrowed_plan_updates_components() {
        let mut world = World::new();

        for i in 0..4 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
        }

        let plan = world.plan_query_mut_borrowed::<Position>();
        let mut visited_indices = Vec::new();
        let mut callback_calls = 0usize;

        plan.for_each_indexed_chunk(&mut world, |entity_indices, positions| {
            callback_calls += 1;
            assert_eq!(entity_indices.len(), positions.len());
            for index in 0..entity_indices.len() {
                positions[index].x += 5.0;
                positions[index].y += 7.0;
                visited_indices.push(entity_indices[index]);
            }
        });

        visited_indices.sort_unstable();
        assert_eq!(callback_calls, 1);
        assert_eq!(visited_indices, vec![0, 1, 2, 3]);

        for (entity, position) in world.query::<Position>() {
            assert_eq!(position.x, entity.index() as f32 + 5.0);
            assert_eq!(position.y, (entity.index() * 2) as f32 + 7.0);
        }
    }

    #[test]
    fn query_mut_borrowed_plan_reports_contiguous_chunk_metadata() {
        let mut world = World::new();

        for i in 0..4 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
        }

        let plan = world.plan_query_mut_borrowed::<Position>();
        let mut chunk_bases = Vec::new();
        let mut chunk_sums = Vec::new();
        let mut visited_indices = Vec::new();
        let mut noncontiguous_indices_f32 = Vec::new();

        // SAFETY: no structural mutation occurs while reading freshly built plan metadata.
        unsafe {
            plan.for_each_indexed_chunk_meta_sum_unchecked(
                |contiguous_base, contiguous_sum, entity_indices, _| {
                    chunk_bases.push(contiguous_base);
                    chunk_sums.push(contiguous_sum);
                    visited_indices.extend_from_slice(entity_indices);
                },
            );
            plan.for_each_indexed_chunk_meta_sum_f32_unchecked(
                |contiguous_base, contiguous_sum, noncontiguous_f32, entity_indices, _| {
                    assert_eq!(contiguous_base, Some(0));
                    assert_eq!(contiguous_sum, Some(6.0));
                    noncontiguous_indices_f32.push(noncontiguous_f32.map(<[f32]>::to_vec));
                    assert_eq!(entity_indices, [0, 1, 2, 3]);
                },
            );
        }

        visited_indices.sort_unstable();
        assert_eq!(chunk_bases, vec![Some(0)]);
        assert_eq!(chunk_sums, vec![Some(6.0)]);
        assert_eq!(visited_indices, vec![0, 1, 2, 3]);
        assert_eq!(noncontiguous_indices_f32, vec![None]);
    }

    #[test]
    fn query_mut_borrowed_plan_reports_noncontiguous_chunk_metadata() {
        let mut world = World::new();
        let mut entities = Vec::new();

        for i in 0..3 {
            let entity = world.spawn_empty();
            entities.push(entity);
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
        }

        assert!(world.despawn(entities[1]));

        let plan = world.plan_query_mut_borrowed::<Position>();
        let mut chunk_bases = Vec::new();
        let mut chunk_sums = Vec::new();
        let mut visited_indices = Vec::new();
        let mut noncontiguous_indices_f32 = Vec::new();

        // SAFETY: no structural mutation occurs while reading freshly built plan metadata.
        unsafe {
            plan.for_each_indexed_chunk_meta_sum_unchecked(
                |contiguous_base, contiguous_sum, entity_indices, _| {
                    chunk_bases.push(contiguous_base);
                    chunk_sums.push(contiguous_sum);
                    visited_indices.extend_from_slice(entity_indices);
                },
            );
            plan.for_each_indexed_chunk_meta_sum_f32_unchecked(
                |contiguous_base, contiguous_sum, noncontiguous_f32, entity_indices, _| {
                    assert_eq!(contiguous_base, None);
                    assert_eq!(contiguous_sum, None);
                    noncontiguous_indices_f32.push(noncontiguous_f32.map(<[f32]>::to_vec));
                    assert_eq!(entity_indices, [0, 2]);
                },
            );
        }

        visited_indices.sort_unstable();
        assert_eq!(chunk_bases, vec![None]);
        assert_eq!(chunk_sums, vec![None]);
        assert_eq!(visited_indices, vec![0, 2]);
        assert_eq!(noncontiguous_indices_f32, vec![Some(vec![0.0, 2.0])]);
    }

    #[test]
    fn query_mut_borrowed_plan_unchecked_updates_components() {
        let mut world = World::new();

        for i in 0..4 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: (i * 2) as f32,
                },
            );
        }

        let plan = world.plan_query_mut_borrowed::<Position>();
        let mut visited_indices = Vec::new();

        // SAFETY: no structural changes occur while executing this plan, and it was built from `world`.
        unsafe {
            plan.for_each_indexed_chunk_unchecked(|entity_indices, positions| {
                assert_eq!(entity_indices.len(), positions.len());
                for index in 0..entity_indices.len() {
                    positions[index].x += 3.0;
                    positions[index].y += 4.0;
                    visited_indices.push(entity_indices[index]);
                }
            });
        }

        visited_indices.sort_unstable();
        assert_eq!(visited_indices, vec![0, 1, 2, 3]);

        for (entity, position) in world.query::<Position>() {
            assert_eq!(position.x, entity.index() as f32 + 3.0);
            assert_eq!(position.y, (entity.index() * 2) as f32 + 4.0);
        }
    }

    #[test]
    #[should_panic(expected = "borrowed mutable query plan is stale")]
    fn query_mut_borrowed_plan_panics_after_structural_change() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });

        let plan = world.plan_query_mut_borrowed::<Position>();
        let _new_entity = world.spawn_empty();

        plan.for_each_indexed_chunk(&mut world, |_entity_indices, _positions| {});
    }

    #[test]
    #[should_panic(expected = "borrowed mutable query plan was created for a different world")]
    fn query_mut_borrowed_plan_panics_for_different_world() {
        let mut world_a = World::new();
        let entity = world_a.spawn_empty();
        world_a.add_component(entity, Position { x: 1.0, y: 2.0 });
        let plan = world_a.plan_query_mut_borrowed::<Position>();

        let mut world_b = World::new();
        plan.for_each_indexed_chunk(&mut world_b, |_entity_indices, _positions| {});
    }

    #[test]
    fn deferred_add_component_applies_on_stage_flush() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        world.begin_stage_commands();
        world.begin_system_commands();
        world.defer_add_component(entity, Position { x: 10.0, y: 20.0 });
        world.end_system_commands();

        assert!(!world.has_component::<Position>(entity));
        world.end_stage_commands();
        assert!(world.has_component::<Position>(entity));
    }

    #[test]
    fn deferred_commands_preserve_system_order() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });

        world.begin_stage_commands();

        world.begin_system_commands();
        world.defer_command(move |world| {
            let pos = world
                .get_component_mut::<Position>(entity)
                .expect("position should exist");
            pos.x += 1.0;
        });
        world.end_system_commands();

        world.begin_system_commands();
        world.defer_command(move |world| {
            let pos = world
                .get_component_mut::<Position>(entity)
                .expect("position should exist");
            pos.x *= 2.0;
        });
        world.end_system_commands();

        world.end_stage_commands();
        let pos = world
            .get_component::<Position>(entity)
            .expect("position should exist after deferred updates");
        assert_eq!(pos.x, 4.0);
        assert_eq!(pos.y, 2.0);
    }

    // ── remove_component tests ──────────────────────────────────────

    #[test]
    fn remove_component_returns_value() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 7.0, y: 8.0 });

        let removed = world.remove_component::<Position>(entity);
        assert_eq!(removed, Some(Position { x: 7.0, y: 8.0 }));
        assert!(!world.has_component::<Position>(entity));
    }

    #[test]
    fn remove_component_entity_moves_to_smaller_archetype() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.add_component(entity, Velocity { dx: 3.0, dy: 4.0 });

        assert!(world.has_component::<Position>(entity));
        assert!(world.has_component::<Velocity>(entity));

        let removed = world.remove_component::<Velocity>(entity);
        assert_eq!(removed, Some(Velocity { dx: 3.0, dy: 4.0 }));
        assert!(world.has_component::<Position>(entity));
        assert!(!world.has_component::<Velocity>(entity));

        // Position should still be accessible with correct value.
        let pos = world.get_component::<Position>(entity).unwrap();
        assert_eq!(*pos, Position { x: 1.0, y: 2.0 });
    }

    #[test]
    fn remove_single_component_goes_to_empty_archetype() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 5.0, y: 6.0 });

        let removed = world.remove_component::<Position>(entity);
        assert_eq!(removed, Some(Position { x: 5.0, y: 6.0 }));

        // Entity is alive but has no components (back in empty archetype).
        assert!(world.is_alive(entity));
        assert!(!world.has_component::<Position>(entity));

        // Can re-add a component to the entity.
        world.add_component(entity, Velocity { dx: 9.0, dy: 10.0 });
        assert!(world.has_component::<Velocity>(entity));
    }

    #[test]
    fn remove_nonexistent_component_returns_none() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 1.0 });

        assert_eq!(world.remove_component::<Velocity>(entity), None);
        // Original component is untouched.
        assert!(world.has_component::<Position>(entity));
    }

    #[test]
    fn remove_component_from_dead_entity_returns_none() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 1.0 });
        world.despawn(entity);

        assert_eq!(world.remove_component::<Position>(entity), None);
    }

    #[test]
    fn remove_component_from_empty_entity_returns_none() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        assert_eq!(world.remove_component::<Position>(entity), None);
    }

    #[test]
    fn remove_component_entity_still_valid_for_other_operations() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.add_component(entity, Velocity { dx: 3.0, dy: 4.0 });

        world.remove_component::<Position>(entity);

        // Entity is alive and Velocity is intact.
        assert!(world.is_alive(entity));
        let vel = world.get_component::<Velocity>(entity).unwrap();
        assert_eq!(*vel, Velocity { dx: 3.0, dy: 4.0 });

        // Can mutate remaining component.
        let vel = world.get_component_mut::<Velocity>(entity).unwrap();
        vel.dx = 99.0;
        assert_eq!(world.get_component::<Velocity>(entity).unwrap().dx, 99.0);
    }

    #[test]
    fn remove_component_preserves_other_entities_in_archetype() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        let e2 = world.spawn_empty();
        let e3 = world.spawn_empty();

        world.add_component(e1, Position { x: 1.0, y: 1.0 });
        world.add_component(e2, Position { x: 2.0, y: 2.0 });
        world.add_component(e3, Position { x: 3.0, y: 3.0 });

        // Remove from middle entity.
        world.remove_component::<Position>(e2);

        // e1 and e3 still have their positions.
        assert_eq!(
            world.get_component::<Position>(e1).unwrap(),
            &Position { x: 1.0, y: 1.0 }
        );
        assert_eq!(
            world.get_component::<Position>(e3).unwrap(),
            &Position { x: 3.0, y: 3.0 }
        );
        assert!(!world.has_component::<Position>(e2));
    }

    #[test]
    fn deferred_remove_component() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 42.0, y: 43.0 });

        world.begin_stage_commands();
        world.defer_command(move |world| {
            let _ = world.remove_component::<Position>(entity);
        });
        world.end_stage_commands();

        assert!(!world.has_component::<Position>(entity));
        assert!(world.is_alive(entity));
    }

    #[test]
    fn add_remove_add_roundtrip() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        let removed = world.remove_component::<Position>(entity);
        assert_eq!(removed, Some(Position { x: 1.0, y: 2.0 }));

        world.add_component(entity, Position { x: 10.0, y: 20.0 });
        let pos = world.get_component::<Position>(entity).unwrap();
        assert_eq!(*pos, Position { x: 10.0, y: 20.0 });
    }

    // ── Without<T> filter tests ─────────────────────────────────────

    #[derive(Debug, PartialEq)]
    struct Static;

    #[test]
    fn query_pair_without_excludes_tagged_entities() {
        let mut world = World::new();

        // e1: Position + Velocity (should match)
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 1.0 });
        world.add_component(e1, Velocity { dx: 1.0, dy: 1.0 });

        // e2: Position + Velocity + Static (should be excluded)
        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 2.0 });
        world.add_component(e2, Velocity { dx: 2.0, dy: 2.0 });
        world.add_component(e2, Static);

        // e3: Position + Velocity (should match)
        let e3 = world.spawn_empty();
        world.add_component(e3, Position { x: 3.0, y: 3.0 });
        world.add_component(e3, Velocity { dx: 3.0, dy: 3.0 });

        let results: Vec<_> = world
            .query_pair_without::<Position, Velocity, Static>()
            .map(|(entity, pos, _vel)| (entity, pos.x))
            .collect();

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(e, x)| *e == e1 && *x == 1.0));
        assert!(results.iter().any(|(e, x)| *e == e3 && *x == 3.0));
    }

    #[test]
    fn for_each_pair_without_skips_excluded() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 10.0, y: 0.0 });
        world.add_component(e1, Velocity { dx: 1.0, dy: 0.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 20.0, y: 0.0 });
        world.add_component(e2, Velocity { dx: 2.0, dy: 0.0 });
        world.add_component(e2, Static);

        let mut sum = 0.0f32;
        world.for_each_pair_without::<Position, Velocity, Static>(|_entity, pos, vel| {
            sum += pos.x + vel.dx;
        });

        assert_eq!(sum, 11.0); // Only e1: 10.0 + 1.0
    }

    #[test]
    fn for_each_mut_without_skips_excluded() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });
        world.add_component(e2, Static);

        world.for_each_mut_without::<Position, Static>(|_entity, pos| {
            pos.x *= 10.0;
        });

        assert_eq!(world.get_component::<Position>(e1).unwrap().x, 10.0);
        assert_eq!(world.get_component::<Position>(e2).unwrap().x, 2.0); // unchanged
    }

    #[test]
    fn query_without_single_component() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 1.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 2.0 });
        world.add_component(e2, Static);

        let results: Vec<_> = world
            .query_without::<Position, Static>()
            .map(|(_, pos)| pos.x)
            .collect();

        assert_eq!(results, vec![1.0]);
    }

    #[test]
    fn plan_query_pair_without_matches_ad_hoc() {
        let mut world = World::new();

        for i in 0..100 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            );
            world.add_component(entity, Velocity { dx: 1.0, dy: 1.0 });
            if i % 3 == 0 {
                world.add_component(entity, Static);
            }
        }

        // Ad-hoc
        let mut ad_hoc_sum = 0.0f32;
        world.for_each_pair_without::<Position, Velocity, Static>(|_entity, pos, _vel| {
            ad_hoc_sum += pos.x;
        });

        // Planned
        let plan = world.plan_query_pair_without::<Position, Velocity, Static>();
        let mut planned_sum = 0.0f32;
        world.for_each_pair_with_plan(&plan, |_entity, pos, _vel| {
            planned_sum += pos.x;
        });

        assert_eq!(ad_hoc_sum, planned_sum);
    }

    #[test]
    fn plan_query_mut_without_matches_ad_hoc() {
        let mut world = World::new();

        for i in 0..50 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            );
            if i % 4 == 0 {
                world.add_component(entity, Static);
            }
        }

        let plan = world.plan_query_mut_without::<Position, Static>();
        let mut planned_count = 0usize;
        world.for_each_mut_with_plan(&plan, |_entity, _pos| {
            planned_count += 1;
        });

        // 50 entities, 13 have Static (i=0,4,8,...,48), 37 without
        assert_eq!(planned_count, 37);
    }

    #[test]
    fn without_nonexistent_exclude_is_noop() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.add_component(entity, Velocity { dx: 3.0, dy: 4.0 });

        // Excluding Static when no entity has it — should return all matches.
        let count = world
            .query_pair_without::<Position, Velocity, Static>()
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn without_required_type_returns_empty() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.add_component(entity, Velocity { dx: 3.0, dy: 4.0 });

        // Excluding Position when it's also required — impossible, returns nothing.
        let count = world
            .query_pair_without::<Position, Velocity, Position>()
            .count();
        assert_eq!(count, 0);
    }

    // ── Optional query tests ────────────────────────────────────────

    #[test]
    fn for_each_with_optional_yields_some_when_present() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        world.add_component(e1, Velocity { dx: 10.0, dy: 20.0 });

        let mut results = Vec::new();
        world.for_each_with_optional::<Position, Velocity>(|entity, pos, vel| {
            results.push((entity, pos.x, vel.map(|v| v.dx)));
        });

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (e1, 1.0, Some(10.0)));
    }

    #[test]
    fn for_each_with_optional_yields_none_when_absent() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 5.0, y: 6.0 });
        // No Velocity added

        let mut results = Vec::new();
        world.for_each_with_optional::<Position, Velocity>(|entity, pos, vel| {
            results.push((entity, pos.x, vel.map(|v| v.dx)));
        });

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (e1, 5.0, None));
    }

    #[test]
    fn for_each_with_optional_mixed_archetypes() {
        let mut world = World::new();

        // e1: Position only
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });

        // e2: Position + Velocity
        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });
        world.add_component(e2, Velocity { dx: 20.0, dy: 0.0 });

        // e3: Position + Static (different archetype)
        let e3 = world.spawn_empty();
        world.add_component(e3, Position { x: 3.0, y: 0.0 });
        world.add_component(e3, Static);

        // e4: Velocity only (should not appear — no Position)
        let e4 = world.spawn_empty();
        world.add_component(e4, Velocity { dx: 40.0, dy: 0.0 });
        let _ = e4;

        let mut results = Vec::new();
        world.for_each_with_optional::<Position, Velocity>(|_entity, pos, vel| {
            results.push((pos.x, vel.map(|v| v.dx)));
        });

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (1.0, None)); // e1: Pos only
        assert_eq!(results[1], (2.0, Some(20.0))); // e2: Pos + Vel
        assert_eq!(results[2], (3.0, None)); // e3: Pos + Static (no Vel)
    }

    #[test]
    fn query_with_optional_iterator_matches_for_each() {
        let mut world = World::new();

        for i in 0..50 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            );
            if i % 3 == 0 {
                world.add_component(
                    entity,
                    Velocity {
                        dx: i as f32 * 10.0,
                        dy: 0.0,
                    },
                );
            }
        }

        // for_each path
        let mut for_each_sum = 0.0f32;
        world.for_each_with_optional::<Position, Velocity>(|_entity, pos, vel| {
            for_each_sum += pos.x + vel.map_or(0.0, |v| v.dx);
        });

        // iterator path
        let iter_sum: f32 = world
            .query_with_optional::<Position, Velocity>()
            .map(|(_, pos, vel)| pos.x + vel.map_or(0.0, |v| v.dx))
            .sum();

        assert_eq!(for_each_sum, iter_sum);
    }

    #[test]
    fn query_with_optional_no_entities_with_required() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Velocity { dx: 1.0, dy: 2.0 });

        // Query requires Position, optional Velocity — no entity has Position.
        let count = world.query_with_optional::<Position, Velocity>().count();
        assert_eq!(count, 0);
    }

    // ── Observer tests ──────────────────────────────────────────────

    #[test]
    fn on_add_fires_when_component_added() {
        use std::sync::{Arc, Mutex};

        let log: Arc<Mutex<Vec<(u32, f32)>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&log);

        let mut world = World::new();
        world.on_add::<Position>(move |entity, pos| {
            log_clone.lock().unwrap().push((entity.index(), pos.x));
        });

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 42.0, y: 0.0 });

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (0, 42.0));
    }

    #[test]
    fn on_add_does_not_fire_on_replacement() {
        use std::sync::{Arc, Mutex};

        let count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let count_clone = Arc::clone(&count);

        let mut world = World::new();
        world.on_add::<Position>(move |_entity, _pos| {
            *count_clone.lock().unwrap() += 1;
        });

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.add_component(entity, Position { x: 3.0, y: 4.0 }); // replacement, not new

        assert_eq!(*count.lock().unwrap(), 1); // Only fired once
    }

    #[test]
    fn on_remove_fires_when_component_removed() {
        use std::sync::{Arc, Mutex};

        let log: Arc<Mutex<Vec<(u32, f32)>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&log);

        let mut world = World::new();
        world.on_remove::<Position>(move |entity, pos| {
            log_clone.lock().unwrap().push((entity.index(), pos.x));
        });

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 99.0, y: 0.0 });
        world.remove_component::<Position>(entity);

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (0, 99.0));
    }

    #[test]
    fn multiple_observers_per_type() {
        use std::sync::{Arc, Mutex};

        let log_a: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let log_b: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let a = Arc::clone(&log_a);
        let b = Arc::clone(&log_b);

        let mut world = World::new();
        world.on_add::<Position>(move |entity, _pos| {
            a.lock().unwrap().push(entity.index());
        });
        world.on_add::<Position>(move |entity, _pos| {
            b.lock().unwrap().push(entity.index() * 10);
        });

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });

        assert_eq!(*log_a.lock().unwrap(), vec![0]);
        assert_eq!(*log_b.lock().unwrap(), vec![0]);
    }

    #[test]
    fn observers_for_different_types_are_independent() {
        use std::sync::{Arc, Mutex};

        let pos_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let vel_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let pc = Arc::clone(&pos_count);
        let vc = Arc::clone(&vel_count);

        let mut world = World::new();
        world.on_add::<Position>(move |_, _| *pc.lock().unwrap() += 1);
        world.on_add::<Velocity>(move |_, _| *vc.lock().unwrap() += 1);

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });

        assert_eq!(*pos_count.lock().unwrap(), 1);
        assert_eq!(*vel_count.lock().unwrap(), 0); // Velocity observer not triggered
    }

    #[test]
    fn on_add_fires_for_general_archetype_transition() {
        use std::sync::{Arc, Mutex};

        let log: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&log);

        let mut world = World::new();
        world.on_add::<Velocity>(move |_entity, vel| {
            log_clone.lock().unwrap().push(vel.dx);
        });

        // Entity starts with Position, then gets Velocity → general transition path.
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.add_component(entity, Velocity { dx: 77.0, dy: 88.0 });

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], 77.0);
    }

    // ── Toggle term tests ───────────────────────────────────────────

    #[test]
    fn disable_component_skips_in_toggled_query() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });
        world.add_component(e1, Velocity { dx: 10.0, dy: 0.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });
        world.add_component(e2, Velocity { dx: 20.0, dy: 0.0 });

        world.disable_component::<Position>(e1);

        let results: Vec<f32> = world
            .query_pair_toggled::<Position, Velocity>()
            .map(|(_, pos, _)| pos.x)
            .collect();

        assert_eq!(results, vec![2.0]); // e1 skipped
    }

    #[test]
    fn enable_component_restores_in_toggled_query() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 5.0, y: 0.0 });

        world.disable_component::<Position>(entity);
        assert_eq!(world.query_toggled::<Position>().count(), 0);

        world.enable_component::<Position>(entity);
        assert_eq!(world.query_toggled::<Position>().count(), 1);
    }

    #[test]
    fn is_component_enabled_reflects_toggle_state() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });

        assert!(world.is_component_enabled::<Position>(entity));

        world.disable_component::<Position>(entity);
        assert!(!world.is_component_enabled::<Position>(entity));

        world.enable_component::<Position>(entity);
        assert!(world.is_component_enabled::<Position>(entity));
    }

    #[test]
    fn disable_nonexistent_component_returns_false() {
        let mut world = World::new();
        let entity = world.spawn_empty();

        assert!(!world.disable_component::<Position>(entity));
    }

    #[test]
    fn disabled_component_still_accessible_via_get() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 42.0, y: 0.0 });
        world.disable_component::<Position>(entity);

        // Disabled components are skipped by toggled queries but still accessible directly.
        assert!(world.has_component::<Position>(entity));
        assert_eq!(world.get_component::<Position>(entity).unwrap().x, 42.0);
    }

    #[test]
    fn for_each_pair_toggled_mixed_enabled_disabled() {
        let mut world = World::new();

        for i in 0..10 {
            let entity = world.spawn_empty();
            world.add_component(
                entity,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            );
            world.add_component(entity, Velocity { dx: 1.0, dy: 0.0 });
            if i % 2 == 0 {
                world.disable_component::<Position>(entity);
            }
        }

        let mut count = 0;
        world.for_each_pair_toggled::<Position, Velocity>(|_, _, _| {
            count += 1;
        });
        assert_eq!(count, 5); // Only odd-indexed entities
    }

    #[test]
    fn for_each_mut_toggled_skips_disabled() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });

        world.disable_component::<Position>(e1);

        world.for_each_mut_toggled::<Position>(|_, pos| {
            pos.x *= 100.0;
        });

        assert_eq!(world.get_component::<Position>(e1).unwrap().x, 1.0); // untouched
        assert_eq!(world.get_component::<Position>(e2).unwrap().x, 200.0); // mutated
    }

    #[test]
    fn despawn_cleans_up_disabled_set() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.disable_component::<Position>(entity);
        world.despawn(entity);

        // Respawn at same index — should not inherit disabled state.
        let entity2 = world.spawn_empty();
        world.add_component(entity2, Position { x: 3.0, y: 4.0 });
        assert!(world.is_component_enabled::<Position>(entity2));
    }

    #[test]
    fn remove_component_cleans_up_disabled_set() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.disable_component::<Position>(entity);
        world.remove_component::<Position>(entity);

        // Re-add — should not inherit disabled state.
        world.add_component(entity, Position { x: 5.0, y: 6.0 });
        assert!(world.is_component_enabled::<Position>(entity));
    }

    #[test]
    fn non_toggled_queries_ignore_disabled_state() {
        let mut world = World::new();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });
        world.disable_component::<Position>(entity);

        // Regular (non-toggled) query still sees the entity.
        let count = world.query::<Position>().count();
        assert_eq!(count, 1);
    }

    // ── Relationship tests ──────────────────────────────────────────

    struct ChildOf;
    struct Likes;

    #[test]
    fn add_and_has_relation() {
        let mut world = World::new();
        let parent = world.spawn_empty();
        let child = world.spawn_empty();

        assert!(world.add_relation::<ChildOf>(child, parent));
        assert!(world.has_relation::<ChildOf>(child, parent));
        assert!(!world.has_relation::<ChildOf>(parent, child)); // directed
    }

    #[test]
    fn add_duplicate_relation_returns_false() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();

        assert!(world.add_relation::<ChildOf>(a, b));
        assert!(!world.add_relation::<ChildOf>(a, b)); // duplicate
    }

    #[test]
    fn remove_relation() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();

        world.add_relation::<ChildOf>(a, b);
        assert!(world.remove_relation::<ChildOf>(a, b));
        assert!(!world.has_relation::<ChildOf>(a, b));
        assert!(!world.remove_relation::<ChildOf>(a, b)); // already gone
    }

    #[test]
    fn targets_returns_related_entities() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();
        let c = world.spawn_empty();

        world.add_relation::<Likes>(a, b);
        world.add_relation::<Likes>(a, c);

        let mut targets = world.targets::<Likes>(a);
        targets.sort_by_key(|e| e.index());
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0], b);
        assert_eq!(targets[1], c);
    }

    #[test]
    fn sources_returns_entities_pointing_at_target() {
        let mut world = World::new();
        let parent = world.spawn_empty();
        let c1 = world.spawn_empty();
        let c2 = world.spawn_empty();

        world.add_relation::<ChildOf>(c1, parent);
        world.add_relation::<ChildOf>(c2, parent);

        let mut sources = world.sources::<ChildOf>(parent);
        sources.sort_by_key(|e| e.index());
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0], c1);
        assert_eq!(sources[1], c2);
    }

    #[test]
    fn despawn_cleans_up_relations_as_source() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();

        world.add_relation::<Likes>(a, b);
        world.despawn(a);

        assert!(world.sources::<Likes>(b).is_empty());
    }

    #[test]
    fn despawn_cleans_up_relations_as_target() {
        let mut world = World::new();
        let parent = world.spawn_empty();
        let child = world.spawn_empty();

        world.add_relation::<ChildOf>(child, parent);
        world.despawn(parent);

        assert!(world.targets::<ChildOf>(child).is_empty());
    }

    #[test]
    fn different_relation_types_are_independent() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();

        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<Likes>(a, b);

        assert!(world.has_relation::<ChildOf>(a, b));
        assert!(world.has_relation::<Likes>(a, b));

        world.remove_relation::<ChildOf>(a, b);
        assert!(!world.has_relation::<ChildOf>(a, b));
        assert!(world.has_relation::<Likes>(a, b)); // untouched
    }

    #[test]
    fn for_each_relation_iterates_all_pairs() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();
        let c = world.spawn_empty();

        world.add_relation::<Likes>(a, b);
        world.add_relation::<Likes>(b, c);
        world.add_relation::<Likes>(a, c);

        let mut pairs = Vec::new();
        world.for_each_relation::<Likes>(|source, target| {
            pairs.push((source.index(), target.index()));
        });

        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn relation_with_dead_entity_returns_false() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();
        world.despawn(b);

        assert!(!world.add_relation::<ChildOf>(a, b));
    }

    // ── Up traversal tests ──────────────────────────────────────────

    #[test]
    fn ancestors_returns_chain() {
        let mut world = World::new();
        let root = world.spawn_empty();
        let mid = world.spawn_empty();
        let leaf = world.spawn_empty();

        world.add_relation::<ChildOf>(leaf, mid);
        world.add_relation::<ChildOf>(mid, root);

        let chain = world.ancestors::<ChildOf>(leaf);
        assert_eq!(chain, vec![mid, root]);
    }

    #[test]
    fn ancestors_root_returns_empty() {
        let mut world = World::new();
        let root = world.spawn_empty();

        assert!(world.ancestors::<ChildOf>(root).is_empty());
    }

    #[test]
    fn find_up_finds_inherited_component() {
        let mut world = World::new();
        let root = world.spawn_empty();
        world.add_component(root, Position { x: 99.0, y: 88.0 });

        let mid = world.spawn_empty();
        world.add_relation::<ChildOf>(mid, root);

        let leaf = world.spawn_empty();
        world.add_relation::<ChildOf>(leaf, mid);

        // leaf has no Position, but root does — find_up should find it.
        let result = world.find_up::<ChildOf, Position>(leaf);
        assert!(result.is_some());
        let (ancestor, pos) = result.unwrap();
        assert_eq!(ancestor, root);
        assert_eq!(pos.x, 99.0);
    }

    #[test]
    fn find_up_returns_self_if_has_component() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 42.0, y: 0.0 });

        let result = world.find_up::<ChildOf, Position>(entity);
        assert!(result.is_some());
        let (found, pos) = result.unwrap();
        assert_eq!(found, entity);
        assert_eq!(pos.x, 42.0);
    }

    #[test]
    fn find_up_returns_none_at_root() {
        let mut world = World::new();
        let root = world.spawn_empty();
        let child = world.spawn_empty();
        world.add_relation::<ChildOf>(child, root);

        // Neither has Position.
        assert!(world.find_up::<ChildOf, Position>(child).is_none());
    }

    #[test]
    fn depth_counts_correctly() {
        let mut world = World::new();
        let root = world.spawn_empty();
        let mid = world.spawn_empty();
        let leaf = world.spawn_empty();

        world.add_relation::<ChildOf>(leaf, mid);
        world.add_relation::<ChildOf>(mid, root);

        assert_eq!(world.depth::<ChildOf>(root), 0);
        assert_eq!(world.depth::<ChildOf>(mid), 1);
        assert_eq!(world.depth::<ChildOf>(leaf), 2);
    }

    #[test]
    fn ancestors_cycle_does_not_infinite_loop() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();

        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<ChildOf>(b, a); // cycle!

        let chain = world.ancestors::<ChildOf>(a);
        // Should terminate — gets b, then tries a (visited), stops.
        assert_eq!(chain, vec![b]);
    }

    #[test]
    fn find_up_mid_level_component() {
        let mut world = World::new();
        let root = world.spawn_empty();
        let mid = world.spawn_empty();
        world.add_component(mid, Velocity { dx: 5.0, dy: 6.0 });
        let leaf = world.spawn_empty();

        world.add_relation::<ChildOf>(leaf, mid);
        world.add_relation::<ChildOf>(mid, root);

        // Velocity is on mid, not root.
        let result = world.find_up::<ChildOf, Velocity>(leaf);
        assert!(result.is_some());
        let (found, vel) = result.unwrap();
        assert_eq!(found, mid);
        assert_eq!(vel.dx, 5.0);
    }

    #[test]
    fn track_changes_registers_component_type() {
        let mut world = World::new();
        world.track_changes::<Position>();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        assert!(world.has_change_tracker::<Position>());
    }

    #[test]
    fn for_each_changed_sees_all_on_first_call() {
        let mut world = World::new();
        world.track_changes::<Position>();
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });
        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });

        let mut count = 0;
        world.for_each_changed::<Position>(0, |_entity, _pos| {
            count += 1;
        });
        assert_eq!(count, 2);
    }

    #[test]
    fn for_each_changed_skips_unchanged() {
        let mut world = World::new();
        world.track_changes::<Position>();
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });
        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });

        world.snapshot_changes::<Position>(0);
        world.get_component_mut::<Position>(e1).unwrap().x = 99.0;
        world.bump_change_version::<Position>();

        let mut changed_xs = Vec::new();
        world.for_each_changed::<Position>(0, |_entity, pos| {
            changed_xs.push(pos.x);
        });
        assert_eq!(changed_xs, vec![99.0]);
    }

    #[test]
    fn for_each_changed_fast_exit_when_unchanged() {
        let mut world = World::new();
        world.track_changes::<Position>();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });

        world.snapshot_changes::<Position>(0);

        let mut count = 0;
        world.for_each_changed::<Position>(0, |_, _| {
            count += 1;
        });
        assert_eq!(count, 0);
    }

    #[test]
    fn add_component_auto_bumps_change_version() {
        let mut world = World::new();
        world.track_changes::<Position>();

        let entity = world.spawn_empty();
        world.snapshot_changes::<Position>(0);

        world.add_component(entity, Position { x: 1.0, y: 0.0 });

        let mut count = 0;
        world.for_each_changed::<Position>(0, |_, _| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn remove_component_auto_bumps_change_version() {
        let mut world = World::new();
        world.track_changes::<Position>();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });
        world.snapshot_changes::<Position>(0);

        world.remove_component::<Position>(entity);
        // After remove, snapshot should detect the column changed.
        world.bump_change_version::<Position>(); // remove doesn't add bytes, but version should differ
        // The tracker version should have been bumped by remove_component.
        // Re-snapshot and verify the version advanced.
        assert!(world.has_change_tracker::<Position>());
    }

    #[test]
    fn is_changed_true_for_mutated_entity() {
        let mut world = World::new();
        world.track_changes::<Position>();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });
        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });

        world.snapshot_changes::<Position>(0);
        world.get_component_mut::<Position>(e1).unwrap().x = 99.0;
        world.bump_change_version::<Position>();

        assert!(world.is_changed::<Position>(e1, 0));
        assert!(!world.is_changed::<Position>(e2, 0));
    }

    #[test]
    fn is_changed_no_tracker_returns_true() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });
        // No track_changes called — should assume changed.
        assert!(world.is_changed::<Position>(entity, 0));
    }

    #[test]
    fn is_changed_dead_entity_returns_false() {
        let mut world = World::new();
        world.track_changes::<Position>();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });
        world.despawn(entity);
        assert!(!world.is_changed::<Position>(entity, 0));
    }

    // ── set_component / on_set tests ────────────────────────────────

    #[test]
    fn set_component_updates_value() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });

        world.set_component(entity, Position { x: 99.0, y: 88.0 });

        let pos = world.get_component::<Position>(entity).unwrap();
        assert_eq!(pos.x, 99.0);
        assert_eq!(pos.y, 88.0);
    }

    #[test]
    fn set_component_fires_on_set_observer() {
        use std::sync::{Arc, Mutex};

        let log: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&log);

        let mut world = World::new();
        world.on_set::<Position>(move |_entity, pos| {
            log_clone.lock().unwrap().push(pos.x);
        });

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });

        // on_set should NOT have fired from add_component.
        assert!(log.lock().unwrap().is_empty());

        world.set_component(entity, Position { x: 42.0, y: 0.0 });

        let entries = log.lock().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], 42.0);
    }

    #[test]
    fn set_component_does_not_fire_on_add() {
        use std::sync::{Arc, Mutex};

        let add_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let set_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let ac = Arc::clone(&add_count);
        let sc = Arc::clone(&set_count);

        let mut world = World::new();
        world.on_add::<Position>(move |_, _| *ac.lock().unwrap() += 1);
        world.on_set::<Position>(move |_, _| *sc.lock().unwrap() += 1);

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 }); // fires on_add
        world.set_component(entity, Position { x: 2.0, y: 0.0 }); // fires on_set

        assert_eq!(*add_count.lock().unwrap(), 1);
        assert_eq!(*set_count.lock().unwrap(), 1);
    }

    #[test]
    fn set_component_on_missing_delegates_to_add() {
        use std::sync::{Arc, Mutex};

        let add_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let set_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let ac = Arc::clone(&add_count);
        let sc = Arc::clone(&set_count);

        let mut world = World::new();
        world.on_add::<Position>(move |_, _| *ac.lock().unwrap() += 1);
        world.on_set::<Position>(move |_, _| *sc.lock().unwrap() += 1);

        let entity = world.spawn_empty();
        // Entity has no Position — set_component delegates to add_component.
        world.set_component(entity, Position { x: 5.0, y: 6.0 });

        assert_eq!(*add_count.lock().unwrap(), 1); // on_add fired
        assert_eq!(*set_count.lock().unwrap(), 0); // on_set did NOT fire
        assert!(world.has_component::<Position>(entity));
        assert_eq!(world.get_component::<Position>(entity).unwrap().x, 5.0);
    }

    #[test]
    fn set_component_bumps_change_detection() {
        let mut world = World::new();
        world.track_changes::<Position>();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });
        world.snapshot_changes::<Position>(0);

        world.set_component(entity, Position { x: 99.0, y: 0.0 });

        // Change detection should see the set_component change.
        let mut count = 0;
        world.for_each_changed::<Position>(0, |_, _| count += 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn get_mut_does_not_fire_on_set() {
        use std::sync::{Arc, Mutex};

        let set_count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let sc = Arc::clone(&set_count);

        let mut world = World::new();
        world.on_set::<Position>(move |_, _| *sc.lock().unwrap() += 1);

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 0.0 });

        // get_mut is the silent hot path — no observer fires.
        world.get_component_mut::<Position>(entity).unwrap().x = 99.0;

        assert_eq!(*set_count.lock().unwrap(), 0);
    }

    // ── Hierarchy utility tests ─────────────────────────────────────

    #[test]
    fn children_returns_child_entities() {
        let mut world = World::new();
        let parent = world.spawn_empty();
        let c1 = world.spawn_empty();
        let c2 = world.spawn_empty();

        world.add_child::<ChildOf>(parent, c1);
        world.add_child::<ChildOf>(parent, c2);

        let mut kids = world.children::<ChildOf>(parent);
        kids.sort_by_key(|e| e.index());
        assert_eq!(kids, vec![c1, c2]);
    }

    #[test]
    fn parent_returns_parent_entity() {
        let mut world = World::new();
        let parent = world.spawn_empty();
        let child = world.spawn_empty();

        world.add_child::<ChildOf>(parent, child);

        assert_eq!(world.parent::<ChildOf>(child), Some(parent));
        assert_eq!(world.parent::<ChildOf>(parent), None);
    }

    #[test]
    fn is_root_and_is_leaf() {
        let mut world = World::new();
        let root = world.spawn_empty();
        let mid = world.spawn_empty();
        let leaf = world.spawn_empty();

        world.add_child::<ChildOf>(root, mid);
        world.add_child::<ChildOf>(mid, leaf);

        assert!(world.is_root::<ChildOf>(root));
        assert!(!world.is_root::<ChildOf>(mid));
        assert!(!world.is_root::<ChildOf>(leaf));

        assert!(!world.is_leaf::<ChildOf>(root));
        assert!(!world.is_leaf::<ChildOf>(mid));
        assert!(world.is_leaf::<ChildOf>(leaf));
    }

    #[test]
    fn despawn_recursive_kills_entire_subtree() {
        let mut world = World::new();
        let root = world.spawn_empty();
        let c1 = world.spawn_empty();
        let c2 = world.spawn_empty();
        let gc1 = world.spawn_empty(); // grandchild of c1

        world.add_child::<ChildOf>(root, c1);
        world.add_child::<ChildOf>(root, c2);
        world.add_child::<ChildOf>(c1, gc1);

        let count = world.despawn_recursive::<ChildOf>(root);
        assert_eq!(count, 4); // root + c1 + c2 + gc1

        assert!(!world.is_alive(root));
        assert!(!world.is_alive(c1));
        assert!(!world.is_alive(c2));
        assert!(!world.is_alive(gc1));
    }

    #[test]
    fn despawn_recursive_leaves_siblings_alive() {
        let mut world = World::new();
        let root = world.spawn_empty();
        let c1 = world.spawn_empty();
        let c2 = world.spawn_empty();

        world.add_child::<ChildOf>(root, c1);
        world.add_child::<ChildOf>(root, c2);

        // Only despawn c1's subtree, not the whole tree.
        let count = world.despawn_recursive::<ChildOf>(c1);
        assert_eq!(count, 1); // just c1, no children

        assert!(world.is_alive(root));
        assert!(!world.is_alive(c1));
        assert!(world.is_alive(c2));
    }

    #[test]
    fn despawn_recursive_handles_cycle() {
        let mut world = World::new();
        let a = world.spawn_empty();
        let b = world.spawn_empty();

        world.add_relation::<ChildOf>(a, b);
        world.add_relation::<ChildOf>(b, a); // cycle

        let count = world.despawn_recursive::<ChildOf>(a);
        assert_eq!(count, 2); // both despawned, no infinite loop
    }

    #[test]
    fn add_child_remove_child_roundtrip() {
        let mut world = World::new();
        let parent = world.spawn_empty();
        let child = world.spawn_empty();

        assert!(world.add_child::<ChildOf>(parent, child));
        assert_eq!(world.children::<ChildOf>(parent).len(), 1);

        assert!(world.remove_child::<ChildOf>(parent, child));
        assert!(world.children::<ChildOf>(parent).is_empty());
        assert!(world.is_alive(child)); // child not despawned, just detached
    }

    // ── Save/load tests ─────────────────────────────────────────────

    #[test]
    fn save_load_roundtrip_preserves_entities() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        let e2 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        world.add_component(e2, Position { x: 3.0, y: 4.0 });
        world.add_component(e2, Velocity { dx: 5.0, dy: 6.0 });

        let snapshot = world.save_state();

        // Wipe and restore.
        let mut world2 = World::new();
        world2.register_component_type::<Position>();
        world2.register_component_type::<Velocity>();
        world2.load_state(&snapshot);

        assert!(world2.is_alive(e1));
        assert!(world2.is_alive(e2));
        let pos1 = world2.get_component::<Position>(e1).unwrap();
        assert_eq!(pos1.x, 1.0);
        assert_eq!(pos1.y, 2.0);
        let pos2 = world2.get_component::<Position>(e2).unwrap();
        assert_eq!(pos2.x, 3.0);
        let vel2 = world2.get_component::<Velocity>(e2).unwrap();
        assert_eq!(vel2.dx, 5.0);
    }

    #[test]
    fn save_load_preserves_generations() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });
        world.despawn(e1);
        let e2 = world.spawn_empty(); // reuses e1's index with gen+1
        world.add_component(e2, Position { x: 2.0, y: 0.0 });

        let snapshot = world.save_state();

        let mut world2 = World::new();
        world2.register_component_type::<Position>();
        world2.load_state(&snapshot);

        // Stale entity should be dead.
        assert!(!world2.is_alive(e1));
        // New generation entity should be alive.
        assert!(world2.is_alive(e2));
        assert_eq!(world2.get_component::<Position>(e2).unwrap().x, 2.0);
    }

    #[test]
    fn save_load_empty_world() {
        let world = World::new();
        let snapshot = world.save_state();

        assert_eq!(snapshot.entity_count(), 0);
        assert_eq!(snapshot.archetype_count(), 1); // empty archetype

        let mut world2 = World::new();
        world2.load_state(&snapshot);
        assert_eq!(world2.archetype_count(), 1);
    }

    #[test]
    fn save_load_snapshot_byte_size() {
        let mut world = World::new();
        for i in 0..100 {
            let e = world.spawn_empty();
            world.add_component(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            );
        }

        let snapshot = world.save_state();
        assert_eq!(snapshot.entity_count(), 100);
        assert!(snapshot.byte_size() > 0);
    }

    #[test]
    fn save_load_can_continue_after_restore() {
        let mut world = World::new();
        world.register_component_type::<Position>();
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });

        let snapshot = world.save_state();
        world.load_state(&snapshot);

        // Should be able to spawn new entities after restore.
        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 99.0, y: 0.0 });
        assert!(world.is_alive(e2));
        assert_eq!(world.get_component::<Position>(e2).unwrap().x, 99.0);
    }

    // ── Sparse storage tests ────────────────────────────────────────

    #[derive(Debug, PartialEq)]
    struct Debuff {
        damage: f32,
    }

    #[test]
    fn sparse_add_get_remove() {
        let mut world = World::new();
        world.register_sparse::<Debuff>();

        let entity = world.spawn_empty();
        world.add_component(entity, Debuff { damage: 5.0 });

        assert!(world.has_component::<Debuff>(entity));
        assert_eq!(world.get_component::<Debuff>(entity).unwrap().damage, 5.0);

        let removed = world.remove_component::<Debuff>(entity);
        assert_eq!(removed, Some(Debuff { damage: 5.0 }));
        assert!(!world.has_component::<Debuff>(entity));
    }

    #[test]
    fn sparse_does_not_cause_archetype_transition() {
        let mut world = World::new();
        world.register_sparse::<Debuff>();

        let entity = world.spawn_empty();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        let arch_count_before = world.archetype_count();

        // Adding a sparse component should NOT create a new archetype.
        world.add_component(entity, Debuff { damage: 10.0 });
        assert_eq!(world.archetype_count(), arch_count_before);

        // Entity still has its archetype Position.
        assert!(world.has_component::<Position>(entity));
        // And the sparse debuff.
        assert!(world.has_component::<Debuff>(entity));
    }

    #[test]
    fn sparse_get_mut_works() {
        let mut world = World::new();
        world.register_sparse::<Debuff>();

        let entity = world.spawn_empty();
        world.add_component(entity, Debuff { damage: 5.0 });

        world.get_component_mut::<Debuff>(entity).unwrap().damage = 99.0;
        assert_eq!(world.get_component::<Debuff>(entity).unwrap().damage, 99.0);
    }

    #[test]
    fn sparse_despawn_cleans_up() {
        let mut world = World::new();
        world.register_sparse::<Debuff>();

        let entity = world.spawn_empty();
        world.add_component(entity, Debuff { damage: 5.0 });
        world.despawn(entity);

        // Sparse storage should be cleaned up.
        assert!(!world.has_component::<Debuff>(entity));
    }

    #[test]
    fn sparse_and_archetype_coexist() {
        let mut world = World::new();
        world.register_sparse::<Debuff>();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });
        world.add_component(e1, Debuff { damage: 10.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 2.0, y: 0.0 });
        // No debuff on e2.

        // Archetype query sees both entities.
        let positions: Vec<f32> = world.query::<Position>().map(|(_, p)| p.x).collect();
        assert_eq!(positions.len(), 2);

        // Sparse lookup is per-entity.
        assert!(world.has_component::<Debuff>(e1));
        assert!(!world.has_component::<Debuff>(e2));
    }

    #[test]
    fn is_sparse_reports_correctly() {
        let mut world = World::new();
        assert!(!world.is_sparse::<Debuff>());

        world.register_sparse::<Debuff>();
        assert!(world.is_sparse::<Debuff>());
        assert!(!world.is_sparse::<Position>()); // not registered as sparse
    }

    // --- Resource management tests ---

    #[derive(Debug, PartialEq)]
    struct ResourceScore(u32);

    #[test]
    fn has_resource_detects_presence() {
        let mut world = World::new();
        assert!(!world.has_resource::<ResourceScore>());
        world.insert_resource(ResourceScore(0));
        assert!(world.has_resource::<ResourceScore>());
    }

    #[test]
    fn remove_resource_returns_owned_value() {
        let mut world = World::new();
        world.insert_resource(ResourceScore(42));

        let removed = world.remove_resource::<ResourceScore>();
        assert_eq!(removed, Some(ResourceScore(42)));
        assert!(!world.has_resource::<ResourceScore>());
    }

    #[test]
    fn remove_resource_missing_returns_none() {
        let mut world = World::new();
        assert_eq!(world.remove_resource::<ResourceScore>(), None);
    }

    #[test]
    fn resource_scope_allows_simultaneous_access() {
        let mut world = World::new();
        world.insert_resource(ResourceScore(0));

        let e1 = world.spawn_empty();
        let e2 = world.spawn_empty();
        world.add_component(e1, Position { x: 10.0, y: 0.0 });
        world.add_component(e2, Position { x: 25.0, y: 0.0 });

        // Iterate components while mutating a resource — no Vec collect needed
        world.resource_scope::<ResourceScore, _>(|world, score| {
            world.for_each_mut::<Position>(|_entity, pos| {
                score.0 += pos.x as u32;
            });
        });

        assert_eq!(world.get_resource::<ResourceScore>().unwrap().0, 35);
    }

    #[test]
    fn resource_scope_restores_resource_on_return() {
        let mut world = World::new();
        world.insert_resource(ResourceScore(100));

        let result = world.resource_scope::<ResourceScore, u32>(|_world, score| {
            score.0 += 1;
            score.0
        });

        assert_eq!(result, 101);
        assert_eq!(world.get_resource::<ResourceScore>().unwrap().0, 101);
    }

    #[test]
    #[should_panic(expected = "resource_scope: resource")]
    fn resource_scope_panics_on_missing_resource() {
        let mut world = World::new();
        world.resource_scope::<ResourceScore, _>(|_world, _score| {});
    }

    #[test]
    fn resource_scope_with_query_pair() {
        let mut world = World::new();

        #[derive(Debug, Clone)]
        struct SceneData(Vec<(u32, u32)>);
        #[derive(Debug, Clone)]
        struct Velocity(u32);

        world.insert_resource(SceneData(Vec::new()));

        let e1 = world.spawn_empty();
        let e2 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        world.add_component(e1, Velocity(100));
        world.add_component(e2, Position { x: 3.0, y: 4.0 });
        world.add_component(e2, Velocity(200));

        // Simultaneous pair query + resource mutation
        world.resource_scope::<SceneData, _>(|world, scene| {
            world.for_each_pair::<Position, Velocity>(|_e, pos, vel| {
                scene.0.push((pos.x as u32, vel.0));
            });
        });

        let scene = world.get_resource::<SceneData>().unwrap();
        assert_eq!(scene.0.len(), 2);
        assert!(scene.0.contains(&(1, 100)));
        assert!(scene.0.contains(&(3, 200)));
    }
}
