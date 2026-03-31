//! Query types for system parameter extraction.
//!
//! [`WorldQuery`] defines how component tuples are fetched from archetypes.
//! [`Query`] is the system parameter that iterates matching archetypes.

use crate::system::SystemAccess;
use crate::unsafe_cell::UnsafeWorldCell;
use crate::param::SystemParam;
use crate::{Archetype, Component, Entity, World};
use std::any::TypeId;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Fetch: per-component column access
// ---------------------------------------------------------------------------

/// How a single component reference is fetched from an archetype.
///
/// # Safety
///
/// Implementations must ensure that `fetch_slice` returns a pointer to valid
/// data for the archetype's entity count, and that the mutability matches
/// the declared access.
pub unsafe trait Fetch: Sized {
    /// The item yielded per entity row.
    type Item<'a>;

    /// Declare component access for the scheduler.
    fn access(access: &mut SystemAccess);

    /// Whether this fetch requires mutable archetype access.
    const MUTABLE: bool;

    /// Check if the archetype contains the required component.
    fn matches(archetype: &Archetype) -> bool;

    /// Get a raw pointer to the component column data.
    ///
    /// # Safety
    /// Caller must ensure the archetype matches (call `matches` first).
    unsafe fn fetch_ptr(archetype: *mut Archetype) -> *mut u8;

    /// The stride (size of one element) for pointer arithmetic.
    fn stride() -> usize;

    /// Convert a raw pointer at a given index to the item type.
    ///
    /// # Safety
    /// Pointer must be valid for the given index within the column.
    unsafe fn read<'a>(ptr: *mut u8, index: usize) -> Self::Item<'a>;
}

// --- &T: immutable component access ---

/// Marker type for immutable component fetch.
pub struct FetchRead<T: Component>(PhantomData<T>);

unsafe impl<T: Component> Fetch for FetchRead<T> {
    type Item<'a> = &'a T;
    const MUTABLE: bool = false;

    fn access(access: &mut SystemAccess) {
        access.reads_component::<T>();
    }

    fn matches(archetype: &Archetype) -> bool {
        archetype.component_set().contains::<T>()
    }

    unsafe fn fetch_ptr(archetype: *mut Archetype) -> *mut u8 {
        let arch = unsafe { &*archetype };
        arch.components::<T>()
            .expect("archetype matched but column missing")
            .as_ptr() as *mut u8
    }

    fn stride() -> usize {
        std::mem::size_of::<T>()
    }

    unsafe fn read<'a>(ptr: *mut u8, index: usize) -> &'a T {
        unsafe { &*ptr.cast::<T>().add(index) }
    }
}

// --- &mut T: mutable component access ---

/// Marker type for mutable component fetch.
pub struct FetchWrite<T: Component>(PhantomData<T>);

unsafe impl<T: Component> Fetch for FetchWrite<T> {
    type Item<'a> = &'a mut T;
    const MUTABLE: bool = true;

    fn access(access: &mut SystemAccess) {
        access.writes_component::<T>();
    }

    fn matches(archetype: &Archetype) -> bool {
        archetype.component_set().contains::<T>()
    }

    unsafe fn fetch_ptr(archetype: *mut Archetype) -> *mut u8 {
        let arch = unsafe { &mut *archetype };
        arch.components_mut::<T>()
            .expect("archetype matched but column missing")
            .as_mut_ptr() as *mut u8
    }

    fn stride() -> usize {
        std::mem::size_of::<T>()
    }

    unsafe fn read<'a>(ptr: *mut u8, index: usize) -> &'a mut T {
        unsafe { &mut *ptr.cast::<T>().add(index) }
    }
}

// --- Option<&T>: optional immutable access ---

/// Marker type for optional immutable component fetch.
pub struct FetchOptionalRead<T: Component>(PhantomData<T>);

unsafe impl<T: Component> Fetch for FetchOptionalRead<T> {
    type Item<'a> = Option<&'a T>;
    const MUTABLE: bool = false;

    fn access(access: &mut SystemAccess) {
        access.reads_component::<T>();
    }

    fn matches(_archetype: &Archetype) -> bool {
        true // Optional always matches — column may or may not exist
    }

    unsafe fn fetch_ptr(archetype: *mut Archetype) -> *mut u8 {
        let arch = unsafe { &*archetype };
        match arch.components::<T>() {
            Some(slice) => slice.as_ptr() as *mut u8,
            None => std::ptr::null_mut(),
        }
    }

    fn stride() -> usize {
        std::mem::size_of::<T>()
    }

    unsafe fn read<'a>(ptr: *mut u8, index: usize) -> Option<&'a T> {
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { &*ptr.cast::<T>().add(index) })
        }
    }
}

// ---------------------------------------------------------------------------
// WorldQuery: tuple of Fetch items
// ---------------------------------------------------------------------------

/// A tuple of component fetches that defines a query's access pattern.
///
/// # Safety
///
/// Implementors must correctly declare all component access and ensure
/// no aliasing between mutable and immutable fetches within the same tuple.
pub unsafe trait WorldQuery: Sized {
    /// The per-archetype fetch state (column pointers).
    type FetchState;

    /// The item yielded per entity.
    type Item<'a>;

    /// Whether any fetch in this query is mutable.
    const MUTABLE: bool;

    /// Declare component access.
    fn access(access: &mut SystemAccess);

    /// Check if an archetype matches all required components.
    fn matches(archetype: &Archetype) -> bool;

    /// Initialize fetch state (column pointers) for an archetype.
    ///
    /// # Safety
    /// Archetype must match (call `matches` first).
    unsafe fn init_fetch(archetype: *mut Archetype) -> Self::FetchState;

    /// Fetch one row from the archetype using initialized state.
    ///
    /// # Safety
    /// Index must be valid for the archetype's entity count.
    unsafe fn fetch<'a>(state: &Self::FetchState, index: usize) -> Self::Item<'a>;
}

// --- Single-fetch WorldQuery impls ---

unsafe impl<T: Component> WorldQuery for &T {
    type FetchState = *mut u8;
    type Item<'a> = &'a T;
    const MUTABLE: bool = false;

    fn access(access: &mut SystemAccess) {
        FetchRead::<T>::access(access);
    }

    fn matches(archetype: &Archetype) -> bool {
        FetchRead::<T>::matches(archetype)
    }

    unsafe fn init_fetch(archetype: *mut Archetype) -> *mut u8 {
        unsafe { FetchRead::<T>::fetch_ptr(archetype) }
    }

    unsafe fn fetch<'a>(state: &*mut u8, index: usize) -> &'a T {
        unsafe { FetchRead::<T>::read(*state, index) }
    }
}

unsafe impl<T: Component> WorldQuery for &mut T {
    type FetchState = *mut u8;
    type Item<'a> = &'a mut T;
    const MUTABLE: bool = true;

    fn access(access: &mut SystemAccess) {
        FetchWrite::<T>::access(access);
    }

    fn matches(archetype: &Archetype) -> bool {
        FetchWrite::<T>::matches(archetype)
    }

    unsafe fn init_fetch(archetype: *mut Archetype) -> *mut u8 {
        unsafe { FetchWrite::<T>::fetch_ptr(archetype) }
    }

    unsafe fn fetch<'a>(state: &*mut u8, index: usize) -> &'a mut T {
        unsafe { FetchWrite::<T>::read(*state, index) }
    }
}

unsafe impl<T: Component> WorldQuery for Option<&T> {
    type FetchState = *mut u8;
    type Item<'a> = Option<&'a T>;
    const MUTABLE: bool = false;

    fn access(access: &mut SystemAccess) {
        FetchOptionalRead::<T>::access(access);
    }

    fn matches(archetype: &Archetype) -> bool {
        FetchOptionalRead::<T>::matches(archetype)
    }

    unsafe fn init_fetch(archetype: *mut Archetype) -> *mut u8 {
        unsafe { FetchOptionalRead::<T>::fetch_ptr(archetype) }
    }

    unsafe fn fetch<'a>(state: &*mut u8, index: usize) -> Option<&'a T> {
        unsafe { FetchOptionalRead::<T>::read(*state, index) }
    }
}

// --- Tuple WorldQuery impls (2-8 components) via macro ---

macro_rules! impl_world_query_tuple {
    ($($idx:tt: $Q:ident),+) => {
        unsafe impl<$($Q: WorldQuery),+> WorldQuery for ($($Q,)+) {
            type FetchState = ($($Q::FetchState,)+);
            type Item<'a> = ($($Q::Item<'a>,)+);
            const MUTABLE: bool = $($Q::MUTABLE ||)+ false;

            fn access(access: &mut SystemAccess) {
                $($Q::access(access);)+
            }

            fn matches(archetype: &Archetype) -> bool {
                $($Q::matches(archetype) &&)+ true
            }

            unsafe fn init_fetch(archetype: *mut Archetype) -> Self::FetchState {
                ($(unsafe { $Q::init_fetch(archetype) },)+)
            }

            unsafe fn fetch<'a>(state: &Self::FetchState, index: usize) -> Self::Item<'a> {
                ($(unsafe { $Q::fetch(&state.$idx, index) },)+)
            }
        }
    };
}

impl_world_query_tuple!(0: A);
impl_world_query_tuple!(0: A, 1: B);
impl_world_query_tuple!(0: A, 1: B, 2: C);
impl_world_query_tuple!(0: A, 1: B, 2: C, 3: D);
impl_world_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_world_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_world_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_world_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);

// ---------------------------------------------------------------------------
// Query<Q>: the system parameter
// ---------------------------------------------------------------------------

/// System parameter for iterating entities matching a component query.
///
/// # Example
///
/// ```ignore
/// #[system]
/// fn physics(query: Query<(&Position, &mut Velocity)>) {
///     for (entity, (pos, vel)) in &mut query {
///         vel.x += pos.x * 0.01;
///     }
/// }
/// ```
pub struct Query<'w, Q: WorldQuery> {
    archetypes: *mut Archetype,
    archetype_count: usize,
    _marker: PhantomData<(&'w (), Q)>,
}

/// Iterator over Query results.
pub struct QueryIter<'w, Q: WorldQuery> {
    archetypes: *mut Archetype,
    archetype_count: usize,
    current_archetype: usize,
    current_row: usize,
    current_len: usize,
    current_entities: *const Entity,
    current_fetch: Option<Q::FetchState>,
    _marker: PhantomData<&'w ()>,
}

impl<'w, Q: WorldQuery> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try current archetype
            if self.current_row < self.current_len {
                let index = self.current_row;
                self.current_row += 1;
                let entity = unsafe { *self.current_entities.add(index) };
                let item = unsafe { Q::fetch(self.current_fetch.as_ref().unwrap(), index) };
                return Some((entity, item));
            }

            // Move to next matching archetype
            self.current_archetype += 1;
            while self.current_archetype < self.archetype_count {
                let arch_ptr = unsafe { self.archetypes.add(self.current_archetype) };
                let arch = unsafe { &*arch_ptr };

                if Q::matches(arch) && !arch.is_empty() {
                    self.current_len = arch.len();
                    self.current_row = 0;
                    self.current_entities = arch.entities().as_ptr();
                    self.current_fetch = Some(unsafe { Q::init_fetch(arch_ptr) });
                    break;
                }

                self.current_archetype += 1;
            }

            if self.current_archetype >= self.archetype_count {
                return None;
            }
        }
    }
}

impl<'w, Q: WorldQuery> IntoIterator for &'w Query<'w, Q> {
    type Item = (Entity, Q::Item<'w>);
    type IntoIter = QueryIter<'w, Q>;

    fn into_iter(self) -> Self::IntoIter {
        let mut iter = QueryIter {
            archetypes: self.archetypes,
            archetype_count: self.archetype_count,
            current_archetype: 0,
            current_row: 0,
            current_len: 0,
            current_entities: std::ptr::null(),
            current_fetch: None,
            _marker: PhantomData,
        };

        // Find first matching archetype
        while iter.current_archetype < iter.archetype_count {
            let arch_ptr = unsafe { iter.archetypes.add(iter.current_archetype) };
            let arch = unsafe { &*arch_ptr };

            if Q::matches(arch) && !arch.is_empty() {
                iter.current_len = arch.len();
                iter.current_entities = arch.entities().as_ptr();
                iter.current_fetch = Some(unsafe { Q::init_fetch(arch_ptr) });
                break;
            }

            iter.current_archetype += 1;
        }

        iter
    }
}

// Also support `&mut query` for the same iteration (Query doesn't change, items may be mut)
impl<'w, Q: WorldQuery> IntoIterator for &'w mut Query<'w, Q> {
    type Item = (Entity, Q::Item<'w>);
    type IntoIter = QueryIter<'w, Q>;

    fn into_iter(self) -> Self::IntoIter {
        // Delegate to the shared implementation
        let query_ref: &'w Query<'w, Q> = self;
        query_ref.into_iter()
    }
}

// --- Query as SystemParam ---

/// Cached archetype match indices for a query (recomputed on structural changes).
pub struct QueryState {
    /// Archetype indices that matched last time (optimization for future use).
    _cached: Vec<usize>,
}

impl SystemParam for Query<'_, &()> {
    // This is a placeholder — the actual impl is done per-Q via the macro.
    // The proc macro generates the correct SystemParam extraction for each
    // concrete Query<Q> type it encounters.
    type Item<'w> = ();
    type State = ();
    fn access(_access: &mut SystemAccess) {}
    fn init(_world: &mut World) -> () {}
    unsafe fn get<'w>(_cell: UnsafeWorldCell<'w>, _state: &'w mut ()) {}
}

// The real Query SystemParam extraction happens in the generated code:
// The proc macro sees `Query<(&A, &mut B)>` and generates:
//   1. SystemAccess declarations from WorldQuery::access
//   2. UnsafeWorldCell::archetypes[_mut]() to get the slice
//   3. Constructs Query { archetypes, archetype_count } directly

/// Helper: construct a Query from an UnsafeWorldCell.
///
/// Called by the generated system runner code.
///
/// # Safety
///
/// The caller must ensure the WorldQuery's access is compatible with
/// other parameters in the system (validated by SystemAccess).
pub unsafe fn query_from_cell<'w, Q: WorldQuery>(
    cell: UnsafeWorldCell<'w>,
) -> Query<'w, Q> {
    if Q::MUTABLE {
        let archetypes = unsafe { cell.archetypes_mut() };
        Query {
            archetypes: archetypes.as_mut_ptr(),
            archetype_count: archetypes.len(),
            _marker: PhantomData,
        }
    } else {
        let archetypes = unsafe { cell.archetypes() };
        Query {
            archetypes: archetypes.as_ptr() as *mut Archetype,
            archetype_count: archetypes.len(),
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    struct Position { x: f32, y: f32 }
    struct Velocity { dx: f32, dy: f32 }

    #[test]
    fn query_iter_single_component() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        let e2 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        world.add_component(e2, Position { x: 3.0, y: 4.0 });

        let query: Query<'_, &Position> = unsafe {
            let cell = UnsafeWorldCell::new(&mut world);
            query_from_cell(cell)
        };

        let results: Vec<_> = (&query).into_iter().map(|(_, p)| p.x).collect();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&1.0));
        assert!(results.contains(&3.0));
    }

    #[test]
    fn query_iter_pair() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        world.add_component(e1, Velocity { dx: 10.0, dy: 20.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 3.0, y: 4.0 });
        // e2 has no Velocity — should not appear in pair query

        let query: Query<'_, (&Position, &Velocity)> = unsafe {
            let cell = UnsafeWorldCell::new(&mut world);
            query_from_cell(cell)
        };

        let results: Vec<_> = (&query).into_iter().collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1.0.x, 1.0);
        assert_eq!(results[0].1.1.dx, 10.0);
    }

    #[test]
    fn query_iter_mutable() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        world.add_component(e1, Velocity { dx: 0.5, dy: 0.0 });

        {
            let query: Query<'_, (&Position, &mut Velocity)> = unsafe {
                let cell = UnsafeWorldCell::new(&mut world);
                query_from_cell(cell)
            };

            for (_entity, (pos, vel)) in &query {
                vel.dx += pos.x;
            }
        }

        let vel = world.get_component::<Velocity>(e1).unwrap();
        assert_eq!(vel.dx, 1.5); // 0.5 + 1.0
    }
}
