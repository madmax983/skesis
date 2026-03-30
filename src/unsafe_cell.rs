//! Raw access to World internals for system parameter extraction.
//!
//! `UnsafeWorldCell` is the foundation of the system parameter machinery.
//! It wraps a `*mut World` and provides unchecked access to disjoint fields
//! (resources, archetypes, events, commands). All methods are `unsafe` —
//! the caller must guarantee that no two calls produce aliasing `&mut` references.
//!
//! Safety is enforced at a higher level:
//! - The scheduler validates `SystemAccess` for parallel batch conflicts
//! - The `#[system]` proc macro generates code accessing non-overlapping fields
//! - Within a single system, parameter types access different World fields

use crate::command::CommandStore;
use crate::event::EventStore;
use crate::resource::ResourceStore;
use crate::{Archetype, Entity, World};
use std::marker::PhantomData;

/// Raw pointer access to `World` internals.
///
/// This is a `Copy` type — cheap to pass around. The lifetime `'w` ties
/// all derived references to the original `&mut World` borrow, preventing
/// use-after-free.
///
/// # Safety
///
/// The caller must guarantee:
/// 1. The World pointer is valid for the lifetime `'w`
/// 2. No two method calls produce aliasing `&mut` references
/// 3. Access patterns match the `SystemAccess` declared for the system
#[derive(Copy, Clone)]
pub struct UnsafeWorldCell<'w> {
    world: *mut World,
    _marker: PhantomData<&'w mut World>,
}

impl<'w> UnsafeWorldCell<'w> {
    /// Create a new cell from a mutable World reference.
    ///
    /// # Safety
    ///
    /// The returned cell must not outlive the World reference, and the caller
    /// must ensure non-aliasing access through the cell's methods.
    pub(crate) unsafe fn new(world: &'w mut World) -> Self {
        Self {
            world: world as *mut World,
            _marker: PhantomData,
        }
    }

    // ---- Resources ----

    /// Get an immutable resource reference.
    ///
    /// # Safety
    /// No concurrent `get_resource_mut::<T>()` call for the same `T`.
    pub unsafe fn get_resource<T: 'static + Send + Sync>(&self) -> Option<&'w T> {
        unsafe { (*self.world).resources_raw().get::<T>() }
    }

    /// Get a mutable resource reference.
    ///
    /// # Safety
    /// No concurrent `get_resource::<T>()` or `get_resource_mut::<T>()` for the same `T`.
    pub unsafe fn get_resource_mut<T: 'static + Send + Sync>(&self) -> Option<&'w mut T> {
        unsafe { (*self.world).resources_raw_mut().get_mut::<T>() }
    }

    // ---- Archetypes ----

    /// Get immutable archetype slice for read-only queries.
    ///
    /// # Safety
    /// No concurrent `archetypes_mut()` call.
    pub unsafe fn archetypes(&self) -> &'w [Archetype] {
        unsafe { (*self.world).archetypes_raw() }
    }

    /// Get mutable archetype slice for queries with `&mut` components.
    ///
    /// # Safety
    /// No concurrent `archetypes()` or `archetypes_mut()` call.
    pub unsafe fn archetypes_mut(&self) -> &'w mut [Archetype] {
        unsafe { (*self.world).archetypes_raw_mut() }
    }

    // ---- Events ----

    /// Get immutable event store reference.
    ///
    /// # Safety
    /// No concurrent `events_mut()` call.
    pub unsafe fn events(&self) -> &'w EventStore {
        unsafe { (*self.world).events_raw() }
    }

    /// Get mutable event store reference.
    ///
    /// # Safety
    /// No concurrent `events()` or `events_mut()` call.
    pub unsafe fn events_mut(&self) -> &'w mut EventStore {
        unsafe { (*self.world).events_raw_mut() }
    }

    // ---- Commands ----

    /// Get mutable command store reference.
    ///
    /// # Safety
    /// No concurrent `commands_mut()` call.
    pub unsafe fn commands_mut(&self) -> &'w mut CommandStore {
        unsafe { (*self.world).commands_raw_mut() }
    }

    // ---- Full world (for Commands that need structural changes) ----

    /// Get a mutable reference to the entire World.
    ///
    /// # Safety
    /// No other references to any World field may be live.
    /// Only valid for exclusive systems.
    pub unsafe fn world_mut(&self) -> &'w mut World {
        unsafe { &mut *self.world }
    }
}
