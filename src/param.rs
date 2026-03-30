//! System parameter extraction.
//!
//! The [`SystemParam`] trait defines how typed parameters are extracted from
//! [`UnsafeWorldCell`] for system execution. Each parameter type declares its
//! access pattern (for scheduler conflict detection) and provides a `get` method
//! that extracts a live reference from the World's internals.
//!
//! # Supported Parameters
//!
//! | Type | Access | Description |
//! |------|--------|-------------|
//! | `Res<T>` | reads resource T | Immutable resource reference |
//! | `ResMut<T>` | writes resource T | Mutable resource reference |
//! | `Query<Q>` | per-component | Archetype iterator |
//! | `Commands` | deferred | Deferred structural changes |
//! | `Local<T>` | private | Per-system persistent state |
//! | `EventReader<E>` | reads events | Event consumption iterator |
//! | `EventWriter<E>` | writes events | Event emission handle |

use crate::command::CommandStore;
use crate::system::SystemAccess;
use crate::unsafe_cell::UnsafeWorldCell;
use crate::World;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// Trait for types that can be extracted as system parameters.
///
/// Implementors declare what they access (for conflict detection) and how to
/// extract a live reference from [`UnsafeWorldCell`].
///
/// # Safety
///
/// `get()` is `unsafe` because it creates references from raw pointers.
/// The caller (the `#[system]` macro runner) must ensure that all parameters
/// in a system have non-overlapping access, validated by [`SystemAccess`].
/// Trait for types that can be extracted as system parameters.
///
/// Uses a GAT (`Item<'w>`) to express the lifetime relationship between the
/// World borrow and the extracted parameter. The `#[system]` macro calls
/// `get()` with the appropriate lifetime from the `UnsafeWorldCell`.
///
/// # Safety
///
/// `get()` is `unsafe` because it creates references from raw pointers.
/// The caller must ensure non-overlapping access, validated by [`SystemAccess`].
pub trait SystemParam {
    /// The concrete parameter type with a lifetime from the World borrow.
    type Item<'w>;

    /// Persistent per-system state (e.g., `Local<T>` storage, query caches).
    type State: Send + Sync + 'static;

    /// Declare what this parameter reads/writes (called once at registration).
    fn access(access: &mut SystemAccess);

    /// Initialize persistent state (called once when the system is first added).
    fn init(world: &mut World) -> Self::State;

    /// Extract the parameter from the world (called once per system execution).
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other live references alias the data
    /// this parameter accesses.
    unsafe fn get<'w>(cell: UnsafeWorldCell<'w>, state: &'w mut Self::State) -> Self::Item<'w>;
}

// ---------------------------------------------------------------------------
// Res<T> — immutable resource access
// ---------------------------------------------------------------------------

/// Immutable reference to a resource stored in the World.
///
/// Derefs to `&T` for ergonomic access.
pub struct Res<'w, T: 'static + Send + Sync> {
    value: &'w T,
}

impl<T: 'static + Send + Sync> Deref for Res<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: 'static + Send + Sync> SystemParam for Res<'_, T> {
    type Item<'w> = Res<'w, T>;
    type State = ();

    fn access(access: &mut SystemAccess) {
        access.reads_resource::<T>();
    }

    fn init(_world: &mut World) -> Self::State {}

    unsafe fn get<'w>(cell: UnsafeWorldCell<'w>, _state: &'w mut Self::State) -> Res<'w, T> {
        let value = unsafe { cell.get_resource::<T>() }
            .unwrap_or_else(|| panic!("Res<{}>: resource not found", std::any::type_name::<T>()));
        Res { value }
    }
}

// ---------------------------------------------------------------------------
// ResMut<T> — mutable resource access
// ---------------------------------------------------------------------------

/// Mutable reference to a resource stored in the World.
///
/// Derefs to `&mut T` for ergonomic access.
pub struct ResMut<'w, T: 'static + Send + Sync> {
    value: &'w mut T,
}

impl<T: 'static + Send + Sync> Deref for ResMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: 'static + Send + Sync> DerefMut for ResMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

impl<T: 'static + Send + Sync> SystemParam for ResMut<'_, T> {
    type Item<'w> = ResMut<'w, T>;
    type State = ();

    fn access(access: &mut SystemAccess) {
        access.writes_resource::<T>();
    }

    fn init(_world: &mut World) -> Self::State {}

    unsafe fn get<'w>(cell: UnsafeWorldCell<'w>, _state: &'w mut Self::State) -> ResMut<'w, T> {
        let value = unsafe { cell.get_resource_mut::<T>() }
            .unwrap_or_else(|| {
                panic!("ResMut<{}>: resource not found", std::any::type_name::<T>())
            });
        ResMut { value }
    }
}

// ---------------------------------------------------------------------------
// Local<T> — per-system persistent state
// ---------------------------------------------------------------------------

/// Per-system persistent state that survives between frames.
///
/// Each system gets its own instance of `T`. The value is initialized via
/// `Default::default()` when the system is first registered.
pub struct Local<'w, T: Default + Send + Sync + 'static> {
    value: &'w mut T,
}

impl<T: Default + Send + Sync + 'static> Deref for Local<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: Default + Send + Sync + 'static> DerefMut for Local<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value
    }
}

impl<T: Default + Send + Sync + 'static> SystemParam for Local<'_, T> {
    type Item<'w> = Local<'w, T>;
    type State = T;

    fn access(_access: &mut SystemAccess) {
        // Local state is private — no conflicts possible.
    }

    fn init(_world: &mut World) -> Self::State {
        T::default()
    }

    unsafe fn get<'w>(
        _cell: UnsafeWorldCell<'w>,
        state: &'w mut Self::State,
    ) -> Local<'w, T> {
        Local { value: state }
    }
}

// ---------------------------------------------------------------------------
// Commands — deferred structural changes
// ---------------------------------------------------------------------------

/// Handle for recording deferred commands (spawn, despawn, add/remove component).
///
/// Commands are buffered during system execution and applied at the end of the
/// stage, preserving deterministic ordering.
pub struct Commands<'w> {
    commands: &'w mut CommandStore,
}

impl Commands<'_> {
    /// Record a deferred command.
    pub fn add(&mut self, command: impl FnOnce(&mut World) + Send + Sync + 'static) {
        self.commands.defer(Box::new(command));
    }
}

impl SystemParam for Commands<'_> {
    type Item<'w> = Commands<'w>;
    type State = ();

    fn access(_access: &mut SystemAccess) {
        // Commands are deferred — no immediate access conflicts.
    }

    fn init(_world: &mut World) -> Self::State {}

    unsafe fn get<'w>(cell: UnsafeWorldCell<'w>, _state: &'w mut Self::State) -> Commands<'w> {
        let commands = unsafe { cell.commands_mut() };
        Commands { commands }
    }
}

// ---------------------------------------------------------------------------
// EventReader<E> — consume events
// ---------------------------------------------------------------------------

/// Iterator over events of type `E` emitted this frame.
pub struct EventReader<'w, E: 'static + Send + Sync> {
    events: &'w crate::EventStore,
    _marker: PhantomData<E>,
}

impl<E: 'static + Send + Sync> EventReader<'_, E> {
    /// Iterate over all events of type E.
    pub fn iter(&self) -> impl Iterator<Item = &E> {
        self.events.iter::<E>()
    }
}

impl<E: 'static + Send + Sync> SystemParam for EventReader<'_, E> {
    type Item<'w> = EventReader<'w, E>;
    type State = ();

    fn access(access: &mut SystemAccess) {
        access.reads_resource::<crate::EventStore>();
    }

    fn init(_world: &mut World) -> Self::State {}

    unsafe fn get<'w>(
        cell: UnsafeWorldCell<'w>,
        _state: &'w mut Self::State,
    ) -> EventReader<'w, E> {
        let events = unsafe { cell.events() };
        EventReader {
            events,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// EventWriter<E> — emit events
// ---------------------------------------------------------------------------

/// Handle for emitting events of type `E`.
pub struct EventWriter<'w, E: 'static + Send + Sync> {
    events: &'w mut crate::EventStore,
    _marker: PhantomData<E>,
}

impl<E: 'static + Send + Sync> EventWriter<'_, E> {
    /// Emit an event.
    pub fn send(&mut self, event: E) {
        self.events.emit(event);
    }
}

impl<E: 'static + Send + Sync> SystemParam for EventWriter<'_, E> {
    type Item<'w> = EventWriter<'w, E>;
    type State = ();

    fn access(access: &mut SystemAccess) {
        access.writes_resource::<crate::EventStore>();
    }

    fn init(_world: &mut World) -> Self::State {}

    unsafe fn get<'w>(
        cell: UnsafeWorldCell<'w>,
        _state: &'w mut Self::State,
    ) -> EventWriter<'w, E> {
        let events = unsafe { cell.events_mut() };
        EventWriter {
            events,
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// SystemParam for tuples (so the macro can extract multiple params at once)
// ---------------------------------------------------------------------------

/// Stores per-system state for all parameters as a type-erased box.
pub struct SystemState {
    state: Box<dyn std::any::Any + Send + Sync>,
}

impl SystemState {
    /// Create a new SystemState wrapping typed state.
    pub fn new<S: Send + Sync + 'static>(state: S) -> Self {
        Self {
            state: Box::new(state),
        }
    }

    /// Downcast to the concrete state type.
    pub fn get_mut<S: 'static>(&mut self) -> &mut S {
        self.state
            .downcast_mut::<S>()
            .expect("SystemState type mismatch")
    }
}
