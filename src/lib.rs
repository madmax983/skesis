//! Skesis — a high-performance Entity Component System.
//!
//! Hybrid storage with archetype-based columnar layout for cache-friendly
//! iteration and sparse sets for rare components. Features anchor+delta
//! change detection, typed relationships with hierarchy traversal, observers,
//! and raw byte world serialization.
//!
//! # Quick Start
//!
//! ```
//! use skesis::World;
//!
//! struct Position { x: f32, y: f32 }
//! struct Velocity { dx: f32, dy: f32 }
//!
//! let mut world = World::new();
//! let entity = world.spawn_empty();
//! world.add_component(entity, Position { x: 0.0, y: 0.0 });
//! world.add_component(entity, Velocity { dx: 1.0, dy: 2.0 });
//!
//! world.for_each_pair::<Position, Velocity>(|entity, pos, vel| {
//!     // update positions from velocities
//! });
//! ```

#![warn(missing_docs)]

mod app;
mod archetype;
mod bundle;
/// Anchor+delta change detection internals.
pub mod change;
mod column;
mod command;
mod component;
mod entity;
mod event;
mod plugin;
mod query;
mod relation;
mod resource;
mod scheduler;
mod snapshot;
mod sparse_set;
mod system;
mod tuple_query;
/// System parameter types (Res, ResMut, Query, Commands, Local, Events).
pub mod param;
/// Raw access to World internals for system parameter extraction.
pub mod unsafe_cell;
mod world;

pub use app::App;
pub use archetype::{Archetype, ArchetypeId, ComponentSet};
pub use bundle::SpawnBundle;
pub use change::ChangeHistory;
pub use command::CommandRecorder;
pub use component::Component;
pub use entity::Entity;
pub use event::EventStore;
pub use plugin::{Plugin, Stage};
pub use relation::Relation;
pub use resource::ResourceStore;
pub use scheduler::plan_stage;
pub use snapshot::WorldSnapshot;
pub use sparse_set::SparseSet;
pub use system::{ParallelSystemFn, SystemAccess, SystemDescriptor, SystemFn};
pub use param::{Commands, EventReader, EventWriter, Local, Res, ResMut, SystemParam, SystemState};
pub use unsafe_cell::UnsafeWorldCell;
pub use tuple_query::QueryTuple;
pub use world::{
    BorrowedMutQueryPlan, BorrowedPairQueryPlan, ComponentView, ComponentViewMut, MutQueryPlan,
    PairQueryPlan, World,
};
