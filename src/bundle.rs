//! Bundle trait for batch entity spawning.
//!
//! Allows spawning an entity with all its components in a single operation,
//! skipping the archetype migration chain that individual `add_component` calls
//! would trigger.
//!
//! ```
//! use skesis::World;
//!
//! struct Position { x: f32, y: f32 }
//! struct Velocity { dx: f32, dy: f32 }
//!
//! let mut world = World::new();
//! let entity = world.spawn_with((
//!     Position { x: 0.0, y: 0.0 },
//!     Velocity { dx: 1.0, dy: 2.0 },
//! ));
//! ```

use crate::archetype::Archetype;
use crate::component::Component;
use crate::{ComponentSet, World};
use std::any::TypeId;

/// A tuple of components that can be spawned together in a single archetype insert.
///
/// Implemented for tuples of 1 to 8 components via macro.
pub trait SpawnBundle: 'static + Send + Sync {
    /// Build the sorted `ComponentSet` for the target archetype.
    fn component_set() -> ComponentSet;

    /// Register column factories for all component types in this bundle.
    fn register(world: &mut World);

    /// Push all component values into the archetype's typed columns.
    ///
    /// The caller must have already called `archetype.track_entity(entity)`
    /// before invoking this method.
    fn push_components(self, archetype: &mut Archetype);

    /// Return the `TypeId`s of all components (for change detection + observers).
    fn type_ids() -> Vec<TypeId>;

    /// Call `f` once per component TypeId, avoiding Vec allocation on hot path.
    fn for_each_type_id(f: impl FnMut(TypeId));
}

macro_rules! impl_spawn_bundle {
    ($($idx:tt: $ty:ident),+) => {
        impl<$($ty: Component),+> SpawnBundle for ($($ty,)+) {
            fn component_set() -> ComponentSet {
                ComponentSet::new()$(.with::<$ty>())+
            }

            fn register(world: &mut World) {
                $(world.register_component_type::<$ty>();)+
            }

            fn push_components(self, archetype: &mut Archetype) {
                $(archetype.push_typed_component(self.$idx);)+
            }

            fn type_ids() -> Vec<TypeId> {
                vec![$(TypeId::of::<$ty>()),+]
            }

            fn for_each_type_id(mut f: impl FnMut(TypeId)) {
                $(f(TypeId::of::<$ty>());)+
            }
        }
    };
}

impl_spawn_bundle!(0: A);
impl_spawn_bundle!(0: A, 1: B);
impl_spawn_bundle!(0: A, 1: B, 2: C);
impl_spawn_bundle!(0: A, 1: B, 2: C, 3: D);
impl_spawn_bundle!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_spawn_bundle!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_spawn_bundle!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_spawn_bundle!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct Pos {
        x: f32,
        y: f32,
    }
    #[allow(dead_code)]
    struct Vel {
        dx: f32,
        dy: f32,
    }
    #[allow(dead_code)]
    struct Health(u32);

    #[test]
    fn spawn_bundle_component_set_is_sorted() {
        let set_ab = <(Pos, Vel)>::component_set();
        let set_ba = <(Vel, Pos)>::component_set();
        // ComponentSet is sorted by TypeId, so order shouldn't matter.
        assert_eq!(set_ab, set_ba);
    }

    #[test]
    fn spawn_bundle_type_ids_includes_all() {
        let ids = <(Pos, Vel, Health)>::type_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&TypeId::of::<Pos>()));
        assert!(ids.contains(&TypeId::of::<Vel>()));
        assert!(ids.contains(&TypeId::of::<Health>()));
    }

    #[test]
    fn spawn_with_single_component() {
        let mut world = World::new();
        let entity = world.spawn_with((Pos { x: 1.0, y: 2.0 },));

        assert!(world.is_alive(entity));
        let pos = world.get_component::<Pos>(entity).unwrap();
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 2.0);
    }

    #[test]
    fn spawn_with_two_components() {
        let mut world = World::new();
        let entity = world.spawn_with((Pos { x: 10.0, y: 20.0 }, Vel { dx: 1.0, dy: 2.0 }));

        assert!(world.is_alive(entity));
        let pos = world.get_component::<Pos>(entity).unwrap();
        assert_eq!(pos.x, 10.0);
        let vel = world.get_component::<Vel>(entity).unwrap();
        assert_eq!(vel.dx, 1.0);
    }

    #[test]
    fn spawn_with_three_components() {
        let mut world = World::new();
        let entity = world.spawn_with((
            Pos { x: 5.0, y: 5.0 },
            Vel { dx: 3.0, dy: 4.0 },
            Health(100),
        ));

        assert!(world.is_alive(entity));
        let h = world.get_component::<Health>(entity).unwrap();
        assert_eq!(h.0, 100);
    }

    #[test]
    fn spawn_with_places_in_correct_archetype() {
        let mut world = World::new();

        // Spawn via add_component path
        let e1 = world.spawn_empty();
        world.add_component(e1, Pos { x: 1.0, y: 1.0 });
        world.add_component(e1, Vel { dx: 1.0, dy: 1.0 });

        // Spawn via bundle path
        let e2 = world.spawn_with((Pos { x: 2.0, y: 2.0 }, Vel { dx: 2.0, dy: 2.0 }));

        // Both should be queryable in the same pair query
        let mut results = Vec::new();
        world.for_each_pair::<Pos, Vel>(|entity, pos, _vel| {
            results.push((entity, pos.x));
        });

        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|(e, _)| *e == e1));
        assert!(results.iter().any(|(e, _)| *e == e2));
    }

    #[test]
    fn spawn_with_respects_entity_reuse() {
        let mut world = World::new();
        let e1 = world.spawn_with((Pos { x: 1.0, y: 1.0 },));
        world.despawn(e1);

        // Next spawn should reuse the freed index with bumped generation.
        let e2 = world.spawn_with((Pos { x: 2.0, y: 2.0 },));
        assert_eq!(e2.index(), e1.index());
        assert_ne!(e2.generation(), e1.generation());
    }
}
