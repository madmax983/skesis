//! Tuple-based queries for arbitrary component counts.
//!
//! Provides `world.query_tuple::<(A, B, C)>()` for ergonomic multi-component queries.
//! Generated via macro for tuples of 1 to 8 components.
//!
//! For maximum hot-path performance on 2-component queries, use the specialized
//! `query_pair` / `for_each_pair` / `BorrowedPairQueryPlan` APIs instead.

use crate::archetype::Archetype;
use crate::component::Component;
use crate::entity::Entity;

/// Trait implemented by component tuples that can be queried from an archetype.
///
/// Each tuple size gets a macro-generated impl that checks all component types
/// against the archetype's component set and extracts typed slices.
pub trait QueryTuple<'a>: Sized {
    /// Check if an archetype contains all required component types.
    fn matches(archetype: &Archetype) -> bool;

    /// Extract component slices from a matching archetype.
    ///
    /// # Panics
    /// Panics if the archetype doesn't contain all required components.
    /// Call `matches` first.
    fn fetch(archetype: &'a Archetype) -> Vec<(Entity, Self)>;
}

macro_rules! impl_query_tuple {
    // Base case: single component
    ($($idx:tt: $ty:ident),+) => {
        #[allow(non_snake_case)]
        impl<'a, $($ty: Component),+> QueryTuple<'a> for ($(&'a $ty,)+) {
            fn matches(archetype: &Archetype) -> bool {
                let cs = archetype.component_set();
                $(cs.contains::<$ty>())&&+
            }

            fn fetch(archetype: &'a Archetype) -> Vec<(Entity, Self)> {
                let entities = archetype.entities();
                $(
                    let $ty = archetype
                        .components::<$ty>()
                        .expect(concat!(
                            "archetype missing column for ",
                            stringify!($ty)
                        ));
                )+

                let mut results = Vec::with_capacity(entities.len());
                for i in 0..entities.len() {
                    results.push((entities[i], ($(&$ty[i],)+)));
                }
                results
            }
        }
    };
}

// Generate impls for tuple sizes 1 through 8.
impl_query_tuple!(0: A);
impl_query_tuple!(0: A, 1: B);
impl_query_tuple!(0: A, 1: B, 2: C);
impl_query_tuple!(0: A, 1: B, 2: C, 3: D);
impl_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_query_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);

#[cfg(test)]
mod tests {
    use crate::World;

    #[derive(Debug, PartialEq)]
    struct Pos {
        x: f32,
    }
    #[derive(Debug, PartialEq)]
    struct Vel {
        dx: f32,
    }
    #[derive(Debug, PartialEq)]
    struct Health(u32);
    #[derive(Debug, PartialEq)]
    struct Team(u8);

    #[test]
    fn query_tuple_two_components() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.add_component(e1, Pos { x: 1.0 });
        world.add_component(e1, Vel { dx: 10.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Pos { x: 2.0 });
        // No Vel — should not appear.

        let results: Vec<_> = world
            .query_tuple::<(&Pos, &Vel)>()
            .map(|(e, (p, v))| (e, p.x, v.dx))
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (e1, 1.0, 10.0));
    }

    #[test]
    fn query_tuple_three_components() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.add_component(e1, Pos { x: 1.0 });
        world.add_component(e1, Vel { dx: 10.0 });
        world.add_component(e1, Health(100));

        let e2 = world.spawn_empty();
        world.add_component(e2, Pos { x: 2.0 });
        world.add_component(e2, Vel { dx: 20.0 });
        // No Health — should not appear.

        let results: Vec<_> = world
            .query_tuple::<(&Pos, &Vel, &Health)>()
            .map(|(e, (p, v, h))| (e, p.x, v.dx, h.0))
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (e1, 1.0, 10.0, 100));
    }

    #[test]
    fn query_tuple_four_components() {
        let mut world = World::new();
        let entity = world.spawn_empty();
        world.add_component(entity, Pos { x: 5.0 });
        world.add_component(entity, Vel { dx: 50.0 });
        world.add_component(entity, Health(200));
        world.add_component(entity, Team(3));

        let results: Vec<_> = world
            .query_tuple::<(&Pos, &Vel, &Health, &Team)>()
            .map(|(_, (p, v, h, t))| (p.x, v.dx, h.0, t.0))
            .collect();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], (5.0, 50.0, 200, 3));
    }

    #[test]
    fn query_tuple_single_component() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.add_component(e1, Pos { x: 42.0 });

        let results: Vec<_> = world
            .query_tuple::<(&Pos,)>()
            .map(|(_, (p,))| p.x)
            .collect();

        assert_eq!(results, vec![42.0]);
    }

    #[test]
    fn query_tuple_no_matches() {
        let mut world = World::new();
        let e1 = world.spawn_empty();
        world.add_component(e1, Pos { x: 1.0 });

        let count = world.query_tuple::<(&Pos, &Vel)>().count();
        assert_eq!(count, 0);
    }
}
