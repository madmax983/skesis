//! Query API tests and compile-time borrow guarantees.
//!
//! ```compile_fail
//! use skesis::World;
//!
//! #[derive(Debug)]
//! struct Position(f32);
//!
//! let mut world = World::new();
//! let entity = world.spawn_empty();
//! world.add_component(entity, Position(1.0));
//!
//! let _first = world.query_mut::<Position>();
//! let _second = world.query_mut::<Position>();
//! ```

#[cfg(test)]
mod tests {
    use crate::World;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn query_pair_returns_entities_with_both_components() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        world.add_component(e1, Velocity { dx: 3.0, dy: 4.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 9.0, y: 9.0 });

        let pairs: Vec<_> = world.query_pair::<Position, Velocity>().collect();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, e1);
        assert_eq!(pairs[0].1, &Position { x: 1.0, y: 2.0 });
        assert_eq!(pairs[0].2, &Velocity { dx: 3.0, dy: 4.0 });
    }

    #[test]
    fn query_mut_updates_components() {
        let mut world = World::new();

        let e1 = world.spawn_empty();
        world.add_component(e1, Position { x: 0.0, y: 0.0 });

        let e2 = world.spawn_empty();
        world.add_component(e2, Position { x: 10.0, y: 10.0 });

        for (_entity, pos) in world.query_mut::<Position>() {
            pos.x += 1.0;
            pos.y += 2.0;
        }

        let values: Vec<_> = world
            .query::<Position>()
            .map(|(_, pos)| (pos.x, pos.y))
            .collect();
        assert!(values.contains(&(1.0, 2.0)));
        assert!(values.contains(&(11.0, 12.0)));
    }
}
