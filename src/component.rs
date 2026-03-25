//! Component trait for ECS.

/// Marker trait for components.
///
/// All component types must implement this. It's auto-implemented for any
/// type that is `'static + Send + Sync`.
pub trait Component: 'static + Send + Sync {}

// Blanket impl for all suitable types
impl<T: 'static + Send + Sync> Component for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::TypeId;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug)]
    struct Velocity {
        _dx: f32,
        _dy: f32,
    }

    #[test]
    fn component_trait_auto_implemented() {
        // Verify these types implement Component
        fn assert_component<T: Component>() {}

        assert_component::<Position>();
        assert_component::<Velocity>();
    }

    #[test]
    fn component_type_ids_are_unique() {
        let pos_id = TypeId::of::<Position>();
        let vel_id = TypeId::of::<Velocity>();
        assert_ne!(pos_id, vel_id);
    }
}
