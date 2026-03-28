//! Typed resource storage.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Type-erased storage for unique resources keyed by concrete type.
#[derive(Default)]
pub struct ResourceStore {
    resources: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl ResourceStore {
    /// Create an empty resource store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a resource by its concrete type.
    pub fn insert<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.resources.insert(TypeId::of::<T>(), Box::new(resource));
    }

    /// Get an immutable reference to a resource by type.
    pub fn get<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.resources.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }

    /// Get a mutable reference to a resource by type.
    pub fn get_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut T> {
        self.resources
            .get_mut(&TypeId::of::<T>())?
            .downcast_mut::<T>()
    }

    /// Check if a resource of the given type exists.
    pub fn has<T: 'static + Send + Sync>(&self) -> bool {
        self.resources.contains_key(&TypeId::of::<T>())
    }

    /// Remove a resource by type, returning the owned value if it existed.
    pub fn remove<T: 'static + Send + Sync>(&mut self) -> Option<T> {
        self.resources
            .remove(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast::<T>().ok())
            .map(|boxed| *boxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Time {
        delta_seconds: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Score(u32);

    #[test]
    fn insert_and_get_resource() {
        let mut resources = ResourceStore::new();
        resources.insert(Time {
            delta_seconds: 0.016,
        });

        let time = resources.get::<Time>().expect("time resource should exist");
        assert_eq!(time.delta_seconds, 0.016);
    }

    #[test]
    fn overwrite_resource_by_type() {
        let mut resources = ResourceStore::new();
        resources.insert(Score(1));
        resources.insert(Score(42));

        assert_eq!(resources.get::<Score>(), Some(&Score(42)));
    }

    #[test]
    fn get_missing_resource_returns_none() {
        let resources = ResourceStore::new();
        assert!(resources.get::<Time>().is_none());
    }

    #[test]
    fn get_mut_resource_allows_updates() {
        let mut resources = ResourceStore::new();
        resources.insert(Score(0));

        let score = resources
            .get_mut::<Score>()
            .expect("score resource should exist");
        score.0 = 7;

        assert_eq!(resources.get::<Score>(), Some(&Score(7)));
    }

    #[test]
    fn has_returns_true_for_existing_resource() {
        let mut resources = ResourceStore::new();
        resources.insert(Score(42));
        assert!(resources.has::<Score>());
        assert!(!resources.has::<Time>());
    }

    #[test]
    fn remove_returns_owned_value() {
        let mut resources = ResourceStore::new();
        resources.insert(Score(99));

        let removed = resources.remove::<Score>();
        assert_eq!(removed, Some(Score(99)));
        assert!(!resources.has::<Score>());
        assert!(resources.get::<Score>().is_none());
    }

    #[test]
    fn remove_missing_returns_none() {
        let mut resources = ResourceStore::new();
        assert_eq!(resources.remove::<Score>(), None);
    }
}
