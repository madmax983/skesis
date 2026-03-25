//! Sparse set storage for components.
//!
//! Efficient for components that are present on few entities.
//! Register a component type as sparse via `world.register_sparse::<T>()`.
//! Sparse components don't cause archetype transitions — the entity stays
//! in its current archetype while the component is stored in a parallel
//! sparse set.

use crate::Entity;
use std::any::Any;

/// Sparse set for storing components.
pub struct SparseSet<T> {
    sparse: Vec<Option<usize>>,
    dense: Vec<Entity>,
    components: Vec<T>,
}

impl<T> SparseSet<T> {
    /// Create a new empty sparse set.
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
            components: Vec::new(),
        }
    }

    /// Insert a component for an entity.
    pub fn insert(&mut self, entity: Entity, component: T) {
        let index = entity.index() as usize;

        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, None);
        }

        if let Some(dense_index) = self.sparse[index] {
            self.components[dense_index] = component;
        } else {
            let dense_index = self.dense.len();
            self.sparse[index] = Some(dense_index);
            self.dense.push(entity);
            self.components.push(component);
        }
    }

    /// Check if an entity has this component.
    pub fn contains(&self, entity: Entity) -> bool {
        let index = entity.index() as usize;
        index < self.sparse.len() && self.sparse[index].is_some()
    }

    /// Get a reference to a component.
    pub fn get(&self, entity: Entity) -> Option<&T> {
        let index = entity.index() as usize;
        let dense_index = *self.sparse.get(index)?.as_ref()?;
        self.components.get(dense_index)
    }

    /// Remove a component for an entity.
    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let index = entity.index() as usize;

        if index >= self.sparse.len() {
            return None;
        }

        let dense_index = self.sparse[index]?;

        let last_dense_index = self.dense.len() - 1;
        let last_entity = self.dense[last_dense_index];

        self.dense.swap_remove(dense_index);
        let component = self.components.swap_remove(dense_index);

        if dense_index < self.dense.len() {
            self.sparse[last_entity.index() as usize] = Some(dense_index);
        }

        self.sparse[index] = None;

        Some(component)
    }

    /// Get a mutable reference to a component.
    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let index = entity.index() as usize;
        let dense_index = *self.sparse.get(index)?.as_ref()?;
        self.components.get_mut(dense_index)
    }

    /// Iterate over all components.
    pub fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.dense
            .iter()
            .zip(self.components.iter())
            .map(|(e, c)| (*e, c))
    }

    /// Get the number of components stored.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Check if the sparse set is empty.
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }
}

impl<T> Default for SparseSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-erased sparse set for storing in World's HashMap.
pub(crate) trait ErasedSparseSet: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn contains_entity(&self, entity: Entity) -> bool;
    fn remove_entity(&mut self, entity: Entity) -> bool;
    fn len(&self) -> usize;
}

impl<T: 'static + Send + Sync> ErasedSparseSet for SparseSet<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn contains_entity(&self, entity: Entity) -> bool {
        self.contains(entity)
    }

    fn remove_entity(&mut self, entity: Entity) -> bool {
        self.remove(entity).is_some()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Create a new empty type-erased sparse set for component type T.
pub(crate) fn erased_sparse_set<T: 'static + Send + Sync>() -> Box<dyn ErasedSparseSet> {
    Box::new(SparseSet::<T>::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Entity;

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[test]
    fn insert_and_get() {
        let mut set = SparseSet::new();
        let entity = Entity::new(0, 0);

        set.insert(entity, Health(100));

        assert_eq!(set.get(entity), Some(&Health(100)));
        assert!(set.contains(entity));
    }

    #[test]
    fn remove() {
        let mut set = SparseSet::new();
        let entity = Entity::new(0, 0);

        set.insert(entity, Health(100));
        let removed = set.remove(entity);

        assert_eq!(removed, Some(Health(100)));
        assert!(!set.contains(entity));
        assert_eq!(set.get(entity), None);
    }

    #[test]
    fn overwrite_existing() {
        let mut set = SparseSet::new();
        let entity = Entity::new(0, 0);

        set.insert(entity, Health(100));
        set.insert(entity, Health(50));

        assert_eq!(set.get(entity), Some(&Health(50)));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn multiple_entities() {
        let mut set = SparseSet::new();
        let e1 = Entity::new(0, 0);
        let e2 = Entity::new(1, 0);
        let e3 = Entity::new(2, 0);

        set.insert(e1, Health(100));
        set.insert(e2, Health(50));
        set.insert(e3, Health(75));

        assert_eq!(set.len(), 3);
        assert_eq!(set.get(e1), Some(&Health(100)));
        assert_eq!(set.get(e2), Some(&Health(50)));
        assert_eq!(set.get(e3), Some(&Health(75)));
    }

    #[test]
    fn iteration() {
        let mut set = SparseSet::new();
        let e1 = Entity::new(0, 0);
        let e2 = Entity::new(1, 0);

        set.insert(e1, Health(100));
        set.insert(e2, Health(50));

        let collected: Vec<_> = set.iter().collect();
        assert_eq!(collected.len(), 2);
    }
}
