//! Archetype storage for hot-path components.

use crate::Entity;
use crate::column::{BoxedColumn, ColumnFactory, ComponentValue, ErasedColumn, TypedColumn};
use std::any::TypeId;
use std::collections::HashMap;

type EntityComponentMap = Vec<(TypeId, ComponentValue)>;
type ArchetypeColumns = HashMap<TypeId, Box<dyn ErasedColumn>>;

/// Unique identifier for an archetype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArchetypeId(pub(crate) u32);

/// Set of component types that define an archetype.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComponentSet {
    types: Vec<TypeId>,
}

impl ComponentSet {
    /// Create a new empty component set.
    pub fn new() -> Self {
        Self { types: Vec::new() }
    }

    /// Add a component type to the set.
    pub fn with<T: 'static>(mut self) -> Self {
        let type_id = TypeId::of::<T>();
        if let Err(index) = self.types.binary_search(&type_id) {
            self.types.insert(index, type_id);
        }
        self
    }

    /// Remove a component type from the set.
    pub fn without<T: 'static>(mut self) -> Self {
        let type_id = TypeId::of::<T>();
        if let Ok(index) = self.types.binary_search(&type_id) {
            self.types.remove(index);
        }
        self
    }

    /// Add a component type by runtime TypeId.
    pub fn with_type_id(mut self, type_id: TypeId) -> Self {
        if let Err(index) = self.types.binary_search(&type_id) {
            self.types.insert(index, type_id);
        }
        self
    }

    /// Check if the set contains a component type.
    pub fn contains<T: 'static>(&self) -> bool {
        self.types.binary_search(&TypeId::of::<T>()).is_ok()
    }

    /// Check if the set contains a component type by runtime TypeId.
    pub fn contains_type_id(&self, type_id: &TypeId) -> bool {
        self.types.binary_search(type_id).is_ok()
    }

    /// Get the number of component types.
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Iterate over component type IDs.
    pub fn iter(&self) -> impl Iterator<Item = &TypeId> {
        self.types.iter()
    }
}

impl Default for ComponentSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage for entities with a specific component set.
pub struct Archetype {
    id: ArchetypeId,
    component_set: ComponentSet,
    entities: Vec<Entity>,
    entity_indices: Vec<u32>,
    components: ArchetypeColumns,
}

impl Archetype {
    /// Create a new archetype with the given ID and component set.
    pub fn new(id: ArchetypeId, component_set: ComponentSet) -> Self {
        Self::new_with_factories(id, component_set, &HashMap::new())
    }

    /// Create a new archetype with registered typed column factories.
    pub(crate) fn new_with_factories(
        id: ArchetypeId,
        component_set: ComponentSet,
        factories: &HashMap<TypeId, ColumnFactory>,
    ) -> Self {
        let mut components = HashMap::new();

        for &type_id in component_set.iter() {
            if let Some(factory) = factories.get(&type_id) {
                components.insert(type_id, factory());
            }
        }

        Self {
            id,
            component_set,
            entities: Vec::new(),
            entity_indices: Vec::new(),
            components,
        }
    }

    /// Get the archetype ID.
    pub fn id(&self) -> ArchetypeId {
        self.id
    }

    /// Get the component set.
    pub fn component_set(&self) -> &ComponentSet {
        &self.component_set
    }

    /// Get the number of entities in this archetype.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if the archetype is empty.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get entity at index.
    pub fn entity(&self, index: usize) -> Option<Entity> {
        self.entities.get(index).copied()
    }

    /// Get all entities stored in this archetype.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Get all dense entity indices stored in this archetype.
    pub fn entity_indices(&self) -> &[u32] {
        &self.entity_indices
    }

    /// Add an entity with components to this archetype.
    pub fn push_entity(&mut self, entity: Entity, components: EntityComponentMap) {
        self.entities.push(entity);
        self.entity_indices.push(entity.index());

        for (type_id, component) in components {
            if let Some(column) = self.components.get_mut(&type_id) {
                column.push_boxed(component);
                continue;
            }

            let mut column = BoxedColumn::new();
            column.push_boxed(component);
            self.components.insert(type_id, Box::new(column));
        }
    }

    /// Add an entity to an archetype that has no component columns.
    #[inline(always)]
    pub fn push_entity_empty(&mut self, entity: Entity) {
        debug_assert!(self.components.is_empty());
        self.entities.push(entity);
        self.entity_indices.push(entity.index());
    }

    /// Add an entity with exactly one typed component to this archetype.
    #[inline(always)]
    pub fn push_entity_single_component<T: 'static + Send + Sync>(
        &mut self,
        entity: Entity,
        component: T,
    ) {
        self.entities.push(entity);
        self.entity_indices.push(entity.index());

        let column = self
            .components
            .get_mut(&TypeId::of::<T>())
            .expect("archetype missing typed column for single-component insertion");
        let typed = column
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("typed column downcast failed during single-component insertion");
        typed.push(component);
    }

    /// Remove an entity at the given index.
    ///
    /// Uses swap-remove for O(1) removal.
    pub fn swap_remove_entity(&mut self, index: usize) -> Option<(Entity, EntityComponentMap)> {
        if index >= self.entities.len() {
            return None;
        }

        let entity = self.entities.swap_remove(index);
        self.entity_indices.swap_remove(index);
        let mut components = Vec::with_capacity(self.components.len());

        for (type_id, column) in &mut self.components {
            if column.len() <= index {
                panic!("column length mismatch while removing entity from archetype");
            }
            components.push((*type_id, column.swap_remove_boxed(index)));
        }

        Some((entity, components))
    }

    /// Remove an entity and drop all of its components without materializing a component map.
    #[inline(always)]
    pub fn swap_remove_entity_discard(&mut self, index: usize) -> Option<Entity> {
        if index >= self.entities.len() {
            return None;
        }

        let entity = self.entities.swap_remove(index);
        self.entity_indices.swap_remove(index);

        for column in self.components.values_mut() {
            if column.len() <= index {
                panic!("column length mismatch while removing entity from archetype");
            }
            column.swap_remove_drop(index);
        }

        Some(entity)
    }

    /// Move a row from this archetype into `destination`, adding one extra component `T`.
    ///
    /// Returns `(destination_row_index, swapped_entity_in_source)` on success.
    pub fn move_row_to_with_added_component<T: 'static + Send + Sync>(
        &mut self,
        index: usize,
        destination: &mut Archetype,
        entity: Entity,
        component: T,
    ) -> Option<(usize, Option<Entity>)> {
        if index >= self.entities.len() {
            return None;
        }

        let destination_row_index = destination.entities.len();
        destination.entities.push(entity);
        destination.entity_indices.push(entity.index());

        for (type_id, source_column) in &mut self.components {
            let destination_column = destination
                .components
                .get_mut(type_id)
                .expect("destination archetype missing source component column");
            source_column.swap_remove_into(index, destination_column.as_mut());
        }

        let destination_typed_column = destination
            .components
            .get_mut(&TypeId::of::<T>())
            .expect("destination archetype missing added component column");
        let destination_typed = destination_typed_column
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("added component column downcast failed");
        destination_typed.push(component);

        let removed_entity = self.entities.swap_remove(index);
        debug_assert_eq!(removed_entity, entity);
        self.entity_indices.swap_remove(index);

        let swapped_entity = if index < self.entities.len() {
            Some(self.entities[index])
        } else {
            None
        };

        Some((destination_row_index, swapped_entity))
    }

    /// Move a row from this archetype into `destination`, dropping one component `T`.
    ///
    /// Returns `(removed_component, destination_row_index, swapped_entity_in_source)` on success.
    pub fn move_row_to_without_component<T: 'static + Send + Sync>(
        &mut self,
        index: usize,
        destination: &mut Archetype,
        entity: Entity,
    ) -> Option<(T, usize, Option<Entity>)> {
        if index >= self.entities.len() {
            return None;
        }

        let destination_row_index = destination.entities.len();
        destination.entities.push(entity);
        destination.entity_indices.push(entity.index());

        let removed_type_id = TypeId::of::<T>();

        // Move all columns except the removed one into the destination.
        for (type_id, source_column) in &mut self.components {
            if *type_id == removed_type_id {
                continue;
            }
            let destination_column = destination
                .components
                .get_mut(type_id)
                .expect("destination archetype missing source component column");
            source_column.swap_remove_into(index, destination_column.as_mut());
        }

        // Extract the removed component value from the dropped column.
        let removed_column = self
            .components
            .get_mut(&removed_type_id)
            .expect("source archetype missing removed component column");
        let removed_typed = removed_column
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("removed component column downcast failed");
        let removed_value = removed_typed.swap_remove(index);

        let removed_entity = self.entities.swap_remove(index);
        debug_assert_eq!(removed_entity, entity);
        self.entity_indices.swap_remove(index);

        let swapped_entity = if index < self.entities.len() {
            Some(self.entities[index])
        } else {
            None
        };

        Some((removed_value, destination_row_index, swapped_entity))
    }

    /// Remove an entity from an archetype that has no component columns.
    #[inline(always)]
    pub fn swap_remove_entity_empty(&mut self, index: usize) -> Option<Entity> {
        if index >= self.entities.len() {
            return None;
        }

        debug_assert!(self.components.is_empty());
        let entity = self.entities.swap_remove(index);
        self.entity_indices.swap_remove(index);
        Some(entity)
    }

    /// Get a raw column reference by `TypeId`, for byte-level change detection.
    pub(crate) fn component_column(&self, type_id: &TypeId) -> Option<&dyn ErasedColumn> {
        self.components.get(type_id).map(|c| c.as_ref())
    }

    /// Restore entity and entity_index arrays from slices (used by snapshot loading).
    pub(crate) fn restore_entities(&mut self, entities: &[Entity], entity_indices: &[u32]) {
        self.entities = entities.to_vec();
        self.entity_indices = entity_indices.to_vec();
    }

    /// Restore a column's data from raw bytes (used by snapshot loading).
    pub(crate) fn restore_column_bytes(&mut self, type_id: &TypeId, bytes: &[u8]) {
        if let Some(column) = self.components.get_mut(type_id) {
            column.restore_from_bytes(bytes);
        }
    }

    /// Get components of a specific type.
    pub fn components<T: 'static + Send + Sync>(&self) -> Option<&[T]> {
        let column = self.components.get(&TypeId::of::<T>())?;
        let typed = column.as_any().downcast_ref::<TypedColumn<T>>()?;
        Some(typed.as_slice())
    }

    /// Get mutable components of a specific type.
    pub fn components_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut [T]> {
        let column = self.components.get_mut(&TypeId::of::<T>())?;
        let typed = column.as_any_mut().downcast_mut::<TypedColumn<T>>()?;
        Some(typed.as_mut_slice())
    }

    /// Get entity slice and mutable component slice for a specific type.
    pub fn entities_and_components_mut<T: 'static + Send + Sync>(
        &mut self,
    ) -> Option<(&[Entity], &mut [T])> {
        let entities = self.entities.as_slice();
        let column = self.components.get_mut(&TypeId::of::<T>())?;
        let typed = column.as_any_mut().downcast_mut::<TypedColumn<T>>()?;
        Some((entities, typed.as_mut_slice()))
    }

    /// Get dense entity-index slice and mutable component slice for a specific type.
    pub fn entity_indices_and_components_mut<T: 'static + Send + Sync>(
        &mut self,
    ) -> Option<(&[u32], &mut [T])> {
        let entity_indices = self.entity_indices.as_slice();
        let column = self.components.get_mut(&TypeId::of::<T>())?;
        let typed = column.as_any_mut().downcast_mut::<TypedColumn<T>>()?;
        Some((entity_indices, typed.as_mut_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Transform;
    struct Sprite;
    struct Velocity;

    #[test]
    fn component_set_builder() {
        let set = ComponentSet::new().with::<Transform>().with::<Sprite>();

        assert_eq!(set.len(), 2);
        assert!(set.contains::<Transform>());
        assert!(set.contains::<Sprite>());
        assert!(!set.contains::<Velocity>());
    }

    #[test]
    fn component_sets_with_same_types_are_equal() {
        let set1 = ComponentSet::new().with::<Transform>().with::<Sprite>();
        let set2 = ComponentSet::new().with::<Sprite>().with::<Transform>();

        assert_eq!(set1, set2);
    }
}

#[cfg(test)]
mod archetype_tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[test]
    fn create_archetype() {
        let component_set = ComponentSet::new().with::<Position>();

        let archetype = Archetype::new(ArchetypeId(0), component_set);

        assert_eq!(archetype.id(), ArchetypeId(0));
        assert_eq!(archetype.len(), 0);
        assert!(archetype.is_empty());
    }

    #[test]
    fn add_entity_to_archetype() {
        use crate::Entity;
        use std::any::{Any, TypeId};

        let component_set = ComponentSet::new().with::<Position>();
        let mut archetype = Archetype::new(ArchetypeId(0), component_set);

        let entity = Entity::new(0, 0);
        let components = vec![(
            TypeId::of::<Position>(),
            Box::new(Position { x: 10.0, y: 20.0 }) as Box<dyn Any + Send + Sync>,
        )];

        archetype.push_entity(entity, components);

        assert_eq!(archetype.len(), 1);
        assert_eq!(archetype.entity(0), Some(entity));
    }

    #[test]
    fn remove_entity_from_archetype() {
        use crate::Entity;
        use std::any::{Any, TypeId};

        let component_set = ComponentSet::new().with::<Position>();
        let mut archetype = Archetype::new(ArchetypeId(0), component_set);

        let entity = Entity::new(0, 0);
        let components = vec![(
            TypeId::of::<Position>(),
            Box::new(Position { x: 10.0, y: 20.0 }) as Box<dyn Any + Send + Sync>,
        )];

        archetype.push_entity(entity, components);

        let removed = archetype.swap_remove_entity(0);
        assert!(removed.is_some());
        assert_eq!(archetype.len(), 0);
    }
}
