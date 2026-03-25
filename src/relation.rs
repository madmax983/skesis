//! Entity relationships — typed directed edges between entities.

use crate::Entity;
use std::any::TypeId;
use std::collections::{HashMap, HashSet};

/// Marker trait for relationship types.
///
/// Relationships are typed directed edges: `(source) --[R]--> (target)`.
/// Any `'static + Send + Sync` type can be a relation.
pub trait Relation: 'static + Send + Sync {}

// Blanket impl for all suitable types.
impl<T: 'static + Send + Sync> Relation for T {}

/// Stores entity relationships indexed by relation type.
pub(crate) struct RelationshipStore {
    /// relation TypeId → set of (source, target) pairs
    edges: HashMap<TypeId, HashSet<(Entity, Entity)>>,
}

impl RelationshipStore {
    pub(crate) fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Add a relationship edge. Returns true if the edge is new.
    pub(crate) fn add<R: Relation>(&mut self, source: Entity, target: Entity) -> bool {
        self.edges
            .entry(TypeId::of::<R>())
            .or_default()
            .insert((source, target))
    }

    /// Remove a relationship edge. Returns true if it existed.
    pub(crate) fn remove<R: Relation>(&mut self, source: Entity, target: Entity) -> bool {
        if let Some(pairs) = self.edges.get_mut(&TypeId::of::<R>()) {
            pairs.remove(&(source, target))
        } else {
            false
        }
    }

    /// Check if a relationship edge exists.
    pub(crate) fn has<R: Relation>(&self, source: Entity, target: Entity) -> bool {
        self.edges
            .get(&TypeId::of::<R>())
            .is_some_and(|pairs| pairs.contains(&(source, target)))
    }

    /// Get the first target for a source entity under relation R, without allocating.
    pub(crate) fn first_target<R: Relation>(&self, source: Entity) -> Option<Entity> {
        self.edges
            .get(&TypeId::of::<R>())
            .and_then(|pairs| pairs.iter().find(|(s, _)| *s == source).map(|(_, t)| *t))
    }

    /// Get all targets for a source entity under relation R.
    pub(crate) fn targets<R: Relation>(&self, source: Entity) -> Vec<Entity> {
        self.edges
            .get(&TypeId::of::<R>())
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(s, _)| *s == source)
                    .map(|(_, t)| *t)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all sources that point to a target entity under relation R.
    pub(crate) fn sources<R: Relation>(&self, target: Entity) -> Vec<Entity> {
        self.edges
            .get(&TypeId::of::<R>())
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(_, t)| *t == target)
                    .map(|(s, _)| *s)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove all edges involving an entity (as source or target), across all relation types.
    pub(crate) fn remove_entity(&mut self, entity: Entity) {
        for pairs in self.edges.values_mut() {
            pairs.retain(|&(s, t)| s != entity && t != entity);
        }
    }

    /// Iterate all (source, target) pairs for a relation type.
    pub(crate) fn iter<R: Relation>(&self) -> impl Iterator<Item = (Entity, Entity)> + '_ {
        self.edges
            .get(&TypeId::of::<R>())
            .into_iter()
            .flat_map(|pairs| pairs.iter().copied())
    }
}
