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

/// Per-relation-type edge storage with forward and reverse adjacency indices.
#[derive(Default)]
struct RelationEdges {
    /// Canonical edge set for O(1) existence checks.
    pairs: HashSet<(Entity, Entity)>,
    /// source → [targets] for O(degree) forward lookup.
    forward: HashMap<Entity, Vec<Entity>>,
    /// target → [sources] for O(degree) reverse lookup.
    reverse: HashMap<Entity, Vec<Entity>>,
}

/// Stores entity relationships indexed by relation type.
pub(crate) struct RelationshipStore {
    edges: HashMap<TypeId, RelationEdges>,
}

impl RelationshipStore {
    pub(crate) fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Add a relationship edge. Returns true if the edge is new.
    pub(crate) fn add<R: Relation>(&mut self, source: Entity, target: Entity) -> bool {
        let rel = self.edges.entry(TypeId::of::<R>()).or_default();
        if rel.pairs.insert((source, target)) {
            rel.forward.entry(source).or_default().push(target);
            rel.reverse.entry(target).or_default().push(source);
            true
        } else {
            false
        }
    }

    /// Remove a relationship edge. Returns true if it existed.
    pub(crate) fn remove<R: Relation>(&mut self, source: Entity, target: Entity) -> bool {
        let Some(rel) = self.edges.get_mut(&TypeId::of::<R>()) else {
            return false;
        };
        if !rel.pairs.remove(&(source, target)) {
            return false;
        }
        if let Some(targets) = rel.forward.get_mut(&source) {
            if let Some(pos) = targets.iter().position(|&t| t == target) {
                targets.swap_remove(pos);
            }
            if targets.is_empty() {
                rel.forward.remove(&source);
            }
        }
        if let Some(sources) = rel.reverse.get_mut(&target) {
            if let Some(pos) = sources.iter().position(|&s| s == source) {
                sources.swap_remove(pos);
            }
            if sources.is_empty() {
                rel.reverse.remove(&target);
            }
        }
        true
    }

    /// Check if a relationship edge exists.
    pub(crate) fn has<R: Relation>(&self, source: Entity, target: Entity) -> bool {
        self.edges
            .get(&TypeId::of::<R>())
            .is_some_and(|rel| rel.pairs.contains(&(source, target)))
    }

    /// Get the first target for a source entity under relation R, without allocating.
    pub(crate) fn first_target<R: Relation>(&self, source: Entity) -> Option<Entity> {
        self.edges
            .get(&TypeId::of::<R>())
            .and_then(|rel| rel.forward.get(&source)?.first().copied())
    }

    /// Get all targets for a source entity under relation R.
    pub(crate) fn targets<R: Relation>(&self, source: Entity) -> Vec<Entity> {
        self.edges
            .get(&TypeId::of::<R>())
            .and_then(|rel| rel.forward.get(&source))
            .cloned()
            .unwrap_or_default()
    }

    /// Get all sources that point to a target entity under relation R.
    pub(crate) fn sources<R: Relation>(&self, target: Entity) -> Vec<Entity> {
        self.edges
            .get(&TypeId::of::<R>())
            .and_then(|rel| rel.reverse.get(&target))
            .cloned()
            .unwrap_or_default()
    }

    /// Check if any source points to this target under relation R, without allocating.
    pub(crate) fn has_any_source<R: Relation>(&self, target: Entity) -> bool {
        self.edges
            .get(&TypeId::of::<R>())
            .is_some_and(|rel| rel.reverse.contains_key(&target))
    }

    /// Remove all edges involving an entity (as source or target), across all relation types.
    pub(crate) fn remove_entity(&mut self, entity: Entity) {
        for rel in self.edges.values_mut() {
            // Clean forward edges: entity as source
            if let Some(targets) = rel.forward.remove(&entity) {
                for target in &targets {
                    if let Some(sources) = rel.reverse.get_mut(target) {
                        if let Some(pos) = sources.iter().position(|&s| s == entity) {
                            sources.swap_remove(pos);
                        }
                        if sources.is_empty() {
                            rel.reverse.remove(target);
                        }
                    }
                    rel.pairs.remove(&(entity, *target));
                }
            }
            // Clean reverse edges: entity as target
            if let Some(sources) = rel.reverse.remove(&entity) {
                for source in &sources {
                    if let Some(targets) = rel.forward.get_mut(source) {
                        if let Some(pos) = targets.iter().position(|&t| t == entity) {
                            targets.swap_remove(pos);
                        }
                        if targets.is_empty() {
                            rel.forward.remove(source);
                        }
                    }
                    rel.pairs.remove(&(*source, entity));
                }
            }
        }
    }

    /// Iterate all (source, target) pairs for a relation type.
    pub(crate) fn iter<R: Relation>(&self) -> impl Iterator<Item = (Entity, Entity)> + '_ {
        self.edges
            .get(&TypeId::of::<R>())
            .into_iter()
            .flat_map(|rel| rel.pairs.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Entity;

    struct ChildOf;

    #[test]
    fn add_and_query_targets() {
        let mut store = RelationshipStore::new();
        let parent = Entity::new(0, 0);
        let c1 = Entity::new(1, 0);
        let c2 = Entity::new(2, 0);
        let c3 = Entity::new(3, 0);

        store.add::<ChildOf>(c1, parent);
        store.add::<ChildOf>(c2, parent);
        store.add::<ChildOf>(c3, parent);

        let targets = store.targets::<ChildOf>(c1);
        assert_eq!(targets, vec![parent]);

        // All three children point to the same parent
        let sources = store.sources::<ChildOf>(parent);
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn add_and_query_sources() {
        let mut store = RelationshipStore::new();
        let target = Entity::new(10, 0);
        let s1 = Entity::new(1, 0);
        let s2 = Entity::new(2, 0);
        let s3 = Entity::new(3, 0);

        store.add::<ChildOf>(s1, target);
        store.add::<ChildOf>(s2, target);
        store.add::<ChildOf>(s3, target);

        let sources = store.sources::<ChildOf>(target);
        assert_eq!(sources.len(), 3);
        assert!(sources.contains(&s1));
        assert!(sources.contains(&s2));
        assert!(sources.contains(&s3));
    }

    #[test]
    fn first_target_returns_result() {
        let mut store = RelationshipStore::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);

        store.add::<ChildOf>(a, b);
        assert_eq!(store.first_target::<ChildOf>(a), Some(b));
        assert_eq!(store.first_target::<ChildOf>(b), None);
    }

    #[test]
    fn remove_cleans_secondary_indices() {
        let mut store = RelationshipStore::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);

        store.add::<ChildOf>(a, b);
        assert!(store.has::<ChildOf>(a, b));

        store.remove::<ChildOf>(a, b);
        assert!(!store.has::<ChildOf>(a, b));
        assert!(store.targets::<ChildOf>(a).is_empty());
        assert!(store.sources::<ChildOf>(b).is_empty());
    }

    #[test]
    fn remove_entity_cleans_all_directions() {
        let mut store = RelationshipStore::new();
        let a = Entity::new(0, 0);
        let b = Entity::new(1, 0);
        let c = Entity::new(2, 0);

        // a -> b, c -> a (a is both source and target)
        store.add::<ChildOf>(a, b);
        store.add::<ChildOf>(c, a);

        store.remove_entity(a);

        assert!(!store.has::<ChildOf>(a, b));
        assert!(!store.has::<ChildOf>(c, a));
        assert!(store.targets::<ChildOf>(a).is_empty());
        assert!(store.sources::<ChildOf>(a).is_empty());
        assert!(store.targets::<ChildOf>(c).is_empty());
        assert!(store.sources::<ChildOf>(b).is_empty());
    }
}
