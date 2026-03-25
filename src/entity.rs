//! Entity identifier with generation for safe reuse.

/// Unique entity identifier with generation counter.
///
/// Generation prevents use-after-free bugs when entity IDs are reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    /// Create a new entity with given index and generation.
    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Get the entity's index (position in storage arrays).
    #[inline(always)]
    pub fn index(self) -> u32 {
        self.index
    }

    /// Get the entity's generation (for validity checking).
    #[inline(always)]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_stores_index_and_generation() {
        let entity = Entity::new(42, 1);
        assert_eq!(entity.index(), 42);
        assert_eq!(entity.generation(), 1);
    }

    #[test]
    fn entities_with_different_generations_are_not_equal() {
        let e1 = Entity::new(10, 1);
        let e2 = Entity::new(10, 2);
        assert_ne!(e1, e2);
    }
}
