//! Event buffering and deterministic stage merge.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Event storage with per-system buffering and deterministic stage merge.
#[derive(Default)]
pub struct EventStore {
    committed: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
    stage_buffer: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
    system_buffer: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
}

impl EventStore {
    /// Create an empty event store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin collecting events for a stage.
    pub fn begin_stage(&mut self) {
        self.committed.clear();
        self.stage_buffer.clear();
    }

    /// Begin collecting events for one system execution.
    pub fn begin_system(&mut self) {
        self.system_buffer.clear();
    }

    /// Emit one event into the current system buffer.
    pub fn emit<T: 'static + Send + Sync>(&mut self, event: T) {
        self.system_buffer
            .entry(TypeId::of::<T>())
            .or_default()
            .push(Box::new(event));
    }

    /// Merge the current system buffer into stage events.
    pub fn end_system(&mut self) {
        for (type_id, mut events) in self.system_buffer.drain() {
            self.stage_buffer
                .entry(type_id)
                .or_default()
                .append(&mut events);
        }
    }

    /// Commit stage events for read access.
    pub fn end_stage(&mut self) {
        std::mem::swap(&mut self.committed, &mut self.stage_buffer);
        self.stage_buffer.clear();
    }

    /// Iterate committed events for a concrete event type without allocation.
    pub fn iter<T: 'static + Send + Sync>(&self) -> impl Iterator<Item = &T> {
        self.committed
            .get(&TypeId::of::<T>())
            .into_iter()
            .flat_map(|events| events.iter().filter_map(|event| event.downcast_ref::<T>()))
    }

    /// Visit committed events for a concrete event type without allocation.
    pub fn for_each<T: 'static + Send + Sync>(&self, mut f: impl FnMut(&T)) {
        for event in self.iter::<T>() {
            f(event);
        }
    }

    /// Read committed events for a concrete event type.
    pub fn read<T: 'static + Send + Sync>(&self) -> Vec<&T> {
        self.iter::<T>().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct HitEvent(u32);

    #[test]
    fn merge_keeps_system_registration_order() {
        let mut events = EventStore::new();

        events.begin_stage();

        events.begin_system();
        events.emit(HitEvent(1));
        events.end_system();

        events.begin_system();
        events.emit(HitEvent(2));
        events.end_system();

        events.end_stage();

        let values: Vec<u32> = events.read::<HitEvent>().into_iter().map(|e| e.0).collect();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn begin_stage_resets_previous_committed_events() {
        let mut events = EventStore::new();

        events.begin_stage();
        events.begin_system();
        events.emit(HitEvent(10));
        events.end_system();
        events.end_stage();

        events.begin_stage();
        events.begin_system();
        events.emit(HitEvent(20));
        events.end_system();
        events.end_stage();

        let values: Vec<u32> = events.read::<HitEvent>().into_iter().map(|e| e.0).collect();
        assert_eq!(values, vec![20]);
    }

    #[test]
    fn read_missing_event_type_returns_empty() {
        let events = EventStore::new();
        let values: Vec<u32> = events.read::<HitEvent>().into_iter().map(|e| e.0).collect();
        assert!(values.is_empty());
    }

    #[test]
    fn for_each_reads_events_in_order_without_allocating() {
        let mut events = EventStore::new();

        events.begin_stage();
        events.begin_system();
        events.emit(HitEvent(7));
        events.emit(HitEvent(8));
        events.end_system();
        events.end_stage();

        let mut values = Vec::new();
        events.for_each::<HitEvent>(|event| values.push(event.0));
        assert_eq!(values, vec![7, 8]);
    }
}
