//! System execution.

use crate::{CommandRecorder, World};
use std::any::TypeId;
use std::collections::BTreeSet;

/// A system function that operates on the world.
pub type SystemFn = fn(&mut World);

/// A parallel-friendly system function with read-only world access and a local command recorder.
pub type ParallelSystemFn = fn(&World, &mut CommandRecorder);

#[derive(Clone, Copy)]
enum SystemKind {
    Exclusive(SystemFn),
    Parallel(ParallelSystemFn),
}

/// Declared access pattern for a system.
#[derive(Debug, Clone, Default)]
pub struct SystemAccess {
    component_reads: BTreeSet<TypeId>,
    component_writes: BTreeSet<TypeId>,
    resource_reads: BTreeSet<TypeId>,
    resource_writes: BTreeSet<TypeId>,
}

impl SystemAccess {
    /// Add a component read.
    pub fn reads_component<T: 'static>(&mut self) {
        self.component_reads.insert(TypeId::of::<T>());
    }

    /// Add a component write.
    pub fn writes_component<T: 'static>(&mut self) {
        self.component_writes.insert(TypeId::of::<T>());
    }

    /// Add a resource read.
    pub fn reads_resource<T: 'static>(&mut self) {
        self.resource_reads.insert(TypeId::of::<T>());
    }

    /// Add a resource write.
    pub fn writes_resource<T: 'static>(&mut self) {
        self.resource_writes.insert(TypeId::of::<T>());
    }

    /// Check if this access set conflicts with another.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        overlaps(&self.component_writes, &other.component_writes)
            || overlaps(&self.component_writes, &other.component_reads)
            || overlaps(&self.component_reads, &other.component_writes)
            || overlaps(&self.resource_writes, &other.resource_writes)
            || overlaps(&self.resource_writes, &other.resource_reads)
            || overlaps(&self.resource_reads, &other.resource_writes)
    }
}

fn overlaps(lhs: &BTreeSet<TypeId>, rhs: &BTreeSet<TypeId>) -> bool {
    lhs.iter().any(|type_id| rhs.contains(type_id))
}

/// System plus metadata required by the scheduler.
#[derive(Clone)]
pub struct SystemDescriptor {
    system: SystemKind,
    access: SystemAccess,
    access_declared: bool,
    registration_index: usize,
    /// This system must run before systems matching these function pointers.
    before_constraints: Vec<usize>,
    /// This system must run after systems matching these function pointers.
    after_constraints: Vec<usize>,
}

impl SystemDescriptor {
    /// Create a system descriptor with empty access metadata.
    pub fn new(system: SystemFn, registration_index: usize) -> Self {
        Self {
            system: SystemKind::Exclusive(system),
            access: SystemAccess::default(),
            access_declared: false,
            registration_index,
            before_constraints: Vec::new(),
            after_constraints: Vec::new(),
        }
    }

    /// Create a parallel system descriptor with empty access metadata.
    pub fn new_parallel(system: ParallelSystemFn, registration_index: usize) -> Self {
        Self {
            system: SystemKind::Parallel(system),
            access: SystemAccess::default(),
            access_declared: false,
            registration_index,
            before_constraints: Vec::new(),
            after_constraints: Vec::new(),
        }
    }

    /// Create a parallel system descriptor with explicitly declared access metadata.
    pub(crate) fn new_parallel_with_declared_access(
        system: ParallelSystemFn,
        registration_index: usize,
    ) -> Self {
        Self {
            system: SystemKind::Parallel(system),
            access: SystemAccess::default(),
            access_declared: true,
            registration_index,
            before_constraints: Vec::new(),
            after_constraints: Vec::new(),
        }
    }

    /// Get the system function.
    pub fn system(&self) -> SystemFn {
        self.exclusive_system()
            .expect("system() is only valid for exclusive systems")
    }

    /// Get the exclusive system function if this descriptor is exclusive.
    pub fn exclusive_system(&self) -> Option<SystemFn> {
        match self.system {
            SystemKind::Exclusive(system) => Some(system),
            SystemKind::Parallel(_) => None,
        }
    }

    /// Get the parallel system function if this descriptor is parallel.
    pub fn parallel_system(&self) -> Option<ParallelSystemFn> {
        match self.system {
            SystemKind::Parallel(system) => Some(system),
            SystemKind::Exclusive(_) => None,
        }
    }

    /// Get the access metadata.
    pub fn access(&self) -> &SystemAccess {
        &self.access
    }

    /// Whether access metadata was explicitly declared for this system.
    pub fn has_declared_access(&self) -> bool {
        self.access_declared
    }

    /// Registration order index.
    pub fn registration_index(&self) -> usize {
        self.registration_index
    }

    /// Declare a component read for this system.
    pub fn reads_component<T: 'static>(mut self) -> Self {
        self.access.reads_component::<T>();
        self.access_declared = true;
        self
    }

    /// Declare a component write for this system.
    pub fn writes_component<T: 'static>(mut self) -> Self {
        self.access.writes_component::<T>();
        self.access_declared = true;
        self
    }

    /// Declare a resource read for this system.
    pub fn reads_resource<T: 'static>(mut self) -> Self {
        self.access.reads_resource::<T>();
        self.access_declared = true;
        self
    }

    /// Declare a resource write for this system.
    pub fn writes_resource<T: 'static>(mut self) -> Self {
        self.access.writes_resource::<T>();
        self.access_declared = true;
        self
    }

    /// Declare that this system must run before the given system.
    pub fn before(mut self, other: SystemFn) -> Self {
        self.before_constraints.push(other as usize);
        self
    }

    /// Declare that this system must run after the given system.
    pub fn after(mut self, other: SystemFn) -> Self {
        self.after_constraints.push(other as usize);
        self
    }

    /// Get the function pointer identity of this system (for ordering lookups).
    pub(crate) fn fn_id(&self) -> usize {
        match self.system {
            SystemKind::Exclusive(f) => f as usize,
            SystemKind::Parallel(f) => f as usize,
        }
    }

    /// Get before constraints (function pointer identities this must run before).
    pub(crate) fn before_constraints(&self) -> &[usize] {
        &self.before_constraints
    }

    /// Get after constraints (function pointer identities this must run after).
    pub(crate) fn after_constraints(&self) -> &[usize] {
        &self.after_constraints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;

    fn test_system(_world: &mut World) {
        // System runs successfully
    }

    #[test]
    fn system_function_signature() {
        let mut world = World::new();
        let system: SystemFn = test_system;

        system(&mut world);
    }

    #[test]
    fn access_conflict_detection_works() {
        struct Position;

        let a = SystemDescriptor::new(test_system, 0).writes_component::<Position>();
        let b = SystemDescriptor::new(test_system, 1).reads_component::<Position>();
        assert!(a.access().conflicts_with(b.access()));
    }

    fn parallel_test_system(_world: &World, _commands: &mut CommandRecorder) {}

    #[test]
    fn declared_access_flag_tracks_metadata_source() {
        struct Position;

        let undeclared = SystemDescriptor::new_parallel(parallel_test_system, 0);
        assert!(!undeclared.has_declared_access());

        let declared_via_builder =
            SystemDescriptor::new_parallel(parallel_test_system, 1).reads_component::<Position>();
        assert!(declared_via_builder.has_declared_access());

        let declared_empty =
            SystemDescriptor::new_parallel_with_declared_access(parallel_test_system, 2);
        assert!(declared_empty.has_declared_access());
    }
}
