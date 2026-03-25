//! Deferred structural command buffering.

use crate::world::World;
use crate::{Component, Entity};

/// Type-erased deferred command.
pub(crate) type DeferredCommand = Box<dyn FnOnce(&mut World) + Send + Sync + 'static>;

/// Thread-local recorder for deferred world commands.
///
/// Commands are recorded without touching shared world state, then merged later.
#[derive(Default)]
pub struct CommandRecorder {
    commands: Vec<DeferredCommand>,
}

impl CommandRecorder {
    /// Create an empty command recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Defer one command to run during stage command flush.
    pub fn defer_command(&mut self, command: impl FnOnce(&mut World) + Send + Sync + 'static) {
        self.commands.push(Box::new(command));
    }

    /// Defer adding or replacing a component on an entity.
    pub fn defer_add_component<T: Component>(&mut self, entity: Entity, component: T) {
        self.defer_command(move |world| {
            world.add_component(entity, component);
        });
    }

    /// Defer setting a component value with observer notification.
    pub fn defer_set_component<T: Component>(&mut self, entity: Entity, component: T) {
        self.defer_command(move |world| {
            world.set_component(entity, component);
        });
    }

    /// Defer removing a component from an entity (value is dropped).
    pub fn defer_remove_component<T: Component>(&mut self, entity: Entity) {
        self.defer_command(move |world| {
            let _ = world.remove_component::<T>(entity);
        });
    }

    /// Defer despawning an entity.
    pub fn defer_despawn(&mut self, entity: Entity) {
        self.defer_command(move |world| {
            let _ = world.despawn(entity);
        });
    }

    /// Defer spawning one empty entity.
    pub fn defer_spawn_empty(&mut self) {
        self.defer_command(|world| {
            let _ = world.spawn_empty();
        });
    }

    pub(crate) fn into_commands(self) -> Vec<DeferredCommand> {
        self.commands
    }
}

/// Per-stage deferred command store with deterministic insertion order.
#[derive(Default)]
pub(crate) struct CommandStore {
    stage_buffer: Vec<DeferredCommand>,
    system_buffer: Vec<DeferredCommand>,
    collecting_system: bool,
}

impl CommandStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Begin a new stage command scope.
    pub(crate) fn begin_stage(&mut self) {
        self.stage_buffer.clear();
        self.system_buffer.clear();
        self.collecting_system = false;
    }

    /// Begin buffering commands for one system execution.
    pub(crate) fn begin_system(&mut self) {
        self.system_buffer.clear();
        self.collecting_system = true;
    }

    /// Defer one command into the active buffer.
    pub(crate) fn defer(&mut self, command: DeferredCommand) {
        if self.collecting_system {
            self.system_buffer.push(command);
        } else {
            self.stage_buffer.push(command);
        }
    }

    /// Merge the current system buffer into stage commands.
    pub(crate) fn end_system(&mut self) {
        self.collecting_system = false;
        self.stage_buffer.append(&mut self.system_buffer);
    }

    /// Merge thread-local recorded commands into stage commands.
    pub(crate) fn extend_stage(&mut self, mut commands: Vec<DeferredCommand>) {
        self.stage_buffer.append(&mut commands);
    }

    /// Drain stage commands in deterministic insertion order.
    pub(crate) fn take_stage(&mut self) -> Vec<DeferredCommand> {
        std::mem::take(&mut self.stage_buffer)
    }
}
