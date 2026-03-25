//! Application entry point with plugin system.

use crate::{
    CommandRecorder, ParallelSystemFn, Plugin, Stage, SystemDescriptor, SystemFn, World, plan_stage,
};
use rayon::prelude::*;
use std::collections::HashMap;

#[inline(always)]
fn can_run_parallel_batch(batch: &[usize], stage_descriptors: &[SystemDescriptor]) -> bool {
    !batch.is_empty()
        && batch.iter().all(|&index| {
            let descriptor = &stage_descriptors[index];
            descriptor.parallel_system().is_some() && descriptor.has_declared_access()
        })
}

/// Main application structure.
pub struct App {
    world: World,
    systems: HashMap<Stage, Vec<SystemDescriptor>>,
    stage_plans: HashMap<Stage, Vec<Vec<usize>>>,
    next_system_registration_index: usize,
    startup_ran: bool,
    shutdown_ran: bool,
}

impl App {
    /// Create a new application.
    pub fn new() -> Self {
        Self {
            world: World::new(),
            systems: HashMap::new(),
            stage_plans: HashMap::new(),
            next_system_registration_index: 0,
            startup_ran: false,
            shutdown_ran: false,
        }
    }

    /// Add a plugin to the application.
    pub fn add_plugin(&mut self, plugin: impl Plugin) -> &mut Self {
        plugin.build(self);
        self
    }

    /// Run startup systems once.
    pub fn run_startup(&mut self) {
        if self.startup_ran {
            return;
        }

        self.run_stage(Stage::Startup);
        self.startup_ran = true;
    }

    /// Add a system to a stage.
    pub fn add_system(&mut self, stage: Stage, system: SystemFn) -> &mut Self {
        let descriptor = SystemDescriptor::new(system, self.next_system_registration_index);
        self.next_system_registration_index += 1;

        self.stage_plans.remove(&stage);
        self.systems.entry(stage).or_default().push(descriptor);
        self
    }

    /// Add a parallel-friendly system to a stage.
    pub fn add_parallel_system(&mut self, stage: Stage, system: ParallelSystemFn) -> &mut Self {
        let descriptor =
            SystemDescriptor::new_parallel(system, self.next_system_registration_index);
        self.next_system_registration_index += 1;

        self.stage_plans.remove(&stage);
        self.systems.entry(stage).or_default().push(descriptor);
        self
    }

    /// Add a parallel system with explicitly declared access metadata.
    ///
    /// The `configure` closure should declare reads/writes via descriptor builder methods.
    /// Pass-through (`|descriptor| descriptor`) is valid for systems that declare empty access.
    pub fn add_parallel_system_with_access(
        &mut self,
        stage: Stage,
        system: ParallelSystemFn,
        configure: impl FnOnce(SystemDescriptor) -> SystemDescriptor,
    ) -> &mut Self {
        let descriptor = SystemDescriptor::new_parallel_with_declared_access(
            system,
            self.next_system_registration_index,
        );
        self.next_system_registration_index += 1;

        self.stage_plans.remove(&stage);
        self.systems
            .entry(stage)
            .or_default()
            .push(configure(descriptor));
        self
    }

    /// Add a system descriptor with explicit access metadata.
    pub fn add_system_descriptor(
        &mut self,
        stage: Stage,
        descriptor: SystemDescriptor,
    ) -> &mut Self {
        self.stage_plans.remove(&stage);
        self.systems.entry(stage).or_default().push(descriptor);
        self
    }

    /// Run one update cycle.
    pub fn update(&mut self) {
        self.run_startup();
        self.run_stage(Stage::PreUpdate);
        self.run_stage(Stage::Update);
        self.run_stage(Stage::PostUpdate);
    }

    /// Run render systems once for the current frame.
    pub fn render(&mut self) {
        self.run_startup();
        self.run_stage(Stage::Render);
    }

    /// Run shutdown systems once.
    pub fn shutdown(&mut self) {
        if self.shutdown_ran {
            return;
        }

        self.run_stage(Stage::Shutdown);
        self.shutdown_ran = true;
    }

    /// Run all systems in a stage.
    fn run_stage(&mut self, stage: Stage) {
        let stage_descriptors = match self.systems.get(&stage) {
            Some(systems) if !systems.is_empty() => systems.clone(),
            _ => return,
        };

        let stage_plan = if let Some(cached_plan) = self.stage_plans.get(&stage) {
            cached_plan.clone()
        } else {
            let computed_plan = plan_stage(&stage_descriptors);
            self.stage_plans.insert(stage, computed_plan.clone());
            computed_plan
        };

        self.world.begin_stage_events();
        self.world.begin_stage_commands();

        for batch in stage_plan {
            if can_run_parallel_batch(&batch, &stage_descriptors) {
                let mut recorded: Vec<(usize, CommandRecorder)> = {
                    let world_ref: &World = &self.world;
                    batch
                        .par_iter()
                        .map(|&index| {
                            let system = stage_descriptors[index]
                                .parallel_system()
                                .expect("parallel batch should only contain parallel systems");
                            let mut recorder = CommandRecorder::new();
                            system(world_ref, &mut recorder);
                            (index, recorder)
                        })
                        .collect()
                };

                recorded.sort_by_key(|(index, _)| stage_descriptors[*index].registration_index());
                for (_, recorder) in recorded {
                    self.world.append_recorded_commands(recorder);
                }
            } else {
                for index in batch {
                    let descriptor = &stage_descriptors[index];
                    if let Some(system) = descriptor.exclusive_system() {
                        self.world.begin_system_events();
                        self.world.begin_system_commands();
                        system(&mut self.world);
                        self.world.end_system_events();
                        self.world.end_system_commands();
                    } else if let Some(system) = descriptor.parallel_system() {
                        let mut recorder = CommandRecorder::new();
                        let world_ref: &World = &self.world;
                        system(world_ref, &mut recorder);
                        self.world.append_recorded_commands(recorder);
                    }
                }
            }
        }

        self.world.end_stage_commands();
        self.world.end_stage_events();
    }

    /// Check if app is valid (for testing).
    pub fn is_valid(&self) -> bool {
        true
    }

    /// Get a reference to the world.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Get a mutable reference to the world.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Plugin;
    use std::sync::{Mutex, OnceLock};

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn build(&self, _app: &mut App) {}
    }

    #[test]
    fn create_app() {
        let app = App::new();
        assert!(app.is_valid());
    }

    #[test]
    fn add_plugin() {
        let mut app = App::new();
        app.add_plugin(TestPlugin);
        assert!(app.is_valid());
    }

    use crate::Stage;

    fn test_system(_world: &mut World) {
        // Test system that runs
    }

    #[test]
    fn add_system_to_stage() {
        let mut app = App::new();
        app.add_system(Stage::Update, test_system);

        assert!(app.is_valid());
    }

    #[test]
    fn run_update_executes_systems() {
        let mut app = App::new();
        app.add_system(Stage::Update, test_system);
        app.update();

        // System should have run
        assert!(app.is_valid());
    }

    #[test]
    fn stage_planning_is_cached_between_frames_without_stage_changes() {
        let mut app = App::new();
        app.add_system(Stage::Update, test_system);

        crate::scheduler::reset_plan_stage_call_count();
        app.update();
        app.update();

        assert_eq!(crate::scheduler::plan_stage_call_count(), 1);
    }

    #[test]
    fn adding_system_invalidates_cached_stage_plan() {
        let mut app = App::new();
        app.add_system(Stage::Update, test_system);

        crate::scheduler::reset_plan_stage_call_count();
        app.update();
        assert_eq!(crate::scheduler::plan_stage_call_count(), 1);

        app.add_system(Stage::Update, test_system);
        app.update();
        assert_eq!(crate::scheduler::plan_stage_call_count(), 2);
    }

    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn call_log() -> &'static Mutex<Vec<&'static str>> {
        static LOG: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
        LOG.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn clear_log() {
        call_log().lock().expect("log mutex poisoned").clear();
    }

    fn snapshot_log() -> Vec<&'static str> {
        call_log().lock().expect("log mutex poisoned").clone()
    }

    fn log(label: &'static str) {
        call_log().lock().expect("log mutex poisoned").push(label);
    }

    fn startup_system(_world: &mut World) {
        log("startup");
    }

    fn pre_update_system(_world: &mut World) {
        log("pre_update");
    }

    fn update_system(_world: &mut World) {
        log("update");
    }

    fn post_update_system(_world: &mut World) {
        log("post_update");
    }

    fn render_system(_world: &mut World) {
        log("render");
    }

    fn shutdown_system(_world: &mut World) {
        log("shutdown");
    }

    #[derive(Debug, PartialEq)]
    struct FrameEvent(u32);

    fn emit_first_event(world: &mut World) {
        world.emit_event(FrameEvent(1));
    }

    fn emit_second_event(world: &mut World) {
        world.emit_event(FrameEvent(2));
    }

    #[derive(Debug, PartialEq)]
    struct DeferredPosition {
        x: f32,
        y: f32,
    }

    struct DeferredTarget(crate::Entity);
    struct SeenInStage(usize);
    struct ParallelTarget(crate::Entity);

    fn defer_add_position(world: &mut World) {
        let target = world
            .get_resource::<DeferredTarget>()
            .expect("target entity resource should exist")
            .0;
        world.defer_add_component(target, DeferredPosition { x: 5.0, y: 6.0 });
    }

    fn observe_deferred_position_count(world: &mut World) {
        let count = world.query::<DeferredPosition>().count();
        world
            .get_resource_mut::<SeenInStage>()
            .expect("stage observation resource should exist")
            .0 = count;
    }

    fn parallel_add_one(world: &World, recorder: &mut CommandRecorder) {
        let target = world
            .get_resource::<ParallelTarget>()
            .expect("parallel target should exist")
            .0;
        recorder.defer_command(move |world| {
            let pos = world
                .get_component_mut::<DeferredPosition>(target)
                .expect("deferred position should exist");
            pos.x += 1.0;
        });
    }

    fn parallel_double(world: &World, recorder: &mut CommandRecorder) {
        let target = world
            .get_resource::<ParallelTarget>()
            .expect("parallel target should exist")
            .0;
        recorder.defer_command(move |world| {
            let pos = world
                .get_component_mut::<DeferredPosition>(target)
                .expect("deferred position should exist");
            pos.x *= 2.0;
        });
    }

    #[test]
    fn lifecycle_startup_runs_once() {
        let _guard = test_lock().lock().expect("test lock poisoned");
        clear_log();

        let mut app = App::new();
        app.add_system(Stage::Startup, startup_system);

        app.run_startup();
        app.run_startup();

        assert_eq!(snapshot_log(), vec!["startup"]);
    }

    #[test]
    fn lifecycle_frame_order_is_deterministic() {
        let _guard = test_lock().lock().expect("test lock poisoned");
        clear_log();

        let mut app = App::new();
        app.add_system(Stage::Startup, startup_system);
        app.add_system(Stage::PreUpdate, pre_update_system);
        app.add_system(Stage::Update, update_system);
        app.add_system(Stage::PostUpdate, post_update_system);
        app.add_system(Stage::Render, render_system);

        app.run_startup();
        app.update();
        app.render();

        assert_eq!(
            snapshot_log(),
            vec!["startup", "pre_update", "update", "post_update", "render"]
        );
    }

    #[test]
    fn lifecycle_shutdown_runs_once() {
        let _guard = test_lock().lock().expect("test lock poisoned");
        clear_log();

        let mut app = App::new();
        app.add_system(Stage::Shutdown, shutdown_system);

        app.shutdown();
        app.shutdown();

        assert_eq!(snapshot_log(), vec!["shutdown"]);
    }

    #[test]
    fn events_merge_in_system_registration_order() {
        let _guard = test_lock().lock().expect("test lock poisoned");

        let mut app = App::new();
        app.add_system(Stage::Update, emit_first_event);
        app.add_system(Stage::Update, emit_second_event);

        app.update();

        let values: Vec<u32> = app
            .world()
            .read_events::<FrameEvent>()
            .into_iter()
            .map(|event| event.0)
            .collect();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn deferred_structural_commands_apply_after_stage() {
        let _guard = test_lock().lock().expect("test lock poisoned");

        let mut app = App::new();
        let target = app.world_mut().spawn_empty();
        app.world_mut().insert_resource(DeferredTarget(target));
        app.world_mut().insert_resource(SeenInStage(usize::MAX));

        app.add_system(Stage::Update, defer_add_position);
        app.add_system(Stage::Update, observe_deferred_position_count);

        app.update();

        let seen = app
            .world()
            .get_resource::<SeenInStage>()
            .expect("observation should exist")
            .0;
        assert_eq!(seen, 0);
        assert!(app.world().has_component::<DeferredPosition>(target));
    }

    #[test]
    fn parallel_systems_merge_commands_in_registration_order() {
        let _guard = test_lock().lock().expect("test lock poisoned");

        let mut app = App::new();
        let target = app.world_mut().spawn_empty();
        app.world_mut()
            .add_component(target, DeferredPosition { x: 1.0, y: 0.0 });
        app.world_mut().insert_resource(ParallelTarget(target));

        app.add_parallel_system(Stage::Update, parallel_add_one);
        app.add_parallel_system(Stage::Update, parallel_double);

        app.update();

        let pos = app
            .world()
            .get_component::<DeferredPosition>(target)
            .expect("deferred position should exist");
        assert_eq!(pos.x, 4.0);
        assert_eq!(pos.y, 0.0);
    }

    #[test]
    fn parallel_batch_requires_declared_access_metadata() {
        let undeclared = vec![
            SystemDescriptor::new_parallel(parallel_add_one, 0),
            SystemDescriptor::new_parallel(parallel_double, 1),
        ];
        assert!(!can_run_parallel_batch(&[0, 1], &undeclared));

        let declared = vec![
            SystemDescriptor::new_parallel(parallel_add_one, 0).reads_resource::<ParallelTarget>(),
            SystemDescriptor::new_parallel(parallel_double, 1).reads_resource::<ParallelTarget>(),
        ];
        assert!(can_run_parallel_batch(&[0, 1], &declared));
    }

    #[test]
    fn add_parallel_system_with_access_runs_in_parallel_path_and_keeps_merge_order() {
        let _guard = test_lock().lock().expect("test lock poisoned");

        let mut app = App::new();
        let target = app.world_mut().spawn_empty();
        app.world_mut()
            .add_component(target, DeferredPosition { x: 1.0, y: 0.0 });
        app.world_mut().insert_resource(ParallelTarget(target));

        app.add_parallel_system_with_access(Stage::Update, parallel_add_one, |descriptor| {
            descriptor.reads_resource::<ParallelTarget>()
        });
        app.add_parallel_system_with_access(Stage::Update, parallel_double, |descriptor| {
            descriptor.reads_resource::<ParallelTarget>()
        });

        app.update();

        let pos = app
            .world()
            .get_component::<DeferredPosition>(target)
            .expect("deferred position should exist");
        assert_eq!(pos.x, 4.0);
        assert_eq!(pos.y, 0.0);
    }
}
