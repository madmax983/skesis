//! Plugin system for extensibility.

use crate::App;

/// Plugin trait for extending the engine.
pub trait Plugin {
    /// Build the plugin, adding systems and resources to the app.
    fn build(&self, app: &mut App);
}

/// Execution stage for systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Runs once at startup.
    Startup,
    /// Before main update (input, events).
    PreUpdate,
    /// Main game logic.
    Update,
    /// After logic (physics resolution, camera).
    PostUpdate,
    /// Rendering.
    Render,
    /// Cleanup.
    Shutdown,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn build(&self, _app: &mut App) {
            // Plugin builds successfully
        }
    }

    #[test]
    fn plugin_trait_compiles() {
        let plugin = TestPlugin;
        let mut app = App::new();

        plugin.build(&mut app);
    }
}
