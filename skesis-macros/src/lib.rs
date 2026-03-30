//! Proc macros for the skesis ECS.
//!
//! Provides the `#[system]` attribute macro for automatic parameter extraction.

use proc_macro::TokenStream;

/// Attribute macro that transforms a function with typed system parameters
/// into a skesis system with automatic borrow splitting.
///
/// # Example
///
/// ```ignore
/// use skesis::prelude::*;
///
/// #[system]
/// fn my_system(scene: ResMut<Scene>, query: Query<(&Position, &mut Velocity)>) {
///     for (entity, (pos, vel)) in &mut query {
///         // simultaneous resource + component access — zero overhead
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn system(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // TODO: implement parameter extraction codegen
    item
}
