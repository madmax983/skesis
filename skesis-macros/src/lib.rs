//! Proc macros for the skesis ECS.
//!
//! Provides the `#[system]` attribute macro for automatic parameter extraction.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, FnArg, ItemFn, Pat, PatType, Type};

/// Attribute macro that transforms a function with typed system parameters
/// into a skesis system with automatic borrow splitting.
///
/// # Supported Parameters
///
/// | Type | Access |
/// |------|--------|
/// | `Res<T>` | Immutable resource |
/// | `ResMut<T>` | Mutable resource |
/// | `Query<Q>` | Component iteration |
/// | `Commands` | Deferred changes |
/// | `Local<T>` | Per-system state |
/// | `EventReader<E>` | Read events |
/// | `EventWriter<E>` | Write events |
///
/// # Example
///
/// ```ignore
/// use skesis::prelude::*;
///
/// #[system]
/// fn physics(mut query: Query<(&Position, &mut Velocity)>, time: Res<DeltaTime>) {
///     for (_entity, (pos, vel)) in &mut query {
///         vel.x += pos.x * time.dt;
///     }
/// }
/// ```
///
/// # Generated Code
///
/// For each `#[system]` function, the macro generates:
/// - A `_access()` function returning `SystemAccess`
/// - A `_init()` function returning typed state
/// - A `_run()` function that extracts params from `UnsafeWorldCell` and calls the body
/// - A `_descriptor()` function returning a `SystemDescriptor` with access metadata
#[proc_macro_attribute]
pub fn system(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let vis = &input.vis;
    let fn_name = &input.sig.ident;
    let fn_body = &input.block;
    let fn_attrs = &input.attrs;

    // Parse parameters
    let params: Vec<ParamInfo> = input
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, ty, .. }) = arg {
                let name = if let Pat::Ident(ident) = pat.as_ref() {
                    ident.ident.clone()
                } else {
                    return None;
                };
                Some(ParamInfo {
                    name,
                    ty: ty.as_ref().clone(),
                    is_mut: matches!(pat.as_ref(), Pat::Ident(pi) if pi.mutability.is_some()),
                })
            } else {
                None
            }
        })
        .collect();

    // Generate the inner function with original signature
    let inner_name = format_ident!("{}_inner", fn_name);
    let inner_params: Vec<_> = input.sig.inputs.iter().collect();

    // Generate access declarations
    let access_stmts: Vec<_> = params
        .iter()
        .map(|p| {
            let ty = &p.ty;
            if is_query_type(ty) {
                // For Query<Q>, we need to call WorldQuery::access on Q
                let inner_ty = extract_query_inner_type(ty);
                quote! {
                    <#inner_ty as ::skesis::WorldQuery>::access(&mut access);
                }
            } else {
                quote! {
                    <#ty as ::skesis::SystemParam>::access(&mut access);
                }
            }
        })
        .collect();

    // Generate state types and init expressions
    let state_types: Vec<_> = params
        .iter()
        .map(|p| {
            let ty = &p.ty;
            if is_query_type(ty) {
                quote! { () } // Query uses no persistent state for now
            } else {
                quote! { <#ty as ::skesis::SystemParam>::State }
            }
        })
        .collect();

    let state_inits: Vec<_> = params
        .iter()
        .map(|p| {
            let ty = &p.ty;
            if is_query_type(ty) {
                quote! { () }
            } else {
                quote! { <#ty as ::skesis::SystemParam>::init(world) }
            }
        })
        .collect();

    // Generate parameter extraction in the runner
    let param_extracts: Vec<_> = params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let name = &p.name;
            let ty = &p.ty;
            let idx = syn::Index::from(i);
            let mutability = if p.is_mut { quote! { mut } } else { quote! {} };

            if is_query_type(ty) {
                quote! {
                    let #mutability #name: #ty = unsafe { ::skesis::query_from_cell(cell) };
                }
            } else {
                quote! {
                    let #mutability #name = unsafe {
                        <#ty as ::skesis::SystemParam>::get(cell, &mut state_tuple.#idx)
                    };
                }
            }
        })
        .collect();

    // Collect parameter names for the inner function call
    let param_names: Vec<_> = params.iter().map(|p| &p.name).collect();

    // Generate the runner function name
    let run_name = format_ident!("{}_run", fn_name);
    let access_name = format_ident!("{}_access", fn_name);
    let init_name = format_ident!("{}_init", fn_name);
    let descriptor_name = format_ident!("{}_descriptor", fn_name);

    // Assemble output — inline the body directly in the runner
    // to avoid lifetime propagation issues with separate inner functions
    let output = quote! {
        /// System access declaration (auto-generated by #[system]).
        #vis fn #access_name() -> ::skesis::SystemAccess {
            let mut access = ::skesis::SystemAccess::default();
            #(#access_stmts)*
            access
        }

        /// System state initialization (auto-generated by #[system]).
        #vis fn #init_name(world: &mut ::skesis::World) -> ::skesis::SystemState {
            ::skesis::SystemState::new((
                #(#state_inits,)*
            ))
        }

        /// System runner (auto-generated by #[system]).
        ///
        /// Creates an `UnsafeWorldCell`, extracts each parameter, and executes
        /// the system body inline. This is the function pointer stored in the scheduler.
        #(#fn_attrs)*
        #vis fn #run_name(world: &mut ::skesis::World, state: &mut ::skesis::SystemState) {
            let state_tuple = state.get_mut::<(#(#state_types,)*)>();

            let cell = unsafe { ::skesis::UnsafeWorldCell::new(world) };
            #(#param_extracts)*

            // Original function body inlined here
            #fn_body
        }
    };

    output.into()
}

/// Parsed parameter info
struct ParamInfo {
    name: syn::Ident,
    ty: Type,
    is_mut: bool,
}

/// Check if a type path looks like `Query<...>`
fn is_query_type(ty: &Type) -> bool {
    if let Type::Path(path) = ty {
        path.path
            .segments
            .last()
            .is_some_and(|seg| seg.ident == "Query")
    } else {
        false
    }
}

/// Extract the inner type Q from Query<Q>
fn extract_query_inner_type(ty: &Type) -> proc_macro2::TokenStream {
    if let Type::Path(path) = ty {
        if let Some(seg) = path.path.segments.last() {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                    return quote! { #inner };
                }
            }
        }
    }
    quote! { () }
}
