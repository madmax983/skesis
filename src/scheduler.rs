//! System scheduling and conflict planning.

use crate::SystemDescriptor;
#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static PLAN_STAGE_CALL_COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Build deterministic execution batches for one stage.
///
/// Each batch contains indices into the input `systems` slice.
/// Systems are first sorted respecting `before`/`after` constraints
/// (topological sort), then grouped into batches where systems within
/// a batch are parallel-compatible according to declared accesses.
pub fn plan_stage(systems: &[SystemDescriptor]) -> Vec<Vec<usize>> {
    #[cfg(test)]
    PLAN_STAGE_CALL_COUNT.with(|count| count.set(count.get().wrapping_add(1)));

    let ordered = topological_sort(systems);

    let mut batches = Vec::new();
    let mut current_batch: Vec<usize> = Vec::new();

    for &index in &ordered {
        let system = &systems[index];

        // Check access conflicts with current batch.
        let conflicts_with_batch = current_batch.iter().any(|existing_index| {
            systems[*existing_index]
                .access()
                .conflicts_with(system.access())
        });

        // Check ordering constraints: if this system must run after any
        // system in the current batch, it must go in a later batch.
        let ordering_conflict = current_batch.iter().any(|&existing_index| {
            let existing = &systems[existing_index];
            // This system has an "after" constraint on existing system.
            system
                .after_constraints()
                .iter()
                .any(|&fn_id| fn_id == existing.fn_id())
            // Or existing system has a "before" constraint naming this system.
            || existing
                .before_constraints()
                .iter()
                .any(|&fn_id| fn_id == system.fn_id())
        });

        if (conflicts_with_batch || ordering_conflict) && !current_batch.is_empty() {
            batches.push(current_batch);
            current_batch = Vec::new();
        }

        current_batch.push(index);
    }

    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    batches
}

/// Topological sort of systems respecting before/after constraints.
///
/// Falls back to registration order when no constraints are present.
/// Panics on cycles (which represent unsatisfiable constraints).
fn topological_sort(systems: &[SystemDescriptor]) -> Vec<usize> {
    use std::collections::{HashMap, VecDeque};

    let n = systems.len();
    if n == 0 {
        return Vec::new();
    }

    // Build adjacency: edges[i] contains indices that must come after i.
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree = vec![0u32; n];

    // Build fn_id → index lookup (O(1) amortized instead of O(n) linear scan).
    let fn_id_to_index: HashMap<usize, usize> = systems
        .iter()
        .enumerate()
        .map(|(i, s)| (s.fn_id(), i))
        .collect();

    for (i, system) in systems.iter().enumerate() {
        // "i.before(other)" means i must run before other → edge i → other.
        for &fn_id in system.before_constraints() {
            if let Some(&j) = fn_id_to_index.get(&fn_id) {
                edges[i].push(j);
                in_degree[j] += 1;
            }
        }
        // "i.after(other)" means other must run before i → edge other → i.
        for &fn_id in system.after_constraints() {
            if let Some(&j) = fn_id_to_index.get(&fn_id) {
                edges[j].push(i);
                in_degree[i] += 1;
            }
        }
    }

    // Kahn's algorithm with registration-order tiebreaking.
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    // Sort by registration index for deterministic output.
    queue.make_contiguous().sort_by_key(|&i| systems[i].registration_index());

    let mut result = Vec::with_capacity(n);

    while let Some(node) = queue.pop_front() {
        result.push(node);

        for &neighbor in &edges[node] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                // Insert sorted by registration index for determinism.
                let reg_idx = systems[neighbor].registration_index();
                let pos = queue.make_contiguous()
                    .binary_search_by_key(&reg_idx, |&i| systems[i].registration_index())
                    .unwrap_or_else(|pos| pos);
                queue.insert(pos, neighbor);
            }
        }
    }

    assert_eq!(
        result.len(),
        n,
        "cycle detected in system ordering constraints"
    );
    result
}

#[cfg(test)]
pub(crate) fn reset_plan_stage_call_count() {
    PLAN_STAGE_CALL_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn plan_stage_call_count() -> usize {
    PLAN_STAGE_CALL_COUNT.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SystemDescriptor, World};

    struct Position;
    struct Velocity;
    struct Health;
    struct AiState;
    struct TimeRes;

    fn sys_a(_world: &mut World) {}
    fn sys_b(_world: &mut World) {}
    fn sys_c(_world: &mut World) {}

    #[test]
    fn non_conflicting_systems_share_batch() {
        let systems = vec![
            SystemDescriptor::new(sys_a, 0)
                .reads_component::<Position>()
                .writes_component::<Velocity>(),
            SystemDescriptor::new(sys_b, 1)
                .reads_component::<Health>()
                .writes_component::<AiState>(),
        ];

        let batches = plan_stage(&systems);
        assert_eq!(batches, vec![vec![0, 1]]);
    }

    #[test]
    fn read_write_conflict_splits_batches() {
        let systems = vec![
            SystemDescriptor::new(sys_a, 0).writes_component::<Position>(),
            SystemDescriptor::new(sys_b, 1).reads_component::<Position>(),
        ];

        let batches = plan_stage(&systems);
        assert_eq!(batches, vec![vec![0], vec![1]]);
    }

    #[test]
    fn write_write_conflict_splits_batches() {
        let systems = vec![
            SystemDescriptor::new(sys_a, 0).writes_resource::<TimeRes>(),
            SystemDescriptor::new(sys_b, 1).writes_resource::<TimeRes>(),
        ];

        let batches = plan_stage(&systems);
        assert_eq!(batches, vec![vec![0], vec![1]]);
    }

    #[test]
    fn planned_execution_keeps_registration_order() {
        let systems = vec![
            SystemDescriptor::new(sys_a, 0).writes_component::<Position>(),
            SystemDescriptor::new(sys_b, 1).reads_component::<Position>(),
            SystemDescriptor::new(sys_c, 2).reads_component::<Velocity>(),
        ];

        let batches = plan_stage(&systems);
        let flattened: Vec<usize> = batches.into_iter().flatten().collect();
        assert_eq!(flattened, vec![0, 1, 2]);
    }

    #[test]
    fn before_constraint_reorders_systems() {
        // Register b before a, but declare b.before(a) — b must run first.
        // Without constraints, order would be [a, b] by registration.
        let systems = vec![
            SystemDescriptor::new(sys_a, 0),
            SystemDescriptor::new(sys_b, 1).before(sys_a),
        ];

        let batches = plan_stage(&systems);
        let flattened: Vec<usize> = batches.into_iter().flatten().collect();
        // sys_b (index 1) should come before sys_a (index 0).
        assert_eq!(flattened, vec![1, 0]);
    }

    #[test]
    fn after_constraint_reorders_systems() {
        // sys_a is registered first but must run after sys_b.
        let systems = vec![
            SystemDescriptor::new(sys_a, 0).after(sys_b),
            SystemDescriptor::new(sys_b, 1),
        ];

        let batches = plan_stage(&systems);
        let flattened: Vec<usize> = batches.into_iter().flatten().collect();
        // sys_b (index 1) should come before sys_a (index 0).
        assert_eq!(flattened, vec![1, 0]);
    }

    #[test]
    fn before_forces_separate_batches() {
        // Two non-conflicting systems, but with ordering constraint.
        let systems = vec![
            SystemDescriptor::new(sys_a, 0).reads_component::<Position>(),
            SystemDescriptor::new(sys_b, 1)
                .reads_component::<Velocity>()
                .after(sys_a),
        ];

        let batches = plan_stage(&systems);
        // Without constraint they'd share a batch (no access conflict).
        // With constraint, sys_b must be in a later batch.
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![0]);
        assert_eq!(batches[1], vec![1]);
    }

    #[test]
    fn no_constraints_preserves_existing_behavior() {
        let systems = vec![
            SystemDescriptor::new(sys_a, 0).reads_component::<Position>(),
            SystemDescriptor::new(sys_b, 1).reads_component::<Velocity>(),
            SystemDescriptor::new(sys_c, 2).reads_component::<Health>(),
        ];

        let batches = plan_stage(&systems);
        // No conflicts, no constraints — all in one batch.
        assert_eq!(batches, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn chain_of_three_systems() {
        // sys_a -> sys_b -> sys_c
        let systems = vec![
            SystemDescriptor::new(sys_a, 0),
            SystemDescriptor::new(sys_b, 1).after(sys_a).before(sys_c),
            SystemDescriptor::new(sys_c, 2),
        ];

        let batches = plan_stage(&systems);
        let flattened: Vec<usize> = batches.into_iter().flatten().collect();
        assert_eq!(flattened, vec![0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "cycle detected")]
    fn cycle_panics() {
        let systems = vec![
            SystemDescriptor::new(sys_a, 0).before(sys_b),
            SystemDescriptor::new(sys_b, 1).before(sys_a),
        ];

        plan_stage(&systems);
    }
}
