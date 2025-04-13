use crate::moves::types::{EvaluatedMove, Move};
use crate::tsplib::{Solution, TsplibInstance};

/// Calculates the cost delta for exchanging the vertex at `pos1` in cycle1
/// with the vertex at `pos2` in cycle2.
///
/// Returns `None` if the move is invalid (e.g., cycles are too small).
pub fn evaluate_inter_route_exchange(
    solution: &Solution,
    instance: &TsplibInstance,
    pos1: usize,
    pos2: usize,
) -> Option<EvaluatedMove> {
    // Ensure cycles are large enough for neighbor calculations
    if solution.cycle1.len() < 2 || solution.cycle2.len() < 2 {
        // Need at least 2 nodes to define edges around the swapped node.
        // If a cycle has only 1 node, swapping it behaves differently.
        // For simplicity, we might disallow swaps involving cycles of size 1.
        // Or handle it as a special case if needed by the problem definition.
        return None; // Or handle differently if required
    }

    // Ensure positions are valid
    if pos1 >= solution.cycle1.len() || pos2 >= solution.cycle2.len() {
        return None; // Invalid position index
    }

    let cycle1 = &solution.cycle1;
    let cycle2 = &solution.cycle2;

    // Vertices to be swapped
    let u = cycle1[pos1];
    let v = cycle2[pos2];

    // --- Calculate cost change in Cycle 1 ---
    let prev_u_pos = if pos1 == 0 {
        cycle1.len() - 1
    } else {
        pos1 - 1
    };
    let next_u_pos = (pos1 + 1) % cycle1.len();
    let prev_u = cycle1[prev_u_pos];
    let next_u = cycle1[next_u_pos];

    // Cost removed from cycle 1
    let cost_removed_c1 = if cycle1.len() > 1 {
        instance.distance(prev_u, u) + instance.distance(u, next_u)
    } else {
        0
    }; // Should not happen due to initial check

    // Cost added to cycle 1 (with v replacing u)
    let cost_added_c1 = if cycle1.len() > 1 {
        instance.distance(prev_u, v) + instance.distance(v, next_u)
    } else {
        0
    };

    // --- Calculate cost change in Cycle 2 ---
    let prev_v_pos = if pos2 == 0 {
        cycle2.len() - 1
    } else {
        pos2 - 1
    };
    let next_v_pos = (pos2 + 1) % cycle2.len();
    let prev_v = cycle2[prev_v_pos];
    let next_v = cycle2[next_v_pos];

    // Cost removed from cycle 2
    let cost_removed_c2 = if cycle2.len() > 1 {
        instance.distance(prev_v, v) + instance.distance(v, next_v)
    } else {
        0
    }; // Should not happen

    // Cost added to cycle 2 (with u replacing v)
    let cost_added_c2 = if cycle2.len() > 1 {
        instance.distance(prev_v, u) + instance.distance(u, next_v)
    } else {
        0
    };

    // Special case: if swapping connects a node to itself (cycle length 1)
    // This delta calculation assumes nodes are distinct. If prev_u == next_u or prev_v == next_v
    // (which happens for cycle length 2), the formula is slightly different.
    // Let's refine for length 2:
    let delta = if cycle1.len() == 2 && cycle2.len() == 2 {
        // C1: remove 2*dist(prev_u, u), add 2*dist(prev_u, v)
        // C2: remove 2*dist(prev_v, v), add 2*dist(prev_v, u)
        (2 * instance.distance(prev_u, v) - 2 * instance.distance(prev_u, u))
            + (2 * instance.distance(prev_v, u) - 2 * instance.distance(prev_v, v))
    } else if cycle1.len() == 2 {
        // C1: remove 2*dist(prev_u, u), add 2*dist(prev_u, v)
        // C2: use standard formula
        (2 * instance.distance(prev_u, v) - 2 * instance.distance(prev_u, u))
            + (cost_added_c2 - cost_removed_c2)
    } else if cycle2.len() == 2 {
        // C1: use standard formula
        // C2: remove 2*dist(prev_v, v), add 2*dist(prev_v, u)
        (cost_added_c1 - cost_removed_c1)
            + (2 * instance.distance(prev_v, u) - 2 * instance.distance(prev_v, v))
    } else {
        // Standard case (both cycles > 2)
        (cost_added_c1 - cost_removed_c1) + (cost_added_c2 - cost_removed_c2)
    };

    Some(EvaluatedMove {
        move_type: Move::InterRouteExchange { pos1, pos2 },
        delta,
    })
}

// --- Optional: Add a function to generate all possible inter-route moves ---
/*
pub fn generate_all_inter_route_moves<'a>(
    solution: &'a Solution,
    instance: &'a TsplibInstance,
) -> impl Iterator<Item = EvaluatedMove> + 'a {
    (0..solution.cycle1.len())
        .flat_map(move |pos1| {
            (0..solution.cycle2.len())
                .filter_map(move |pos2| {
                     evaluate_inter_route_exchange(solution, instance, pos1, pos2)
                })
        })
}
*/
