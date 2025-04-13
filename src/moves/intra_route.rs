// Implementation for intra-route move evaluations

use crate::moves::types::{CycleId, EvaluatedMove, Move};
use crate::tsplib::{Solution, TsplibInstance};

/// Selects the appropriate cycle vector from the solution.
fn get_cycle_vec(solution: &Solution, cycle: CycleId) -> &Vec<usize> {
    match cycle {
        CycleId::Cycle1 => &solution.cycle1,
        CycleId::Cycle2 => &solution.cycle2,
    }
}

/// Calculates the cost delta for exchanging vertices at `pos1` and `pos2`
/// within the specified `cycle`.
///
/// Returns `None` if the move is invalid (e.g., `pos1 == pos2` or cycle too small).
pub fn evaluate_intra_route_vertex_exchange(
    solution: &Solution,
    instance: &TsplibInstance,
    cycle: CycleId,
    pos1: usize,
    pos2: usize,
) -> Option<EvaluatedMove> {
    let cycle_vec = get_cycle_vec(solution, cycle);
    let n = cycle_vec.len();

    if pos1 == pos2 || n < 2 || pos1 >= n || pos2 >= n {
        return None; // Invalid move
    }

    // Ensure pos1 < pos2 for easier neighbor calculation
    let (pos1, pos2) = (pos1.min(pos2), pos1.max(pos2));

    let v1 = cycle_vec[pos1];
    let v2 = cycle_vec[pos2];

    let prev1 = cycle_vec[if pos1 == 0 { n - 1 } else { pos1 - 1 }];
    let next1 = cycle_vec[(pos1 + 1) % n];
    let prev2 = cycle_vec[if pos2 == 0 { n - 1 } else { pos2 - 1 }];
    let next2 = cycle_vec[(pos2 + 1) % n];

    // Handle different exchange cases
    if pos2 == pos1 + 1 {
        // Case 1: Normal adjacency (nodes at pos1 and pos1+1)
        // Cycle: ..., prev1, v1, v2, next2, ... swapped to ..., prev1, v2, v1, next2, ...
        let cost_removed =
            instance.distance(prev1, v1) + instance.distance(v1, v2) + instance.distance(v2, next2);
        let cost_added =
            instance.distance(prev1, v2) + instance.distance(v2, v1) + instance.distance(v1, next2);
        let delta = cost_added - cost_removed;
        Some(EvaluatedMove {
            move_type: Move::IntraRouteVertexExchange { cycle, pos1, pos2 },
            delta,
        })
    } else if pos1 == 0 && pos2 == n - 1 {
        // Case 2: Wrapped adjacency (nodes at 0 and n-1)
        // Cycle: v1, next1, ..., prev2, v2 swapped to v2, next1, ..., prev2, v1
        // Edges removed: (v2, v1), (v1, next1), (prev2, v2)
        // Edges added: (v1, v2), (v2, next1), (prev2, v1)
        let cost_removed =
            instance.distance(v2, v1) + instance.distance(v1, next1) + instance.distance(prev2, v2);
        let cost_added =
            instance.distance(v1, v2) + instance.distance(v2, next1) + instance.distance(prev2, v1);
        let delta = cost_added - cost_removed;
        Some(EvaluatedMove {
            move_type: Move::IntraRouteVertexExchange { cycle, pos1, pos2 },
            delta,
        })
    } else {
        // Case 3: Non-adjacent nodes
        let cost_removed = instance.distance(prev1, v1)
            + instance.distance(v1, next1)
            + instance.distance(prev2, v2)
            + instance.distance(v2, next2);
        let cost_added = instance.distance(prev1, v2)
            + instance.distance(v2, next1)
            + instance.distance(prev2, v1)
            + instance.distance(v1, next2);
        let delta = cost_added - cost_removed;
        Some(EvaluatedMove {
            move_type: Move::IntraRouteVertexExchange { cycle, pos1, pos2 },
            delta,
        })
    }
}

/// Calculates the cost delta for exchanging edges `(v_i, v_{i+1})` and `(v_j, v_{j+1})`
/// within the specified `cycle`, where `i = pos1`, `j = pos2`.
/// This is equivalent to reversing the path between `pos1+1` and `pos2`.
///
/// Assumes `pos1 < pos2` and the move is valid (cycle size >= 3).
/// Returns `None` if the move is invalid.
pub fn evaluate_intra_route_edge_exchange(
    solution: &Solution,
    instance: &TsplibInstance,
    cycle: CycleId,
    pos1: usize,
    pos2: usize,
) -> Option<EvaluatedMove> {
    let cycle_vec = get_cycle_vec(solution, cycle);
    let n = cycle_vec.len();

    // Need at least 3 nodes to exchange two distinct edges.
    // Also ensure pos1 and pos2 are not adjacent or the same.
    if n < 3
        || pos1 == pos2
        || (pos1 + 1) % n == pos2
        || (pos2 + 1) % n == pos1
        || pos1 >= n
        || pos2 >= n
    {
        return None;
    }

    // Ensure pos1 < pos2 conceptually, handling wrap-around
    let (pos1, pos2) = if pos1 < pos2 {
        (pos1, pos2)
    } else {
        (pos2, pos1)
    };

    let vi = cycle_vec[pos1];
    let vi_plus_1 = cycle_vec[(pos1 + 1) % n];
    let vj = cycle_vec[pos2];
    let vj_plus_1 = cycle_vec[(pos2 + 1) % n];

    // Cost removed: dist(v_i, v_{i+1}) + dist(v_j, v_{j+1})
    let cost_removed = instance.distance(vi, vi_plus_1) + instance.distance(vj, vj_plus_1);

    // Cost added: dist(v_i, v_j) + dist(v_{i+1}, v_{j+1})
    // Note: This specific edge exchange corresponds to a 2-opt move in TSP.
    // It connects (vi, vj) and (vi+1, vj+1). Other edge exchanges are possible.
    // The `apply` function reverses the path, which matches this delta.
    let cost_added = instance.distance(vi, vj) + instance.distance(vi_plus_1, vj_plus_1);

    let delta = cost_added - cost_removed;

    Some(EvaluatedMove {
        move_type: Move::IntraRouteEdgeExchange { cycle, pos1, pos2 }, // Store original indices
        delta,
    })
}

// --- Optional: Add functions to generate all possible intra-route moves ---
/*
pub fn generate_all_intra_route_vertex_moves<'a>(
    solution: &'a Solution,
    instance: &'a TsplibInstance,
    cycle: CycleId,
) -> impl Iterator<Item = EvaluatedMove> + 'a {
    let cycle_vec = get_cycle_vec(solution, cycle);
    let n = cycle_vec.len();
    (0..n)
        .flat_map(move |pos1| {
            (pos1 + 1..n)
                .filter_map(move |pos2| {
                     evaluate_intra_route_vertex_exchange(solution, instance, cycle, pos1, pos2)
                })
        })
}

pub fn generate_all_intra_route_edge_moves<'a>(
    solution: &'a Solution,
    instance: &'a TsplibInstance,
    cycle: CycleId,
) -> impl Iterator<Item = EvaluatedMove> + 'a {
     let cycle_vec = get_cycle_vec(solution, cycle);
    let n = cycle_vec.len();
    if n < 3 {
        // Need to return an empty iterator explicitly if using impl Trait
        // A simple way is to use an empty slice iterator
        return Vec::new().into_iter().filter_map(|_| None); // Type inference helps here
    }
    (0..n)
        .flat_map(move |pos1| {
            // Ensure pos2 is not pos1 or adjacent to pos1
            (0..n)
                .filter(move |&pos2| pos1 != pos2 && (pos1 + 1) % n != pos2 && (pos2 + 1) % n != pos1)
                .filter_map(move |pos2| {
                     // Ensure pos1 < pos2 for the evaluation function's assumption
                     let (p1, p2) = (pos1.min(pos2), pos1.max(pos2));
                     evaluate_intra_route_edge_exchange(solution, instance, cycle, p1, p2)
                 })
        })
        // This generates duplicates (i,j) and (j,i), need to unique or generate differently
        // A better way: iterate i from 0 to n-1, j from i+2 to n-1 (handle wrap around)
        // (0..n).flat_map(move |i| {
        //     ((i + 2)..n).filter(move |&j| (i==0 && j == n-1) == false) // avoid adjacent wrap around
        //         .filter_map(move |j| evaluate_intra_route_edge_exchange(solution, instance, cycle, i, j))
        // })
}
*/
