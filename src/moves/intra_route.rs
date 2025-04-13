use crate::moves::types::{CycleId, EvaluatedMove, Move};
use crate::tsplib::{Solution, TsplibInstance};

pub fn evaluate_intra_route_vertex_exchange(
    solution: &Solution,
    instance: &TsplibInstance,
    cycle: CycleId,
    pos1: usize,
    pos2: usize,
) -> Option<EvaluatedMove> {
    let cycle_vec = solution.get_cycle(cycle);
    let n = cycle_vec.len();

    // Need at least 2 nodes to swap.
    if n < 2 || pos1 == pos2 || pos1 >= n || pos2 >= n {
        return None; // Invalid move
    }

    // Ensure pos1 < pos2 for easier neighbor calculation, doesn't affect result
    let (pos1, pos2) = (pos1.min(pos2), pos1.max(pos2));

    let v1 = cycle_vec[pos1];
    let v2 = cycle_vec[pos2];

    // Calculate delta based on adjacency
    let delta = if n == 2 {
        // Only two nodes, swapping them doesn't change the cycle or cost.
        0
    } else if pos2 == pos1 + 1 || (pos1 == 0 && pos2 == n - 1) {
        // Adjacent nodes (including wrap-around)
        // Find neighbours correctly considering wrap-around for both cases
        let prev1 = cycle_vec[if pos1 == 0 { n - 1 } else { pos1 - 1 }];
        let next2 = cycle_vec[(pos2 + 1) % n]; // next of v2

        // If adjacent: ..., prev1, v1, v2, next2, ... swapped to ..., prev1, v2, v1, next2, ...
        // Edges removed: (prev1, v1), (v1, v2), (v2, next2)
        // Edges added:   (prev1, v2), (v2, v1), (v1, next2)
        // Delta = Added - Removed
        (instance.distance(prev1, v2) + instance.distance(v2, v1) + instance.distance(v1, next2))
            - (instance.distance(prev1, v1)
                + instance.distance(v1, v2)
                + instance.distance(v2, next2))
    } else {
        // Non-adjacent nodes
        let prev1 = cycle_vec[if pos1 == 0 { n - 1 } else { pos1 - 1 }];
        let next1 = cycle_vec[(pos1 + 1) % n]; // Should exist since n > 2 and not adjacent
        let prev2 = cycle_vec[if pos2 == 0 { n - 1 } else { pos2 - 1 }]; // Should exist
        let next2 = cycle_vec[(pos2 + 1) % n];

        // Edges removed: (prev1, v1), (v1, next1), (prev2, v2), (v2, next2)
        // Edges added:   (prev1, v2), (v2, next1), (prev2, v1), (v1, next2)
        // Delta = Added - Removed
        (instance.distance(prev1, v2)
            + instance.distance(v2, next1)
            + instance.distance(prev2, v1)
            + instance.distance(v1, next2))
            - (instance.distance(prev1, v1)
                + instance.distance(v1, next1)
                + instance.distance(prev2, v2)
                + instance.distance(v2, next2))
    };

    Some(EvaluatedMove {
        move_type: Move::IntraRouteVertexExchange { v1, v2, cycle }, // Use correct field names
        delta,
    })
}

/// Calculates the cost delta for exchanging edges `(a, b)` and `(c, d)`
/// within the specified `cycle`, where `a=cycle[pos1]`, `b=cycle[pos1+1]`,
/// `c=cycle[pos2]`, `d=cycle[pos2+1]`.
/// This is a 2-opt move.
///
/// Assumes `pos1` and `pos2` represent the *start* indices of the edges to be removed.
/// Returns `None` if the move is invalid (e.g., cycle size < 3, adjacent edges).
pub fn evaluate_intra_route_edge_exchange(
    solution: &Solution,
    instance: &TsplibInstance,
    cycle: CycleId,
    pos1: usize, // Index of node `a`
    pos2: usize, // Index of node `c`
) -> Option<EvaluatedMove> {
    let cycle_vec = solution.get_cycle(cycle);
    let n = cycle_vec.len();

    // Need at least 3 nodes for non-degenerate 2-opt.
    // Ensure pos1 and pos2 are valid indices.
    // Ensure edges are not adjacent or overlapping.
    if n < 3
        || pos1 >= n
        || pos2 >= n
        || pos1 == pos2
        || (pos1 + 1) % n == pos2
        || (pos2 + 1) % n == pos1
    {
        return None;
    }

    // Nodes defining the edges to be removed: (a, b) and (c, d)
    let a = cycle_vec[pos1];
    let b = cycle_vec[(pos1 + 1) % n];
    let c = cycle_vec[pos2];
    let d = cycle_vec[(pos2 + 1) % n];

    // Cost removed: dist(a, b) + dist(c, d)
    let cost_removed = instance.distance(a, b) + instance.distance(c, d);

    // Cost added: dist(a, c) + dist(b, d)
    let cost_added = instance.distance(a, c) + instance.distance(b, d);

    let delta = cost_added - cost_removed;

    Some(EvaluatedMove {
        move_type: Move::IntraRouteEdgeExchange { a, b, c, d, cycle }, // Use correct field names
        delta,
    })
}

/// Calculates the cost delta for a specific candidate 2-opt move:
/// removing edges (a, a_next) and (b, b_next) and adding (a, b) and (a_next, b_next).
/// This is used in the Candidate Moves strategy. It considers performing a
/// 2-opt move by removing edges (a, a_next) and (b, b_next), and adding
/// edges (a, b) and (a_next, b_next).
/// `pos_a` is the index of node `a`, `pos_b` is the index of node `b`.
pub fn evaluate_candidate_intra_route_edge_exchange(
    solution: &Solution,
    instance: &TsplibInstance,
    cycle_id: CycleId,
    pos_a: usize,
    pos_b: usize,
) -> Option<EvaluatedMove> {
    let cycle_vec = solution.get_cycle(cycle_id);
    let n = cycle_vec.len();

    // Basic validation
    if n < 3 || pos_a >= n || pos_b >= n || pos_a == pos_b {
        return None;
    }

    let a = cycle_vec[pos_a];
    let b = cycle_vec[pos_b];

    let pos_a_next = (pos_a + 1) % n;
    let pos_b_next = (pos_b + 1) % n;

    // Ensure the edges we intend to remove are not adjacent or overlapping
    // i.e., a_next != b and b_next != a
    if pos_a_next == pos_b || pos_b_next == pos_a {
        return None; // Invalid 2-opt move topology for these positions
    }

    let a_next = cycle_vec[pos_a_next];
    let b_next = cycle_vec[pos_b_next];

    // Cost removed: dist(a, a_next) + dist(b, b_next)
    let cost_removed = instance.distance(a, a_next) + instance.distance(b, b_next);

    // Cost added: dist(a, b) + dist(a_next, b_next)
    let cost_added = instance.distance(a, b) + instance.distance(a_next, b_next);

    let delta = cost_added - cost_removed;

    // Store the move in the standard IntraRouteEdgeExchange format.
    // Removed edges were (a, a_next) and (b, b_next).
    // Apply function expects { a: w, b: x, c: y, d: z } where removed edges are (w, x) and (y, z).
    Some(EvaluatedMove {
        move_type: Move::IntraRouteEdgeExchange {
            a,         // w = a
            b: a_next, // x = a_next
            c: b,      // y = b
            d: b_next, // z = b_next
            cycle: cycle_id,
        },
        delta,
    })
}
