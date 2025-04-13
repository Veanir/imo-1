use crate::tsplib::Solution;

/// Represents the type of a cycle (0 or 1) a vertex belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CycleId {
    Cycle1,
    Cycle2,
}

/// Represents a specific move that can be applied to a solution.
/// Stores node IDs involved in the move for later validation (Move List).
#[derive(Debug, Clone, PartialEq, Eq, Hash)] // Add Eq, Hash for use in sets/maps if needed
pub enum Move {
    /// Exchange nodes `v1` (originally from cycle1) and `v2` (originally from cycle2).
    InterRouteExchange { v1: usize, v2: usize },
    /// Exchange nodes `v1` and `v2` within the specified `cycle`.
    IntraRouteVertexExchange {
        v1: usize,
        v2: usize,
        cycle: CycleId, // Cycle where the swap happens
    },
    /// Exchange edges `(a, b)` and `(c, d)` within the specified `cycle`.
    /// Nodes `a, b, c, d` define the edges *removed* by the 2-opt move.
    /// The move effectively adds edges `(a, c)` and `(b, d)`, reversing the path between `b` and `c`.
    IntraRouteEdgeExchange {
        a: usize,
        b: usize, // node after a in the original cycle
        c: usize,
        d: usize, // node after c in the original cycle
        cycle: CycleId,
    },
}

/// Represents a move along with its calculated cost change (delta).
/// A negative delta indicates an improvement (cost reduction).
#[derive(Debug, Clone)]
pub struct EvaluatedMove {
    pub move_type: Move,
    pub delta: i32,
}

impl Move {
    /// Applies the move to a given solution, modifying it in place.
    /// Note: This requires finding the current positions of the nodes involved.
    pub fn apply(&self, solution: &mut Solution) {
        match self {
            Move::InterRouteExchange { v1, v2 } => {
                // Find current positions of v1 and v2
                let pos1_opt = solution.find_node(*v1);
                let pos2_opt = solution.find_node(*v2);

                if let (Some((CycleId::Cycle1, pos1)), Some((CycleId::Cycle2, pos2))) =
                    (pos1_opt, pos2_opt)
                {
                    solution.cycle1[pos1] = *v2;
                    solution.cycle2[pos2] = *v1;
                } else if let (Some((CycleId::Cycle2, pos1)), Some((CycleId::Cycle1, pos2))) =
                    (pos1_opt, pos2_opt)
                {
                    // Handle case where nodes might have already swapped implicitly by other moves
                    solution.cycle2[pos1] = *v2;
                    solution.cycle1[pos2] = *v1;
                } else {
                    // Error: Nodes not found in the expected cycles. This might indicate an issue
                    // if a move is applied after the solution state has diverged significantly.
                    // For robust implementation, might need error handling or re-validation.
                    eprintln!(
                        "Warning: InterRouteExchange apply failed. Nodes {} or {} not found in expected cycles.",
                        v1, v2
                    );
                }
            }
            Move::IntraRouteVertexExchange { v1, v2, cycle } => {
                if let (Some((c1, pos1)), Some((c2, pos2))) =
                    (solution.find_node(*v1), solution.find_node(*v2))
                {
                    if c1 == *cycle && c2 == *cycle {
                        // Both nodes found in the correct cycle, perform swap
                        let cycle_vec = solution.get_cycle_mut(*cycle);
                        cycle_vec.swap(pos1, pos2);
                    } else {
                        eprintln!(
                            "Warning: IntraRouteVertexExchange apply failed. Nodes {} or {} not in cycle {:?}.",
                            v1, v2, cycle
                        );
                    }
                } else {
                    eprintln!(
                        "Warning: IntraRouteVertexExchange apply failed. Nodes {} or {} not found.",
                        v1, v2
                    );
                }
            }
            Move::IntraRouteEdgeExchange {
                a,
                b,
                c,
                d: _,
                cycle,
            } => {
                // The core 2-opt logic involves reversing the path between b and c.
                // We need to find the current positions of b and c in the specified cycle.
                if let (Some((cb, pos_b)), Some((cc, pos_c))) =
                    (solution.find_node(*b), solution.find_node(*c))
                {
                    if cb == *cycle && cc == *cycle {
                        // Nodes b and c found in the correct cycle
                        let cycle_vec = solution.get_cycle_mut(*cycle);
                        let n = cycle_vec.len();
                        if n < 2 {
                            return;
                        } // Cannot reverse in empty or single-node cycle

                        // Determine the range to reverse [idx1..=idx2]
                        // Ensure indices are treated cyclically for reversal.
                        // We reverse the sequence starting *after* node `a` (which is at `pos_b-1` or wrap around)
                        // up to node `c` (at `pos_c`). The edge `(c, d)` means `d` is `pos_c+1`.
                        // The Python code reverses from `i+1` to `j`, where `i` is index of `a`, `j` is index of `c`.
                        // So we reverse from `pos_b` to `pos_c`.

                        let mut start = pos_b;
                        let mut end = pos_c;

                        // Handle cyclic reversal correctly
                        if start > end {
                            // Reverse path wraps around the end of the vector
                            // Example: [0, 1, 2, 3, 4], reverse b=3 to c=1 => reverse [3, 4, 0, 1]
                            let mut temp_slice = Vec::with_capacity(n);
                            // Add elements from start to end of vec
                            temp_slice.extend_from_slice(&cycle_vec[start..]);
                            // Add elements from start of vec to end
                            temp_slice.extend_from_slice(&cycle_vec[..=end]);
                            // Reverse the combined slice
                            temp_slice.reverse();
                            // Copy back
                            let mut temp_iter = temp_slice.into_iter();
                            for i in start..n {
                                cycle_vec[i] = temp_iter.next().unwrap();
                            }
                            for i in 0..=end {
                                cycle_vec[i] = temp_iter.next().unwrap();
                            }
                        } else {
                            // Normal slice reversal
                            cycle_vec[start..=end].reverse();
                        }
                    } else {
                        eprintln!(
                            "Warning: IntraRouteEdgeExchange apply failed. Nodes {} or {} not in cycle {:?}.",
                            b, c, cycle
                        );
                    }
                } else {
                    eprintln!(
                        "Warning: IntraRouteEdgeExchange apply failed. Nodes {} or {} not found.",
                        b, c
                    );
                }
            }
        }
    }
}
