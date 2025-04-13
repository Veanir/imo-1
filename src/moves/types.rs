use crate::tsplib::Solution;

/// Represents the type of a cycle (0 or 1) a vertex belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CycleId {
    Cycle1,
    Cycle2,
}

/// Represents a specific move that can be applied to a solution.
#[derive(Debug, Clone, PartialEq)]
pub enum Move {
    /// Exchange vertex at `pos1` in `cycle1` with vertex at `pos2` in `cycle2`.
    InterRouteExchange { pos1: usize, pos2: usize },
    /// Exchange vertices at `pos1` and `pos2` within the specified `cycle`.
    IntraRouteVertexExchange {
        cycle: CycleId,
        pos1: usize,
        pos2: usize,
    },
    /// Exchange edges `(v_i, v_i+1)` and `(v_j, v_j+1)` within the specified `cycle`.
    /// `pos1` is the index of `v_i`, `pos2` is the index of `v_j`.
    /// Assumes `pos1 < pos2`.
    IntraRouteEdgeExchange {
        cycle: CycleId,
        pos1: usize,
        pos2: usize,
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
    /// Returns the cost delta of the move.
    /// Note: This currently recalculates cost. For efficiency, it should ideally
    /// apply the change and return a pre-calculated delta or allow delta calculation
    /// separately.
    pub fn apply(&self, solution: &mut Solution) {
        match self {
            Move::InterRouteExchange { pos1, pos2 } => {
                // This is tricky because we're swapping *vertices*, not positions.
                // We need the actual vertex values at these positions.
                let vertex1 = solution.cycle1[*pos1];
                let vertex2 = solution.cycle2[*pos2];
                solution.cycle1[*pos1] = vertex2;
                solution.cycle2[*pos2] = vertex1;
                // Note: This assumes the task means swapping vertices at specific *indices*.
                // If it means swapping specific *vertex values* regardless of position,
                // the implementation would need to find those vertices first.
                // Also, this basic swap might not be the most efficient way if delta calculation is needed.
            }
            Move::IntraRouteVertexExchange { cycle, pos1, pos2 } => match cycle {
                CycleId::Cycle1 => solution.cycle1.swap(*pos1, *pos2),
                CycleId::Cycle2 => solution.cycle2.swap(*pos1, *pos2),
            },
            Move::IntraRouteEdgeExchange { cycle, pos1, pos2 } => {
                // This involves reversing the sub-path between pos1+1 and pos2
                match cycle {
                    CycleId::Cycle1 => {
                        if pos1 < pos2 {
                            // Ensure indices are valid for slicing
                            if *pos1 + 1 <= *pos2 && *pos2 < solution.cycle1.len() {
                                solution.cycle1[(*pos1 + 1)..=*pos2].reverse();
                            }
                        }
                    }
                    CycleId::Cycle2 => {
                        if pos1 < pos2 {
                            // Ensure indices are valid for slicing
                            if *pos1 + 1 <= *pos2 && *pos2 < solution.cycle2.len() {
                                solution.cycle2[(*pos1 + 1)..=*pos2].reverse();
                            }
                        }
                    }
                }
            }
        }
    }
}
