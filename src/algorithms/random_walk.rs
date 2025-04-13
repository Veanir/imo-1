use crate::algorithm::ProgressCallback;
use crate::algorithm::TspAlgorithm;
use crate::moves::types::{CycleId, Move};
use crate::tsplib::{Solution, TsplibInstance};
use crate::utils::generate_random_solution;
use rand::{Rng, thread_rng};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct RandomWalk {
    // Time limit based on the slowest Local Search variant (set during experiment setup)
    // For now, we can use a fixed number of iterations or a fixed duration.
    max_iterations: usize, // Or use duration: time_limit: Duration,
}

impl Default for RandomWalk {
    fn default() -> Self {
        Self {
            // Default to a reasonable number of iterations for now.
            // This should ideally be adjusted based on LS runtime.
            max_iterations: 10000, // Example value
        }
    }
}

impl RandomWalk {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }

    /// Generates a random valid move for the current solution.
    fn generate_random_move(&self, solution: &Solution, rng: &mut impl Rng) -> Option<Move> {
        let n1 = solution.cycle1.len();
        let n2 = solution.cycle2.len();

        // Avoid moves if cycles are too small
        if n1 + n2 < 3 {
            return None;
        }

        // Choose move type randomly (proportional to possibility?)
        // 0: Inter-route Exchange
        // 1: Intra-route Vertex Exchange
        // 2: Intra-route Edge Exchange
        let move_type_choice = rng.gen_range(0..=2);

        match move_type_choice {
            0 if n1 > 0 && n2 > 0 => {
                // Inter-route Exchange
                let pos1 = rng.gen_range(0..n1);
                let pos2 = rng.gen_range(0..n2);
                Some(Move::InterRouteExchange { pos1, pos2 })
            }
            1 => {
                // Intra-route Vertex Exchange
                let cycle_choice = if n1 >= 2 && (n2 < 2 || rng.gen_bool(0.5)) {
                    CycleId::Cycle1
                } else if n2 >= 2 {
                    CycleId::Cycle2
                } else {
                    return None; // Neither cycle is large enough
                };
                let n = if cycle_choice == CycleId::Cycle1 {
                    n1
                } else {
                    n2
                };
                if n < 2 {
                    return None;
                } // Should not happen based on above logic, but safe check
                let pos1 = rng.gen_range(0..n);
                let mut pos2 = rng.gen_range(0..n);
                while pos1 == pos2 {
                    // Ensure distinct positions
                    pos2 = rng.gen_range(0..n);
                }
                Some(Move::IntraRouteVertexExchange {
                    cycle: cycle_choice,
                    pos1,
                    pos2,
                })
            }
            2 => {
                // Intra-route Edge Exchange
                let cycle_choice = if n1 >= 3 && (n2 < 3 || rng.gen_bool(0.5)) {
                    CycleId::Cycle1
                } else if n2 >= 3 {
                    CycleId::Cycle2
                } else {
                    return None; // Neither cycle is large enough
                };
                let n = if cycle_choice == CycleId::Cycle1 {
                    n1
                } else {
                    n2
                };
                if n < 3 {
                    return None;
                } // Need at least 3 nodes

                // Select two distinct, non-adjacent positions
                let pos1 = rng.gen_range(0..n);
                let mut pos2 = rng.gen_range(0..n);
                while pos1 == pos2 || (pos1 + 1) % n == pos2 || (pos2 + 1) % n == pos1 {
                    pos2 = rng.gen_range(0..n);
                }
                // Ensure pos1 < pos2 for consistency if needed by apply/evaluate logic
                let (pos1, pos2) = (pos1.min(pos2), pos1.max(pos2));

                Some(Move::IntraRouteEdgeExchange {
                    cycle: cycle_choice,
                    pos1,
                    pos2,
                })
            }
            _ => None, // Handles cases where conditions for chosen move type aren't met
        }
    }
}

impl TspAlgorithm for RandomWalk {
    fn name(&self) -> &str {
        "Random Walk"
    }

    fn solve_with_feedback(
        &self,
        instance: &TsplibInstance,
        progress_callback: ProgressCallback,
    ) -> Solution {
        let mut current_solution = generate_random_solution(instance);
        let mut best_solution = current_solution.clone();
        let mut best_cost = best_solution.calculate_cost(instance);
        let mut rng = thread_rng();

        for i in 0..self.max_iterations {
            // Update progress every N iterations to avoid excessive updates
            if i % 100 == 0 || i == self.max_iterations - 1 {
                // Update every 100 iter + last
                progress_callback(format!(
                    "[Iter: {}/{}] Best Cost: {}",
                    i + 1,
                    self.max_iterations,
                    best_cost
                ));
            }

            if let Some(random_move) = self.generate_random_move(&current_solution, &mut rng) {
                random_move.apply(&mut current_solution);
                let current_cost = current_solution.calculate_cost(instance);
                if current_cost < best_cost {
                    best_cost = current_cost;
                    best_solution = current_solution.clone();
                }
            } else {
                break;
            }
        }
        progress_callback(format!("[Finished] Final Best Cost: {}", best_cost));
        best_solution
    }
}
