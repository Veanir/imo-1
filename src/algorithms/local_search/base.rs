// This file will contain the core LocalSearch implementation

use crate::algorithm::ProgressCallback;
use crate::algorithm::TspAlgorithm;
use crate::algorithms::constructive::weighted_regret_cycle::WeightedRegretCycle; // Example, might need a generic Heuristic trait later
use crate::moves::inter_route::evaluate_inter_route_exchange;
use crate::moves::intra_route::{
    evaluate_intra_route_edge_exchange, evaluate_intra_route_vertex_exchange,
};
use crate::moves::types::{CycleId, EvaluatedMove, Move};
use crate::tsplib::{Solution, TsplibInstance};
use crate::utils::generate_random_solution; // Assuming this utility exists
use rand::seq::SliceRandom; // For Greedy variant
use rand::thread_rng; // For Greedy variant

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchVariant {
    Steepest,
    Greedy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeighborhoodType {
    VertexExchange, // Intra-route: exchange vertices within a cycle
    EdgeExchange,   // Intra-route: exchange edges within a cycle
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitialSolutionType {
    Random,
    Heuristic(HeuristicAlgorithm), // Placeholder for different heuristics
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeuristicAlgorithm {
    WeightedRegret,
    // Add other heuristics here if needed
}

pub struct LocalSearch {
    variant: SearchVariant,
    neighborhood: NeighborhoodType,
    initial_solution_type: InitialSolutionType,
    // Store the generated name
    name_str: String,
}

impl LocalSearch {
    pub fn new(
        variant: SearchVariant,
        neighborhood: NeighborhoodType,
        initial_solution_type: InitialSolutionType,
    ) -> Self {
        let name_str = format!(
            "Local Search ({:?}, {:?}, Init: {:?})",
            variant, neighborhood, initial_solution_type
        );
        Self {
            variant,
            neighborhood,
            initial_solution_type,
            name_str,
        }
    }

    fn generate_initial_solution(&self, instance: &TsplibInstance) -> Solution {
        match self.initial_solution_type {
            InitialSolutionType::Random => generate_random_solution(instance),
            InitialSolutionType::Heuristic(heuristic) => match heuristic {
                HeuristicAlgorithm::WeightedRegret => {
                    let constructive_algo = WeightedRegretCycle::default();
                    // Provide a dummy callback as progress isn't needed here
                    let mut dummy_callback = |_: String| {};
                    constructive_algo.solve_with_feedback(instance, &mut dummy_callback)
                }
            },
        }
    }
}

impl TspAlgorithm for LocalSearch {
    fn name(&self) -> &str {
        &self.name_str
    }

    fn solve_with_feedback(
        &self,
        instance: &TsplibInstance,
        progress_callback: ProgressCallback,
    ) -> Solution {
        let mut current_solution = self.generate_initial_solution(instance);
        let mut current_cost = current_solution.calculate_cost(instance);
        let mut rng = thread_rng();
        let mut iteration = 0;

        loop {
            iteration += 1;
            progress_callback(format!("[Iter: {}] Cost: {}", iteration, current_cost));

            let mut best_evaluated_move: Option<EvaluatedMove> = None;
            let mut found_improving_move = false;

            // Generate and evaluate all possible moves in the neighborhood
            let mut possible_moves: Vec<EvaluatedMove> = Vec::new();

            // 1. Inter-route moves (exchange vertices between cycles)
            for pos1 in 0..current_solution.cycle1.len() {
                for pos2 in 0..current_solution.cycle2.len() {
                    if let Some(evaluated_move) =
                        evaluate_inter_route_exchange(&current_solution, instance, pos1, pos2)
                    {
                        possible_moves.push(evaluated_move);
                    }
                }
            }

            // 2. Intra-route moves (within each cycle)
            for cycle_id in [CycleId::Cycle1, CycleId::Cycle2] {
                let cycle_vec = match cycle_id {
                    CycleId::Cycle1 => &current_solution.cycle1,
                    CycleId::Cycle2 => &current_solution.cycle2,
                };
                let n = cycle_vec.len();

                match self.neighborhood {
                    NeighborhoodType::VertexExchange => {
                        if n >= 2 {
                            for pos1 in 0..n {
                                for pos2 in pos1 + 1..n {
                                    if let Some(evaluated_move) =
                                        evaluate_intra_route_vertex_exchange(
                                            &current_solution,
                                            instance,
                                            cycle_id,
                                            pos1,
                                            pos2,
                                        )
                                    {
                                        possible_moves.push(evaluated_move);
                                    }
                                }
                            }
                        }
                    }
                    NeighborhoodType::EdgeExchange => {
                        if n >= 3 {
                            // Iterate i from 0..n, j from i+2..n (avoid adjacent and wrap-around)
                            for pos1 in 0..n {
                                for pos2_offset in 2..n {
                                    let pos2 = (pos1 + pos2_offset) % n;
                                    // Ensure we don't wrap around to adjacent edge (pos1=0, pos2=n-1)
                                    if pos1 == 0 && pos2 == n - 1 {
                                        continue;
                                    }
                                    // Ensure pos1 < pos2 conceptually for evaluate function
                                    let (p1, p2) = (pos1.min(pos2), pos1.max(pos2));

                                    if let Some(evaluated_move) = evaluate_intra_route_edge_exchange(
                                        &current_solution,
                                        instance,
                                        cycle_id,
                                        p1, // Use ensured smaller index
                                        p2, // Use ensured larger index
                                    ) {
                                        possible_moves.push(evaluated_move);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            match self.variant {
                SearchVariant::Steepest => {
                    // Find the best *improving* move (most negative delta)
                    for evaluated_move in possible_moves {
                        if evaluated_move.delta < 0 {
                            if best_evaluated_move.is_none()
                                || evaluated_move.delta
                                    < best_evaluated_move.as_ref().unwrap().delta
                            {
                                best_evaluated_move = Some(evaluated_move);
                            }
                        }
                    }
                    if best_evaluated_move.is_some() {
                        found_improving_move = true;
                    }
                }
                SearchVariant::Greedy => {
                    possible_moves.shuffle(&mut rng);
                    for evaluated_move in possible_moves {
                        if evaluated_move.delta < 0 {
                            best_evaluated_move = Some(evaluated_move);
                            found_improving_move = true;
                            break; // Apply the first improving move found
                        }
                    }
                }
            }

            if found_improving_move {
                let applied_move = best_evaluated_move.unwrap();
                // Apply the move
                applied_move.move_type.apply(&mut current_solution);
                // Update the cost using the calculated delta
                current_cost += applied_move.delta;
            } else {
                progress_callback(format!("[Finished] Final Cost: {}", current_cost));
                break;
            }
        }

        current_solution
    }
}
