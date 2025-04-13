// This file will contain the core LocalSearch implementation

use crate::algorithm::ProgressCallback;
use crate::algorithm::TspAlgorithm;
use crate::algorithms::constructive::weighted_regret_cycle::WeightedRegretCycle; // Example, might need a generic Heuristic trait later
use crate::moves::inter_route::evaluate_inter_route_exchange;
use crate::moves::intra_route::{
    evaluate_candidate_intra_route_edge_exchange, // Added for Candidate search
    evaluate_intra_route_edge_exchange,
    evaluate_intra_route_vertex_exchange,
};
use crate::moves::types::{CycleId, EvaluatedMove, Move};
use crate::tsplib::{Solution, TsplibInstance};
use crate::utils::generate_random_solution; // Assuming this utility exists
use rand::seq::SliceRandom; // For Greedy variant
use rand::thread_rng; // For Greedy variant
use std::collections::{BinaryHeap, HashSet}; // Added for MoveList

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchVariant {
    Steepest,
    Greedy,
    CandidateSteepest(usize), // Steepest search considering only k-nearest neighbors
    MoveListSteepest,         // Steepest search using a sorted list of moves
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
        let name_str = match variant {
            SearchVariant::CandidateSteepest(k) => format!(
                "Local Search (Candidate k={}, {:?}, Init: {:?})",
                k, neighborhood, initial_solution_type
            ),
            SearchVariant::MoveListSteepest => format!(
                "Local Search (MoveListSteepest, {:?}, Init: {:?})",
                neighborhood, initial_solution_type
            ),
            _ => format!(
                "Local Search ({:?}, {:?}, Init: {:?})",
                variant, neighborhood, initial_solution_type
            ),
        };
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

        // --- MoveListSteepest specific initialization ---
        // Initialize the move list *outside* the loop for MoveListSteepest
        let mut move_list: Vec<EvaluatedMove> = if self.variant == SearchVariant::MoveListSteepest {
            let mut initial_moves = self.generate_all_improving_moves(instance, &current_solution);
            initial_moves.sort_unstable_by(|a, b| a.delta.cmp(&b.delta)); // Sort by delta ascending
            initial_moves
        } else {
            Vec::new() // Not used for other variants
        };
        // Use BTreeSet for efficient removal and sorted iteration?
        // Or stick with Vec + sort/binary_search for insertion? Let's try Vec first.

        loop {
            iteration += 1;
            let cost_before_iter = current_cost;
            progress_callback(format!("[Iter: {}] Cost: {}", iteration, current_cost));

            let mut best_evaluated_move: Option<EvaluatedMove> = None;
            let mut found_improving_move = false;
            let mut best_move_index: Option<usize> = None; // For MoveListSteepest removal

            // --- Move Generation / Selection Phase ---
            match self.variant {
                SearchVariant::Steepest | SearchVariant::CandidateSteepest(_) => {
                    // Generate all improving moves *for this iteration*
                    let current_improving_moves =
                        if let SearchVariant::CandidateSteepest(k) = self.variant {
                            self.generate_candidate_moves(instance, &current_solution, k)
                        } else {
                            self.generate_all_improving_moves(instance, &current_solution)
                        };

                    // Find the best *improving* move among those generated
                    if let Some(best_move) = current_improving_moves.iter().min_by_key(|m| m.delta)
                    {
                        if best_move.delta < 0 {
                            // Ensure it's actually improving
                            best_evaluated_move = Some(best_move.clone());
                            found_improving_move = true;
                        }
                    }
                }
                SearchVariant::Greedy => {
                    // Generate all improving moves *for this iteration*
                    let mut current_improving_moves =
                        self.generate_all_improving_moves(instance, &current_solution);
                    current_improving_moves.shuffle(&mut rng);
                    // Find the *first* improving move after shuffle
                    if let Some(first_move) =
                        current_improving_moves.into_iter().find(|m| m.delta < 0)
                    {
                        best_evaluated_move = Some(first_move);
                        found_improving_move = true;
                    }
                }
                SearchVariant::MoveListSteepest => {
                    // Iterate through the *persistent* move list to find the first valid move
                    for (index, evaluated_move) in move_list.iter().enumerate() {
                        // Check if delta is still improving (might have become 0 or positive due to other changes)
                        // Re-evaluating delta is complex, rely on is_move_valid for now.
                        if evaluated_move.delta < 0
                            && self.is_move_valid(&current_solution, &evaluated_move.move_type)
                        {
                            // Found the best *valid* move according to the sorted list
                            best_evaluated_move = Some(evaluated_move.clone());
                            best_move_index = Some(index); // Store index for removal
                            found_improving_move = true;
                            break; // Apply this move
                        }
                        // Moves that are no longer valid will be skipped and eventually removed.
                    }
                }
            }

            // --- Apply the selected move (if any) ---
            if found_improving_move {
                let applied_move = best_evaluated_move.unwrap(); // Safe due to found_improving_move flag
                let cost_before_apply = current_cost;

                // Apply the move
                applied_move.move_type.apply(&mut current_solution);
                current_cost += applied_move.delta;

                // Sanity check cost calculation
                let real_cost_after_apply = current_solution.calculate_cost(instance);
                if (real_cost_after_apply as f64 - current_cost as f64).abs() > 1e-6 {
                    // Use tolerance for float comparison
                    eprintln!(
                        "[WARN] Cost mismatch after apply! Iter: {}, Move: {:?}, Delta: {}, Cost before: {}, Incremental cost: {}, Real cost: {}",
                        iteration,
                        applied_move.move_type,
                        applied_move.delta,
                        cost_before_apply,
                        current_cost,
                        real_cost_after_apply
                    );
                    current_cost = real_cost_after_apply; // Correct the cost
                }

                // --- Update Move List for MoveListSteepest ---
                if self.variant == SearchVariant::MoveListSteepest {
                    // 1. Remove the applied move
                    if let Some(index) = best_move_index {
                        move_list.remove(index);
                    } else {
                        eprintln!("[WARN] Applied move index not found in move list!");
                        // Potential issue if the list was modified unexpectedly
                    }

                    // 2. Identify affected nodes (needs helper function)
                    let affected_nodes =
                        self.identify_affected_nodes(&applied_move.move_type, &current_solution);

                    // 3. Remove potentially invalid moves (those involving affected nodes)
                    // This is an approximation to avoid full re-validation
                    move_list.retain(|m| !self.move_involves_nodes(&m.move_type, &affected_nodes));

                    // 4. Generate new potential moves around affected nodes (needs helper function)
                    let new_moves = self.generate_moves_around_nodes(
                        instance,
                        &current_solution,
                        &affected_nodes,
                    );

                    // 5. Merge new moves into the sorted list
                    // Simple approach: add all and re-sort. Better: insert maintaining order.
                    // Let's use insertion while maintaining sort (like insertion sort logic)
                    // Need to handle potential duplicates (e.g. a new move identical to one already in list)
                    let mut unique_new_moves = HashSet::new(); // Track move types to avoid duplicates for now
                    for existing_move in &move_list {
                        unique_new_moves.insert(existing_move.move_type.clone()); // Use clone if Move doesn't impl Copy
                    }

                    for new_move in new_moves {
                        if new_move.delta < 0 && unique_new_moves.insert(new_move.move_type.clone())
                        {
                            // Find insertion point using binary search based on delta
                            match move_list
                                .binary_search_by(|probe| probe.delta.cmp(&new_move.delta))
                            {
                                Ok(pos) => move_list.insert(pos, new_move), // Insert among equals
                                Err(pos) => move_list.insert(pos, new_move), // Insert at sorted position
                            }
                        }
                    }
                    // Alternative: Add all then sort (simpler but potentially slower if list is large)
                    // move_list.extend(new_moves.into_iter().filter(|m| m.delta < 0));
                    // move_list.sort_unstable_by(|a, b| a.delta.cmp(&b.delta));
                    // move_list.dedup_by(|a, b| a.move_type == b.move_type); // Requires PartialEq on Move
                }

                // Check if cost actually improved
                if current_cost >= cost_before_iter {
                    // Compare with cost *before* this iteration started
                    // Allow moves with delta 0 if they enable future improvements? For now, stop.
                    progress_callback(format!(
                        "[Finished] No cost improvement (Cost: {} >= {}). Final Cost: {}",
                        current_cost, cost_before_iter, current_cost
                    ));
                    break; // Stop if cost doesn't decrease
                }
            } else {
                // No improving move found (either none exist or none are valid in MoveListSteepest)
                progress_callback(format!(
                    "[Finished] Local optimum found / No valid moves. Final Cost: {}",
                    current_cost
                ));
                break;
            }
        } // end loop

        current_solution
    }
}

// Helper methods for LocalSearch
impl LocalSearch {
    /// Generates all potentially improving moves (delta < 0) for the current solution state.
    fn generate_all_improving_moves(
        &self,
        instance: &TsplibInstance,
        solution: &Solution,
    ) -> Vec<EvaluatedMove> {
        let mut moves = Vec::new();

        // 1. Inter-route moves
        for pos1 in 0..solution.cycle1.len() {
            for pos2 in 0..solution.cycle2.len() {
                if let Some(m) = evaluate_inter_route_exchange(solution, instance, pos1, pos2) {
                    if m.delta < 0 {
                        moves.push(m);
                    }
                }
            }
        }

        // 2. Intra-route moves
        for cycle_id in [CycleId::Cycle1, CycleId::Cycle2] {
            let cycle_vec = solution.get_cycle(cycle_id);
            let n = cycle_vec.len();
            match self.neighborhood {
                NeighborhoodType::VertexExchange => {
                    if n >= 2 {
                        for pos1 in 0..n {
                            for pos2 in pos1 + 1..n {
                                if let Some(m) = evaluate_intra_route_vertex_exchange(
                                    solution, instance, cycle_id, pos1, pos2,
                                ) {
                                    if m.delta < 0 {
                                        moves.push(m);
                                    }
                                }
                            }
                        }
                    }
                }
                NeighborhoodType::EdgeExchange => {
                    if n >= 3 {
                        for pos1 in 0..n {
                            for pos2_offset in 2..n {
                                // Generate non-adjacent pairs
                                let pos2 = (pos1 + pos2_offset) % n;
                                if pos1 < pos2 || (pos2 == 0 && pos1 == n - 1) {
                                    // Avoid duplicates
                                    if !(pos1 == 0 && pos2 == n - 1) {
                                        // Avoid adjacent wrap-around
                                        if let Some(m) = evaluate_intra_route_edge_exchange(
                                            solution, instance, cycle_id, pos1, pos2,
                                        ) {
                                            if m.delta < 0 {
                                                moves.push(m);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        moves
    }

    /// Generates candidate moves (delta < 0) based on k-nearest neighbors.
    fn generate_candidate_moves(
        &self,
        instance: &TsplibInstance,
        solution: &Solution,
        k: usize,
    ) -> Vec<EvaluatedMove> {
        let mut moves = Vec::new();
        // Assumes instance.precompute_nearest_neighbors(k) was called.
        for node_a in 0..instance.dimension {
            let neighbors = instance.get_nearest_neighbors(node_a);
            let node_a_info_opt = solution.find_node(node_a);
            if node_a_info_opt.is_none() {
                continue;
            }
            let (cycle_a, pos_a) = node_a_info_opt.unwrap();

            for &node_b in neighbors {
                if node_a == node_b {
                    continue;
                }
                let node_b_info_opt = solution.find_node(node_b);
                if node_b_info_opt.is_none() {
                    continue;
                }
                let (cycle_b, pos_b) = node_b_info_opt.unwrap();

                if cycle_a != cycle_b {
                    let (actual_pos_a, actual_pos_b) = if cycle_a == CycleId::Cycle1 {
                        (pos_a, pos_b)
                    } else {
                        (pos_b, pos_a)
                    };
                    if let Some(m) = evaluate_inter_route_exchange(
                        solution,
                        instance,
                        actual_pos_a,
                        actual_pos_b,
                    ) {
                        if m.delta < 0 {
                            moves.push(m);
                        }
                    }
                } else {
                    match self.neighborhood {
                        NeighborhoodType::EdgeExchange => {
                            if let Some(m) = evaluate_candidate_intra_route_edge_exchange(
                                solution, instance, cycle_a, pos_a, pos_b,
                            ) {
                                if m.delta < 0 {
                                    moves.push(m);
                                }
                            }
                        }
                        NeighborhoodType::VertexExchange => {
                            if let Some(m) = evaluate_intra_route_vertex_exchange(
                                solution, instance, cycle_a, pos_a, pos_b,
                            ) {
                                if m.delta < 0 {
                                    moves.push(m);
                                }
                            }
                        }
                    }
                }
            }
        }
        moves
    }

    /// Checks if a move is valid in the current solution state.
    /// Used by MoveListSteepest.
    fn is_move_valid(&self, solution: &Solution, move_type: &Move) -> bool {
        match move_type {
            Move::InterRouteExchange { v1, v2 } => {
                // Valid if v1 is in one cycle and v2 is in the other.
                let info1 = solution.find_node(*v1);
                let info2 = solution.find_node(*v2);
                match (info1, info2) {
                    (Some((c1, _)), Some((c2, _))) => c1 != c2,
                    _ => false, // One or both nodes not found
                }
            }
            Move::IntraRouteVertexExchange { v1, v2, cycle } => {
                // Valid if v1 and v2 are both found in the specified cycle.
                let info1 = solution.find_node(*v1);
                let info2 = solution.find_node(*v2);
                match (info1, info2) {
                    (Some((c1, _)), Some((c2, _))) => c1 == *cycle && c2 == *cycle,
                    _ => false,
                }
            }
            Move::IntraRouteEdgeExchange { a, b, c, d, cycle } => {
                // Valid if edge (a, b) and edge (c, d) both exist in the specified cycle.
                // Note: has_edge checks both directions, but for 2-opt validity,
                // we need the specific directed edges a->b and c->d.
                let edge1_check = solution.check_edge_in_cycle(solution.get_cycle(*cycle), *a, *b);
                let edge2_check = solution.check_edge_in_cycle(solution.get_cycle(*cycle), *c, *d);
                // Check if edges exist in the correct direction (result is Some(1))
                edge1_check == Some(1) && edge2_check == Some(1)
            }
        }
    }

    /// **Placeholder:** Identifies nodes potentially affected by an applied move.
    /// This needs a proper implementation based on the move type.
    fn identify_affected_nodes(&self, applied_move: &Move, solution: &Solution) -> HashSet<usize> {
        let mut affected = HashSet::new();
        // TODO: Implement logic based on Python's SearchMemory.next_moves
        // Example:
        match applied_move {
            Move::InterRouteExchange { v1, v2 } => {
                // Nodes involved and their neighbors *after* the move
                affected.insert(*v1);
                affected.insert(*v2);
                // Add neighbors... (requires looking up positions in solution)
            }
            Move::IntraRouteVertexExchange { v1, v2, cycle } => {
                affected.insert(*v1);
                affected.insert(*v2);
                // Add neighbors in the cycle...
            }
            Move::IntraRouteEdgeExchange { a, b, c, d, cycle } => {
                // These are the nodes *before* the 2-opt reverse.
                // Affected nodes are a, b, c, d and their neighbors *after* the reverse.
                affected.insert(*a);
                affected.insert(*b);
                affected.insert(*c);
                affected.insert(*d);
                // Add neighbors after reverse...
            }
        }
        // Placeholder: return empty set until implemented
        affected
    }

    /// **Placeholder:** Checks if a move involves any of the specified nodes.
    /// This needs a proper implementation.
    fn move_involves_nodes(&self, move_type: &Move, affected_nodes: &HashSet<usize>) -> bool {
        if affected_nodes.is_empty() {
            return false;
        } // Optimization
        // TODO: Implement logic to check if nodes in move_type intersect with affected_nodes
        match move_type {
            Move::InterRouteExchange { v1, v2 } => {
                affected_nodes.contains(v1) || affected_nodes.contains(v2)
            }
            Move::IntraRouteVertexExchange { v1, v2, .. } => {
                affected_nodes.contains(v1) || affected_nodes.contains(v2)
            }
            Move::IntraRouteEdgeExchange { a, b, c, d, .. } => {
                affected_nodes.contains(a)
                    || affected_nodes.contains(b)
                    || affected_nodes.contains(c)
                    || affected_nodes.contains(d)
            } // Add other move types if necessary
        }
        // Placeholder
        // false
    }

    /// **Placeholder:** Generates new potential improving moves involving the affected nodes.
    /// This needs a proper implementation based on Python's SearchMemory.next_moves logic.
    fn generate_moves_around_nodes(
        &self,
        instance: &TsplibInstance,
        solution: &Solution,
        affected_nodes: &HashSet<usize>,
    ) -> Vec<EvaluatedMove> {
        let mut new_moves = Vec::new();
        if affected_nodes.is_empty() {
            return new_moves;
        }

        // TODO: Re-implement parts of generate_all_improving_moves, but only
        // considering pairs where at least one node is in affected_nodes.
        // Remember to check for delta < 0.

        // Example Sketch:
        // for &node_a in affected_nodes {
        //     // Find node_a position
        //     if let Some((cycle_id_a, pos_a)) = solution.find_node(node_a) {
        //         // 1. Try Inter-route moves with all nodes in the *other* cycle
        //         let other_cycle_id = if cycle_id_a == CycleId::Cycle1 { CycleId::Cycle2 } else { CycleId::Cycle1 };
        //         let other_cycle = solution.get_cycle(other_cycle_id);
        //         for pos_b in 0..other_cycle.len() {
        //              // Evaluate inter-route move between (node_a, pos_a) and (other_cycle[pos_b], pos_b)
        //              // Add to new_moves if delta < 0
        //         }
        //
        //         // 2. Try Intra-route moves within the *same* cycle
        //         let same_cycle = solution.get_cycle(cycle_id_a);
        //         for pos_b in 0..same_cycle.len() {
        //              let node_b = same_cycle[pos_b];
        //              if node_a == node_b { continue; }
        //              // Evaluate intra-route move (Vertex or Edge) between (node_a, pos_a) and (node_b, pos_b)
        //              // Ensure not generating duplicate moves already handled by iterating affected_nodes
        //              // Add to new_moves if delta < 0
        //         }
        //     }
        // }

        // Placeholder: return empty vec until implemented
        new_moves
    }
}

// Need to ensure Move implements PartialEq, Eq, Hash, Clone for HashSet usage
// Need to ensure EvaluatedMove implements necessary traits if used in complex collections
