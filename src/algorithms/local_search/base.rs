use crate::algorithm::ProgressCallback;
use crate::algorithm::TspAlgorithm;
use crate::algorithms::constructive::weighted_regret_cycle::WeightedRegretCycle;
use crate::moves::inter_route::evaluate_inter_route_exchange;
use crate::moves::intra_route::{
    evaluate_candidate_intra_route_edge_exchange, evaluate_intra_route_edge_exchange,
    evaluate_intra_route_vertex_exchange,
};
use crate::moves::types::{CycleId, EvaluatedMove, Move};
use crate::tsplib::{Solution, TsplibInstance};
use crate::utils::generate_random_solution;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchVariant {
    Steepest,
    Greedy,
    CandidateSteepest(usize),
    MoveListSteepest,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeighborhoodType {
    VertexExchange,
    EdgeExchange,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitialSolutionType {
    Random,
    Heuristic(HeuristicAlgorithm),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HeuristicAlgorithm {
    WeightedRegret,
}

pub struct LocalSearch {
    variant: SearchVariant,
    neighborhood: NeighborhoodType,
    initial_solution_type: InitialSolutionType,
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

        let mut move_list: Vec<EvaluatedMove> = Vec::new();
        if self.variant == SearchVariant::MoveListSteepest {
            move_list = self.generate_all_improving_moves(instance, &current_solution);
            move_list.sort_unstable_by_key(|m| m.delta);
        }

        loop {
            iteration += 1;
            let cost_before_iter = current_cost;
            progress_callback(format!("[Iter: {}] Cost: {}", iteration, current_cost));

            let mut best_evaluated_move: Option<EvaluatedMove> = None;
            let mut found_improving_move = false;
            let mut best_move_index_in_list: Option<usize> = None;

            let mut current_improving_moves: Vec<EvaluatedMove> = Vec::new();

            match self.variant {
                SearchVariant::Steepest | SearchVariant::Greedy => {
                    current_improving_moves =
                        self.generate_all_improving_moves(instance, &current_solution);
                }
                SearchVariant::CandidateSteepest(k) => {
                    current_improving_moves =
                        self.generate_candidate_moves(instance, &current_solution, k);
                }
                SearchVariant::MoveListSteepest => {}
            }

            best_evaluated_move = None;
            found_improving_move = false;

            match self.variant {
                SearchVariant::Steepest | SearchVariant::CandidateSteepest(_) => {
                    best_evaluated_move = current_improving_moves
                        .iter()
                        .min_by_key(|m| m.delta)
                        .cloned();

                    if best_evaluated_move.is_some() {
                        found_improving_move = true;
                    }
                }
                SearchVariant::Greedy => {
                    current_improving_moves.shuffle(&mut rng);
                    if let Some(first_move) = current_improving_moves.into_iter().next() {
                        best_evaluated_move = Some(first_move);
                        found_improving_move = true;
                    }
                }
                SearchVariant::MoveListSteepest => {
                    for (index, evaluated_move) in move_list.iter().enumerate() {
                        if evaluated_move.delta < 0
                            && self.is_move_valid(&current_solution, &evaluated_move.move_type)
                        {
                            best_evaluated_move = Some(evaluated_move.clone());
                            found_improving_move = true;
                            best_move_index_in_list = Some(index);
                            break;
                        }
                    }
                }
            }

            if found_improving_move {
                let applied_move = best_evaluated_move.unwrap();
                let cost_before_apply = current_cost;
                applied_move.move_type.apply(&mut current_solution);
                current_cost += applied_move.delta;

                let real_cost_after_apply = current_solution.calculate_cost(instance);
                if real_cost_after_apply != current_cost {
                    eprintln!(
                        "[WARN] Cost mismatch after apply! Iter: {}, Move: {:?}, Delta: {}, Cost before: {}, Incremental cost: {}, Real cost: {}",
                        iteration,
                        applied_move.move_type,
                        applied_move.delta,
                        cost_before_apply,
                        current_cost,
                        real_cost_after_apply
                    );
                    current_cost = real_cost_after_apply;
                }

                if self.variant == SearchVariant::MoveListSteepest {
                    if let Some(applied_index) = best_move_index_in_list {
                        move_list.remove(applied_index);

                        let affected_nodes = self
                            .identify_affected_nodes(&applied_move.move_type, &current_solution);

                        move_list
                            .retain(|m| !self.move_involves_nodes(&m.move_type, &affected_nodes));

                        let new_potential_moves = self.generate_moves_around_nodes(
                            instance,
                            &current_solution,
                            &affected_nodes,
                        );

                        let mut existing_moves_set: HashSet<Move> =
                            move_list.iter().map(|em| em.move_type.clone()).collect();
                        for new_move in new_potential_moves {
                            if new_move.delta < 0
                                && !existing_moves_set.contains(&new_move.move_type)
                            {
                                move_list.push(new_move);
                                existing_moves_set
                                    .insert(move_list.last().unwrap().move_type.clone());
                            }
                        }

                        move_list.sort_unstable_by_key(|m| m.delta);
                    } else {
                        eprintln!("[WARN] MoveListSteepest applied a move but had no index?");
                    }
                }
                if current_cost >= cost_before_iter {
                    progress_callback(format!(
                        "[Finished] No significant cost improvement. Final Cost: {}",
                        current_cost
                    ));
                    break;
                }
            } else {
                progress_callback(format!(
                    "[Finished] Local optimum found or no improving moves. Final Cost: {}",
                    current_cost
                ));
                break;
            }
        }

        current_solution
    }
}

impl LocalSearch {
    fn get_neighbors(&self, solution: &Solution, node: usize) -> (Option<usize>, Option<usize>) {
        if let Some((cycle_id, pos)) = solution.find_node(node) {
            let cycle = solution.get_cycle(cycle_id);
            let n = cycle.len();
            if n <= 1 {
                (None, None)
            } else {
                let pred_pos = if pos == 0 { n - 1 } else { pos - 1 };
                let succ_pos = (pos + 1) % n;
                (Some(cycle[pred_pos]), Some(cycle[succ_pos]))
            }
        } else {
            (None, None)
        }
    }

    fn generate_all_improving_moves(
        &self,
        instance: &TsplibInstance,
        solution: &Solution,
    ) -> Vec<EvaluatedMove> {
        let mut moves = Vec::new();

        for pos1 in 0..solution.cycle1.len() {
            for pos2 in 0..solution.cycle2.len() {
                if let Some(m) = evaluate_inter_route_exchange(solution, instance, pos1, pos2) {
                    if m.delta < 0 {
                        moves.push(m);
                    }
                }
            }
        }

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
                                let pos2 = (pos1 + pos2_offset) % n;
                                if pos1 < pos2 || (pos2 == 0 && pos1 == n - 1) {
                                    if !(pos1 == 0 && pos2 == n - 1) {
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

    fn generate_candidate_moves(
        &self,
        instance: &TsplibInstance,
        solution: &Solution,
        k: usize,
    ) -> Vec<EvaluatedMove> {
        let mut moves = Vec::new();
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

    fn is_move_valid(&self, solution: &Solution, move_type: &Move) -> bool {
        match move_type {
            Move::InterRouteExchange { v1, v2 } => {
                let info1 = solution.find_node(*v1);
                let info2 = solution.find_node(*v2);
                match (info1, info2) {
                    (Some((c1, _)), Some((c2, _))) => c1 != c2,
                    _ => false,
                }
            }
            Move::IntraRouteVertexExchange { v1, v2, cycle } => {
                let info1 = solution.find_node(*v1);
                let info2 = solution.find_node(*v2);
                match (info1, info2) {
                    (Some((c1, _)), Some((c2, _))) => c1 == *cycle && c2 == *cycle,
                    _ => false,
                }
            }
            Move::IntraRouteEdgeExchange { a, b, c, d, cycle } => {
                let edge1_check = solution.check_edge_in_cycle(solution.get_cycle(*cycle), *a, *b);
                let edge2_check = solution.check_edge_in_cycle(solution.get_cycle(*cycle), *c, *d);
                edge1_check == Some(1) && edge2_check == Some(1)
            }
        }
    }

    fn identify_affected_nodes(&self, applied_move: &Move, solution: &Solution) -> HashSet<usize> {
        let mut affected = HashSet::new();

        let mut add_node_and_neighbors = |node: usize, affected: &mut HashSet<usize>| {
            let _newly_inserted = affected.insert(node);
            if let (Some(pred), Some(succ)) = self.get_neighbors(solution, node) {
                affected.insert(pred);
                affected.insert(succ);
            }
        };

        match applied_move {
            Move::InterRouteExchange { v1, v2 } => {
                add_node_and_neighbors(*v1, &mut affected);
                add_node_and_neighbors(*v2, &mut affected);
            }
            Move::IntraRouteVertexExchange { v1, v2, .. } => {
                add_node_and_neighbors(*v1, &mut affected);
                add_node_and_neighbors(*v2, &mut affected);
            }
            Move::IntraRouteEdgeExchange { a, b, c, d, .. } => {
                add_node_and_neighbors(*a, &mut affected);
                add_node_and_neighbors(*b, &mut affected);
                add_node_and_neighbors(*c, &mut affected);
                add_node_and_neighbors(*d, &mut affected);
            }
        }
        affected
    }

    fn move_involves_nodes(&self, move_type: &Move, affected_nodes: &HashSet<usize>) -> bool {
        if affected_nodes.is_empty() {
            return false;
        }
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
            }
        }
    }

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

        let mut considered_vertex_pairs = HashSet::new();
        let mut considered_inter_pairs = HashSet::new();

        for &node_a in affected_nodes {
            if let Some((cycle_id_a, pos_a)) = solution.find_node(node_a) {
                let other_cycle_id = if cycle_id_a == CycleId::Cycle1 {
                    CycleId::Cycle2
                } else {
                    CycleId::Cycle1
                };
                let other_cycle = solution.get_cycle(other_cycle_id);
                for pos_b in 0..other_cycle.len() {
                    let node_b = other_cycle[pos_b];
                    let pair = if node_a < node_b {
                        (node_a, node_b)
                    } else {
                        (node_b, node_a)
                    };
                    if considered_inter_pairs.insert(pair) {
                        let (eval_pos1, eval_pos2) = if cycle_id_a == CycleId::Cycle1 {
                            (pos_a, pos_b)
                        } else {
                            (pos_b, pos_a)
                        };
                        if let Some(m) =
                            evaluate_inter_route_exchange(solution, instance, eval_pos1, eval_pos2)
                        {
                            if m.delta < 0 {
                                new_moves.push(m);
                            }
                        }
                    }
                }

                let same_cycle = solution.get_cycle(cycle_id_a);
                let n = same_cycle.len();
                for pos_b in 0..n {
                    let node_b = same_cycle[pos_b];
                    if node_a == node_b {
                        continue;
                    }

                    match self.neighborhood {
                        NeighborhoodType::VertexExchange => {
                            let pair = if node_a < node_b {
                                (node_a, node_b)
                            } else {
                                (node_b, node_a)
                            };
                            if considered_vertex_pairs.insert(pair) {
                                if let Some(m) = evaluate_intra_route_vertex_exchange(
                                    solution, instance, cycle_id_a, pos_a, pos_b,
                                ) {
                                    if m.delta < 0 {
                                        new_moves.push(m);
                                    }
                                }
                            }
                        }
                        NeighborhoodType::EdgeExchange => {
                            let diff = (pos_a as isize - pos_b as isize).abs();
                            if n >= 3 && diff != 1 && diff != (n - 1) as isize {
                                if let Some(m) = evaluate_intra_route_edge_exchange(
                                    solution, instance, cycle_id_a, pos_a, pos_b,
                                ) {
                                    if m.delta < 0 {
                                        new_moves.push(m);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        new_moves
    }
}
