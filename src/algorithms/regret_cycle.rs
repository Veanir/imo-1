use crate::tsplib::{TsplibInstance, Solution};
use crate::algorithm::TspAlgorithm;
use rand::Rng;

pub struct RegretCycle {
    pub k_regret: usize, // k value for k-regret (2 for this task)
}

impl RegretCycle {
    pub fn new() -> Self {
        Self { k_regret: 2 }
    }

    fn find_max_distance_pair(&self, instance: &TsplibInstance) -> (usize, usize) {
        let n = instance.size();
        (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .max_by_key(|&(i, j)| instance.distance(i, j))
            .unwrap_or((0, 1))
    }

    fn find_nearest(&self, from: usize, available: &[usize], instance: &TsplibInstance) -> usize {
        available
            .iter()
            .min_by_key(|&&vertex| instance.distance(from, vertex))
            .copied()
            .unwrap_or(available[0])
    }

    fn calculate_insertion_cost(
        &self,
        vertex: usize,
        pos: usize,
        cycle: &[usize],
        instance: &TsplibInstance,
    ) -> i32 {
        if cycle.is_empty() {
            return 0;
        }
        if cycle.len() == 1 {
            return instance.distance(cycle[0], vertex) * 2;
        }

        let prev = cycle[if pos == 0 { cycle.len() - 1 } else { pos - 1 }];
        let next = cycle[pos % cycle.len()];

        instance.distance(prev, vertex) + 
        instance.distance(vertex, next) - 
        instance.distance(prev, next)
    }

    // Calculate regret value and best insertion position for a vertex
    fn calculate_regret(&self, vertex: usize, cycle: &[usize], instance: &TsplibInstance) -> (i32, usize) {
        if cycle.is_empty() {
            return (0, 0);
        }

        // Calculate costs for all possible insertion positions
        let mut costs: Vec<(usize, i32)> = (0..=cycle.len())
            .map(|pos| (pos, self.calculate_insertion_cost(vertex, pos, cycle, instance)))
            .collect();

        // Sort by cost (best/lowest first)
        costs.sort_by_key(|&(_, cost)| cost);

        // Calculate regret as difference between k-th best and best insertion
        let best_cost = costs[0].1;
        let k_best_cost = costs.get(self.k_regret - 1).map_or(best_cost, |&(_, cost)| cost);
        let regret = k_best_cost - best_cost;

        (regret, costs[0].0) // Return (regret value, best position)
    }

    // Select the best vertex based on regret and return its best insertion position
    fn select_best_vertex(
        &self,
        cycle: &[usize],
        available: &[usize],
        instance: &TsplibInstance,
    ) -> Option<(usize, usize)> {
        if available.is_empty() {
            return None;
        }

        available.iter()
            .map(|&vertex| {
                let (regret, pos) = self.calculate_regret(vertex, cycle, instance);
                (vertex, pos, regret)
            })
            .max_by_key(|&(_, _, regret)| regret) // Choose vertex with highest regret
            .map(|(v, p, _)| (v, p))
    }
}

impl TspAlgorithm for RegretCycle {
    fn name(&self) -> &str {
        "2-Regret Cycle"
    }

    fn solve(&self, instance: &TsplibInstance) -> Solution {
        let n = instance.size();
        let (start1, start2) = self.find_max_distance_pair(instance);
        
        // Initialize cycles with starting vertices
        let mut cycle1 = vec![start1];
        let mut cycle2 = vec![start2];
        
        // Create set of available vertices (excluding starting vertices)
        let mut available: Vec<usize> = (0..n).filter(|&x| x != start1 && x != start2).collect();
        
        // Add initial vertices to each cycle if possible
        if !available.is_empty() {
            let nearest1 = self.find_nearest(start1, &available, instance);
            cycle1.push(nearest1);
            available.retain(|&x| x != nearest1);
            
            if !available.is_empty() {
                let nearest2 = self.find_nearest(start2, &available, instance);
                cycle2.push(nearest2);
                available.retain(|&x| x != nearest2);
            }
        }
        
        // Alternate between cycles until all vertices are assigned
        let mut current_cycle = 1; // Start with cycle 1
        
        while !available.is_empty() {
            if current_cycle == 1 {
                // Add to cycle 1
                if let Some((best_vertex, best_pos)) = self.select_best_vertex(&cycle1, &available, instance) {
                    cycle1.insert(best_pos, best_vertex);
                    available.retain(|&x| x != best_vertex);
                }
                current_cycle = 2;
            } else {
                // Add to cycle 2
                if let Some((best_vertex, best_pos)) = self.select_best_vertex(&cycle2, &available, instance) {
                    cycle2.insert(best_pos, best_vertex);
                    available.retain(|&x| x != best_vertex);
                }
                current_cycle = 1;
            }
        }

        Solution::new(cycle1, cycle2)
    }
}
