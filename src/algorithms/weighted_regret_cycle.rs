use crate::tsplib::{TsplibInstance, Solution};
use crate::algorithm::TspAlgorithm;

pub struct WeightedRegretCycle {
    pub k_regret: usize,
    pub regret_weight: f64,
    pub greedy_weight: f64,
}

impl WeightedRegretCycle {
    pub fn new(regret_weight: f64, greedy_weight: f64) -> Self {
        Self {
            k_regret: 2,
            regret_weight,
            greedy_weight,
        }
    }

    pub fn default() -> Self {
        // Default weights as per task description
        Self::new(1.0, -1.0)
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

    fn calculate_weighted_score(&self, vertex: usize, cycle: &[usize], instance: &TsplibInstance) -> (f64, usize) {
        if cycle.is_empty() {
            return (0.0, 0);
        }

        // Calculate costs for all possible insertion positions
        let mut costs: Vec<(usize, i32)> = (0..=cycle.len())
            .map(|pos| (pos, self.calculate_insertion_cost(vertex, pos, cycle, instance)))
            .collect();

        // Sort by cost (best/lowest first)
        costs.sort_by_key(|&(_, cost)| cost);

        // Calculate regret component (k-best - best)
        let best_cost = costs[0].1;
        let k_best_cost = costs.get(self.k_regret - 1).map_or(best_cost, |&(_, cost)| cost);
        let regret = k_best_cost - best_cost;

        // Calculate weighted score
        let weighted_score = 
            self.regret_weight * regret as f64 +  // Regret component
            self.greedy_weight * best_cost as f64; // Greedy component

        (weighted_score, costs[0].0) // Return (weighted score, best position)
    }

    // Select the best vertex based on weighted score and return its best insertion position
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
                let (score, pos) = self.calculate_weighted_score(vertex, cycle, instance);
                (vertex, pos, score)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) // Choose vertex with highest weighted score
            .map(|(v, p, _)| (v, p))
    }
}

impl TspAlgorithm for WeightedRegretCycle {
    fn name(&self) -> &str {
        "Weighted 2-Regret Cycle"
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
