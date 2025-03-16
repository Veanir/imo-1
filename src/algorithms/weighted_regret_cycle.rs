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

    fn build_cycle(
        &self,
        start: usize,
        mut available: Vec<usize>,
        target_size: usize,
        instance: &TsplibInstance,
    ) -> Vec<usize> {
        // Initialize cycle with start vertex
        let mut cycle = vec![start];

        // Find nearest vertex to start and add it
        if !available.is_empty() {
            let nearest = self.find_nearest(start, &available, instance);
            cycle.push(nearest);
            available.retain(|&x| x != nearest);
        }

        // Keep adding vertices until target size is reached
        while cycle.len() < target_size && !available.is_empty() {
            // Calculate weighted scores for all available vertices
            let (best_vertex, best_pos) = available.iter()
                .map(|&vertex| {
                    let (score, pos) = self.calculate_weighted_score(vertex, &cycle, instance);
                    (vertex, pos, score)
                })
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) // Choose vertex with highest weighted score
                .map(|(v, p, _)| (v, p))
                .unwrap();

            // Insert chosen vertex at its best position
            cycle.insert(best_pos, best_vertex);
            available.retain(|&x| x != best_vertex);
        }

        cycle
    }
}

impl TspAlgorithm for WeightedRegretCycle {
    fn name(&self) -> &str {
        "Weighted 2-Regret Cycle"
    }

    fn solve(&self, instance: &TsplibInstance) -> Solution {
        let n = instance.size();
        let (start1, start2) = self.find_max_distance_pair(instance);
        
        // Create two complementary sets of available vertices
        let mut vertices: Vec<usize> = (0..n).filter(|&x| x != start1 && x != start2).collect();
        let (available1, available2) = vertices.iter()
            .enumerate()
            .fold((Vec::new(), Vec::new()), |(mut odd, mut even), (idx, &v)| {
                if idx % 2 == 0 {
                    even.push(v);
                } else {
                    odd.push(v);
                }
                (odd, even)
            });

        // Build cycles with their respective available vertices
        let cycle1 = self.build_cycle(start1, available1, (n + 1) / 2, instance);
        let cycle2 = self.build_cycle(start2, available2, n / 2, instance);

        Solution::new(cycle1, cycle2)
    }
}
