use crate::tsplib::{TsplibInstance, Solution};
use crate::algorithm::TspAlgorithm;
use rand::Rng;

pub struct GreedyCycle;

impl GreedyCycle {
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

        // Cost of new edges minus cost of removed edge
        instance.distance(prev, vertex) + 
        instance.distance(vertex, next) - 
        instance.distance(prev, next)
    }

    fn find_best_insertion(
        &self,
        vertex: usize,
        cycle: &[usize],
        instance: &TsplibInstance,
    ) -> (usize, i32) {
        if cycle.is_empty() {
            return (0, 0);
        }

        (0..=cycle.len())
            .map(|pos| (pos, self.calculate_insertion_cost(vertex, pos, cycle, instance)))
            .min_by_key(|&(_, cost)| cost)
            .unwrap()
    }

    // Select the best vertex and its insertion position based on minimum cost
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
                let (pos, cost) = self.find_best_insertion(vertex, cycle, instance);
                (vertex, pos, cost)
            })
            .min_by_key(|&(_, _, cost)| cost)
            .map(|(v, p, _)| (v, p))
    }
}

impl TspAlgorithm for GreedyCycle {
    fn name(&self) -> &str {
        "Greedy Cycle"
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
