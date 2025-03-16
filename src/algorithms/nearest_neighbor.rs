use crate::tsplib::{TsplibInstance, Solution};
use crate::algorithm::TspAlgorithm;


pub struct NearestNeighbor;

impl NearestNeighbor {
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
}

impl TspAlgorithm for NearestNeighbor {
    fn name(&self) -> &str {
        "Nearest Neighbor"
    }

    fn solve(&self, instance: &TsplibInstance) -> Solution {
        let n = instance.size();
        let (start1, start2) = self.find_max_distance_pair(instance);
        
        // Initialize cycles with starting vertices
        let mut cycle1 = vec![start1];
        let mut cycle2 = vec![start2];
        
        // Create set of available vertices (excluding starting vertices)
        let mut available: Vec<usize> = (0..n).filter(|&x| x != start1 && x != start2).collect();
        
        // Alternate between cycles until all vertices are assigned
        let mut current_cycle = 1; // Start with cycle 1
        
        while !available.is_empty() {
            if current_cycle == 1 {
                // Add nearest vertex to the last vertex in cycle 1
                let last = *cycle1.last().unwrap();
                let nearest = self.find_nearest(last, &available, instance);
                cycle1.push(nearest);
                available.retain(|&x| x != nearest);
                current_cycle = 2;
            } else {
                // Add nearest vertex to the last vertex in cycle 2
                let last = *cycle2.last().unwrap();
                let nearest = self.find_nearest(last, &available, instance);
                cycle2.push(nearest);
                available.retain(|&x| x != nearest);
                current_cycle = 1;
            }
        }

        Solution::new(cycle1, cycle2)
    }
} 