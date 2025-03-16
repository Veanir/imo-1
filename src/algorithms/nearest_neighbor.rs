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

    fn build_cycle(
        start: usize,
        mut available: Vec<usize>,
        target_size: usize,
        instance: &TsplibInstance,
    ) -> Vec<usize> {
        let mut cycle = vec![start];
        
        while cycle.len() < target_size && !available.is_empty() {
            let last = cycle.last().unwrap();
            let nearest = NearestNeighbor::find_nearest(&NearestNeighbor, *last, &available, instance);
            cycle.push(nearest);
            available.retain(|&x| x != nearest);
        }
        
        cycle
    }
}

impl TspAlgorithm for NearestNeighbor {
    fn name(&self) -> &str {
        "Nearest Neighbor"
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
        let cycle1 = Self::build_cycle(start1, available1, (n + 1) / 2, instance);
        let cycle2 = Self::build_cycle(start2, available2, n / 2, instance);

        Solution::new(cycle1, cycle2)
    }
} 