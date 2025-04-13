use lazy_static::lazy_static;
use regex::Regex;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use thiserror::Error;

// Re-export CycleId here for convenience in other modules
pub use crate::moves::types::CycleId;

#[derive(Debug, Error)]
pub enum TsplibError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Invalid format: {0}")]
    Format(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EdgeWeightType {
    Explicit,
    Euc2D,
    Ceil2D,
    Geo,
    Att,
}

#[derive(Debug, Clone)]
pub struct TsplibInstance {
    pub name: String,
    pub dimension: usize,
    pub edge_weight_type: EdgeWeightType,
    pub coordinates: Vec<(f64, f64)>,
    distances: Vec<Vec<i32>>,
    nearest_neighbors: Vec<Vec<usize>>,
}

impl TsplibInstance {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TsplibError> {
        lazy_static! {
            static ref KEYWORD_RE: Regex = Regex::new(r"^([A-Za-z_]+)\s*:\s*(.+)$").unwrap();
            static ref NODE_COORD_RE: Regex = Regex::new(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s*$").unwrap();
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut name = String::new();
        let mut dimension = 0;
        let mut edge_weight_type = None;
        let mut coordinates = Vec::new();
        let mut in_node_coord_section = false;

        while let Some(line) = lines.next() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with("COMMENT") {
                continue;
            }

            if line == "NODE_COORD_SECTION" {
                in_node_coord_section = true;
                continue;
            }

            if in_node_coord_section {
                if let Some(caps) = NODE_COORD_RE.captures(line) {
                    let x = caps[2].parse::<f64>().map_err(|e| {
                        TsplibError::Parse(format!("Failed to parse x coordinate: {}", e))
                    })?;
                    let y = caps[3].parse::<f64>().map_err(|e| {
                        TsplibError::Parse(format!("Failed to parse y coordinate: {}", e))
                    })?;
                    coordinates.push((x, y));
                } else {
                    in_node_coord_section = false;
                }
            } else if let Some(caps) = KEYWORD_RE.captures(line) {
                let key = caps[1].to_string();
                let value = caps[2].trim().to_string();

                match key.as_str() {
                    "NAME" => name = value,
                    "DIMENSION" => {
                        dimension = value.parse().map_err(|e| {
                            TsplibError::Parse(format!("Failed to parse dimension: {}", e))
                        })?;
                    }
                    "EDGE_WEIGHT_TYPE" => {
                        edge_weight_type = Some(match value.as_str() {
                            "EXPLICIT" => EdgeWeightType::Explicit,
                            "EUC_2D" => EdgeWeightType::Euc2D,
                            "CEIL_2D" => EdgeWeightType::Ceil2D,
                            "GEO" => EdgeWeightType::Geo,
                            "ATT" => EdgeWeightType::Att,
                            _ => {
                                return Err(TsplibError::Format(format!(
                                    "Unsupported EDGE_WEIGHT_TYPE: {}",
                                    value
                                )));
                            }
                        });
                    }
                    _ => {} // Ignore other keywords for now
                }
            }
        }

        let edge_weight_type = edge_weight_type
            .ok_or_else(|| TsplibError::Format("Missing EDGE_WEIGHT_TYPE".to_string()))?;

        if coordinates.is_empty() {
            return Err(TsplibError::Format("No coordinates found".to_string()));
        }

        if coordinates.len() != dimension {
            return Err(TsplibError::Format(format!(
                "Number of coordinates ({}) does not match dimension ({})",
                coordinates.len(),
                dimension
            )));
        }

        // Calculate distance matrix
        let mut instance = Self {
            name,
            dimension,
            edge_weight_type,
            coordinates,
            distances: vec![vec![0; dimension]; dimension],
            nearest_neighbors: vec![Vec::new(); dimension],
        };
        instance.calculate_distance_matrix();
        Ok(instance)
    }

    // Calculate and store the complete distance matrix
    fn calculate_distance_matrix(&mut self) {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                self.distances[i][j] = self.calculate_distance(i, j);
            }
        }
    }

    pub fn distance(&self, i: usize, j: usize) -> i32 {
        self.distances[i][j]
    }

    fn calculate_distance(&self, i: usize, j: usize) -> i32 {
        if i == j {
            return 0;
        }

        let (x1, y1) = self.coordinates[i];
        let (x2, y2) = self.coordinates[j];

        match self.edge_weight_type {
            EdgeWeightType::Euc2D => {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let dist = (dx * dx + dy * dy).sqrt();
                // Use mathematical rounding: round to nearest integer
                dist.round() as i32
            }
            _ => panic!("Only EUC_2D is supported for this task"),
        }
    }

    // Get the dimension of the instance
    pub fn size(&self) -> usize {
        self.dimension
    }

    /// Precomputes the k-nearest neighbors for each node based on the distance matrix.
    /// Stores the result internally.
    pub fn precompute_nearest_neighbors(&mut self, k: usize) {
        if k == 0 || k >= self.dimension {
            eprintln!(
                "Warning: Invalid k value ({}) for nearest neighbors. Must be 0 < k < dimension.",
                k
            );
            // Optionally clear or set to default
            self.nearest_neighbors = vec![Vec::new(); self.dimension];
            return;
        }

        // Check if already computed for this k (simple check based on length of first node's list)
        if !self.nearest_neighbors[0].is_empty() && self.nearest_neighbors[0].len() == k {
            return; // Already computed
        }

        self.nearest_neighbors = vec![Vec::with_capacity(k); self.dimension];

        for i in 0..self.dimension {
            let mut neighbors: Vec<_> = (0..self.dimension)
                .filter(|&j| i != j) // Exclude self
                .map(|j| (j, self.distances[i][j])) // Map to (node_index, distance)
                .collect();

            // Sort by distance (ascending)
            neighbors.sort_unstable_by_key(|&(_, dist)| dist);

            // Take the first k neighbors (indices only)
            self.nearest_neighbors[i] = neighbors.into_iter().take(k).map(|(idx, _)| idx).collect();
        }
    }

    /// Gets the precomputed k-nearest neighbors for a given node.
    /// Panics if `precompute_nearest_neighbors` was not called appropriately before.
    pub fn get_nearest_neighbors(&self, node_id: usize) -> &[usize] {
        // Ensure precomputation happened (basic check)
        if self.nearest_neighbors.is_empty() || self.nearest_neighbors[0].is_empty() {
            // Or if the outer vec has size 0 (although initialized with dimension)
            panic!(
                "Nearest neighbors requested but not precomputed. Call precompute_nearest_neighbors first."
            );
        }
        if node_id >= self.dimension {
            panic!(
                "Invalid node_id ({}) requested for nearest neighbors.",
                node_id
            );
        }
        &self.nearest_neighbors[node_id]
    }
}

// Represents a solution with two cycles
#[derive(Debug, Clone)]
pub struct Solution {
    pub cycle1: Vec<usize>,
    pub cycle2: Vec<usize>,
}

impl Solution {
    pub fn new(cycle1: Vec<usize>, cycle2: Vec<usize>) -> Self {
        Self { cycle1, cycle2 }
    }

    // Calculate total cost of the solution
    pub fn calculate_cost(&self, instance: &TsplibInstance) -> i32 {
        let cost1 = self.calculate_cycle_cost(&self.cycle1, instance);
        let cost2 = self.calculate_cycle_cost(&self.cycle2, instance);
        cost1 + cost2
    }

    // Calculate cost of a single cycle
    fn calculate_cycle_cost(&self, cycle: &[usize], instance: &TsplibInstance) -> i32 {
        if cycle.is_empty() {
            return 0;
        }
        let mut cost = 0;
        for i in 0..cycle.len() {
            let from = cycle[i];
            let to = cycle[(i + 1) % cycle.len()];
            cost += instance.distance(from, to);
        }
        cost
    }

    // Validate if the solution is correct (all vertices used exactly once)
    pub fn is_valid(&self, instance: &TsplibInstance) -> bool {
        let mut used = vec![false; instance.size()];
        let mut count = 0;

        // Check cycle1
        for &v in &self.cycle1 {
            if v >= instance.size() || used[v] {
                return false;
            }
            used[v] = true;
            count += 1;
        }

        // Check cycle2
        for &v in &self.cycle2 {
            if v >= instance.size() || used[v] {
                return false;
            }
            used[v] = true;
            count += 1;
        }

        // Check if all vertices are used
        count == instance.size() && used.iter().all(|&x| x)
    }

    /// Finds the cycle and index of a given node ID.
    pub fn find_node(&self, node_id: usize) -> Option<(CycleId, usize)> {
        if let Some(pos) = self.cycle1.iter().position(|&n| n == node_id) {
            Some((CycleId::Cycle1, pos))
        } else if let Some(pos) = self.cycle2.iter().position(|&n| n == node_id) {
            Some((CycleId::Cycle2, pos))
        } else {
            None
        }
    }

    /// Returns an immutable reference to the specified cycle vector.
    pub fn get_cycle(&self, cycle_id: CycleId) -> &Vec<usize> {
        match cycle_id {
            CycleId::Cycle1 => &self.cycle1,
            CycleId::Cycle2 => &self.cycle2,
        }
    }

    /// Returns a mutable reference to the specified cycle vector.
    pub fn get_cycle_mut(&mut self, cycle_id: CycleId) -> &mut Vec<usize> {
        match cycle_id {
            CycleId::Cycle1 => &mut self.cycle1,
            CycleId::Cycle2 => &mut self.cycle2,
        }
    }

    /// Checks if an edge exists between nodes `a` and `b` in either cycle.
    /// Returns `Some((cycle_id, direction))` or `None`.
    /// `direction` is +1 if edge is (a, b), -1 if edge is (b, a).
    pub fn has_edge(&self, a: usize, b: usize) -> Option<(CycleId, i8)> {
        if let Some(direction) = self.check_edge_in_cycle(&self.cycle1, a, b) {
            Some((CycleId::Cycle1, direction))
        } else if let Some(direction) = self.check_edge_in_cycle(&self.cycle2, a, b) {
            Some((CycleId::Cycle2, direction))
        } else {
            None
        }
    }

    /// Helper for `has_edge`. Checks for edge (a, b) or (b, a) in a single cycle.
    pub fn check_edge_in_cycle(&self, cycle: &[usize], a: usize, b: usize) -> Option<i8> {
        let n = cycle.len();
        if n < 2 {
            return None;
        }
        for i in 0..n {
            let u = cycle[i];
            let v = cycle[(i + 1) % n];
            if u == a && v == b {
                return Some(1); // Found (a, b)
            }
            if u == b && v == a {
                return Some(-1); // Found (b, a)
            }
        }
        None // Edge not found
    }
}
