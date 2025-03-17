use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use thiserror::Error;
use regex::Regex;
use lazy_static::lazy_static;

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
                    let x = caps[2].parse::<f64>()
                        .map_err(|e| TsplibError::Parse(format!("Failed to parse x coordinate: {}", e)))?;
                    let y = caps[3].parse::<f64>()
                        .map_err(|e| TsplibError::Parse(format!("Failed to parse y coordinate: {}", e)))?;
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
                        dimension = value.parse()
                            .map_err(|e| TsplibError::Parse(format!("Failed to parse dimension: {}", e)))?;
                    }
                    "EDGE_WEIGHT_TYPE" => {
                        edge_weight_type = Some(match value.as_str() {
                            "EXPLICIT" => EdgeWeightType::Explicit,
                            "EUC_2D" => EdgeWeightType::Euc2D,
                            "CEIL_2D" => EdgeWeightType::Ceil2D,
                            "GEO" => EdgeWeightType::Geo,
                            "ATT" => EdgeWeightType::Att,
                            _ => return Err(TsplibError::Format(format!("Unsupported EDGE_WEIGHT_TYPE: {}", value))),
                        });
                    }
                    _ => {} // Ignore other keywords for now
                }
            }
        }

        let edge_weight_type = edge_weight_type.ok_or_else(|| 
            TsplibError::Format("Missing EDGE_WEIGHT_TYPE".to_string()))?;

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
            _ => panic!("Only EUC_2D is supported for this task")
        }
    }

    // Get the dimension of the instance
    pub fn size(&self) -> usize {
        self.dimension
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
        
        // Check cycle1
        for &v in &self.cycle1 {
            if v >= instance.size() || used[v] {
                return false;
            }
            used[v] = true;
        }
        
        // Check cycle2
        for &v in &self.cycle2 {
            if v >= instance.size() || used[v] {
                return false;
            }
            used[v] = true;
        }
        
        // Check if all vertices are used
        used.iter().all(|&x| x)
    }
}

fn deg_to_rad(deg: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let deg = deg.round();
    pi * (deg + 5.0 / 3.0) / 180.0
} 