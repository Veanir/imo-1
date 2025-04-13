use crate::tsplib::{Solution, TsplibInstance};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

// Type alias for the progress callback
pub type ProgressCallback<'a> = &'a mut dyn FnMut(String);

// Trait that all TSP algorithms must implement
pub trait TspAlgorithm {
    fn name(&self) -> &str;

    // Original solve method (optional, can be removed if not needed elsewhere)
    // fn solve(&self, instance: &TsplibInstance) -> Solution;

    /// Solves the instance, optionally providing status updates via callback.
    fn solve_with_feedback(
        &self,
        instance: &TsplibInstance,
        progress_callback: ProgressCallback,
    ) -> Solution;
}

// Results of a single algorithm run
#[derive(Debug)]
pub struct RunResult {
    pub cost: i32,
    pub solution: Solution,
    pub time_ms: u128,
}

// Statistics for multiple runs
#[derive(Debug)]
pub struct ExperimentStats {
    pub algorithm_name: String,
    pub instance_name: String,
    pub min_cost: i32,
    pub max_cost: i32,
    pub avg_cost: f64,
    pub best_solution: Solution,
    pub avg_time_ms: f64,
    pub num_runs: usize,
}

// Run experiment multiple times and collect statistics
pub fn run_experiment(
    algorithm: &dyn TspAlgorithm,
    instance: &TsplibInstance,
    num_runs: usize,
) -> ExperimentStats {
    if num_runs == 0 {
        return ExperimentStats {
            algorithm_name: algorithm.name().to_string(),
            instance_name: instance.name.clone(),
            min_cost: 0,
            max_cost: 0,
            avg_cost: 0.0,
            best_solution: Solution::new(vec![], vec![]),
            avg_time_ms: 0.0,
            num_runs: 0,
        };
    }

    let mut results = Vec::with_capacity(num_runs);

    // Create a progress bar
    let pb = ProgressBar::new(num_runs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                // Added {msg} to display the custom message
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
            )
            .unwrap()
            .progress_chars("# >-"), // Changed filler char for clarity
    );
    pb.set_prefix(format!("Running {}", algorithm.name()));
    pb.set_message("Starting..."); // Initial message

    // Run the algorithm multiple times
    for run_index in 0..num_runs {
        let start = Instant::now();

        // Create the callback closure which updates the progress bar message
        let mut callback = |status: String| {
            // Include run number in the message
            pb.set_message(format!("[Run {}/{}] {}", run_index + 1, num_runs, status));
        };

        // Call the solve method with the callback
        let solution = algorithm.solve_with_feedback(instance, &mut callback);
        let elapsed = start.elapsed();

        // Validate solution
        assert!(
            solution.is_valid(instance),
            "Invalid solution produced by {}",
            algorithm.name()
        );

        let result = RunResult {
            cost: solution.calculate_cost(instance),
            solution,
            time_ms: elapsed.as_millis(),
        };
        results.push(result);
        pb.inc(1);
        // Clear message after each run completes incrementing
        pb.set_message("Done run.");
    }
    pb.finish_with_message("Finished all runs."); // Final message

    // Calculate statistics
    let mut min_cost = i32::MAX;
    let mut max_cost = i32::MIN;
    let mut sum_cost: i64 = 0;
    let mut sum_time: u128 = 0;
    let mut best_solution = None;

    for result in &results {
        if result.cost < min_cost {
            min_cost = result.cost;
            best_solution = Some(result.solution.clone());
        }
        max_cost = max_cost.max(result.cost);
        sum_cost += result.cost as i64;
        sum_time += result.time_ms;
    }

    let final_best_solution = best_solution.expect("Best solution should exist if num_runs > 0");

    ExperimentStats {
        algorithm_name: algorithm.name().to_string(),
        instance_name: instance.name.clone(),
        min_cost,
        max_cost,
        avg_cost: sum_cost as f64 / num_runs as f64,
        best_solution: final_best_solution,
        avg_time_ms: sum_time as f64 / num_runs as f64,
        num_runs,
    }
}

// Helper function to format experiment results as a table row
pub fn format_stats_row(stats: &ExperimentStats) -> String {
    if stats.num_runs == 0 {
        return format!("| {} | No runs executed | N/A |", stats.algorithm_name);
    }
    format!(
        "| {} | {:.2} ({} - {}) | {:.2} |",
        stats.algorithm_name, stats.avg_cost, stats.min_cost, stats.max_cost, stats.avg_time_ms
    )
}
