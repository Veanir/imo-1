mod algorithm;
mod algorithms;
mod moves;
mod tsplib;
mod utils;
mod visualization;

use algorithm::{TspAlgorithm, format_stats_row, run_experiment};
use algorithms::constructive::weighted_regret_cycle::WeightedRegretCycle;
use algorithms::local_search::base::{
    HeuristicAlgorithm, InitialSolutionType, LocalSearch, NeighborhoodType, SearchVariant,
};
use algorithms::random_walk::RandomWalk;
use std::fs::create_dir_all;
use std::path::Path;
use tsplib::TsplibInstance;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading instances...");

    // Create output directory for visualizations
    create_dir_all("output")?;

    // Load both instances
    let instances = [
        (
            "kroa200",
            TsplibInstance::from_file(Path::new("tsplib/kroa200.tsp")),
        ),
        (
            "krob200",
            TsplibInstance::from_file(Path::new("tsplib/krob200.tsp")),
        ),
    ];

    // Create algorithms for Lab 2
    let algorithms: Vec<Box<dyn TspAlgorithm>> = vec![
        // --- Reference Algorithms ---
        Box::new(WeightedRegretCycle::default()), // Best from Lab 1 (assuming)
        Box::new(RandomWalk::default()),          // Default iterations for now
        // --- Local Search Variants ---
        // Start with Random solution
        Box::new(LocalSearch::new(
            SearchVariant::Steepest,
            NeighborhoodType::VertexExchange,
            InitialSolutionType::Random,
        )),
        Box::new(LocalSearch::new(
            SearchVariant::Steepest,
            NeighborhoodType::EdgeExchange,
            InitialSolutionType::Random,
        )),
        Box::new(LocalSearch::new(
            SearchVariant::Greedy,
            NeighborhoodType::VertexExchange,
            InitialSolutionType::Random,
        )),
        Box::new(LocalSearch::new(
            SearchVariant::Greedy,
            NeighborhoodType::EdgeExchange,
            InitialSolutionType::Random,
        )),
        // Start with Heuristic solution (WeightedRegret)
        Box::new(LocalSearch::new(
            SearchVariant::Steepest,
            NeighborhoodType::VertexExchange,
            InitialSolutionType::Heuristic(HeuristicAlgorithm::WeightedRegret),
        )),
        Box::new(LocalSearch::new(
            SearchVariant::Steepest,
            NeighborhoodType::EdgeExchange,
            InitialSolutionType::Heuristic(HeuristicAlgorithm::WeightedRegret),
        )),
        Box::new(LocalSearch::new(
            SearchVariant::Greedy,
            NeighborhoodType::VertexExchange,
            InitialSolutionType::Heuristic(HeuristicAlgorithm::WeightedRegret),
        )),
        Box::new(LocalSearch::new(
            SearchVariant::Greedy,
            NeighborhoodType::EdgeExchange,
            InitialSolutionType::Heuristic(HeuristicAlgorithm::WeightedRegret),
        )),
    ];

    // Collect results for summary
    let mut all_results = Vec::new();
    let mut slowest_ls_avg_time: f64 = 0.0; // To potentially adjust RandomWalk later

    // Run experiments for each instance
    for (name, instance_result) in instances.iter() {
        println!("\nProcessing instance: {}", name);

        match instance_result {
            Ok(instance) => {
                for algorithm in &algorithms {
                    // TODO: Adjust RandomWalk iterations/time based on slowest LS run
                    // This requires a more complex setup: run LS first, find max time,
                    // then run RW with that time/equivalent iterations.
                    // For now, we use the default iterations set in RandomWalk::default().

                    println!("  Running algorithm: {}", algorithm.name());
                    let stats = run_experiment(&**algorithm, instance, 2);

                    // Track slowest LS average time
                    if algorithm.name().contains("Local Search") {
                        slowest_ls_avg_time = slowest_ls_avg_time.max(stats.avg_time_ms);
                    }

                    all_results.push((name.to_string(), stats));

                    // Create visualization for the best solution
                    let safe_algo_name = algorithm
                        .name()
                        .replace(|c: char| !c.is_alphanumeric() && c != '-', "_")
                        .replace("__", "_");
                    let output_path = format!("output/{}_{}.png", name, safe_algo_name);
                    visualization::plot_solution(
                        instance,
                        &all_results.last().unwrap().1.best_solution,
                        &format!("{} - {}", algorithm.name(), name),
                        Path::new(&output_path),
                    )?;
                }
            }
            Err(e) => println!("Error loading {}: {}", name, e),
        }
    }

    // Print summary table
    println!("\nSummary of Results:");
    println!("| Instance | Algorithm | Cost (min - max) | Time (ms) |");
    println!("|----------|-----------|------------------|-----------|");
    for (instance_name, stats) in all_results {
        println!(
            "| {} | {}",
            instance_name,
            format_stats_row(&stats).trim_start_matches("| ")
        );
    }

    println!("\nVisualizations have been saved to the 'output' directory.");
    Ok(())
}
