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

    create_dir_all("output")?;

    let mut instances = [
        (
            "kroa200",
            TsplibInstance::from_file(Path::new("tsplib/kroa200.tsp")),
        ),
        (
            "krob200",
            TsplibInstance::from_file(Path::new("tsplib/krob200.tsp")),
        ),
    ];

    let algorithms: Vec<Box<dyn TspAlgorithm>> = vec![
        Box::new(WeightedRegretCycle::default()),
        Box::new(LocalSearch::new(
            SearchVariant::Steepest,
            NeighborhoodType::EdgeExchange,
            InitialSolutionType::Random,
        )),
        Box::new(LocalSearch::new(
            SearchVariant::MoveListSteepest,
            NeighborhoodType::EdgeExchange,
            InitialSolutionType::Random,
        )),
        Box::new(LocalSearch::new(
            SearchVariant::CandidateSteepest(10),
            NeighborhoodType::EdgeExchange,
            InitialSolutionType::Random,
        )),
    ];

    let mut all_results = Vec::new();
    let mut slowest_ls_avg_time: f64 = 0.0;

    for (name, instance_result) in instances.iter_mut() {
        println!("\nProcessing instance: {}", name);

        match instance_result {
            Ok(instance) => {
                println!("  Precomputing nearest neighbors (k=10)...");
                instance.precompute_nearest_neighbors(10);

                for algorithm in &algorithms {
                    println!("  Running algorithm: {}", algorithm.name());
                    let stats = run_experiment(&**algorithm, instance, 100);

                    if algorithm.name().contains("Local Search") {
                        slowest_ls_avg_time = slowest_ls_avg_time.max(stats.avg_time_ms);
                    }

                    all_results.push((name.to_string(), stats));

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
