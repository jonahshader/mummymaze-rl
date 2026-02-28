use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use mummymaze::batch;
use mummymaze::error::Result;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "mummymaze-cli", about = "Mummy Maze batch solver and Markov analyzer")]
struct Cli {
    /// Directory containing B-*.dat maze files
    maze_dir: PathBuf,

    /// Output CSV path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of threads (0 = all cores)
    #[arg(short, long, default_value = "0")]
    jobs: usize,

    /// Analyze a single file (e.g. B-0.dat)
    #[arg(long)]
    file: Option<String>,

    /// Sublevel index (used with --file)
    #[arg(long, default_value = "0")]
    sublevel: usize,

    /// BFS-only mode (skip Markov analysis)
    #[arg(long)]
    bfs_only: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(file) = &cli.file {
        // Single level mode
        let path = cli.maze_dir.join(file);
        if cli.bfs_only {
            let moves = batch::solve_one(&path, cli.sublevel)?;
            match moves {
                Some(m) => println!("{} sub {}: {} moves", file, cli.sublevel, m),
                None => println!("{} sub {}: unsolvable", file, cli.sublevel),
            }
        } else {
            let r = batch::analyze_one(&path, cli.sublevel)?;
            println!(
                "{} sub {} (grid {}): {} states, win_prob={:.6}, expected_steps={:.2}, bfs={}, dead_end={:.4}, branching={:.2}, optimal_paths={}, greedy_dev={}, safety={}",
                r.file_stem,
                r.sublevel,
                r.grid_size,
                r.n_states,
                r.win_prob,
                r.expected_steps,
                r.bfs_moves
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "unsolvable".into()),
                r.difficulty.dead_end_ratio,
                r.difficulty.avg_branching_factor,
                r.difficulty.n_optimal_solutions,
                r.difficulty.greedy_deviation_count
                    .map(|m| m.to_string())
                    .unwrap_or_default(),
                r.difficulty.path_safety
                    .map(|s| format!("{:.4}", s))
                    .unwrap_or_default(),
            );
        }
        return Ok(());
    }

    // Batch mode
    let all_levels = {
        // Count levels for progress bar
        let mut count = 0usize;
        for entry in std::fs::read_dir(&cli.maze_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "dat")
                && path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("B-"))
            {
                if let Ok((hdr, _)) = mummymaze::parse::parse_file(&path) {
                    count += hdr.num_sublevels as usize;
                }
            }
        }
        count
    };

    let pb = ProgressBar::new(all_levels as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:60.cyan/blue} {pos}/{len} ({per_sec}) {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let results = batch::analyze_all(&cli.maze_dir, cli.jobs, Some(&pb))?;
    pb.finish_with_message("done");

    // Summary
    let solved = results.iter().filter(|r| r.bfs_moves.is_some()).count();
    let unsolvable = results.iter().filter(|r| r.bfs_moves.is_none()).count();
    eprintln!(
        "\n{} levels: {} solved, {} unsolvable",
        results.len(),
        solved,
        unsolvable
    );

    // Write CSV
    if let Some(out_path) = &cli.output {
        let mut wtr = csv::Writer::from_path(out_path)?;
        wtr.write_record(["file", "sublevel", "grid_size", "n_states", "win_prob", "expected_steps", "bfs_moves", "dead_end_ratio", "avg_branching_factor", "n_optimal_solutions", "greedy_deviation_count", "path_safety"])?;
        for r in &results {
            wtr.write_record(&[
                r.file_stem.clone(),
                r.sublevel.to_string(),
                r.grid_size.to_string(),
                r.n_states.to_string(),
                format!("{:.6}", r.win_prob),
                format!("{:.2}", r.expected_steps),
                r.bfs_moves
                    .map(|m| m.to_string())
                    .unwrap_or_default(),
                format!("{:.6}", r.difficulty.dead_end_ratio),
                format!("{:.2}", r.difficulty.avg_branching_factor),
                r.difficulty.n_optimal_solutions.to_string(),
                r.difficulty.greedy_deviation_count
                    .map(|m| m.to_string())
                    .unwrap_or_default(),
                r.difficulty.path_safety
                    .map(|s| format!("{:.6}", s))
                    .unwrap_or_default(),
            ])?;
        }
        wtr.flush()?;
        eprintln!("Results written to {}", out_path.display());
    } else {
        // Print to stdout
        println!("file,sublevel,grid_size,n_states,win_prob,expected_steps,bfs_moves,dead_end_ratio,avg_branching_factor,n_optimal_solutions,greedy_deviation_count,path_safety");
        for r in &results {
            println!(
                "{},{},{},{},{:.6},{:.2},{},{:.6},{:.2},{},{},{}",
                r.file_stem,
                r.sublevel,
                r.grid_size,
                r.n_states,
                r.win_prob,
                r.expected_steps,
                r.bfs_moves
                    .map(|m| m.to_string())
                    .unwrap_or_default(),
                r.difficulty.dead_end_ratio,
                r.difficulty.avg_branching_factor,
                r.difficulty.n_optimal_solutions,
                r.difficulty.greedy_deviation_count
                    .map(|m| m.to_string())
                    .unwrap_or_default(),
                r.difficulty.path_safety
                    .map(|s| format!("{:.6}", s))
                    .unwrap_or_default(),
            );
        }
    }

    Ok(())
}
