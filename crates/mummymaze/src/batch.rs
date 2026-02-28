//! Rayon parallel orchestration for batch analysis.

use crate::error::Result;
use crate::graph::build_graph;
use crate::markov::analyze;
use crate::metrics;
use crate::parse::{self, Level};
use crate::solver;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LevelAnalysis {
    pub file_stem: String,
    pub sublevel: usize,
    pub grid_size: i32,
    pub n_states: usize,
    pub win_prob: f64,
    pub expected_steps: f64,
    pub bfs_moves: Option<u32>,
    pub difficulty: metrics::DifficultyMetrics,
}

/// Analyze a single level: BFS solve + full state graph + Markov chain + difficulty metrics.
pub fn analyze_level(file_stem: &str, sublevel: usize, lev: &Level) -> Result<LevelAnalysis> {
    let bfs = solver::solve(lev);
    let graph = build_graph(lev);
    let markov = analyze(&graph)?;
    let diff = metrics::compute(&graph, lev, &bfs);

    Ok(LevelAnalysis {
        file_stem: file_stem.to_string(),
        sublevel,
        grid_size: lev.grid_size,
        n_states: markov.n_transient,
        win_prob: markov.win_prob,
        expected_steps: markov.expected_steps,
        bfs_moves: bfs.moves,
        difficulty: diff,
    })
}

/// Gather all (file_stem, sublevel_index, Level) triples from a maze directory.
pub fn collect_levels(maze_dir: &Path) -> Result<Vec<(String, usize, Level)>> {
    let mut dat_files: Vec<PathBuf> = std::fs::read_dir(maze_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "dat")
                && p.file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.starts_with("B-"))
        })
        .collect();

    dat_files.sort_by(|a, b| {
        let a_num: i32 = a
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_prefix("B-")
            .unwrap()
            .parse()
            .unwrap_or(0);
        let b_num: i32 = b
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_prefix("B-")
            .unwrap()
            .parse()
            .unwrap_or(0);
        a_num.cmp(&b_num)
    });

    let mut all_levels = Vec::new();
    for path in &dat_files {
        let parsed = parse::parse_file(path);
        let (_, levels) = match parsed {
            Ok(v) => v,
            Err(_) => continue, // skip empty/corrupt files (e.g. B-100.dat is 0 bytes)
        };
        let stem = path.file_stem().unwrap().to_str().unwrap().to_string();
        for (i, lev) in levels.into_iter().enumerate() {
            all_levels.push((stem.clone(), i, lev));
        }
    }

    Ok(all_levels)
}

/// Analyze all levels in parallel using rayon.
///
/// `jobs`: number of threads. 0 = use rayon default (all cores).
pub fn analyze_all(
    maze_dir: &Path,
    jobs: usize,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<Vec<LevelAnalysis>> {
    let all_levels = collect_levels(maze_dir)?;

    if jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .ok(); // Ignore error if pool already built
    }

    let results: Vec<Result<LevelAnalysis>> = all_levels
        .par_iter()
        .map(|(stem, sub_idx, lev)| {
            let r = analyze_level(stem, *sub_idx, lev);
            if let Some(pb) = progress {
                pb.inc(1);
            }
            r
        })
        .collect();

    // Collect results, propagating first error
    let mut out = Vec::with_capacity(results.len());
    for r in results {
        out.push(r?);
    }

    // Sort by file stem then sublevel to ensure deterministic output
    out.sort_by(|a, b| {
        let a_num: i32 = a
            .file_stem
            .strip_prefix("B-")
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
        let b_num: i32 = b
            .file_stem
            .strip_prefix("B-")
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
        a_num.cmp(&b_num).then(a.sublevel.cmp(&b.sublevel))
    });

    Ok(out)
}

/// Analyze a single level by file path and sublevel index.
pub fn analyze_one(path: &Path, sublevel: usize) -> Result<LevelAnalysis> {
    let (_, levels) = parse::parse_file(path)?;
    if sublevel >= levels.len() {
        return Err(crate::error::MummyMazeError::Parse(format!(
            "sublevel {} out of range (file has {})",
            sublevel,
            levels.len()
        )));
    }
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    analyze_level(&stem, sublevel, &levels[sublevel])
}

/// Solve all levels and return action sequences. Parallel via rayon.
/// Returns sorted Vec of (file_stem, sublevel, Option<Vec<Action>>).
pub fn solve_all_with_actions(
    maze_dir: &Path,
    jobs: usize,
) -> Result<Vec<(String, usize, Option<Vec<crate::game::Action>>)>> {
    let all_levels = collect_levels(maze_dir)?;

    if jobs > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .ok();
    }

    let mut results: Vec<(String, usize, Option<Vec<crate::game::Action>>)> = all_levels
        .par_iter()
        .map(|(stem, sub_idx, lev)| {
            let bfs = solver::solve(lev);
            (stem.clone(), *sub_idx, bfs.actions)
        })
        .collect();

    results.sort_by(|a, b| {
        let a_num: i32 = a.0.strip_prefix("B-").unwrap_or("0").parse().unwrap_or(0);
        let b_num: i32 = b.0.strip_prefix("B-").unwrap_or("0").parse().unwrap_or(0);
        a_num.cmp(&b_num).then(a.1.cmp(&b.1))
    });

    Ok(results)
}

/// BFS-only solve for a single level.
pub fn solve_one(path: &Path, sublevel: usize) -> Result<Option<u32>> {
    let (_, levels) = parse::parse_file(path)?;
    if sublevel >= levels.len() {
        return Err(crate::error::MummyMazeError::Parse(format!(
            "sublevel {} out of range (file has {})",
            sublevel,
            levels.len()
        )));
    }
    Ok(solver::solve(&levels[sublevel]).moves)
}
