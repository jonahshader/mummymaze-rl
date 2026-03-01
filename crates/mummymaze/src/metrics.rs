//! Difficulty metrics computed from the state graph and BFS solution.

use crate::game::{Action, State, step};
use crate::graph::{StateGraph, StateKey};
use crate::parse::Level;
use crate::solver::SolveResult;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct DifficultyMetrics {
    /// Fraction of reachable states from which WIN is unreachable.
    pub dead_end_ratio: f64,
    /// Mean number of valid actions per transient state.
    pub avg_branching_factor: f64,
    /// Number of distinct shortest action sequences reaching WIN.
    pub n_optimal_solutions: u64,
    /// Steps on the optimal path where the action doesn't decrease Manhattan distance to exit.
    /// None if unsolvable.
    pub greedy_deviation_count: Option<u32>,
    /// Average fraction of actions leading to winnable states along the optimal path.
    /// None if unsolvable.
    pub path_safety: Option<f64>,
}

/// Compute all difficulty metrics for a level.
pub fn compute(graph: &StateGraph, lev: &Level, solve: &SolveResult) -> DifficultyMetrics {
    let winning = winning_set(graph);
    let dead_end_ratio = compute_dead_end_ratio(graph, &winning);
    let avg_branching_factor = compute_avg_branching_factor(graph);
    let n_optimal_solutions = count_optimal_solutions(graph, solve);

    let (greedy_deviation_count, path_safety) = if let Some(ref actions) = solve.actions {
        let (greedy, safety) = replay_path_metrics(graph, lev, actions, &winning);
        (Some(greedy), Some(safety))
    } else {
        (None, None)
    };

    DifficultyMetrics {
        dead_end_ratio,
        avg_branching_factor,
        n_optimal_solutions,
        greedy_deviation_count,
        path_safety,
    }
}

/// Backward BFS from WIN to find all states that can reach a win.
pub fn winning_set(graph: &StateGraph) -> FxHashSet<State> {
    // Build reverse adjacency: for each transient destination, record the source.
    let mut reverse: FxHashMap<State, Vec<State>> = FxHashMap::default();
    let mut win_predecessors: Vec<State> = Vec::new();

    for (src, transitions) in &graph.transitions {
        for &(_action, dest) in transitions {
            match dest {
                StateKey::Win => {
                    win_predecessors.push(*src);
                }
                StateKey::Transient(dst) => {
                    reverse.entry(dst).or_default().push(*src);
                }
                StateKey::Dead => {}
            }
        }
    }

    // BFS backward from all states with a WIN transition.
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();
    for s in win_predecessors {
        if visited.insert(s) {
            queue.push_back(s);
        }
    }

    while let Some(cur) = queue.pop_front() {
        if let Some(preds) = reverse.get(&cur) {
            for &p in preds {
                if visited.insert(p) {
                    queue.push_back(p);
                }
            }
        }
    }

    visited
}

fn compute_dead_end_ratio(graph: &StateGraph, winning: &FxHashSet<State>) -> f64 {
    let n = graph.n_transient;
    if n == 0 {
        return 0.0;
    }
    let dead_ends = n - winning.len();
    dead_ends as f64 / n as f64
}

fn compute_avg_branching_factor(graph: &StateGraph) -> f64 {
    let n = graph.n_transient;
    if n == 0 {
        return 0.0;
    }
    let total_actions: usize = graph.transitions.values().map(|v| v.len()).sum();
    total_actions as f64 / n as f64
}

/// Count the number of distinct shortest action sequences that reach WIN.
/// Uses BFS on the state graph with distance and path-count tracking.
fn count_optimal_solutions(graph: &StateGraph, solve: &SolveResult) -> u64 {
    let optimal_depth = match solve.moves {
        Some(d) => d,
        None => return 0,
    };

    // BFS tracking (distance, path_count) per state in a single map.
    let mut info: FxHashMap<State, (u32, u64)> = FxHashMap::default();
    info.insert(graph.start, (0, 1));

    let mut queue = VecDeque::new();
    queue.push_back(graph.start);

    let mut win_count: u64 = 0;

    while let Some(cur) = queue.pop_front() {
        let (cur_dist, cur_count) = info[&cur];

        // Don't expand beyond optimal depth
        if cur_dist >= optimal_depth {
            continue;
        }

        if let Some(transitions) = graph.transitions.get(&cur) {
            for &(_action, dest) in transitions {
                let next_dist = cur_dist + 1;
                match dest {
                    StateKey::Win => {
                        if next_dist == optimal_depth {
                            win_count = win_count.saturating_add(cur_count);
                        }
                    }
                    StateKey::Transient(ns) => {
                        if let Some(entry) = info.get_mut(&ns) {
                            if next_dist == entry.0 {
                                // Same distance — add path count
                                entry.1 = entry.1.saturating_add(cur_count);
                            }
                            // If next_dist > entry.0, skip (not shortest)
                        } else {
                            // First time reaching this state
                            info.insert(ns, (next_dist, cur_count));
                            queue.push_back(ns);
                        }
                    }
                    StateKey::Dead => {}
                }
            }
        }
    }

    win_count
}

/// Compute greedy_deviation_count and path_safety in a single replay pass.
fn replay_path_metrics(
    graph: &StateGraph,
    lev: &Level,
    actions: &[Action],
    winning: &FxHashSet<State>,
) -> (u32, f64) {
    let n_steps = actions.len();
    if n_steps == 0 {
        return (0, 1.0);
    }

    let mut state = State::from_level(lev);
    let exit_row = lev.exit_row;
    let exit_col = lev.exit_col;
    let mut deviations = 0u32;
    let mut total_safety = 0.0;

    for &action in actions {
        // Greedy deviation: check if action decreases Manhattan distance
        let old_dist = (state.player_row - exit_row).abs() + (state.player_col - exit_col).abs();

        // Path safety: fraction of actions leading to winnable states
        if let Some(transitions) = graph.transitions.get(&state) {
            let total = transitions.len() as f64;
            if total > 0.0 {
                let safe = transitions
                    .iter()
                    .filter(|&&(_a, dest)| match dest {
                        StateKey::Win => true,
                        StateKey::Transient(ns) => winning.contains(&ns),
                        StateKey::Dead => false,
                    })
                    .count() as f64;
                total_safety += safe / total;
            }
        }

        // Advance state
        step(lev, &mut state, action);

        let new_dist = (state.player_row - exit_row).abs() + (state.player_col - exit_col).abs();
        if new_dist >= old_dist {
            deviations += 1;
        }
    }

    (deviations, total_safety / n_steps as f64)
}
