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
    let (n_optimal_solutions, greedy_deviation_count, path_safety) =
        optimal_path_metrics(graph, lev, solve, &winning);

    DifficultyMetrics {
        dead_end_ratio,
        avg_branching_factor,
        n_optimal_solutions,
        greedy_deviation_count,
        path_safety,
    }
}

/// Whether a transition destination leads to a winnable state.
pub fn is_winnable(dest: StateKey, winning: &FxHashSet<State>) -> bool {
    match dest {
        StateKey::Win => true,
        StateKey::Transient(ns) => winning.contains(&ns),
        StateKey::Dead => false,
    }
}

/// Compute the safety of a state: fraction of actions leading to winnable states.
pub fn state_safety(
    transitions: &[(Action, StateKey)],
    winning: &FxHashSet<State>,
) -> f64 {
    let total = transitions.len() as f64;
    if total == 0.0 {
        return 0.0;
    }
    let safe = transitions
        .iter()
        .filter(|&&(_, dest)| is_winnable(dest, winning))
        .count() as f64;
    safe / total
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

/// Compute optimal path metrics: n_optimal_solutions, greedy_deviation_count, and path_safety.
///
/// path_safety is the expected safety under a uniformly random optimal path.
/// For each state on any optimal path, we compute its safety (fraction of actions
/// leading to winnable states) and weight it by `fwd_count * bwd_count` — the number
/// of optimal paths passing through it. This makes the metric rotation-invariant.
fn optimal_path_metrics(
    graph: &StateGraph,
    lev: &Level,
    solve: &SolveResult,
    winning: &FxHashSet<State>,
) -> (u64, Option<u32>, Option<f64>) {
    let optimal_depth = match solve.moves {
        Some(d) => d,
        None => return (0, None, None),
    };

    // Forward BFS: (distance_from_start, path_count_from_start) per state.
    let mut fwd: FxHashMap<State, (u32, u64)> = FxHashMap::default();
    fwd.insert(graph.start, (0, 1));
    let mut queue = VecDeque::new();
    queue.push_back(graph.start);

    // Track which states have a WIN transition at optimal depth, with their fwd counts.
    let mut win_predecessors: Vec<(State, u64)> = Vec::new();

    while let Some(cur) = queue.pop_front() {
        let (cur_dist, cur_count) = fwd[&cur];
        if cur_dist >= optimal_depth {
            continue;
        }
        if let Some(transitions) = graph.transitions.get(&cur) {
            for &(_action, dest) in transitions {
                let next_dist = cur_dist + 1;
                match dest {
                    StateKey::Win => {
                        if next_dist == optimal_depth {
                            win_predecessors.push((cur, cur_count));
                        }
                    }
                    StateKey::Transient(ns) => {
                        if let Some(entry) = fwd.get_mut(&ns) {
                            if next_dist == entry.0 {
                                entry.1 = entry.1.saturating_add(cur_count);
                            }
                        } else {
                            fwd.insert(ns, (next_dist, cur_count));
                            queue.push_back(ns);
                        }
                    }
                    StateKey::Dead => {}
                }
            }
        }
    }

    let n_optimal_solutions: u64 = win_predecessors.iter().map(|&(_, c)| c).sum();

    // Backward BFS from WIN along optimal edges: (distance_to_win, path_count_to_win).
    // A backward edge from dst to src exists if fwd_dist[src] + 1 == fwd_dist[dst]
    // (i.e., it's on a shortest path).
    let mut bwd: FxHashMap<State, (u32, u64)> = FxHashMap::default();
    let mut bwd_queue = VecDeque::new();

    // States with direct WIN transitions at optimal depth get bwd distance 1.
    // bwd_count = number of WIN transitions from this state on optimal paths.
    for &(s, _) in &win_predecessors {
        let win_actions = graph.transitions.get(&s).map_or(0u64, |tr| {
            tr.iter().filter(|&&(_, d)| d == StateKey::Win).count() as u64
        });
        if let Some(entry) = bwd.get_mut(&s) {
            entry.1 = entry.1.saturating_add(win_actions);
        } else {
            bwd.insert(s, (1, win_actions));
            bwd_queue.push_back(s);
        }
    }

    // Build reverse optimal-edge adjacency: for each state, who are its optimal predecessors?
    let mut reverse_optimal: FxHashMap<State, Vec<State>> = FxHashMap::default();
    for (&src, transitions) in &graph.transitions {
        let src_dist = match fwd.get(&src) {
            Some(&(d, _)) => d,
            None => continue,
        };
        if src_dist >= optimal_depth {
            continue;
        }
        for &(_action, dest) in transitions {
            if let StateKey::Transient(dst) = dest {
                if let Some(&(dst_dist, _)) = fwd.get(&dst) {
                    if dst_dist == src_dist + 1 {
                        reverse_optimal.entry(dst).or_default().push(src);
                    }
                }
            }
        }
    }

    // Propagate backward path counts.
    while let Some(cur) = bwd_queue.pop_front() {
        let (cur_bwd_dist, cur_bwd_count) = bwd[&cur];
        if let Some(preds) = reverse_optimal.get(&cur) {
            // For each predecessor: count how many optimal forward edges go from pred to cur.
            for &pred in preds {
                let n_edges = graph.transitions.get(&pred).map_or(0u64, |tr| {
                    tr.iter()
                        .filter(|&&(_, d)| d == StateKey::Transient(cur))
                        .count() as u64
                });
                let contrib = cur_bwd_count.saturating_mul(n_edges);
                if let Some(entry) = bwd.get_mut(&pred) {
                    if cur_bwd_dist + 1 == entry.0 {
                        entry.1 = entry.1.saturating_add(contrib);
                    }
                } else {
                    bwd.insert(pred, (cur_bwd_dist + 1, contrib));
                    bwd_queue.push_back(pred);
                }
            }
        }
    }

    // Compute weighted path_safety over all states on optimal paths.
    // Weight of state s = fwd_count[s] * bwd_count[s] (number of optimal paths through s).
    let mut weighted_safety = 0.0f64;
    let mut total_weight = 0u64;

    for (&state, &(fwd_dist, fwd_count)) in &fwd {
        if let Some(&(bwd_dist, bwd_count)) = bwd.get(&state) {
            if fwd_dist + bwd_dist == optimal_depth {
                let weight = fwd_count.saturating_mul(bwd_count);
                if let Some(transitions) = graph.transitions.get(&state) {
                    weighted_safety += weight as f64 * state_safety(transitions, winning);
                }
                total_weight = total_weight.saturating_add(weight);
            }
        }
    }

    let path_safety = if total_weight > 0 {
        Some(weighted_safety / total_weight as f64)
    } else {
        Some(1.0)
    };

    // Greedy deviation count (replay-based, uses specific solution path).
    let greedy_deviation_count = solve.actions.as_ref().map(|actions| {
        let mut state = State::from_level(lev);
        let exit_row = lev.exit_row;
        let exit_col = lev.exit_col;
        let mut deviations = 0u32;
        for &action in actions {
            let old_dist =
                (state.player_row - exit_row).abs() + (state.player_col - exit_col).abs();
            step(lev, &mut state, action);
            let new_dist =
                (state.player_row - exit_row).abs() + (state.player_col - exit_col).abs();
            if new_dist >= old_dist {
                deviations += 1;
            }
        }
        deviations
    });

    (n_optimal_solutions, greedy_deviation_count, path_safety)
}
