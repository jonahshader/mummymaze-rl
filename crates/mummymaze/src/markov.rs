//! Absorbing Markov chain solver — sparse iterative Gauss-Seidel.
//!
//! Given a StateGraph, computes exact win probability and expected steps
//! to termination under a uniform-random valid-action policy.
//!
//! Memory: O(5n) instead of O(n²) — stores Q as adjacency list.
//! Convergence: (I-Q) is strictly diagonally dominant (rows sum to <1),
//! so Gauss-Seidel is guaranteed to converge.

use crate::error::{MummyMazeError, Result};
use crate::game::State;
use crate::graph::{StateGraph, StateKey};
use rustc_hash::FxHashMap;

const MAX_ITERATIONS: usize = 100_000;
const TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct MarkovResult {
    pub win_prob: f64,
    pub expected_steps: f64,
    pub n_transient: usize,
}

/// Solve the absorbing Markov chain defined by the state graph.
///
/// Assigns uniform probability 1/k over the k valid actions at each state.
/// Returns exact win probability and expected steps from the start state.
pub fn analyze(graph: &StateGraph) -> Result<MarkovResult> {
    let n = graph.n_transient;
    if n == 0 {
        return Ok(MarkovResult {
            win_prob: 0.0,
            expected_steps: 0.0,
            n_transient: 0,
        });
    }

    // Map transient states to integer indices
    let mut state_to_idx: FxHashMap<State, usize> = FxHashMap::default();
    let mut idx = 0usize;
    for s in graph.transitions.keys() {
        state_to_idx.insert(*s, idx);
        idx += 1;
    }

    let start_idx = state_to_idx[&graph.start];

    // Build sparse Q (transient -> transient) and win absorption vector
    // q_rows[i] = vec of (j, prob) for transitions from state i to state j (j != i)
    let mut q_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut win_absorb: Vec<f64> = vec![0.0; n];
    // diag[i] = (I-Q)[i,i] = 1 - Q[i,i]
    let mut diag: Vec<f64> = vec![1.0; n];

    for (s, s_idx) in &state_to_idx {
        let action_map = &graph.transitions[s];
        let k = action_map.len();
        if k == 0 {
            continue;
        }
        let prob = 1.0 / k as f64;

        for &(_, next_key) in action_map {
            match next_key {
                StateKey::Win => {
                    win_absorb[*s_idx] += prob;
                }
                StateKey::Dead => {
                    // absorbed into dead state, contributes nothing
                }
                StateKey::Transient(next_state) => {
                    let j = state_to_idx[&next_state];
                    if j == *s_idx {
                        // Self-loop: Q[i,i] increases, so diag (I-Q)[i,i] decreases
                        diag[*s_idx] -= prob;
                    } else {
                        q_rows[*s_idx].push((j, prob));
                    }
                }
            }
        }
    }

    // Identify trapped states (diag ≈ 0 means all transitions are self-loops)
    let mut trapped = vec![false; n];
    for i in 0..n {
        if diag[i].abs() < 1e-15 {
            trapped[i] = true;
        }
    }

    // Solve (I-Q)x = win_absorb for win probabilities using Gauss-Seidel
    let mut x = vec![0.0f64; n];
    for _ in 0..MAX_ITERATIONS {
        let mut max_diff = 0.0f64;
        for i in 0..n {
            if trapped[i] {
                continue; // x[i] stays 0 (can never reach absorption)
            }
            let mut sum = win_absorb[i];
            for &(j, qij) in &q_rows[i] {
                sum += qij * x[j];
            }
            let new_val = sum / diag[i];
            let diff = (new_val - x[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            x[i] = new_val;
        }
        if max_diff < TOLERANCE {
            // Converged for win_prob. Now solve for expected steps.
            let mut t = vec![0.0f64; n];
            // Initialize trapped states to a large finite value (they never absorb)
            for i in 0..n {
                if trapped[i] {
                    t[i] = f64::INFINITY;
                }
            }

            for iter2 in 0..MAX_ITERATIONS {
                let mut max_diff2 = 0.0f64;
                for i in 0..n {
                    if trapped[i] {
                        continue;
                    }
                    let mut sum = 1.0;
                    for &(j, qij) in &q_rows[i] {
                        if trapped[j] {
                            // Neighbor is trapped → contributes infinity.
                            // But in practice the chain can absorb through
                            // other paths, so we just skip the trapped neighbor.
                            // This gives a finite expected time conditional on
                            // not entering the trapped component.
                            // Actually that's wrong — if there's a nonzero prob
                            // of reaching a trapped state, expected time IS infinite.
                            // Let's propagate infinity.
                            sum = f64::INFINITY;
                            break;
                        }
                        sum += qij * t[j];
                    }
                    let new_val = if sum.is_infinite() {
                        f64::INFINITY
                    } else {
                        sum / diag[i]
                    };
                    let diff = if new_val.is_infinite() && t[i].is_infinite() {
                        0.0
                    } else {
                        (new_val - t[i]).abs()
                    };
                    if diff > max_diff2 {
                        max_diff2 = diff;
                    }
                    t[i] = new_val;
                }
                if max_diff2 < TOLERANCE {
                    return Ok(MarkovResult {
                        win_prob: x[start_idx],
                        expected_steps: t[start_idx],
                        n_transient: n,
                    });
                }
                if iter2 == MAX_ITERATIONS - 1 {
                    return Err(MummyMazeError::ConvergenceFailure(MAX_ITERATIONS));
                }
            }
            unreachable!();
        }
    }
    Err(MummyMazeError::ConvergenceFailure(MAX_ITERATIONS))
}

#[derive(Debug, Clone)]
pub struct FullMarkovResult {
    pub win_prob: f64,
    pub expected_steps: f64,
    pub state_win_probs: FxHashMap<State, f64>,
}

/// Like `analyze()`, but returns per-state win probabilities for all transient states.
pub fn analyze_full(graph: &StateGraph) -> Result<FullMarkovResult> {
    let n = graph.n_transient;
    if n == 0 {
        return Ok(FullMarkovResult {
            win_prob: 0.0,
            expected_steps: 0.0,
            state_win_probs: FxHashMap::default(),
        });
    }

    // Map transient states to integer indices
    let mut state_to_idx: FxHashMap<State, usize> = FxHashMap::default();
    let mut idx_to_state: Vec<State> = Vec::with_capacity(n);
    let mut idx = 0usize;
    for s in graph.transitions.keys() {
        state_to_idx.insert(*s, idx);
        idx_to_state.push(*s);
        idx += 1;
    }

    let start_idx = state_to_idx[&graph.start];

    // Build sparse Q and win absorption vector (same as analyze())
    let mut q_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut win_absorb: Vec<f64> = vec![0.0; n];
    let mut diag: Vec<f64> = vec![1.0; n];

    for (s, s_idx) in &state_to_idx {
        let action_map = &graph.transitions[s];
        let k = action_map.len();
        if k == 0 {
            continue;
        }
        let prob = 1.0 / k as f64;

        for &(_, next_key) in action_map {
            match next_key {
                StateKey::Win => {
                    win_absorb[*s_idx] += prob;
                }
                StateKey::Dead => {}
                StateKey::Transient(next_state) => {
                    let j = state_to_idx[&next_state];
                    if j == *s_idx {
                        diag[*s_idx] -= prob;
                    } else {
                        q_rows[*s_idx].push((j, prob));
                    }
                }
            }
        }
    }

    let mut trapped = vec![false; n];
    for i in 0..n {
        if diag[i].abs() < 1e-15 {
            trapped[i] = true;
        }
    }

    // Solve (I-Q)x = win_absorb for win probabilities
    let mut x = vec![0.0f64; n];
    for _ in 0..MAX_ITERATIONS {
        let mut max_diff = 0.0f64;
        for i in 0..n {
            if trapped[i] {
                continue;
            }
            let mut sum = win_absorb[i];
            for &(j, qij) in &q_rows[i] {
                sum += qij * x[j];
            }
            let new_val = sum / diag[i];
            let diff = (new_val - x[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            x[i] = new_val;
        }
        if max_diff < TOLERANCE {
            // Solve for expected steps
            let mut t = vec![0.0f64; n];
            for i in 0..n {
                if trapped[i] {
                    t[i] = f64::INFINITY;
                }
            }

            for iter2 in 0..MAX_ITERATIONS {
                let mut max_diff2 = 0.0f64;
                for i in 0..n {
                    if trapped[i] {
                        continue;
                    }
                    let mut sum = 1.0;
                    for &(j, qij) in &q_rows[i] {
                        if trapped[j] {
                            sum = f64::INFINITY;
                            break;
                        }
                        sum += qij * t[j];
                    }
                    let new_val = if sum.is_infinite() {
                        f64::INFINITY
                    } else {
                        sum / diag[i]
                    };
                    let diff = if new_val.is_infinite() && t[i].is_infinite() {
                        0.0
                    } else {
                        (new_val - t[i]).abs()
                    };
                    if diff > max_diff2 {
                        max_diff2 = diff;
                    }
                    t[i] = new_val;
                }
                if max_diff2 < TOLERANCE {
                    // Build the result map
                    let mut state_win_probs = FxHashMap::default();
                    for (i, &state) in idx_to_state.iter().enumerate() {
                        state_win_probs.insert(state, x[i]);
                    }

                    return Ok(FullMarkovResult {
                        win_prob: x[start_idx],
                        expected_steps: t[start_idx],
                        state_win_probs,
                    });
                }
                if iter2 == MAX_ITERATIONS - 1 {
                    return Err(MummyMazeError::ConvergenceFailure(MAX_ITERATIONS));
                }
            }
            unreachable!();
        }
    }
    Err(MummyMazeError::ConvergenceFailure(MAX_ITERATIONS))
}
