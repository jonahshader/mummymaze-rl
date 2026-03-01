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

/// Sparse representation of an absorbing Markov chain built from a state graph.
///
/// Precomputes the sparse Q matrix and absorption vector once; individual
/// quantities (win probabilities, expected steps) can then be solved independently.
pub struct MarkovChain {
    /// Sparse off-diagonal rows of Q: q_rows[i] = vec of (j, prob).
    q_rows: Vec<Vec<(usize, f64)>>,
    /// Win absorption probability per state.
    win_absorb: Vec<f64>,
    /// Diagonal of (I - Q): diag[i] = 1 - Q[i,i].
    diag: Vec<f64>,
    /// States where all transitions are self-loops (diag ≈ 0).
    trapped: Vec<bool>,
    /// Index of the start state.
    pub start_idx: usize,
    /// Map from index back to game state.
    pub idx_to_state: Vec<State>,
}

impl MarkovChain {
    /// Build the sparse Markov chain representation from a state graph.
    pub fn from_graph(graph: &StateGraph) -> Self {
        let indices = graph.state_indices();
        let state_to_idx = &indices.state_to_idx;
        let idx_to_state = indices.idx_to_state;
        let n = idx_to_state.len();

        let start_idx = state_to_idx[&graph.start];

        // Build sparse Q (transient -> transient) and win absorption vector
        // q_rows[i] = vec of (j, prob) for transitions from state i to state j (j != i)
        let mut q_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut win_absorb: Vec<f64> = vec![0.0; n];
        // diag[i] = (I-Q)[i,i] = 1 - Q[i,i]
        let mut diag: Vec<f64> = vec![1.0; n];

        for (s, s_idx) in state_to_idx {
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

        MarkovChain {
            q_rows,
            win_absorb,
            diag,
            trapped,
            start_idx,
            idx_to_state,
        }
    }

    /// Number of transient states.
    pub fn n_states(&self) -> usize {
        self.idx_to_state.len()
    }

    /// Solve (I-Q)x = win_absorb for per-state win probabilities.
    pub fn solve_win_probs(&self) -> Result<Vec<f64>> {
        let n = self.n_states();
        let mut x = vec![0.0f64; n];
        for _ in 0..MAX_ITERATIONS {
            let mut max_diff = 0.0f64;
            for i in 0..n {
                if self.trapped[i] {
                    continue; // x[i] stays 0 (can never reach absorption)
                }
                let mut sum = self.win_absorb[i];
                for &(j, qij) in &self.q_rows[i] {
                    sum += qij * x[j];
                }
                let new_val = sum / self.diag[i];
                let diff = (new_val - x[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                x[i] = new_val;
            }
            if max_diff < TOLERANCE {
                return Ok(x);
            }
        }
        Err(MummyMazeError::ConvergenceFailure(MAX_ITERATIONS))
    }

    /// Solve (I-Q)t = 1 for per-state expected steps to absorption.
    pub fn solve_expected_steps(&self) -> Result<Vec<f64>> {
        let n = self.n_states();
        let mut t = vec![0.0f64; n];
        // Initialize trapped states to infinity (they never absorb)
        for i in 0..n {
            if self.trapped[i] {
                t[i] = f64::INFINITY;
            }
        }

        for iter in 0..MAX_ITERATIONS {
            let mut max_diff = 0.0f64;
            for i in 0..n {
                if self.trapped[i] {
                    continue;
                }
                let mut sum = 1.0;
                for &(j, qij) in &self.q_rows[i] {
                    if self.trapped[j] {
                        // Neighbor is trapped → nonzero prob of never absorbing.
                        // Expected time is infinite.
                        sum = f64::INFINITY;
                        break;
                    }
                    sum += qij * t[j];
                }
                let new_val = if sum.is_infinite() {
                    f64::INFINITY
                } else {
                    sum / self.diag[i]
                };
                let diff = if new_val.is_infinite() && t[i].is_infinite() {
                    0.0
                } else {
                    (new_val - t[i]).abs()
                };
                if diff > max_diff {
                    max_diff = diff;
                }
                t[i] = new_val;
            }
            if max_diff < TOLERANCE {
                return Ok(t);
            }
            if iter == MAX_ITERATIONS - 1 {
                return Err(MummyMazeError::ConvergenceFailure(MAX_ITERATIONS));
            }
        }
        Err(MummyMazeError::ConvergenceFailure(MAX_ITERATIONS))
    }

    /// Build a map from State to its solved value for all transient states.
    pub fn per_state_map(&self, values: &[f64]) -> FxHashMap<State, f64> {
        self.idx_to_state
            .iter()
            .enumerate()
            .map(|(i, &state)| (state, values[i]))
            .collect()
    }
}
