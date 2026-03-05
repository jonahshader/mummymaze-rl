//! Absorbing Markov chain solver — sparse iterative Gauss-Seidel.
//!
//! Given a StateGraph, computes exact win probability and expected steps
//! to termination under a uniform-random valid-action policy.
//!
//! Memory: O(5n) instead of O(n²) — stores Q as adjacency list.
//! Convergence: (I-Q) is strictly diagonally dominant (rows sum to <1),
//! so Gauss-Seidel is guaranteed to converge.

use crate::error::{MummyMazeError, Result};
use crate::game::{Action, State};
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

/// Computes action probability for a single action given the state's transition list.
type ProbFn<'a> = &'a dyn Fn(&State, Action, usize, usize) -> f64;

impl MarkovChain {
    /// Shared construction logic: builds the sparse Markov chain using a caller-provided
    /// function that returns the probability for each (state, action) pair.
    ///
    /// `prob_fn(state, action, action_index, k)` returns the probability for the given
    /// action, where `action_index` is its position in the transition list and `k` is
    /// the total number of valid actions for that state.
    fn build(graph: &StateGraph, prob_fn: ProbFn<'_>) -> Self {
        let indices = graph.state_indices();
        let state_to_idx = &indices.state_to_idx;
        let idx_to_state = indices.idx_to_state;
        let n = idx_to_state.len();

        let start_idx = state_to_idx[&graph.start];

        let mut q_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut win_absorb: Vec<f64> = vec![0.0; n];
        let mut diag: Vec<f64> = vec![1.0; n];

        for (s, s_idx) in state_to_idx {
            let action_map = &graph.transitions[s];
            let k = action_map.len();
            if k == 0 {
                continue;
            }

            for (idx, &(action, next_key)) in action_map.iter().enumerate() {
                let prob = prob_fn(s, action, idx, k);
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

        MarkovChain {
            q_rows,
            win_absorb,
            diag,
            trapped,
            start_idx,
            idx_to_state,
        }
    }

    /// Build the sparse Markov chain representation from a state graph.
    pub fn from_graph(graph: &StateGraph) -> Self {
        Self::build(graph, &|_state, _action, _idx, k| 1.0 / k as f64)
    }

    /// Build the sparse Markov chain using arbitrary per-state action probabilities.
    ///
    /// `policy` maps each state to raw (unnormalized) probabilities for the 5 actions
    /// [N, S, E, W, Wait]. Only actions present in the state's transition map are kept;
    /// the rest are zeroed and the remaining probs are renormalized. Falls back to
    /// uniform 1/k for states not in the policy map or when all kept probs are ~0.
    pub fn from_graph_with_policy(
        graph: &StateGraph,
        policy: &FxHashMap<State, [f64; 5]>,
    ) -> Self {
        // Pre-compute per-state normalization sums so the prob_fn closure is cheap.
        // For each state in the graph, sum the policy probs over valid actions only.
        let sums: FxHashMap<State, f64> = graph
            .transitions
            .iter()
            .filter_map(|(s, action_map)| {
                let raw = policy.get(s)?;
                let sum: f64 = action_map
                    .iter()
                    .map(|(action, _)| raw[action.to_index() as usize])
                    .sum();
                Some((*s, sum))
            })
            .collect();

        Self::build(graph, &|state, action, _idx, k| {
            let uniform = 1.0 / k as f64;
            let Some(raw) = policy.get(state) else {
                return uniform;
            };
            let sum = sums[state];
            if sum > 1e-30 {
                raw[action.to_index() as usize] / sum
            } else {
                uniform
            }
        })
    }

    /// Number of transient states.
    pub fn n_states(&self) -> usize {
        self.idx_to_state.len()
    }

    /// Solve (I-Q)x = win_absorb for per-state win probabilities.
    pub fn solve_win_probs(&self) -> Result<Vec<f64>> {
        self.solve_win_probs_tol(TOLERANCE, MAX_ITERATIONS)
    }

    /// Solve (I-Q)x = win_absorb with custom convergence parameters.
    pub fn solve_win_probs_tol(&self, tol: f64, max_iter: usize) -> Result<Vec<f64>> {
        let n = self.n_states();
        let mut x = vec![0.0f64; n];
        for _ in 0..max_iter {
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
            if max_diff < tol {
                return Ok(x);
            }
        }
        Err(MummyMazeError::ConvergenceFailure(max_iter))
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
