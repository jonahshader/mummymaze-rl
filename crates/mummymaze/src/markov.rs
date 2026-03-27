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
use rustc_hash::{FxHashMap, FxHashSet};

const MAX_ITERATIONS: usize = 100_000;
const TOLERANCE: f64 = 1e-12;
/// Threshold for detecting trapped states (all-self-loop diagonal ≈ 0).
/// Looser than PIVOT_ZERO because diag values accumulate float rounding error.
const TRAPPED_THRESHOLD: f64 = 1e-15;
/// Threshold below which a pivot or normalization sum is treated as zero.
const PIVOT_ZERO: f64 = 1e-30;

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
    /// Index of the start state (None if start is not winnable).
    pub start_idx: Option<usize>,
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
    ///
    /// Only states in `winnable` are included in the chain. Transitions to states
    /// outside this set are treated as absorption into DEAD (probability lost).
    /// `k` still counts ALL valid actions (including those leading to unwinnable
    /// states) since the agent doesn't know which states are dead ends.
    fn build(graph: &StateGraph, winnable: &FxHashSet<State>, prob_fn: ProbFn<'_>) -> Self {
        // Build index mapping for winnable states only.
        let mut state_to_idx: FxHashMap<State, usize> = FxHashMap::default();
        let mut idx_to_state: Vec<State> = Vec::with_capacity(winnable.len());
        for s in graph.transitions.keys() {
            if winnable.contains(s) {
                state_to_idx.insert(*s, idx_to_state.len());
                idx_to_state.push(*s);
            }
        }
        let n = idx_to_state.len();

        let start_idx = state_to_idx.get(&graph.start).copied();

        let mut q_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut win_absorb: Vec<f64> = vec![0.0; n];
        let mut diag: Vec<f64> = vec![1.0; n];

        for (&s, &s_idx) in &state_to_idx {
            let action_map = &graph.transitions[&s];
            let k = action_map.len();
            if k == 0 {
                continue;
            }

            for (idx, &(action, next_key)) in action_map.iter().enumerate() {
                let prob = prob_fn(&s, action, idx, k);
                match next_key {
                    StateKey::Win => {
                        win_absorb[s_idx] += prob;
                    }
                    StateKey::Dead => {}
                    StateKey::Transient(next_state) => {
                        if let Some(&j) = state_to_idx.get(&next_state) {
                            if j == s_idx {
                                diag[s_idx] -= prob;
                            } else {
                                q_rows[s_idx].push((j, prob));
                            }
                        }
                        // else: next_state is unwinnable, treat as DEAD (absorbed)
                    }
                }
            }
        }

        let mut trapped = vec![false; n];
        for i in 0..n {
            if diag[i].abs() < TRAPPED_THRESHOLD {
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

    /// Compute the set of states that can reach WIN (backward BFS).
    fn winnable_set(graph: &StateGraph) -> FxHashSet<State> {
        let mut reverse: FxHashMap<State, Vec<State>> = FxHashMap::default();
        let mut win_predecessors: Vec<State> = Vec::new();

        for (src, transitions) in &graph.transitions {
            for &(_action, dest) in transitions {
                match dest {
                    StateKey::Win => win_predecessors.push(*src),
                    StateKey::Transient(dst) => reverse.entry(dst).or_default().push(*src),
                    StateKey::Dead => {}
                }
            }
        }

        let mut visited = FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();
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

    /// Build the sparse Markov chain representation from a state graph.
    ///
    /// Trims unwinnable states: only states that can reach WIN are included.
    pub fn from_graph(graph: &StateGraph) -> Self {
        let winnable = Self::winnable_set(graph);
        Self::build(graph, &winnable, &|_state, _action, _idx, k| {
            1.0 / k as f64
        })
    }

    /// Build the sparse Markov chain using arbitrary per-state action probabilities.
    ///
    /// `policy` maps each state to raw (unnormalized) probabilities for the 5 actions
    /// [N, S, E, W, Wait]. Only actions present in the state's transition map are kept;
    /// the rest are zeroed and the remaining probs are renormalized. Falls back to
    /// uniform 1/k for states not in the policy map or when all kept probs are ~0.
    ///
    /// Trims unwinnable states: only states that can reach WIN are included.
    pub fn from_graph_with_policy(
        graph: &StateGraph,
        policy: &FxHashMap<State, [f64; 5]>,
    ) -> Self {
        let winnable = Self::winnable_set(graph);

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

        Self::build(graph, &winnable, &|state, action, _idx, k| {
            let uniform = 1.0 / k as f64;
            let Some(raw) = policy.get(state) else {
                return uniform;
            };
            let sum = sums[state];
            if sum > PIVOT_ZERO {
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
    /// Falls back to direct Gaussian elimination if Gauss-Seidel doesn't converge.
    pub fn solve_win_probs(&self) -> Result<Vec<f64>> {
        self.solve_with_rhs(&self.win_absorb)
    }

    /// Solve (I-Q)x = win_absorb with custom convergence parameters.
    ///
    /// Uses a combined absolute + relative convergence criterion so that
    /// extremely small win probabilities (e.g. 1e-34) can still converge.
    pub fn solve_win_probs_tol(&self, tol: f64, max_iter: usize) -> Result<Vec<f64>> {
        let n = self.n_states();
        let mut x = vec![0.0f64; n];
        for _ in 0..max_iter {
            let mut converged = true;
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
                // Converged when absolute diff < tol OR relative diff < tol
                if diff > tol && diff > tol * new_val.abs() {
                    converged = false;
                }
                x[i] = new_val;
            }
            if converged {
                return Ok(x);
            }
        }
        Err(MummyMazeError::ConvergenceFailure(max_iter))
    }

    /// Solve the adjoint system (I-Q^T)λ = e_start for per-state importance weights.
    ///
    /// λ[s] is the expected number of visits to state s starting from the start state
    /// under the current policy. Used for computing dV_start/dπ in the surrogate
    /// gradient trick (implicit differentiation).
    pub fn solve_adjoint(&self) -> Result<Vec<f64>> {
        let n = self.n_states();
        let Some(start) = self.start_idx else {
            return Ok(vec![0.0; n]);
        };

        // Build transposed adjacency: q_cols[j] = vec of (i, Q[i,j])
        // so that Q^T[j,i] = Q[i,j] → row j of Q^T has entries from column j of Q.
        let mut q_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (i, row) in self.q_rows.iter().enumerate() {
            for &(j, qij) in row {
                q_cols[j].push((i, qij));
            }
        }

        // Gauss-Seidel: (I - Q^T)λ = e_start
        // λ[j] = (e_start[j] + Σ_i Q^T[j,i] * λ[i]) / diag[j]
        // Note: diag[j] = 1 - Q[j,j] = 1 - Q^T[j,j] (same diagonal)
        let mut lam = vec![0.0f64; n];
        for _ in 0..MAX_ITERATIONS {
            let mut converged = true;
            for j in 0..n {
                if self.trapped[j] {
                    continue;
                }
                let rhs = if j == start { 1.0 } else { 0.0 };
                let mut sum = rhs;
                for &(i, qij) in &q_cols[j] {
                    sum += qij * lam[i]; // Q^T[j,i] = Q[i,j] = qij
                }
                let new_val = sum / self.diag[j];
                let diff = (new_val - lam[j]).abs();
                if diff > TOLERANCE && diff > TOLERANCE * new_val.abs() {
                    converged = false;
                }
                lam[j] = new_val;
            }
            if converged {
                return Ok(lam);
            }
        }

        // Gauss-Seidel failed — fall back to direct solve on (I - Q^T).
        self.solve_adjoint_direct()
    }

    /// Direct Gaussian elimination fallback for the adjoint system (I - Q^T) λ = e_start.
    fn solve_adjoint_direct(&self) -> Result<Vec<f64>> {
        let n = self.n_states();
        let Some(start) = self.start_idx else {
            return Ok(vec![0.0; n]);
        };

        // Build dense augmented matrix [(I - Q^T) | e_start] row by row.
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            if self.trapped[i] {
                a[i][i] = 1.0;
                a[i][n] = 0.0;
                continue;
            }
            a[i][i] = self.diag[i]; // 1 - Q[i,i] = 1 - Q^T[i,i]
            // Off-diagonal: -(Q^T[i,j]) = -(Q[j,i])
            // Iterate over all rows j, looking for entries that target column i.
            for (j, row) in self.q_rows.iter().enumerate() {
                for &(col, qjcol) in row {
                    if col == i {
                        a[i][j] -= qjcol; // Q^T[i,j] = Q[j,i] = qjcol
                    }
                }
            }
            a[i][n] = if i == start { 1.0 } else { 0.0 };
        }

        // Forward elimination with partial pivoting.
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..n {
                let v = a[row][col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < PIVOT_ZERO {
                continue;
            }
            a.swap(col, max_row);
            let pivot = a[col][col];
            for row in (col + 1)..n {
                let factor = a[row][col] / pivot;
                if factor == 0.0 {
                    continue;
                }
                for j in col..=n {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }

        // Back substitution.
        let mut lam = vec![0.0f64; n];
        for i in (0..n).rev() {
            if a[i][i].abs() < PIVOT_ZERO {
                lam[i] = 0.0;
                continue;
            }
            let mut sum = a[i][n];
            for j in (i + 1)..n {
                sum -= a[i][j] * lam[j];
            }
            lam[i] = sum / a[i][i];
        }
        Ok(lam)
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

    /// Solve for log10(win_prob) per state, avoiding f64 underflow.
    ///
    /// First runs the standard solver. If any winnable state underflows to
    /// zero, re-solves with `win_absorb` pre-scaled by 10^200. Since
    /// `(I-Q)x = b` is linear, scaling `b` by `C` scales `x` by `C`,
    /// so `log10(x) = log10(scaled_x) - log10(C)`.
    ///
    /// Repeats with increasing scale until no state underflows.
    ///
    /// Returns log10 values. States that are truly unreachable from WIN
    /// (shouldn't exist after trimming) get `f64::NEG_INFINITY`.
    pub fn solve_log_win_probs(&self) -> Result<Vec<f64>> {
        const LOG10_SCALE_STEP: f64 = 200.0;
        const SCALE_FACTOR: f64 = 1e200;
        const MAX_RESCALES: usize = 10; // supports log10(win_prob) down to ~-2000

        let mut log10_offset = 0.0f64;
        let mut scaled_win_absorb = self.win_absorb.clone();

        for _ in 0..=MAX_RESCALES {
            let x = self.solve_with_rhs(&scaled_win_absorb)?;

            // Check if any winnable state underflowed to zero.
            let has_underflow = x.iter().enumerate().any(|(i, &v)| {
                !self.trapped[i] && v == 0.0 && self.has_path_to_win(i)
            });

            if !has_underflow {
                return Ok(x
                    .iter()
                    .map(|&v| {
                        if v > 0.0 {
                            v.log10() - log10_offset
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect());
            }

            // Scale up RHS for next attempt.
            for v in &mut scaled_win_absorb {
                *v *= SCALE_FACTOR;
            }
            log10_offset += LOG10_SCALE_STEP;
        }

        // Exhausted rescales — return best effort.
        let x = self.solve_with_rhs(&scaled_win_absorb)?;
        Ok(x
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    v.log10() - log10_offset
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect())
    }

    /// Solve for the start state's win probability using the log-space solver.
    ///
    /// Returns `(win_prob, log10_win_prob)`. Always converges.
    pub fn start_log_win_prob(&self) -> Result<(f64, f64)> {
        let log_wp = self.solve_log_win_probs()?;
        let log_p = self.start_idx.map_or(f64::NEG_INFINITY, |i| log_wp[i]);
        Ok((10.0f64.powf(log_p).clamp(0.0, 1.0), log_p.min(0.0)))
    }

    /// Run Gauss-Seidel with a given RHS vector (scaled win_absorb).
    /// Falls back to direct Gaussian elimination if iterative solver fails to converge.
    fn solve_with_rhs(&self, win_absorb: &[f64]) -> Result<Vec<f64>> {
        let n = self.n_states();
        let mut x = vec![0.0f64; n];
        for _ in 0..MAX_ITERATIONS {
            let mut converged = true;
            for i in 0..n {
                if self.trapped[i] {
                    continue;
                }
                let mut sum = win_absorb[i];
                for &(j, qij) in &self.q_rows[i] {
                    sum += qij * x[j];
                }
                let new_val = sum / self.diag[i];
                let diff = (new_val - x[i]).abs();
                if diff > TOLERANCE && diff > TOLERANCE * new_val.abs() {
                    converged = false;
                }
                x[i] = new_val;
            }
            if converged {
                return Ok(x);
            }
        }
        // Gauss-Seidel failed — fall back to direct solve.
        self.solve_with_rhs_direct(win_absorb)
    }

    /// Direct Gaussian elimination with partial pivoting for (I-Q)x = win_absorb.
    /// O(n³) but exact; used as fallback when iterative solver doesn't converge.
    fn solve_with_rhs_direct(&self, win_absorb: &[f64]) -> Result<Vec<f64>> {
        let n = self.n_states();

        // Build dense augmented matrix [I-Q | b] row by row.
        let mut a = vec![vec![0.0f64; n + 1]; n];
        for i in 0..n {
            if self.trapped[i] {
                // Trapped state: set identity row, RHS = 0 (solution x[i] = 0).
                a[i][i] = 1.0;
                a[i][n] = 0.0;
                continue;
            }
            a[i][i] = self.diag[i]; // 1 - Q[i,i]
            for &(j, qij) in &self.q_rows[i] {
                a[i][j] = -qij; // off-diagonal: -(Q[i,j])
            }
            a[i][n] = win_absorb[i];
        }

        // Forward elimination with partial pivoting.
        for col in 0..n {
            // Find pivot.
            let mut max_row = col;
            let mut max_val = a[col][col].abs();
            for row in (col + 1)..n {
                let v = a[row][col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < PIVOT_ZERO {
                // Singular column — skip (trapped or degenerate).
                continue;
            }
            a.swap(col, max_row);

            let pivot = a[col][col];
            for row in (col + 1)..n {
                let factor = a[row][col] / pivot;
                if factor == 0.0 {
                    continue;
                }
                for j in col..=n {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }

        // Back substitution.
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            if a[i][i].abs() < PIVOT_ZERO {
                x[i] = 0.0;
                continue;
            }
            let mut sum = a[i][n];
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = (sum / a[i][i]).max(0.0);
        }

        Ok(x)
    }

    /// Check if state i has any path to a state with win_absorb > 0 (BFS through Q).
    fn has_path_to_win(&self, start: usize) -> bool {
        if self.win_absorb[start] > 0.0 {
            return true;
        }
        let n = self.n_states();
        let mut visited = vec![false; n];
        visited[start] = true;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(cur) = queue.pop_front() {
            for &(j, _) in &self.q_rows[cur] {
                if self.win_absorb[j] > 0.0 {
                    return true;
                }
                if !visited[j] {
                    visited[j] = true;
                    queue.push_back(j);
                }
            }
        }
        false
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::build_graph;
    use crate::parse;
    use std::path::Path;

    #[test]
    fn log_solver_matches_regular_solver() {
        use std::hash::{Hash, Hasher};

        let maze_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mazes");
        if !maze_dir.exists() {
            eprintln!("Skipping: mazes/ not found");
            return;
        }

        let mut max_err: f64 = 0.0;
        let mut tested = 0;
        let mut skipped = 0;

        for entry in std::fs::read_dir(&maze_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().map_or(true, |e| e != "dat") {
                continue;
            }
            let Ok((_, levels)) = parse::parse_file(&path) else {
                continue;
            };

            for (i, lev) in levels.iter().enumerate() {
                // Sample ~1% of levels for speed
                let mut hasher = std::hash::DefaultHasher::new();
                path.hash(&mut hasher);
                i.hash(&mut hasher);
                if hasher.finish() % 100 != 0 {
                    skipped += 1;
                    continue;
                }

                let bfs = crate::solver::solve(lev);
                if bfs.moves.is_none() {
                    continue;
                }

                let graph = build_graph(lev);
                let chain = MarkovChain::from_graph(&graph);

                let Some(start) = chain.start_idx else {
                    continue;
                };

                let wp = chain.solve_win_probs().unwrap();
                let log_wp = chain.solve_log_win_probs().unwrap();

                let p = wp[start];
                let log_p = log_wp[start];

                let expected_log = p.log10();
                let err = (expected_log - log_p).abs();
                if err > max_err {
                    max_err = err;
                    if err > 0.01 {
                        eprintln!(
                            "  {:?} p={:.6e} log10(p)={:.4} log_solver={:.4} err={:.6}",
                            path.file_stem().unwrap(),
                            p,
                            expected_log,
                            log_p,
                            err
                        );
                    }
                }
                tested += 1;
            }
        }
        eprintln!("Tested {tested} solvable levels (skipped {skipped}). Max log10 error: {max_err:.6e}");
        assert!(
            max_err < 0.1,
            "log solver diverged: max error {max_err:.6e}"
        );
    }
}
