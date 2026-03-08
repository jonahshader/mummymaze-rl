//! User-configurable fitness expressions for the GA.
//!
//! Parses a math expression string (e.g. `-win_prob + bfs_moves / 1000`) and
//! evaluates it against a set of named metrics. Uses `fasteval` for parsing
//! and evaluation.

use crate::graph::StateGraph;
use crate::metrics::{self, DifficultyMetrics};
use crate::parse::Level;
use crate::solver::SolveResult;

/// All variables available in fitness expressions.
pub const VARIABLES: &[(&str, &str)] = &[
    ("win_prob", "Win probability under uniform-random policy (0–1)"),
    ("log_win_prob", "log10(win_prob) — no underflow, always finite for solvable levels"),
    ("bfs_moves", "Optimal BFS solution length"),
    ("n_states", "Number of reachable transient states"),
    ("dead_end_ratio", "Fraction of states from which winning is impossible (0–1)"),
    ("avg_branching", "Mean number of valid actions per state"),
    ("n_optimal", "Number of distinct shortest action sequences"),
    ("greedy_deviation", "Steps on optimal path that don't reduce Manhattan distance"),
    ("path_safety", "Fraction of safe actions along optimal paths (0–1)"),
];

/// Variable names that require the expensive `metrics::compute()` call.
const EXPENSIVE_VARS: &[&str] = &[
    "dead_end_ratio",
    "avg_branching",
    "n_optimal",
    "greedy_deviation",
    "path_safety",
];

// Preset fitness expressions.
pub const PRESET_DEFAULT: &str = "-win_prob + bfs_moves / 1000";
pub const PRESET_HARD: &str = "-win_prob";
pub const PRESET_COMPLEX: &str = "-win_prob + n_states / 10000 + bfs_moves / 1000";
pub const PRESET_DECEPTIVE: &str = "-win_prob + greedy_deviation / 10";
pub const PRESET_TRAP: &str = "-win_prob - path_safety + dead_end_ratio";

pub const PRESETS: &[(&str, &str)] = &[
    ("Default", PRESET_DEFAULT),
    ("Hard", PRESET_HARD),
    ("Complex", PRESET_COMPLEX),
    ("Deceptive", PRESET_DECEPTIVE),
    ("Trap-heavy", PRESET_TRAP),
];

/// Metric values for fitness evaluation.
#[derive(Debug, Clone)]
pub struct FitnessVars {
    pub win_prob: f64,
    pub log_win_prob: f64,
    pub bfs_moves: f64,
    pub n_states: f64,
    pub dead_end_ratio: f64,
    pub avg_branching: f64,
    pub n_optimal: f64,
    pub greedy_deviation: f64,
    pub path_safety: f64,
}

impl FitnessVars {
    /// Build from basic GA metrics (cheap — no difficulty metrics needed).
    pub fn basic(win_prob: f64, log_win_prob: f64, bfs_moves: u32, n_states: usize) -> Self {
        FitnessVars {
            win_prob,
            log_win_prob,
            bfs_moves: bfs_moves as f64,
            n_states: n_states as f64,
            dead_end_ratio: 0.0,
            avg_branching: 0.0,
            n_optimal: 0.0,
            greedy_deviation: 0.0,
            path_safety: 1.0,
        }
    }

    /// Populate difficulty metrics from a `DifficultyMetrics` struct.
    pub fn with_difficulty(mut self, dm: &DifficultyMetrics) -> Self {
        self.dead_end_ratio = dm.dead_end_ratio;
        self.avg_branching = dm.avg_branching_factor;
        self.n_optimal = dm.n_optimal_solutions as f64;
        self.greedy_deviation = dm.greedy_deviation_count.unwrap_or(0) as f64;
        self.path_safety = dm.path_safety.unwrap_or(1.0);
        self
    }

    /// Look up a variable by name.
    fn get(&self, name: &str) -> Option<f64> {
        match name {
            "win_prob" => Some(self.win_prob),
            "log_win_prob" => Some(self.log_win_prob),
            "bfs_moves" => Some(self.bfs_moves),
            "n_states" => Some(self.n_states),
            "dead_end_ratio" => Some(self.dead_end_ratio),
            "avg_branching" => Some(self.avg_branching),
            "n_optimal" => Some(self.n_optimal),
            "greedy_deviation" => Some(self.greedy_deviation),
            "path_safety" => Some(self.path_safety),
            _ => None,
        }
    }
}

/// A parsed fitness expression that can be evaluated against metric variables.
#[derive(Debug, Clone)]
pub struct FitnessExpr {
    expression: String,
    /// Whether the expression references any expensive difficulty metrics.
    pub needs_difficulty_metrics: bool,
}

impl FitnessExpr {
    /// Parse and validate a fitness expression.
    ///
    /// Does a test evaluation with all variables set to 1.0 to catch syntax
    /// errors and undefined variable references.
    pub fn parse(expr: &str) -> Result<Self, String> {
        let expr = expr.trim().to_string();
        if expr.is_empty() {
            return Err("empty expression".to_string());
        }

        // Test-evaluate to catch errors early.
        let test_vars = FitnessVars {
            win_prob: 1.0,
            log_win_prob: 0.0,
            bfs_moves: 1.0,
            n_states: 1.0,
            dead_end_ratio: 1.0,
            avg_branching: 1.0,
            n_optimal: 1.0,
            greedy_deviation: 1.0,
            path_safety: 1.0,
        };
        eval_expr(&expr, &test_vars)?;

        let needs_difficulty_metrics = EXPENSIVE_VARS.iter().any(|v| expr.contains(v));

        Ok(FitnessExpr {
            expression: expr,
            needs_difficulty_metrics,
        })
    }

    /// Evaluate the expression with the given variables.
    pub fn eval(&self, vars: &FitnessVars) -> f64 {
        eval_expr(&self.expression, vars).unwrap_or(f64::NEG_INFINITY)
    }

    /// The expression string.
    pub fn expression(&self) -> &str {
        &self.expression
    }

    /// Compute FitnessVars for a level, conditionally including difficulty metrics.
    pub fn compute_vars(
        &self,
        graph: &StateGraph,
        level: &Level,
        solve: &SolveResult,
        win_prob: f64,
        log_win_prob: f64,
    ) -> FitnessVars {
        let vars =
            FitnessVars::basic(win_prob, log_win_prob, solve.moves.unwrap_or(0), graph.n_transient);
        if self.needs_difficulty_metrics {
            let dm = metrics::compute(graph, level, solve);
            vars.with_difficulty(&dm)
        } else {
            vars
        }
    }
}

impl Default for FitnessExpr {
    fn default() -> Self {
        FitnessExpr::parse(PRESET_DEFAULT).unwrap()
    }
}

/// Evaluate an expression string with the given variables using fasteval.
fn eval_expr(expr: &str, vars: &FitnessVars) -> Result<f64, String> {
    let mut ns = |name: &str, _args: Vec<f64>| -> Option<f64> { vars.get(name) };

    let val = fasteval::ez_eval(expr, &mut ns).map_err(|e| format!("{e}"))?;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_default() {
        let expr = FitnessExpr::parse(PRESET_DEFAULT).unwrap();
        assert!(!expr.needs_difficulty_metrics);
    }

    #[test]
    fn test_parse_with_difficulty() {
        let expr = FitnessExpr::parse(PRESET_TRAP).unwrap();
        assert!(expr.needs_difficulty_metrics);
    }

    #[test]
    fn test_eval_basic() {
        let expr = FitnessExpr::parse("-win_prob + bfs_moves / 1000").unwrap();
        let vars = FitnessVars::basic(0.5, 0.5f64.log10(), 20, 100);
        let result = expr.eval(&vars);
        assert!((result - (-0.5 + 20.0 / 1000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_error_undefined_var() {
        let result = FitnessExpr::parse("typo_var + 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_syntax() {
        let result = FitnessExpr::parse("1 +");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_empty() {
        let result = FitnessExpr::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_all_presets_parse() {
        for (name, expr) in PRESETS {
            FitnessExpr::parse(expr).unwrap_or_else(|e| panic!("preset {name:?} failed: {e}"));
        }
    }

    #[test]
    fn test_math_functions() {
        // fasteval supports: +, -, *, /, %, ^, abs(), log(), ceil(), floor(), etc.
        let expr = FitnessExpr::parse("abs(-win_prob) + bfs_moves ^ 0.5").unwrap();
        let vars = FitnessVars::basic(0.5, 0.5f64.log10(), 16, 100);
        let result = expr.eval(&vars);
        assert!((result - (0.5 + 4.0)).abs() < 1e-10);
    }
}
