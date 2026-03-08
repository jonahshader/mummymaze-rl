//! PyO3 bindings for the mummymaze crate.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use crate::batch::{self, LevelAnalysis};
use crate::game::State;
use crate::parse::Level;

type StateTuple = (i32, i32, i32, i32, bool, i32, i32, bool, i32, i32, bool, bool);

/// Convert a State to a Python-compatible tuple.
fn state_to_tuple(s: &State) -> StateTuple {
    (
        s.player_row, s.player_col,
        s.mummy1_row, s.mummy1_col, s.mummy1_alive,
        s.mummy2_row, s.mummy2_col, s.mummy2_alive,
        s.scorpion_row, s.scorpion_col, s.scorpion_alive,
        s.gate_open,
    )
}

fn action_name(a: crate::game::Action) -> &'static str {
    match a {
        crate::game::Action::North => "N",
        crate::game::Action::South => "S",
        crate::game::Action::East => "E",
        crate::game::Action::West => "W",
        crate::game::Action::Wait => "wait",
    }
}

/// Convert a LevelAnalysis to a Python dict.
fn analysis_to_dict(py: Python<'_>, r: &LevelAnalysis) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("file", &r.file_stem)?;
    dict.set_item("sublevel", r.sublevel)?;
    dict.set_item("grid_size", r.grid_size)?;
    dict.set_item("n_states", r.n_states)?;
    dict.set_item("win_prob", r.win_prob)?;
    dict.set_item("expected_steps", r.expected_steps)?;
    dict.set_item("bfs_moves", r.bfs_moves)?;
    dict.set_item("dead_end_ratio", r.difficulty.dead_end_ratio)?;
    dict.set_item("avg_branching_factor", r.difficulty.avg_branching_factor)?;
    dict.set_item("n_optimal_solutions", r.difficulty.n_optimal_solutions)?;
    dict.set_item("greedy_deviation_count", r.difficulty.greedy_deviation_count)?;
    dict.set_item("path_safety", r.difficulty.path_safety)?;
    Ok(dict.into())
}

/// Extract numpy arrays into Vec<[i32; 12]> and Vec<[f32; 5]>.
fn extract_policy_arrays(
    state_tuples: &PyReadonlyArray2<i32>,
    action_probs: &PyReadonlyArray2<f32>,
) -> (Vec<[i32; 12]>, Vec<[f32; 5]>) {
    let st_array = state_tuples.as_array();
    let ap_array = action_probs.as_array();
    let n = st_array.nrows();

    let mut st_vec: Vec<[i32; 12]> = Vec::with_capacity(n);
    for row in st_array.rows() {
        let mut arr = [0i32; 12];
        for (i, &v) in row.iter().enumerate() {
            arr[i] = v;
        }
        st_vec.push(arr);
    }

    let mut ap_vec: Vec<[f32; 5]> = Vec::with_capacity(n);
    for row in ap_array.rows() {
        let mut arr = [0f32; 5];
        for (i, &v) in row.iter().enumerate() {
            arr[i] = v;
        }
        ap_vec.push(arr);
    }

    (st_vec, ap_vec)
}

// ---------------------------------------------------------------------------
// PyLevel — Python-accessible wrapper around Level
// ---------------------------------------------------------------------------

/// A parsed or constructed Mummy Maze level.
///
/// Construct via `Level.from_file()` or `Level.from_edges()`, then pass to
/// `solve()`, `analyze()`, `build_graph()`, or `policy_win_prob()`.
#[pyclass(name = "Level")]
#[derive(Clone)]
struct PyLevel {
    inner: Level,
}

#[pymethods]
impl PyLevel {
    /// Parse a single level from a .dat file.
    #[staticmethod]
    #[pyo3(signature = (dat_path, sublevel))]
    fn from_file(dat_path: &str, sublevel: usize) -> PyResult<Self> {
        let path = Path::new(dat_path);
        let (_, levels) = crate::parse::parse_file(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        if sublevel >= levels.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("sublevel out of range"));
        }
        Ok(PyLevel { inner: levels.into_iter().nth(sublevel).unwrap() })
    }

    /// Construct a level from edge-array walls and entity positions.
    ///
    /// Args:
    ///     grid_size: 6, 8, or 10
    ///     flip: True for red mummies, False for white
    ///     h_walls: flat list of (n+1)*n bools, row-major
    ///     v_walls: flat list of n*(n+1) bools, row-major
    ///     exit_side: "N", "S", "E", or "W"
    ///     exit_pos: position along the exit side (0-indexed)
    ///     player: (row, col)
    ///     mummy1: (row, col)
    ///     mummy2: (row, col) or None
    ///     scorpion: (row, col) or None
    ///     traps: list of (row, col), up to 2
    ///     gate: (row, col) or None
    ///     key: (row, col) or None
    #[staticmethod]
    #[pyo3(signature = (grid_size, flip, h_walls, v_walls, exit_side, exit_pos, player, mummy1, mummy2=None, scorpion=None, traps=vec![], gate=None, key=None))]
    fn from_edges(
        grid_size: i32,
        flip: bool,
        h_walls: Vec<bool>,
        v_walls: Vec<bool>,
        exit_side: &str,
        exit_pos: i32,
        player: (i32, i32),
        mummy1: (i32, i32),
        mummy2: Option<(i32, i32)>,
        scorpion: Option<(i32, i32)>,
        traps: Vec<(i32, i32)>,
        gate: Option<(i32, i32)>,
        key: Option<(i32, i32)>,
    ) -> Self {
        PyLevel {
            inner: Level::from_edges(
                grid_size, flip, &h_walls, &v_walls,
                exit_side, exit_pos, player, mummy1,
                mummy2, scorpion, &traps, gate, key,
            ),
        }
    }

    #[getter]
    fn grid_size(&self) -> i32 {
        self.inner.grid_size
    }

    #[getter]
    fn flip(&self) -> bool {
        self.inner.flip
    }

    fn __repr__(&self) -> String {
        let l = &self.inner;
        format!(
            "Level(grid_size={}, flip={}, player=({},{}), mummy1=({},{}))",
            l.grid_size, l.flip, l.player_row, l.player_col, l.mummy1_row, l.mummy1_col,
        )
    }
}

// ---------------------------------------------------------------------------
// Per-level operations (take PyLevel)
// ---------------------------------------------------------------------------

/// Parse all levels from a .dat file.
#[pyfunction]
#[pyo3(signature = (dat_path,))]
fn parse_file(dat_path: &str) -> PyResult<Vec<PyLevel>> {
    let path = Path::new(dat_path);
    let (_, levels) = crate::parse::parse_file(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(levels.into_iter().map(|l| PyLevel { inner: l }).collect())
}

/// BFS solve. Returns move count or None if unsolvable.
#[pyfunction]
#[pyo3(signature = (level,))]
fn solve(level: &PyLevel) -> Option<u32> {
    crate::solver::solve(&level.inner).moves
}

/// BFS solve returning the full action sequence as a list of ints.
/// Action indices: N=0, S=1, E=2, W=3, Wait=4. Returns None if unsolvable.
#[pyfunction]
#[pyo3(signature = (level,))]
fn solve_actions(level: &PyLevel) -> Option<Vec<u32>> {
    crate::solver::solve(&level.inner)
        .actions
        .map(|acts| acts.iter().map(|a| a.to_index() as u32).collect())
}

/// Full analysis: BFS + state graph + Markov chain + difficulty metrics.
#[pyfunction]
#[pyo3(signature = (level, label="", sublevel=0))]
fn analyze(py: Python<'_>, level: &PyLevel, label: &str, sublevel: usize) -> PyResult<PyObject> {
    let result = batch::analyze_level(label, sublevel, &level.inner)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    analysis_to_dict(py, &result)
}

/// Build and return the full state graph.
/// Returns: {"states": [...], "edges": [(src_idx, action, dst), ...], "start_idx": int}
#[pyfunction]
#[pyo3(signature = (level,))]
fn build_graph(py: Python<'_>, level: &PyLevel) -> PyResult<PyObject> {
    let graph = crate::graph::build_graph(&level.inner);
    let indices = graph.state_indices();
    let state_to_idx = &indices.state_to_idx;
    let state_list = &indices.idx_to_state;

    let py_states: Vec<StateTuple> = state_list.iter().map(state_to_tuple).collect();

    let mut py_edges: Vec<PyObject> = Vec::new();
    for (s, transitions) in &graph.transitions {
        let src_idx = state_to_idx[s];
        for &(action, dest) in transitions {
            let dst: PyObject = match dest {
                crate::graph::StateKey::Transient(ns) => state_to_idx[&ns].into_pyobject(py)?.into(),
                crate::graph::StateKey::Win => "WIN".into_pyobject(py)?.into(),
                crate::graph::StateKey::Dead => "DEAD".into_pyobject(py)?.into(),
            };
            let edge = (src_idx, action_name(action), dst);
            py_edges.push(edge.into_pyobject(py)?.into());
        }
    }

    let start_idx = state_to_idx[&graph.start];

    let dict = PyDict::new(py);
    dict.set_item("states", py_states)?;
    dict.set_item("edges", py_edges)?;
    dict.set_item("start_idx", start_idx)?;
    Ok(dict.into())
}

/// Compute exact win probability under an arbitrary policy for a single level.
///
/// Args:
///     level: Level object
///     state_tuples: numpy (n_states, 12) i32
///     action_probs: numpy (n_states, 5) f32
///
/// Returns win probability (f64), or NaN if solver fails to converge.
#[pyfunction]
#[pyo3(signature = (level, state_tuples, action_probs))]
fn policy_win_prob(
    level: &PyLevel,
    state_tuples: PyReadonlyArray2<i32>,
    action_probs: PyReadonlyArray2<f32>,
) -> PyResult<f64> {
    use crate::markov::MarkovChain;
    use rustc_hash::FxHashMap;

    let (st_vec, ap_vec) = extract_policy_arrays(&state_tuples, &action_probs);

    let mut policy: FxHashMap<State, [f64; 5]> =
        FxHashMap::with_capacity_and_hasher(st_vec.len(), Default::default());
    for (tuple, probs) in st_vec.iter().zip(ap_vec.iter()) {
        let state = State::from_i32_array(tuple);
        policy.insert(state, probs.map(|p| p as f64));
    }

    let graph = crate::graph::build_graph(&level.inner);
    let chain = MarkovChain::from_graph_with_policy(&graph, &policy);
    match chain.solve_win_probs_tol(1e-10, 200_000) {
        Ok(win_probs) => Ok(chain.start_idx.map_or(0.0, |i| win_probs[i])),
        Err(_) => Ok(f64::NAN),
    }
}

// ---------------------------------------------------------------------------
// Batch operations (take lists of PyLevel or directory paths)
// ---------------------------------------------------------------------------

/// Compute exact win probability under an arbitrary policy for a batch of levels.
///
/// Args:
///     levels: list of Level objects
///     state_tuples: numpy (total_states, 12) i32 — all levels concatenated
///     action_probs: numpy (total_states, 5) f32
///     offsets: list of n_levels+1 ints slicing into the flat arrays
///
/// Returns list of f64 win probabilities, one per level.
#[pyfunction]
#[pyo3(signature = (levels, state_tuples, action_probs, offsets))]
fn policy_win_prob_batch(
    py: Python<'_>,
    levels: Vec<PyRef<'_, PyLevel>>,
    state_tuples: PyReadonlyArray2<i32>,
    action_probs: PyReadonlyArray2<f32>,
    offsets: Vec<usize>,
) -> PyResult<Vec<f64>> {
    let (st_vec, ap_vec) = extract_policy_arrays(&state_tuples, &action_probs);
    let level_refs: Vec<&Level> = levels.iter().map(|l| &l.inner).collect();

    py.allow_threads(|| {
        batch::policy_win_prob_batch(&level_refs, &st_vec, &ap_vec, &offsets)
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Analyze all levels in a directory. Releases the GIL, uses rayon internally.
#[pyfunction]
#[pyo3(signature = (maze_dir, jobs=0))]
fn analyze_all(py: Python<'_>, maze_dir: &str, jobs: usize) -> PyResult<Vec<PyObject>> {
    let path = Path::new(maze_dir);
    let results = py
        .allow_threads(|| batch::analyze_all(path, jobs, None))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut out = Vec::with_capacity(results.len());
    for r in &results {
        out.push(analysis_to_dict(py, r)?);
    }
    Ok(out)
}

/// Solve all levels in a directory, returning action sequences. Releases the GIL.
/// Returns list of dicts: {"file": str, "sublevel": int, "actions": list[int] | None}
#[pyfunction]
#[pyo3(signature = (maze_dir, jobs=0))]
fn solve_all_actions(py: Python<'_>, maze_dir: &str, jobs: usize) -> PyResult<Vec<PyObject>> {
    let path = Path::new(maze_dir);
    let results = py
        .allow_threads(|| batch::solve_all_with_actions(path, jobs))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut out = Vec::with_capacity(results.len());
    for (stem, sub_idx, actions) in &results {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("file", stem)?;
        dict.set_item("sublevel", sub_idx)?;
        dict.set_item("actions", actions.as_ref().map(|a| a.iter().map(|x| x.to_index() as u32).collect::<Vec<u32>>()))?;
        out.push(dict.into());
    }
    Ok(out)
}

/// Compute best (distance-reducing) actions for every winnable state across all levels.
/// Releases the GIL, uses rayon internally.
#[pyfunction]
#[pyo3(signature = (maze_dir, jobs=0))]
fn best_actions_all(py: Python<'_>, maze_dir: &str, jobs: usize) -> PyResult<Vec<PyObject>> {
    let path = Path::new(maze_dir);
    let results = py
        .allow_threads(|| batch::best_actions_all(path, jobs))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut out = Vec::with_capacity(results.len());
    for (stem, sub_idx, grid_size, state_actions) in &results {
        let dict = PyDict::new(py);
        dict.set_item("file", stem)?;
        dict.set_item("sublevel", sub_idx)?;
        dict.set_item("grid_size", grid_size)?;

        let states: Vec<StateTuple> = state_actions
            .iter()
            .map(|(s, _)| state_to_tuple(s))
            .collect();
        dict.set_item("states", states)?;

        let masks: Vec<u8> = state_actions.iter().map(|(_, m)| *m).collect();
        dict.set_item("action_masks", masks)?;

        out.push(dict.into());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Fitness expression evaluation
// ---------------------------------------------------------------------------

/// Evaluate a fitness expression against a dict of metric values.
///
/// Available variables: win_prob, bfs_moves, n_states, dead_end_ratio,
/// avg_branching, n_optimal, greedy_deviation, path_safety.
///
/// >>> eval_fitness("-win_prob + bfs_moves / 1000", {"win_prob": 0.5, "bfs_moves": 20})
/// -0.48
#[pyfunction]
#[pyo3(signature = (expr, metrics))]
fn eval_fitness(expr: &str, metrics: &Bound<'_, PyDict>) -> PyResult<f64> {
    use crate::ga::fitness::{FitnessExpr, FitnessVars};

    let get = |key: &str| -> f64 {
        metrics
            .get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(0.0)
    };

    let wp = get("win_prob");
    let vars = FitnessVars {
        win_prob: wp,
        log_win_prob: if wp > 0.0 { wp.log10() } else { f64::NEG_INFINITY },
        bfs_moves: get("bfs_moves"),
        n_states: get("n_states"),
        dead_end_ratio: get("dead_end_ratio"),
        avg_branching: get("avg_branching"),
        n_optimal: get("n_optimal"),
        greedy_deviation: get("greedy_deviation"),
        path_safety: get("path_safety"),
    };

    let fitness = FitnessExpr::parse(expr)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(fitness.eval(&vars))
}

/// Return the list of available fitness variable names and descriptions.
///
/// >>> fitness_variables()
/// [("win_prob", "Win probability under uniform-random policy (0–1)"), ...]
#[pyfunction]
fn fitness_variables() -> Vec<(&'static str, &'static str)> {
    crate::ga::fitness::VARIABLES.to_vec()
}

/// Return the list of built-in fitness presets.
///
/// >>> fitness_presets()
/// [("Default", "-win_prob + bfs_moves / 1000"), ...]
#[pyfunction]
fn fitness_presets() -> Vec<(&'static str, &'static str)> {
    crate::ga::fitness::PRESETS.to_vec()
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn mummymaze_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLevel>()?;
    m.add_function(wrap_pyfunction!(parse_file, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_actions, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(policy_win_prob, m)?)?;
    m.add_function(wrap_pyfunction!(policy_win_prob_batch, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_all, m)?)?;
    m.add_function(wrap_pyfunction!(solve_all_actions, m)?)?;
    m.add_function(wrap_pyfunction!(best_actions_all, m)?)?;
    m.add_function(wrap_pyfunction!(eval_fitness, m)?)?;
    m.add_function(wrap_pyfunction!(fitness_variables, m)?)?;
    m.add_function(wrap_pyfunction!(fitness_presets, m)?)?;
    Ok(())
}
