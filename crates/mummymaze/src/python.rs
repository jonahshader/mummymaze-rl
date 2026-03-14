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

    /// Hash all gameplay-relevant fields to a u64 fingerprint.
    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }

    /// Canonical fingerprint under dihedral symmetry.
    /// Gate-free levels use all 8 D4 symmetries; gate levels use identity,
    /// rot180, h_mirror, v_mirror.
    fn canonical_fingerprint(&self) -> u64 {
        self.inner.canonical_fingerprint()
    }

    /// Whether this level has a gate (and key).
    #[getter]
    fn has_gate(&self) -> bool {
        self.inner.has_gate
    }

    /// Apply a dihedral symmetry transform (0..8 for the 8 D4 elements).
    fn apply_dihedral(&self, sym: u8) -> PyLevel {
        PyLevel { inner: self.inner.apply_dihedral(sym) }
    }

    /// Serialize level to a Python dict (all fields).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let l = &self.inner;
        let dict = PyDict::new(py);
        dict.set_item("grid_size", l.grid_size)?;
        dict.set_item("flip", l.flip)?;

        let (h_walls, v_walls) = l.to_edges();
        dict.set_item("h_walls", h_walls)?;
        dict.set_item("v_walls", v_walls)?;
        dict.set_item("exit_side", l.exit_side_str())?;
        dict.set_item("exit_pos", l.exit_pos())?;

        dict.set_item("player", (l.player_row, l.player_col))?;
        dict.set_item("mummy1", (l.mummy1_row, l.mummy1_col))?;
        dict.set_item("mummy2", if l.has_mummy2 { Some((l.mummy2_row, l.mummy2_col)) } else { None })?;
        dict.set_item("scorpion", if l.has_scorpion { Some((l.scorpion_row, l.scorpion_col)) } else { None })?;

        let mut traps: Vec<(i32, i32)> = Vec::new();
        if l.trap_count >= 1 { traps.push((l.trap1_row, l.trap1_col)); }
        if l.trap_count >= 2 { traps.push((l.trap2_row, l.trap2_col)); }
        dict.set_item("traps", traps)?;

        dict.set_item("gate", if l.has_gate { Some((l.gate_row, l.gate_col)) } else { None })?;
        dict.set_item("key", if l.has_gate { Some((l.key_row, l.key_col)) } else { None })?;

        Ok(dict)
    }

    /// Serialize level to a JSON string (internal representation).
    ///
    /// This uses serde to serialize the full Rust Level struct, including
    /// cell bitmask walls. The Rust viewer can deserialize this directly.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Reconstruct a Level from a dict (as returned by to_dict).
    #[staticmethod]
    #[pyo3(signature = (d,))]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let grid_size: i32 = d.get_item("grid_size")?.unwrap().extract()?;
        let flip: bool = d.get_item("flip")?.unwrap().extract()?;
        let h_walls: Vec<bool> = d.get_item("h_walls")?.unwrap().extract()?;
        let v_walls: Vec<bool> = d.get_item("v_walls")?.unwrap().extract()?;
        let exit_side: String = d.get_item("exit_side")?.unwrap().extract()?;
        let exit_pos: i32 = d.get_item("exit_pos")?.unwrap().extract()?;
        let player: (i32, i32) = d.get_item("player")?.unwrap().extract()?;
        let mummy1: (i32, i32) = d.get_item("mummy1")?.unwrap().extract()?;
        let mummy2: Option<(i32, i32)> = d.get_item("mummy2")?.unwrap().extract()?;
        let scorpion: Option<(i32, i32)> = d.get_item("scorpion")?.unwrap().extract()?;
        let traps: Vec<(i32, i32)> = d.get_item("traps")?.unwrap().extract()?;
        let gate: Option<(i32, i32)> = d.get_item("gate")?.unwrap().extract()?;
        let key: Option<(i32, i32)> = d.get_item("key")?.unwrap().extract()?;

        Ok(PyLevel {
            inner: Level::from_edges(
                grid_size, flip, &h_walls, &v_walls,
                &exit_side, exit_pos, player, mummy1,
                mummy2, scorpion, &traps, gate, key,
            ),
        })
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
    // Use log-space solver — always converges, even for very low win probs.
    match chain.start_log_win_prob() {
        Ok((wp, _log_p)) => Ok(wp),
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

/// Set "states" and "action_masks" on a dict from a state-actions list.
fn set_state_actions(dict: &Bound<'_, PyDict>, state_actions: &[(crate::game::State, u8)]) -> PyResult<()> {
    let states: Vec<StateTuple> = state_actions.iter().map(|(s, _)| state_to_tuple(s)).collect();
    dict.set_item("states", states)?;
    let masks: Vec<u8> = state_actions.iter().map(|(_, m)| *m).collect();
    dict.set_item("action_masks", masks)?;
    Ok(())
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
        set_state_actions(&dict, state_actions)?;
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
    let pwp = get("policy_win_prob");
    let vars = FitnessVars {
        win_prob: wp,
        log_win_prob: get("log_win_prob"),
        policy_win_prob: pwp,
        log_policy_win_prob: get("log_policy_win_prob"),
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
// Best actions for arbitrary level lists
// ---------------------------------------------------------------------------

/// Compute best (distance-reducing) actions for every winnable state in a list of levels.
/// Returns dicts with `level_idx`, `grid_size`, `states`, `action_masks` (no `file`/`sublevel`).
#[pyfunction]
#[pyo3(signature = (levels,))]
fn best_actions_for_levels(py: Python<'_>, levels: Vec<PyRef<'_, PyLevel>>) -> PyResult<Vec<PyObject>> {
    use crate::graph::build_graph as bg;
    use rayon::prelude::*;

    let inner_levels: Vec<&Level> = levels.iter().map(|l| &l.inner).collect();

    let results: Vec<Option<(i32, Vec<(crate::game::State, u8)>)>> = py.allow_threads(|| {
        inner_levels
            .par_iter()
            .map(|lev| {
                let graph = bg(lev);
                let optimal = graph.best_actions_per_state();
                if optimal.is_empty() {
                    None
                } else {
                    Some((lev.grid_size, optimal))
                }
            })
            .collect()
    });

    let mut out = Vec::new();
    for (i, result) in results.into_iter().enumerate() {
        if let Some((grid_size, state_actions)) = result {
            let dict = PyDict::new(py);
            dict.set_item("level_idx", i)?;
            dict.set_item("grid_size", grid_size)?;
            set_state_actions(&dict, &state_actions)?;
            out.push(dict.into());
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// GA primitives (for Python-orchestrated GA loop)
// ---------------------------------------------------------------------------

/// Mutate a level using weighted random mutation operators.
///
/// Args:
///     level: Level to mutate
///     seed: RNG seed
///     w_wall, w_move_entity, w_move_player, w_add_entity, w_remove_entity,
///     w_move_exit: relative mutation weights
///     extra_wall_prob: probability of an extra wall flip after the primary mutation
///
/// Returns a new mutated Level.
#[pyfunction]
#[pyo3(signature = (level, seed, w_wall=5.0, w_move_entity=3.0, w_move_player=2.0,
                    w_add_entity=1.0, w_remove_entity=1.0, w_move_exit=1.0,
                    extra_wall_prob=0.3))]
fn mutate(
    level: &PyLevel,
    seed: u64,
    w_wall: f64,
    w_move_entity: f64,
    w_move_player: f64,
    w_add_entity: f64,
    w_remove_entity: f64,
    w_move_exit: f64,
    extra_wall_prob: f64,
) -> PyLevel {
    use crate::ga::GaConfig;
    use rand::SeedableRng;

    let config = GaConfig {
        w_wall,
        w_move_entity,
        w_move_player,
        w_add_entity,
        w_remove_entity,
        w_move_exit,
        extra_wall_prob,
        ..GaConfig::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    PyLevel {
        inner: crate::ga::mutate_with_config(&level.inner, &mut rng, &config),
    }
}

/// Mutate a batch of levels. Each level gets a deterministic RNG derived from
/// `base_seed + index`.
///
/// Returns a list of mutated Levels (same length as input).
#[pyfunction]
#[pyo3(signature = (levels, base_seed, w_wall=5.0, w_move_entity=3.0, w_move_player=2.0,
                    w_add_entity=1.0, w_remove_entity=1.0, w_move_exit=1.0,
                    extra_wall_prob=0.3))]
fn mutate_batch(
    py: Python<'_>,
    levels: Vec<PyRef<'_, PyLevel>>,
    base_seed: u64,
    w_wall: f64,
    w_move_entity: f64,
    w_move_player: f64,
    w_add_entity: f64,
    w_remove_entity: f64,
    w_move_exit: f64,
    extra_wall_prob: f64,
) -> Vec<PyLevel> {
    use crate::ga::GaConfig;
    use rand::SeedableRng;

    let config = GaConfig {
        w_wall,
        w_move_entity,
        w_move_player,
        w_add_entity,
        w_remove_entity,
        w_move_exit,
        extra_wall_prob,
        ..GaConfig::default()
    };
    let inner_levels: Vec<&Level> = levels.iter().map(|l| &l.inner).collect();

    py.allow_threads(|| {
        inner_levels
            .iter()
            .enumerate()
            .map(|(i, lev)| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
                PyLevel {
                    inner: crate::ga::mutate_with_config(lev, &mut rng, &config),
                }
            })
            .collect()
    })
}

/// Crossover two levels to produce an offspring.
///
/// Args:
///     a, b: parent levels (must have the same grid_size)
///     mode: "swap_entities", "region", "wall_patch", or "feature_level"
///     seed: RNG seed
#[pyfunction]
#[pyo3(signature = (a, b, mode="swap_entities", seed=0))]
fn ga_crossover(
    a: &PyLevel,
    b: &PyLevel,
    mode: &str,
    seed: u64,
) -> PyResult<PyLevel> {
    use crate::ga::{crossover as cx, CrossoverMode};
    use rand::SeedableRng;

    let cx_mode = match mode {
        "swap_entities" => CrossoverMode::SwapEntities,
        "region" => CrossoverMode::Region,
        "wall_patch" => CrossoverMode::WallPatch,
        "feature_level" => CrossoverMode::FeatureLevel,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown crossover mode: {mode:?}. Use: swap_entities, region, wall_patch, feature_level"
            )));
        }
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    Ok(PyLevel {
        inner: cx(&a.inner, &b.inner, &mut rng, cx_mode),
    })
}

/// Evaluate a batch of levels: parallel BFS + state graph + Markov chain.
///
/// Returns a list of dicts for solvable levels:
///     {"level_idx": int, "level": Level, "bfs_moves": int, "n_states": int,
///      "win_prob": float, "log_win_prob": float}
///
/// If `fitness_expr` is provided, also includes "fitness": float and any
/// difficulty metrics the expression requires.
///
/// Unsolvable levels are omitted from the output.
/// Releases the GIL, uses rayon for parallelism.
#[pyfunction]
#[pyo3(signature = (levels, fitness_expr=None))]
fn ga_evaluate_batch(
    py: Python<'_>,
    levels: Vec<PyRef<'_, PyLevel>>,
    fitness_expr: Option<&str>,
) -> PyResult<Vec<PyObject>> {
    use crate::ga::fitness::FitnessExpr;
    use crate::graph::build_graph;
    use crate::markov::MarkovChain;
    use rayon::prelude::*;

    let expr = match fitness_expr {
        Some(e) => FitnessExpr::parse(e)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
        None => FitnessExpr::default(),
    };

    let inner_levels: Vec<(usize, &Level)> = levels
        .iter()
        .enumerate()
        .map(|(i, l)| (i, &l.inner))
        .collect();

    struct EvalResult {
        level_idx: usize,
        level: Level,
        bfs_moves: u32,
        n_states: usize,
        win_prob: f64,
        log_win_prob: f64,
        fitness: f64,
    }

    let results: Vec<EvalResult> = py.allow_threads(|| {
        inner_levels
            .par_iter()
            .filter_map(|&(idx, lev)| {
                let solve = crate::solver::solve(lev);
                let moves = solve.moves?;
                let graph = build_graph(lev);
                let chain = MarkovChain::from_graph(&graph);
                let start_idx = chain.start_idx?;
                let log_win_prob = chain.solve_log_win_probs().ok()?;
                let lwp = log_win_prob[start_idx];
                let wp = 10.0f64.powf(lwp).max(0.0);

                let vars = expr.compute_vars(&graph, lev, &solve, wp, lwp);
                let fitness = expr.eval(&vars);

                Some(EvalResult {
                    level_idx: idx,
                    level: lev.clone(),
                    bfs_moves: moves,
                    n_states: graph.n_transient,
                    win_prob: wp,
                    log_win_prob: lwp,
                    fitness,
                })
            })
            .collect()
    });

    let mut out = Vec::with_capacity(results.len());
    for r in results {
        let dict = PyDict::new(py);
        dict.set_item("level_idx", r.level_idx)?;
        dict.set_item("level", Py::new(py, PyLevel { inner: r.level })?)?;
        dict.set_item("bfs_moves", r.bfs_moves)?;
        dict.set_item("n_states", r.n_states)?;
        dict.set_item("win_prob", r.win_prob)?;
        dict.set_item("log_win_prob", r.log_win_prob)?;
        dict.set_item("fitness", r.fitness)?;
        out.push(dict.into());
    }
    Ok(out)
}

/// Replay a sequence of actions on a level, returning the state after each step.
///
/// Returns a list of dicts, one per step (including the initial state at index 0):
///     {"player": (r, c), "mummy1": (r, c, alive), "mummy2": (r, c, alive),
///      "scorpion": (r, c, alive), "gate_open": bool, "result": "ok"|"win"|"dead"}
///
/// The list has length `len(actions) + 1` (initial state + one per action).
#[pyfunction]
#[pyo3(signature = (level, actions))]
fn replay_actions(py: Python<'_>, level: &PyLevel, actions: Vec<u32>) -> PyResult<Vec<PyObject>> {
    use crate::game::{Action, StepResult, step};

    let lev = &level.inner;
    let mut state = State::from_level(lev);
    let mut frames = Vec::with_capacity(actions.len() + 1);

    let state_to_dict = |py: Python<'_>, s: &State, result: &str| -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("player", (s.player_row, s.player_col))?;
        dict.set_item("mummy1", (s.mummy1_row, s.mummy1_col, s.mummy1_alive))?;
        dict.set_item("mummy2", (s.mummy2_row, s.mummy2_col, s.mummy2_alive))?;
        dict.set_item("scorpion", (s.scorpion_row, s.scorpion_col, s.scorpion_alive))?;
        dict.set_item("gate_open", s.gate_open)?;
        dict.set_item("result", result)?;
        Ok(dict.into())
    };

    // Initial state
    frames.push(state_to_dict(py, &state, "ok")?);

    for &action_idx in &actions {
        let action = match action_idx {
            0 => Action::North,
            1 => Action::South,
            2 => Action::East,
            3 => Action::West,
            4 => Action::Wait,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Invalid action index: {action_idx}. Use 0-4 (N/S/E/W/Wait)")
                ));
            }
        };
        let result = step(lev, &mut state, action);
        let result_str = match result {
            StepResult::Ok => "ok",
            StepResult::Win => "win",
            StepResult::Dead => "dead",
        };
        frames.push(state_to_dict(py, &state, result_str)?);
    }

    Ok(frames)
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
    m.add_function(wrap_pyfunction!(best_actions_for_levels, m)?)?;
    m.add_function(wrap_pyfunction!(mutate, m)?)?;
    m.add_function(wrap_pyfunction!(mutate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ga_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(ga_evaluate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(replay_actions, m)?)?;

    Ok(())
}
