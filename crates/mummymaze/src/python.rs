//! PyO3 bindings for the mummymaze crate.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use crate::batch::{self, LevelAnalysis};

/// Convert a LevelAnalysis to a Python dict.
fn level_analysis_to_dict(py: Python<'_>, r: &LevelAnalysis) -> PyResult<PyObject> {
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

/// Analyze a single level, returning a dict with all results.
#[pyfunction]
#[pyo3(signature = (dat_path, sublevel))]
fn analyze_level(py: Python<'_>, dat_path: &str, sublevel: usize) -> PyResult<PyObject> {
    let path = Path::new(dat_path);
    let result = batch::analyze_one(path, sublevel)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    level_analysis_to_dict(py, &result)
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
        out.push(level_analysis_to_dict(py, r)?);
    }
    Ok(out)
}

/// BFS-only solve for a single level. Returns move count.
#[pyfunction]
#[pyo3(signature = (dat_path, sublevel))]
fn solve_level(_py: Python<'_>, dat_path: &str, sublevel: usize) -> PyResult<Option<u32>> {
    let path = Path::new(dat_path);
    batch::solve_one(path, sublevel)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// BFS solve returning the full action sequence as a list of ints.
/// Action indices match JAX env: N=0, S=1, E=2, W=3, Wait=4.
/// Returns None if unsolvable.
#[pyfunction]
#[pyo3(signature = (dat_path, sublevel))]
fn solve_level_actions(_py: Python<'_>, dat_path: &str, sublevel: usize) -> PyResult<Option<Vec<u32>>> {
    let path = Path::new(dat_path);
    let (_, levels) = crate::parse::parse_file(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    if sublevel >= levels.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err("sublevel out of range"));
    }
    let result = crate::solver::solve(&levels[sublevel]);
    Ok(result.actions.map(|acts| acts.iter().map(|a| a.to_index() as u32).collect()))
}

/// Solve all levels, returning action sequences. Releases the GIL, uses rayon internally.
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

/// Build and return the full state graph as a Python dict.
/// Returns: {"states": [state_tuple, ...], "edges": [(src_idx, action_name, dst)], "start_idx": int}
/// where dst is an int (index into states) or "WIN" or "DEAD".
/// state_tuple = (pr, pc, m1r, m1c, m1_alive, m2r, m2c, m2_alive, sr, sc, s_alive, gate_open)
#[pyfunction]
#[pyo3(signature = (dat_path, sublevel))]
fn build_graph(py: Python<'_>, dat_path: &str, sublevel: usize) -> PyResult<PyObject> {
    let path = Path::new(dat_path);
    let (_, levels) = crate::parse::parse_file(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    if sublevel >= levels.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err("sublevel out of range"));
    }
    let graph = crate::graph::build_graph(&levels[sublevel]);

    // Map states to indices
    let indices = graph.state_indices();
    let state_to_idx = &indices.state_to_idx;
    let state_list = &indices.idx_to_state;

    // Convert states to Python tuples
    let py_states: Vec<(i32,i32,i32,i32,bool,i32,i32,bool,i32,i32,bool,bool)> = state_list.iter().map(|s| {
        (s.player_row, s.player_col,
         s.mummy1_row, s.mummy1_col, s.mummy1_alive,
         s.mummy2_row, s.mummy2_col, s.mummy2_alive,
         s.scorpion_row, s.scorpion_col, s.scorpion_alive,
         s.gate_open)
    }).collect();

    // Build edges
    let action_name = |a: crate::game::Action| -> &'static str {
        match a {
            crate::game::Action::North => "N",
            crate::game::Action::South => "S",
            crate::game::Action::East => "E",
            crate::game::Action::West => "W",
            crate::game::Action::Wait => "wait",
        }
    };

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

/// Python module definition
#[pymodule]
fn mummymaze_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_level, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_all, m)?)?;
    m.add_function(wrap_pyfunction!(solve_level, m)?)?;
    m.add_function(wrap_pyfunction!(solve_level_actions, m)?)?;
    m.add_function(wrap_pyfunction!(solve_all_actions, m)?)?;
    m.add_function(wrap_pyfunction!(build_graph, m)?)?;
    Ok(())
}
