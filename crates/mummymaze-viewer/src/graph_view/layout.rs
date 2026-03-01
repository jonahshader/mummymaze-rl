//! CPU layout algorithms for initial node positioning.

use mummymaze::game::State;
use mummymaze::graph::{StateGraph, StateKey};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Compute BFS depth from the start state for each node.
pub fn bfs_depths(graph: &StateGraph) -> FxHashMap<State, u32> {
    let mut depths = FxHashMap::default();
    depths.insert(graph.start, 0);
    let mut queue = VecDeque::new();
    queue.push_back(graph.start);

    while let Some(cur) = queue.pop_front() {
        let d = depths[&cur];
        if let Some(transitions) = graph.transitions.get(&cur) {
            for &(_action, dest) in transitions {
                if let StateKey::Transient(ns) = dest {
                    if !depths.contains_key(&ns) {
                        depths.insert(ns, d + 1);
                        queue.push_back(ns);
                    }
                }
            }
        }
    }
    depths
}

/// Random initial positions scattered in a circle.
pub fn random_positions(n: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut positions = Vec::with_capacity(n);
    // Simple LCG pseudo-random
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let radius = (n as f32).sqrt() * 2.0;
    for _ in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = ((rng >> 33) as f32 / (u32::MAX as f32) - 0.5) * radius * 2.0;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let y = ((rng >> 33) as f32 / (u32::MAX as f32) - 0.5) * radius * 2.0;
        positions.push([x, y]);
    }
    positions
}

/// BFS layer layout: nodes arranged by BFS depth, spread horizontally.
pub fn bfs_layer_positions(
    state_indices: &FxHashMap<State, usize>,
    depths: &FxHashMap<State, u32>,
    n: usize,
) -> Vec<[f32; 2]> {
    let max_depth = depths.values().copied().max().unwrap_or(0);
    let mut positions = vec![[0.0f32; 2]; n];

    // Group nodes by depth
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); (max_depth + 1) as usize];
    for (state, &idx) in state_indices {
        if let Some(&d) = depths.get(state) {
            layers[d as usize].push(idx);
        }
    }

    let spacing_y = 3.0;
    for (d, layer) in layers.iter().enumerate() {
        let n_in_layer = layer.len();
        if n_in_layer == 0 {
            continue;
        }
        let spacing_x = 2.0;
        let width = (n_in_layer as f32 - 1.0) * spacing_x;
        for (i, &idx) in layer.iter().enumerate() {
            positions[idx] = [
                i as f32 * spacing_x - width / 2.0,
                -(d as f32 * spacing_y), // top to bottom
            ];
        }
    }
    positions
}

/// Radial tree layout: BFS tree with depth as radius.
pub fn radial_tree_positions(
    state_indices: &FxHashMap<State, usize>,
    depths: &FxHashMap<State, u32>,
    n: usize,
) -> Vec<[f32; 2]> {
    let max_depth = depths.values().copied().max().unwrap_or(0);
    let mut positions = vec![[0.0f32; 2]; n];

    // Group by depth
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); (max_depth + 1) as usize];
    for (state, &idx) in state_indices {
        if let Some(&d) = depths.get(state) {
            layers[d as usize].push(idx);
        }
    }

    let ring_spacing = 3.0;
    for (d, layer) in layers.iter().enumerate() {
        if d == 0 {
            // Start node at center
            for &idx in layer {
                positions[idx] = [0.0, 0.0];
            }
            continue;
        }
        let radius = d as f32 * ring_spacing;
        let n_in_ring = layer.len();
        for (i, &idx) in layer.iter().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * (i as f32 / n_in_ring as f32);
            positions[idx] = [radius * angle.cos(), radius * angle.sin()];
        }
    }
    positions
}
