use mummymaze::graph::{StateGraph, StateKey};
use rustc_hash::FxHashMap;

use super::gpu::GraphBuffers;
use super::layout;
use super::types::*;
use super::{metric_color, GraphView, LayoutMode, NodeKind};

use std::sync::Arc;

impl GraphView {
    pub(super) fn build_from_graph(
        &mut self,
        graph: &StateGraph,
        chain: &mummymaze::markov::MarkovChain,
    ) {
        let mut state_to_idx = FxHashMap::default();
        let mut idx_to_state: Vec<Option<mummymaze::game::State>> = Vec::new();
        let mut node_kinds: Vec<NodeKind> = Vec::new();
        let mut idx = 0usize;

        // Start state first
        state_to_idx.insert(graph.start, idx);
        idx_to_state.push(Some(graph.start));
        node_kinds.push(NodeKind::Transient);
        idx += 1;

        // Other transient states
        for state in graph.transitions.keys() {
            if *state != graph.start {
                state_to_idx.insert(*state, idx);
                idx_to_state.push(Some(*state));
                node_kinds.push(NodeKind::Transient);
                idx += 1;
            }
        }

        // Create per-source terminal nodes: one WIN and/or one DEAD per source state.
        let mut win_terminal_for: FxHashMap<mummymaze::game::State, usize> = FxHashMap::default();
        let mut dead_terminal_for: FxHashMap<mummymaze::game::State, usize> =
            FxHashMap::default();

        for (state, transitions) in &graph.transitions {
            let mut needs_win = false;
            let mut needs_dead = false;
            for &(_, dest) in transitions {
                match dest {
                    StateKey::Win => needs_win = true,
                    StateKey::Dead => needs_dead = true,
                    _ => {}
                }
            }
            if needs_win {
                win_terminal_for.insert(*state, idx);
                idx_to_state.push(Some(*state));
                node_kinds.push(NodeKind::Win);
                idx += 1;
            }
            if needs_dead {
                dead_terminal_for.insert(*state, idx);
                idx_to_state.push(Some(*state));
                node_kinds.push(NodeKind::Dead);
                idx += 1;
            }
        }

        let n_nodes = idx;

        // Build edges + edge lookup
        let mut edges: Vec<EdgeGpu> = Vec::new();
        let mut edge_lookup: FxHashMap<(u32, u32), Vec<usize>> = FxHashMap::default();

        for (state, transitions) in &graph.transitions {
            let src = state_to_idx[state];
            for &(_, dest) in transitions {
                let dst = match dest {
                    StateKey::Transient(ns) => state_to_idx[&ns],
                    StateKey::Win => win_terminal_for[state],
                    StateKey::Dead => dead_terminal_for[state],
                };
                let edge_idx = edges.len();
                edge_lookup
                    .entry((src as u32, dst as u32))
                    .or_default()
                    .push(edge_idx);
                edges.push(EdgeGpu {
                    src: src as u32,
                    dst: dst as u32,
                });
            }
        }

        self.edge_lookup = edge_lookup;
        self.n_edges = edges.len();

        // --- Compute per-node metrics ---
        let win_probs = chain.solve_win_probs().ok().map(|v| chain.per_state_map(&v));
        let expected_steps = chain.solve_expected_steps().ok().map(|v| chain.per_state_map(&v));
        let depths = graph.bfs_depths();
        let winning = mummymaze::metrics::winning_set(graph);

        // Compute per-state safety: fraction of actions leading to winnable states
        let state_safety: FxHashMap<mummymaze::game::State, f64> = graph
            .transitions
            .iter()
            .map(|(state, transitions)| {
                (*state, mummymaze::metrics::state_safety(transitions, &winning))
            })
            .collect();

        // Find ranges for normalization
        let max_depth = depths.values().copied().max().unwrap_or(1).max(1) as f32;
        let max_expected = expected_steps
            .as_ref()
            .map(|m| {
                m.values()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(1.0f64, f64::max)
            })
            .unwrap_or(1.0) as f32;

        // Build normalized metric vectors (one per ColorMetric, indexed by node)
        // Terminal nodes get sentinel values (handled specially in recolor)
        let mut metric_win_prob = vec![0.0f32; n_nodes];
        let mut metric_expected = vec![0.0f32; n_nodes];
        let mut metric_depth = vec![0.0f32; n_nodes];
        let mut metric_safety = vec![0.0f32; n_nodes];

        for i in 0..n_nodes {
            if node_kinds[i] != NodeKind::Transient {
                continue;
            }
            let state = idx_to_state[i].unwrap();
            metric_win_prob[i] = win_probs
                .as_ref()
                .and_then(|m| m.get(&state))
                .copied()
                .unwrap_or(0.0) as f32;
            // Expected steps: invert so shorter = green, longer = red
            let raw_expected = expected_steps
                .as_ref()
                .and_then(|m| m.get(&state))
                .copied()
                .unwrap_or(f64::INFINITY) as f32;
            metric_expected[i] = if raw_expected.is_finite() {
                1.0 - (raw_expected / max_expected).min(1.0)
            } else {
                0.0 // unreachable/infinite → worst
            };
            metric_depth[i] =
                1.0 - (depths.get(&state).copied().unwrap_or(0) as f32 / max_depth).min(1.0);
            metric_safety[i] = state_safety.get(&state).copied().unwrap_or(0.0) as f32;
        }

        self.node_metrics = vec![metric_win_prob, metric_expected, metric_depth, metric_safety];
        self.start_node_idx = Some(0); // start state is always index 0

        // Compute positions based on layout mode
        let mut positions = match self.layout_mode {
            LayoutMode::ForceDirected => layout::random_positions(n_nodes, 42),
            LayoutMode::BfsLayers => {
                let mut pos = layout::bfs_layer_positions(&state_to_idx, &depths, n_nodes);
                let spacing_y = 3.0;
                for (&parent, &ti) in win_terminal_for.iter().chain(dead_terminal_for.iter()) {
                    let pi = state_to_idx[&parent];
                    let parent_pos = pos[pi];
                    let parent_depth = depths.get(&parent).copied().unwrap_or(0);
                    pos[ti] = [
                        parent_pos[0],
                        -((parent_depth + 1) as f32 * spacing_y),
                        0.0,
                    ];
                }
                pos
            }
            LayoutMode::BfsCylinder => {
                let mut pos = layout::bfs_cylinder_positions(&state_to_idx, &depths, n_nodes);
                let spacing_y = 3.0;
                let min_radius = 2.0;
                for (&parent, &ti) in win_terminal_for.iter().chain(dead_terminal_for.iter()) {
                    let pi = state_to_idx[&parent];
                    let pp = pos[pi];
                    let parent_depth = depths.get(&parent).copied().unwrap_or(0);
                    let child_depth = parent_depth + 1;
                    // Place terminal at same angle as parent but one layer down
                    let r_xz = (pp[0] * pp[0] + pp[2] * pp[2]).sqrt();
                    if r_xz > 0.001 {
                        let angle = pp[2].atan2(pp[0]);
                        pos[ti] = [
                            r_xz * angle.cos(),
                            -(child_depth as f32 * spacing_y),
                            r_xz * angle.sin(),
                        ];
                    } else {
                        // Parent at origin (depth 0), push terminal outward
                        let angle = ti as f32 * 2.4;
                        pos[ti] = [
                            min_radius * angle.cos(),
                            -(child_depth as f32 * spacing_y),
                            min_radius * angle.sin(),
                        ];
                    }
                }
                pos
            }
            LayoutMode::RadialTree => {
                let mut pos = layout::radial_tree_positions(&state_to_idx, &depths, n_nodes);
                let ring_spacing = 3.0;
                for (&parent, &ti) in win_terminal_for.iter().chain(dead_terminal_for.iter()) {
                    let pi = state_to_idx[&parent];
                    let pp = pos[pi];
                    let parent_r = (pp[0] * pp[0] + pp[1] * pp[1]).sqrt();
                    if parent_r > 0.001 {
                        let scale = (parent_r + ring_spacing) / parent_r;
                        pos[ti] = [pp[0] * scale, pp[1] * scale, 0.0];
                    } else {
                        let angle = ti as f32 * 2.4;
                        pos[ti] = [
                            ring_spacing * angle.cos(),
                            ring_spacing * angle.sin(),
                            0.0,
                        ];
                    }
                }
                pos
            }
        };

        // For 2D layouts, jitter terminals with same parent slightly so they don't overlap
        if !self.layout_mode.is_3d() {
            for (&parent, &wi) in &win_terminal_for {
                if let Some(&di) = dead_terminal_for.get(&parent) {
                    positions[wi][0] -= 0.8;
                    positions[di][0] += 0.8;
                }
            }
        }

        // Build NodeGpu (positions + zero velocity)
        let node_data: Vec<NodeGpu> = positions
            .iter()
            .map(|p| NodeGpu {
                pos: [p[0], p[1], p[2], 0.0],
                vel: [0.0, 0.0, 0.0, 0.0],
            })
            .collect();

        // Recreate GPU buffers sized to this graph
        let n_edges_u32 = edges.len() as u32;
        let new_buffers = GraphBuffers::new(
            &self.render_state.device,
            &self.pipelines,
            n_nodes as u32,
            n_edges_u32,
        );
        let queue = &self.render_state.queue;
        queue.write_buffer(&new_buffers.node_buf, 0, bytemuck::cast_slice(&node_data));
        if !edges.is_empty() {
            queue.write_buffer(&new_buffers.edge_buf, 0, bytemuck::cast_slice(&edges));
        }

        // Upload sim params once
        let sim_params = SimParams {
            n_nodes: n_nodes as u32,
            n_edges: n_edges_u32,
            repel: 1.0,
            attract: 1.0,
            decay: 0.92,
            speed_limit: 20.0,
            _pad: [0.0; 2],
        };
        queue.write_buffer(
            &new_buffers.sim_params_buf,
            0,
            bytemuck::bytes_of(&sim_params),
        );

        self.buffers = Some(Arc::new(new_buffers));
        self.initial_positions = positions;
        self.state_to_idx = state_to_idx;
        self.idx_to_state = idx_to_state;
        self.node_kinds = node_kinds;
        self.sim_running = self.layout_mode == LayoutMode::ForceDirected;

        // Upload initial node colors based on active metric
        self.recolor_nodes();

        // Fit the appropriate camera
        if self.layout_mode.is_3d() {
            self.cam_3d.fit(&self.initial_positions);
        } else {
            self.cam_2d.fit(&self.initial_positions);
        }
    }

    /// Recompute node colors from the active metric and upload to GPU.
    pub(super) fn recolor_nodes(&self) {
        let Some(buffers) = &self.buffers else {
            return;
        };
        let n_nodes = buffers.n_nodes as usize;
        let metric_idx = self.color_metric as usize;
        let values = match self.node_metrics.get(metric_idx) {
            Some(v) => v,
            None => return,
        };
        let start_idx = self.start_node_idx.unwrap_or(usize::MAX);

        let node_info: Vec<NodeInfo> = (0..n_nodes)
            .map(|i| {
                match self.node_kinds[i] {
                    NodeKind::Win => {
                        return NodeInfo {
                            color: [0.2, 0.9, 0.2, 1.0],
                            flags: FLAG_WIN,
                            bfs_depth: u32::MAX,
                            radius: 0.6,
                            _pad: 0.0,
                        };
                    }
                    NodeKind::Dead => {
                        return NodeInfo {
                            color: [0.7, 0.1, 0.1, 1.0],
                            flags: FLAG_DEAD,
                            bfs_depth: u32::MAX,
                            radius: 0.6,
                            _pad: 0.0,
                        };
                    }
                    NodeKind::Transient => {}
                }

                let t = values[i];
                let is_start = i == start_idx;
                let is_zero = t <= 0.0;

                let color = if is_start {
                    [0.3, 0.55, 1.0, 1.0]
                } else if is_zero {
                    [0.25, 0.25, 0.25, 0.8]
                } else {
                    metric_color(t)
                };

                let radius = if is_start {
                    0.8
                } else if is_zero {
                    0.35
                } else {
                    0.5
                };

                let mut flags = 0u32;
                if is_start {
                    flags |= FLAG_START;
                }

                NodeInfo {
                    color,
                    flags,
                    bfs_depth: 0,
                    radius,
                    _pad: 0.0,
                }
            })
            .collect();

        self.render_state.queue.write_buffer(
            &buffers.node_info_buf,
            0,
            bytemuck::cast_slice(&node_info),
        );
    }
}
