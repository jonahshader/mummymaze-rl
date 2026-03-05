//! State graph visualization with GPU-accelerated rendering and force-directed layout.

mod gpu;
mod layout;
mod math;
mod shaders;
pub mod types;

use eframe::egui;
use eframe::wgpu;
use egui_wgpu::{Callback, RenderState};
use gpu::{GraphBuffers, GraphPaintCallback, GraphPipelines};
use math::{OrbitalCamera, PanZoomCamera};
use mummymaze::game::{Action, State};
use mummymaze::graph::{StateGraph, StateKey};
use mummymaze::parse::Level;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use types::*;

/// Blue→White→Orange diverging gradient (t in 0..1).
/// Avoids red/green so those are reserved exclusively for WIN/DEAD terminals.
fn metric_color(t: f32) -> [f32; 4] {
    // 0.0 = dark orange [0.8, 0.4, 0.15]
    // 0.5 = near-white  [0.9, 0.9, 0.9]
    // 1.0 = blue         [0.2, 0.5, 0.9]
    let (r, g, b) = if t < 0.5 {
        let s = t * 2.0; // 0..1 within orange→white
        (
            0.8 + 0.1 * s,
            0.4 + 0.5 * s,
            0.15 + 0.75 * s,
        )
    } else {
        let s = (t - 0.5) * 2.0; // 0..1 within white→blue
        (
            0.9 - 0.7 * s,
            0.9 - 0.4 * s,
            0.9,
        )
    };
    [r, g, b, 1.0]
}

/// Convert [f32;4] color to egui Color32.
fn color32(c: [f32; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
        (c[3] * 255.0) as u8,
    )
}

/// Wraps a `Level` in an `Arc` to avoid cloning on every hover frame.
type SharedLevel = Arc<Level>;

/// Layout algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    ForceDirected,
    BfsLayers,
    RadialTree,
}

impl LayoutMode {
    fn label(self) -> &'static str {
        match self {
            LayoutMode::ForceDirected => "Force-Directed",
            LayoutMode::BfsLayers => "BFS Layers",
            LayoutMode::RadialTree => "Radial Tree",
        }
    }

    fn is_3d(self) -> bool {
        self == LayoutMode::ForceDirected
    }
}

/// What kind of node this is in the graph visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    Transient,
    Win,
    Dead,
}

/// Which metric to use for node coloring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMetric {
    WinProb,
    ExpectedSteps,
    BfsDepth,
    Safety,
}

impl ColorMetric {
    fn label(self) -> &'static str {
        match self {
            ColorMetric::WinProb => "Win %",
            ColorMetric::ExpectedSteps => "E[steps]",
            ColorMetric::BfsDepth => "BFS Depth",
            ColorMetric::Safety => "Safety",
        }
    }

    const ALL: [ColorMetric; 4] = [
        ColorMetric::WinProb,
        ColorMetric::ExpectedSteps,
        ColorMetric::BfsDepth,
        ColorMetric::Safety,
    ];
}

/// CPU-side state for the graph view.
pub struct GraphView {
    render_state: RenderState,
    pipelines: Arc<GraphPipelines>,
    buffers: Option<Arc<GraphBuffers>>,

    /// Initial node positions from layout (not updated by GPU force simulation).
    /// Accurate for 2D layouts (BFS/Radial); stale for force-directed after first frame.
    /// Used for: 2D hit-testing, initial camera fitting.
    initial_positions: Vec<[f32; 3]>,
    /// Map from graph State to node index
    state_to_idx: FxHashMap<State, usize>,
    /// Map from node index to graph State (terminal nodes store parent state)
    idx_to_state: Vec<Option<State>>,
    /// Per-node classification (transient, win terminal, dead terminal)
    node_kinds: Vec<NodeKind>,
    /// Level associated with current graph
    level: Option<SharedLevel>,

    // --- Node coloring ---
    /// Per-node normalized metric values [0..1] for each ColorMetric.
    /// Indexed as `node_metrics[metric_index][node_index]`.
    node_metrics: Vec<Vec<f32>>,
    /// Currently active coloring metric.
    color_metric: ColorMetric,
    /// Start node index (for special coloring).
    start_node_idx: Option<usize>,

    // Cameras (each mode uses its native camera)
    cam_2d: PanZoomCamera,
    cam_3d: OrbitalCamera,

    // Simulation state
    layout_mode: LayoutMode,
    sim_running: bool,
    iterations_per_frame: u32,

    // Hover
    hovered_node: Option<usize>,
    /// Hit-test params to send to GPU this frame (3D mode only)
    pending_hit_test: Option<HitTestParams>,
    /// Whether a GPU hit-test readback is in flight from the previous frame
    hit_test_in_flight: bool,
    /// Which level index we last built the graph for
    loaded_level_idx: Option<usize>,

    // --- Tracked node / auto-follow ---
    /// Index of the node being tracked (current game state)
    tracked_node_idx: Option<usize>,
    /// Whether a GPU readback for tracked node position is in flight
    tracked_node_in_flight: bool,
    /// Last readback result for tracked node position (3D mode)
    tracked_node_pos: Option<[f32; 3]>,
    /// Whether camera auto-follows the tracked node
    auto_follow: bool,
    /// Smoothed follow target position
    follow_target: Option<[f32; 3]>,
    /// Whether the follow lerp still needs frames to converge
    follow_animating: bool,

    // --- Current node index (for shader highlight) ---
    current_node_idx: Option<u32>,

    // --- Walk edge highlighting ---
    /// Lookup from (src_node_idx, dst_node_idx) to edge indices in the edge buffer
    edge_lookup: FxHashMap<(u32, u32), Vec<usize>>,
    /// Number of edges (for sizing highlight buffer)
    n_edges: usize,
    /// Whether highlight data has been uploaded and needs display
    has_walk_highlight: bool,

    // --- Click handling ---
    /// Node clicked this frame (set by handle_interaction, consumed by main.rs)
    clicked_node: Option<usize>,
    /// Whether a click occurred this frame on the graph viewport
    click_pending: bool,
}

impl GraphView {
    pub fn new(render_state: RenderState) -> Self {
        let pipelines = Arc::new(GraphPipelines::new(&render_state));
        GraphView {
            render_state,
            pipelines,
            buffers: None,
            initial_positions: Vec::new(),
            state_to_idx: FxHashMap::default(),
            idx_to_state: Vec::new(),
            node_kinds: Vec::new(),
            level: None,
            node_metrics: Vec::new(),
            color_metric: ColorMetric::WinProb,
            start_node_idx: None,
            cam_2d: PanZoomCamera::new(),
            cam_3d: OrbitalCamera::new(),
            layout_mode: LayoutMode::ForceDirected,
            sim_running: false,
            iterations_per_frame: 5,
            hovered_node: None,
            pending_hit_test: None,
            hit_test_in_flight: false,
            loaded_level_idx: None,
            tracked_node_idx: None,
            tracked_node_in_flight: false,
            tracked_node_pos: None,
            auto_follow: true,
            follow_target: None,
            follow_animating: false,
            current_node_idx: None,
            edge_lookup: FxHashMap::default(),
            n_edges: 0,
            has_walk_highlight: false,
            clicked_node: None,
            click_pending: false,
        }
    }

    pub fn is_loaded(&self, level_idx: usize) -> bool {
        self.loaded_level_idx == Some(level_idx)
    }

    /// Build the graph for a given level and upload to GPU.
    pub fn load_level(
        &mut self,
        level: &Level,
        level_idx: usize,
        graph: &StateGraph,
        chain: &mummymaze::markov::MarkovChain,
    ) {
        self.loaded_level_idx = Some(level_idx);
        self.level = Some(Arc::new(level.clone()));
        self.has_walk_highlight = false;
        self.current_node_idx = None;
        self.tracked_node_idx = None;
        self.tracked_node_pos = None;
        self.follow_target = None;
        self.auto_follow = true;
        self.build_from_graph(graph, chain);
    }

    /// Update the tracked/current node on the graph. Does NOT change auto-follow state.
    pub fn set_current_state(&mut self, state: State) {
        if let Some(&idx) = self.state_to_idx.get(&state) {
            self.tracked_node_idx = Some(idx);
            self.current_node_idx = Some(idx as u32);
            if self.auto_follow {
                self.follow_animating = true;
            }
        } else {
            self.tracked_node_idx = None;
            self.current_node_idx = None;
        }
    }

    /// Re-engage auto-follow (called only on player moves).
    pub fn reengage_auto_follow(&mut self) {
        self.auto_follow = true;
        self.follow_animating = true;
    }

    /// Update walk highlight from gameplay history.
    pub fn update_walk_highlight(&mut self, history: &[(Action, State)], current_state: State) {
        if self.n_edges == 0 {
            return;
        }

        let mut highlight_data = vec![0u32; self.n_edges];

        // Build pairs of consecutive states from the walk
        // history stores (action, state_before_action)
        // So the walk is: history[0].1 -> history[1].1 -> ... -> current_state
        for i in 0..history.len() {
            let src_state = history[i].1;
            let dst_state = if i + 1 < history.len() {
                history[i + 1].1
            } else {
                current_state
            };

            let src_idx = self.state_to_idx.get(&src_state).copied();
            let dst_idx = self.state_to_idx.get(&dst_state).copied();

            if let (Some(si), Some(di)) = (src_idx, dst_idx) {
                let key = (si as u32, di as u32);
                if let Some(edge_indices) = self.edge_lookup.get(&key) {
                    for &ei in edge_indices {
                        if ei < highlight_data.len() {
                            highlight_data[ei] = 1;
                        }
                    }
                }
            }
        }

        // Upload to GPU
        if let Some(ref buffers) = self.buffers {
            self.render_state.queue.write_buffer(
                &buffers.edge_highlight_buf,
                0,
                bytemuck::cast_slice(&highlight_data),
            );
            self.has_walk_highlight = true;
        }
    }

    /// Take the clicked node index (consumed once).
    pub fn take_clicked_node(&mut self) -> Option<usize> {
        self.clicked_node.take()
    }

    /// Get the state for a node index (parent state for terminals).
    pub fn node_state(&self, idx: usize) -> Option<State> {
        self.idx_to_state.get(idx).copied().flatten()
    }

    /// Check if a node is a terminal (Win/Dead).
    pub fn is_terminal(&self, idx: usize) -> bool {
        matches!(
            self.node_kinds.get(idx),
            Some(NodeKind::Win) | Some(NodeKind::Dead)
        )
    }

    fn build_from_graph(&mut self, graph: &StateGraph, chain: &mummymaze::markov::MarkovChain) {
        let mut state_to_idx = FxHashMap::default();
        let mut idx_to_state: Vec<Option<State>> = Vec::new();
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
        let mut win_terminal_for: FxHashMap<State, usize> = FxHashMap::default();
        let mut dead_terminal_for: FxHashMap<State, usize> = FxHashMap::default();

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
        let state_safety: FxHashMap<State, f64> = graph
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
            metric_depth[i] = 1.0 - (depths.get(&state).copied().unwrap_or(0) as f32 / max_depth).min(1.0);
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
    fn recolor_nodes(&self) {
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

    /// Synchronously read back last frame's GPU hit-test result.
    fn poll_hit_test_result(&mut self) {
        if !self.hit_test_in_flight {
            return;
        }
        self.hit_test_in_flight = false;

        let Some(buffers) = &self.buffers else {
            return;
        };

        let staging = &buffers.hit_test_staging_buf;
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.render_state.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let packed = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging.unmap();

        if packed == 0xFFFF_FFFF {
            self.hovered_node = None;
        } else {
            let idx = (packed & 0xFFFF) as usize;
            if idx < buffers.n_nodes as usize {
                self.hovered_node = Some(idx);
            } else {
                self.hovered_node = None;
            }
        }
    }

    /// Read back tracked node position from GPU (3D force-directed mode).
    fn poll_tracked_node_pos(&mut self) {
        if !self.tracked_node_in_flight {
            return;
        }
        self.tracked_node_in_flight = false;

        let Some(buffers) = &self.buffers else {
            return;
        };

        let staging = &buffers.tracked_node_staging_buf;
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.render_state.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let bytes: [u8; 16] = data[0..16].try_into().unwrap();
        drop(data);
        staging.unmap();

        let floats: [f32; 4] = bytemuck::cast(bytes);
        self.tracked_node_pos = Some([floats[0], floats[1], floats[2]]);
    }

    /// Update auto-follow camera smoothly toward the tracked node.
    /// Uses frame-rate-independent exponential decay: `alpha = 1 - e^(-speed * dt)`.
    fn update_auto_follow(&mut self, dt: f32) {
        if !self.auto_follow {
            self.follow_animating = false;
            return;
        }
        let Some(tracked_idx) = self.tracked_node_idx else {
            self.follow_animating = false;
            return;
        };

        // Get the node's current position
        let node_pos = if self.layout_mode.is_3d() {
            // Use GPU readback position for force-directed mode
            self.tracked_node_pos
        } else {
            // Use CPU-side initial positions for 2D modes
            self.initial_positions.get(tracked_idx).copied()
        };

        let Some(pos) = node_pos else {
            self.follow_animating = false;
            return;
        };

        // Frame-rate-independent exponential decay
        // speed=8.0 gives ~63% closure per 1/8s, visually similar to old alpha=0.12 at 60fps
        let alpha = 1.0 - (-8.0_f32 * dt).exp();
        let target = self.follow_target.get_or_insert(pos);
        let dx = pos[0] - target[0];
        let dy = pos[1] - target[1];
        let dz = pos[2] - target[2];
        target[0] += dx * alpha;
        target[1] += dy * alpha;
        target[2] += dz * alpha;

        // Check if we've converged (distance < threshold)
        let dist_sq = dx * dx + dy * dy + dz * dz;
        self.follow_animating = dist_sq > 0.0001;

        if self.layout_mode.is_3d() {
            self.cam_3d.target = glam::Vec3::from_array(*target);
        } else {
            self.cam_2d.pan = [target[0], target[1]];
        }
    }

    /// Main draw function, called from the combined view.
    pub fn draw(&mut self, ui: &mut egui::Ui, selected_level_idx: Option<usize>) {
        // Read back GPU results from previous frame
        self.poll_hit_test_result();
        self.poll_tracked_node_pos();

        // Update camera auto-follow (dt-scaled)
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        self.update_auto_follow(dt);

        self.draw_toolbar(ui, selected_level_idx);
        ui.separator();

        // Clamp to clip rect so toolbar overflow doesn't inflate the painter width
        let available = ui.available_rect_before_wrap().intersect(ui.clip_rect());
        let (response, painter) =
            ui.allocate_painter(available.size(), egui::Sense::click_and_drag());
        let rect = response.rect;

        if rect.width() > 0.0 && rect.height() > 0.0 {
            if self.layout_mode.is_3d() {
                self.cam_3d.aspect = rect.width() / rect.height();
            } else {
                self.cam_2d.aspect = rect.width() / rect.height();
            }
        }

        self.handle_interaction(ui, &response);

        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 35));

        if let Some(buffers) = &self.buffers {
            if buffers.n_nodes > 0 {
                let mut camera = if self.layout_mode.is_3d() {
                    self.cam_3d.to_uniform()
                } else {
                    self.cam_2d.to_uniform()
                };

                // Pack current_node_idx into camera_right.w (bitcast u32 -> f32)
                let current_idx = self.current_node_idx.unwrap_or(u32::MAX);
                camera.camera_right[3] = f32::from_bits(current_idx);

                // Fill in view_proj for the hit-test params
                let mut hit_test_params = self.pending_hit_test.take();
                if let Some(ref mut params) = hit_test_params {
                    params.view_proj = camera.view_proj;
                    self.hit_test_in_flight = true;
                }

                // Set up tracked node readback for 3D mode
                let tracked_node_idx = if self.layout_mode.is_3d() {
                    if let Some(idx) = self.tracked_node_idx {
                        self.tracked_node_in_flight = true;
                        Some(idx as u32)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let callback = GraphPaintCallback {
                    pipelines: Arc::clone(&self.pipelines),
                    buffers: Arc::clone(buffers),
                    camera,
                    run_compute: self.sim_running,
                    iterations_per_frame: self.iterations_per_frame,
                    hit_test_params,
                    tracked_node_idx,
                };

                painter.add(Callback::new_paint_callback(rect, callback));

                if self.sim_running || self.follow_animating {
                    ui.ctx().request_repaint();
                }
            }
        }

        self.draw_hover_tooltip(ui, &response);
    }

    fn draw_toolbar(&mut self, ui: &mut egui::Ui, _selected_level_idx: Option<usize>) {
        ui.horizontal(|ui: &mut egui::Ui| {
            ui.label("Layout:");
            let prev_mode = self.layout_mode;
            egui::ComboBox::from_id_salt("layout_mode")
                .selected_text(self.layout_mode.label())
                .show_ui(ui, |ui: &mut egui::Ui| {
                    ui.selectable_value(
                        &mut self.layout_mode,
                        LayoutMode::ForceDirected,
                        LayoutMode::ForceDirected.label(),
                    );
                    ui.selectable_value(
                        &mut self.layout_mode,
                        LayoutMode::BfsLayers,
                        LayoutMode::BfsLayers.label(),
                    );
                    ui.selectable_value(
                        &mut self.layout_mode,
                        LayoutMode::RadialTree,
                        LayoutMode::RadialTree.label(),
                    );
                });
            if self.layout_mode != prev_mode {
                self.loaded_level_idx = None;
            }

            ui.separator();

            if self.layout_mode == LayoutMode::ForceDirected {
                let label = if self.sim_running { "Pause" } else { "Resume" };
                if ui.button(label).clicked() {
                    self.sim_running = !self.sim_running;
                }
                if ui.button("Reset").clicked() {
                    self.loaded_level_idx = None;
                }
            }

            ui.separator();

            let n = self.buffers.as_ref().map(|b| b.n_nodes).unwrap_or(0);
            let e = self.buffers.as_ref().map(|b| b.n_edges).unwrap_or(0);
            ui.label(format!("{n} nodes, {e} edges"));

            if self.sim_running {
                ui.label("(simulating)");
            }

            ui.separator();

            // Color metric dropdown + gradient bar
            ui.label("Color:");
            let prev_metric = self.color_metric;
            egui::ComboBox::from_id_salt("color_metric")
                .selected_text(self.color_metric.label())
                .show_ui(ui, |ui: &mut egui::Ui| {
                    for m in ColorMetric::ALL {
                        ui.selectable_value(&mut self.color_metric, m, m.label());
                    }
                });
            if self.color_metric != prev_metric {
                self.recolor_nodes();
            }

            // Gradient bar
            let bar_width = 60.0;
            let bar_height = 8.0;
            let (bar_rect, _) = ui.allocate_exact_size(
                egui::Vec2::new(bar_width, bar_height),
                egui::Sense::hover(),
            );
            let steps = 16;
            let step_w = bar_width / steps as f32;
            for s in 0..steps {
                let t = s as f32 / (steps - 1) as f32;
                let c = color32(metric_color(t));
                let x0 = bar_rect.min.x + s as f32 * step_w;
                let seg = egui::Rect::from_min_size(
                    egui::Pos2::new(x0, bar_rect.min.y),
                    egui::Vec2::new(step_w, bar_height),
                );
                ui.painter().rect_filled(seg, 0.0, c);
            }
        });
    }

    fn handle_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let rect = response.rect;

        // Detect clicks for node navigation
        if response.clicked() {
            self.click_pending = true;
        }

        if self.layout_mode.is_3d() {
            self.handle_3d_interaction(ui, response, rect);
        } else {
            self.handle_2d_interaction(ui, response, rect);
        }

        // Process click
        if self.click_pending {
            self.click_pending = false;
            if let Some(hovered) = self.hovered_node {
                self.clicked_node = Some(hovered);
            }
        }
    }

    fn handle_3d_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
    ) {
        // Orbit: left-drag (trackball — rotate around camera-local axes)
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            let right = self.cam_3d.right();
            let up = self.cam_3d.up();
            let rot_x = glam::Quat::from_axis_angle(up, -delta.x * 0.005);
            let rot_y = glam::Quat::from_axis_angle(right, -delta.y * 0.005);
            self.cam_3d.orientation = (rot_x * rot_y * self.cam_3d.orientation).normalize();
            self.auto_follow = false;
        }

        // Pan: right-drag or middle-drag
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = response.drag_delta();
            let speed = self.cam_3d.distance * 0.002;
            let r = self.cam_3d.right();
            let u = self.cam_3d.up();
            self.cam_3d.target -= r * delta.x * speed - u * delta.y * speed;
            self.auto_follow = false;
        }

        // Dolly: scroll wheel
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            self.cam_3d.distance *= (-scroll * 0.003).exp();
            self.cam_3d.distance = self.cam_3d.distance.clamp(0.1, 10000.0);
        }

        // Hover hit test — store cursor/rect info; view_proj is filled in draw()
        if let Some(cursor) = response.hover_pos() {
            if let Some(buffers) = &self.buffers {
                self.pending_hit_test = Some(HitTestParams {
                    view_proj: [[0.0; 4]; 4], // filled by draw() before dispatch
                    cursor: [cursor.x, cursor.y],
                    half_size: [rect.width() / 2.0, rect.height() / 2.0],
                    rect_center: [rect.center().x, rect.center().y],
                    threshold_sq: 100.0,
                    n_nodes: buffers.n_nodes,
                });
            }
        } else {
            self.pending_hit_test = None;
            self.hovered_node = None;
        }
    }

    fn handle_2d_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
    ) {
        // Pan: any drag
        if response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = response.drag_delta();
            self.cam_2d.pan[0] -=
                delta.x * self.cam_2d.aspect / (self.cam_2d.zoom * rect.width() / 2.0);
            self.cam_2d.pan[1] += delta.y / (self.cam_2d.zoom * rect.height() / 2.0);
            self.auto_follow = false;
        }

        // Zoom toward cursor
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let factor = (scroll * 0.005).exp();
            if let Some(cursor) = response.hover_pos() {
                self.cam_2d.zoom_at(
                    factor,
                    [cursor.x, cursor.y],
                    [rect.center().x, rect.center().y],
                    [rect.width() / 2.0, rect.height() / 2.0],
                );
            } else {
                self.cam_2d.zoom *= factor;
                self.cam_2d.zoom = self.cam_2d.zoom.clamp(0.001, 100.0);
            }
        }

        // Hover hit test (world-space, same as old 2D approach)
        self.hovered_node = None;
        if let Some(cursor) = response.hover_pos() {
            let clip = PanZoomCamera::screen_to_clip(
                [cursor.x, cursor.y],
                [rect.center().x, rect.center().y],
                [rect.width() / 2.0, rect.height() / 2.0],
            );
            let world = self.cam_2d.clip_to_world(clip, self.cam_2d.zoom);

            let hit_r_sq = 1.0f32;
            let mut best_dist_sq = f32::MAX;
            for (i, pos) in self.initial_positions.iter().enumerate() {
                let dx = pos[0] - world[0];
                let dy = pos[1] - world[1];
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < hit_r_sq && dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    self.hovered_node = Some(i);
                }
            }
        }
    }

    fn draw_hover_tooltip(&self, _ui: &egui::Ui, response: &egui::Response) {
        let Some(node_idx) = self.hovered_node else {
            return;
        };
        let Some(ref level) = self.level else {
            return;
        };
        let Some(&kind) = self.node_kinds.get(node_idx) else {
            return;
        };

        // Get the state for this node (parent state for terminals)
        let state = match self.idx_to_state.get(node_idx) {
            Some(Some(s)) => *s,
            _ => return,
        };

        let label = match kind {
            NodeKind::Win => Some("WIN"),
            NodeKind::Dead => Some("DEAD"),
            NodeKind::Transient => None,
        };

        // Win% is metric index 0 (WinProb), get raw value for tooltip
        let win_prob = self
            .node_metrics
            .first()
            .and_then(|v| v.get(node_idx))
            .copied()
            .unwrap_or(0.0);
        let level = Arc::clone(level);

        response.clone().on_hover_ui(move |ui: &mut egui::Ui| {
            if let Some(l) = label {
                ui.label(l);
            }
            ui.label(format!("Win prob: {:.1}%", win_prob * 100.0));

            let size = egui::Vec2::new(200.0, 200.0);
            let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
            crate::render::draw_maze_state(&painter, resp.rect, &level, &state);
        });
    }
}
