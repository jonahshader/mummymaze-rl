//! State graph visualization with GPU-accelerated rendering and force-directed layout.

mod build;
mod gpu;
mod interaction;
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
use mummymaze::graph::StateGraph;
use mummymaze::parse::Level;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use types::*;

/// Blue→White→Orange diverging gradient (t in 0..1).
/// Avoids red/green so those are reserved exclusively for WIN/DEAD terminals.
pub(super) fn metric_color(t: f32) -> [f32; 4] {
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
pub(super) fn color32(c: [f32; 4]) -> egui::Color32 {
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
    BfsCylinder,
    RadialTree,
}

impl LayoutMode {
    fn label(self) -> &'static str {
        match self {
            LayoutMode::ForceDirected => "Force-Directed",
            LayoutMode::BfsLayers => "BFS Layers",
            LayoutMode::BfsCylinder => "BFS Cylinder",
            LayoutMode::RadialTree => "Radial Tree",
        }
    }

    fn is_3d(self) -> bool {
        matches!(self, LayoutMode::ForceDirected | LayoutMode::BfsCylinder)
    }
}

/// What kind of node this is in the graph visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NodeKind {
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
    pub(super) render_state: RenderState,
    pub(super) pipelines: Arc<GraphPipelines>,
    pub(super) buffers: Option<Arc<GraphBuffers>>,

    /// Initial node positions from layout (not updated by GPU force simulation).
    /// Accurate for 2D layouts (BFS/Radial); stale for force-directed after first frame.
    /// Used for: 2D hit-testing, initial camera fitting.
    pub(super) initial_positions: Vec<[f32; 3]>,
    /// Map from graph State to node index
    pub(super) state_to_idx: FxHashMap<State, usize>,
    /// Map from node index to graph State (terminal nodes store parent state)
    pub(super) idx_to_state: Vec<Option<State>>,
    /// Per-node classification (transient, win terminal, dead terminal)
    pub(super) node_kinds: Vec<NodeKind>,
    /// Level associated with current graph
    pub(super) level: Option<SharedLevel>,

    // --- Cached topology (invariant across layout modes) ---
    /// BFS depth per state, for position computation.
    pub(super) cached_depths: FxHashMap<State, u32>,
    /// Per-source WIN terminal node indices.
    pub(super) cached_win_terminals: FxHashMap<State, usize>,
    /// Per-source DEAD terminal node indices.
    pub(super) cached_dead_terminals: FxHashMap<State, usize>,
    /// Edge list for GPU re-upload on layout change.
    pub(super) cached_edges: Vec<types::EdgeGpu>,

    // --- Node coloring ---
    /// Per-node normalized metric values [0..1] for each ColorMetric.
    /// Indexed as `node_metrics[metric_index][node_index]`.
    pub(super) node_metrics: Vec<Vec<f32>>,
    /// Currently active coloring metric.
    pub(super) color_metric: ColorMetric,
    /// Start node index (for special coloring).
    pub(super) start_node_idx: Option<usize>,

    // Cameras (each mode uses its native camera)
    pub(super) cam_2d: PanZoomCamera,
    pub(super) cam_3d: OrbitalCamera,

    // Simulation state
    pub(super) layout_mode: LayoutMode,
    pub(super) sim_running: bool,
    pub(super) iterations_per_frame: u32,

    // Hover
    pub(super) hovered_node: Option<usize>,
    /// Hit-test params to send to GPU this frame (3D mode only)
    pub(super) pending_hit_test: Option<HitTestParams>,
    /// Whether a GPU hit-test readback is in flight from the previous frame
    pub(super) hit_test_in_flight: bool,
    /// Which level index we last built the graph for
    pub(super) loaded_level_idx: Option<usize>,

    // --- Tracked node / auto-follow ---
    /// Index of the node being tracked (current game state)
    pub(super) tracked_node_idx: Option<usize>,
    /// Whether a GPU readback for tracked node position is in flight
    pub(super) tracked_node_in_flight: bool,
    /// Last readback result for tracked node position (3D mode)
    pub(super) tracked_node_pos: Option<[f32; 3]>,
    /// Whether camera auto-follows the tracked node
    pub(super) auto_follow: bool,
    /// Smoothed follow target position
    pub(super) follow_target: Option<[f32; 3]>,
    /// Whether the follow lerp still needs frames to converge
    pub(super) follow_animating: bool,

    // --- Current node index (for shader highlight) ---
    pub(super) current_node_idx: Option<u32>,

    // --- Walk edge highlighting ---
    /// Lookup from (src_node_idx, dst_node_idx) to edge indices in the edge buffer
    pub(super) edge_lookup: FxHashMap<(u32, u32), Vec<usize>>,
    /// Number of edges (for sizing highlight buffer)
    pub(super) n_edges: usize,
    /// Whether highlight data has been uploaded and needs display
    pub(super) has_walk_highlight: bool,

    // --- Click handling ---
    /// Node clicked this frame (set by handle_interaction, consumed by main.rs)
    pub(super) clicked_node: Option<usize>,
    /// Whether a click occurred this frame on the graph viewport
    pub(super) click_pending: bool,
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
            cached_depths: FxHashMap::default(),
            cached_win_terminals: FxHashMap::default(),
            cached_dead_terminals: FxHashMap::default(),
            cached_edges: Vec::new(),
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

    /// Rebuild positions and GPU buffers for the current layout mode.
    /// Reuses cached topology/metrics — only recomputes spatial layout.
    fn rebuild_layout(&mut self) {
        if self.state_to_idx.is_empty() {
            return;
        }
        self.rebuild_positions();
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
                            let edge = &self.cached_edges[ei];
                            if key.0 == edge.src && key.1 == edge.dst {
                                highlight_data[ei] |= 1; // forward highlight
                            } else {
                                highlight_data[ei] |= 2; // reverse highlight
                            }
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
                        LayoutMode::BfsCylinder,
                        LayoutMode::BfsCylinder.label(),
                    );
                    ui.selectable_value(
                        &mut self.layout_mode,
                        LayoutMode::RadialTree,
                        LayoutMode::RadialTree.label(),
                    );
                });
            if self.layout_mode != prev_mode {
                self.rebuild_layout();
            }

            ui.separator();

            if self.layout_mode == LayoutMode::ForceDirected {
                let label = if self.sim_running { "Pause" } else { "Resume" };
                if ui.button(label).clicked() {
                    self.sim_running = !self.sim_running;
                }
                if ui.button("Reset").clicked() {
                    self.rebuild_layout();
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
}
