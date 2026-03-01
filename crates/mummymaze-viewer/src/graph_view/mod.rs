//! State graph visualization with GPU-accelerated rendering and force-directed layout.

mod gpu;
mod layout;
mod math;
mod shaders;
pub mod types;

use eframe::egui;
use egui_wgpu::{Callback, RenderState};
use glam::{Mat4, Vec3};
use gpu::{GraphBuffers, GraphPaintCallback, GraphPipelines};
use math::{project_to_screen, OrbitalCamera, PanZoomCamera};
use mummymaze::game::State;
use mummymaze::graph::{StateGraph, StateKey};
use mummymaze::parse::Level;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use types::*;

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

/// CPU-side state for the graph view.
pub struct GraphView {
    render_state: RenderState,
    pipelines: Arc<GraphPipelines>,
    buffers: Option<Arc<GraphBuffers>>,

    // CPU mirror of node positions (for hit testing)
    positions: Vec<[f32; 3]>,
    /// Map from graph State to node index
    state_to_idx: FxHashMap<State, usize>,
    /// Map from node index to graph State (None for WIN/DEAD virtual nodes)
    idx_to_state: Vec<Option<State>>,
    /// Per-state win probabilities (from analyze_full)
    state_win_probs: FxHashMap<State, f64>,
    /// Level associated with current graph
    level: Option<SharedLevel>,
    /// Index of WIN virtual node (if present)
    win_idx: Option<usize>,
    /// Index of DEAD virtual node (if present)
    dead_idx: Option<usize>,

    // Cameras (each mode uses its native camera)
    cam_2d: PanZoomCamera,
    cam_3d: OrbitalCamera,

    // Simulation state
    layout_mode: LayoutMode,
    sim_running: bool,
    iterations_per_frame: u32,

    // Hover
    hovered_node: Option<usize>,
    /// Which level index we last built the graph for
    loaded_level_idx: Option<usize>,
}

impl GraphView {
    pub fn new(render_state: RenderState) -> Self {
        let pipelines = Arc::new(GraphPipelines::new(&render_state));
        GraphView {
            render_state,
            pipelines,
            buffers: None,
            positions: Vec::new(),
            state_to_idx: FxHashMap::default(),
            idx_to_state: Vec::new(),
            state_win_probs: FxHashMap::default(),
            level: None,
            win_idx: None,
            dead_idx: None,
            cam_2d: PanZoomCamera::new(),
            cam_3d: OrbitalCamera::new(),
            layout_mode: LayoutMode::ForceDirected,
            sim_running: false,
            iterations_per_frame: 5,
            hovered_node: None,
            loaded_level_idx: None,
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
        state_win_probs: FxHashMap<State, f64>,
    ) {
        self.loaded_level_idx = Some(level_idx);
        self.level = Some(Arc::new(level.clone()));
        self.state_win_probs = state_win_probs;
        self.build_from_graph(graph);
    }

    fn build_from_graph(&mut self, graph: &StateGraph) {
        // Assign indices: transient states first, then WIN and DEAD virtual nodes
        let mut state_to_idx = FxHashMap::default();
        let mut idx_to_state: Vec<Option<State>> = Vec::new();
        let mut idx = 0usize;

        // Start state first
        state_to_idx.insert(graph.start, idx);
        idx_to_state.push(Some(graph.start));
        idx += 1;

        for state in graph.transitions.keys() {
            if *state != graph.start {
                state_to_idx.insert(*state, idx);
                idx_to_state.push(Some(*state));
                idx += 1;
            }
        }

        // Check if we need WIN and DEAD virtual nodes
        let mut has_win = false;
        let mut has_dead = false;
        for transitions in graph.transitions.values() {
            for &(_, dest) in transitions {
                match dest {
                    StateKey::Win => has_win = true,
                    StateKey::Dead => has_dead = true,
                    _ => {}
                }
            }
        }

        self.win_idx = if has_win {
            let i = idx;
            idx_to_state.push(None);
            idx += 1;
            Some(i)
        } else {
            None
        };

        self.dead_idx = if has_dead {
            let i = idx;
            idx_to_state.push(None);
            idx += 1;
            Some(i)
        } else {
            None
        };

        let win_idx = self.win_idx;
        let dead_idx = self.dead_idx;

        let n_nodes = idx;

        // Build edges
        let mut edges: Vec<EdgeGpu> = Vec::new();
        for (state, transitions) in &graph.transitions {
            let src = state_to_idx[state];
            for &(_, dest) in transitions {
                let dst = match dest {
                    StateKey::Transient(ns) => state_to_idx[&ns],
                    StateKey::Win => {
                        if let Some(wi) = win_idx {
                            wi
                        } else {
                            continue;
                        }
                    }
                    StateKey::Dead => {
                        if let Some(di) = dead_idx {
                            di
                        } else {
                            continue;
                        }
                    }
                };
                edges.push(EdgeGpu {
                    src: src as u32,
                    dst: dst as u32,
                });
            }
        }

        // Compute BFS depths (only needed for BFS/radial layouts and node info)
        let depths = if self.layout_mode != LayoutMode::ForceDirected {
            graph.bfs_depths()
        } else {
            FxHashMap::default()
        };

        // Compute positions based on layout mode
        let mut positions = match self.layout_mode {
            LayoutMode::ForceDirected => layout::random_positions(n_nodes, 42),
            LayoutMode::BfsLayers => {
                layout::bfs_layer_positions(&state_to_idx, &depths, n_nodes)
            }
            LayoutMode::RadialTree => {
                layout::radial_tree_positions(&state_to_idx, &depths, n_nodes)
            }
        };

        // Place virtual nodes outside the layout
        if !self.layout_mode.is_3d() {
            let max_depth = depths.values().copied().max().unwrap_or(0);
            let outer = (max_depth + 2) as f32 * 3.0;
            if let Some(wi) = win_idx {
                match self.layout_mode {
                    LayoutMode::BfsLayers => positions[wi] = [-3.0, -outer, 0.0],
                    _ => positions[wi] = [outer, 0.0, 0.0],
                }
            }
            if let Some(di) = dead_idx {
                match self.layout_mode {
                    LayoutMode::BfsLayers => positions[di] = [3.0, -outer, 0.0],
                    _ => positions[di] = [-outer, 0.0, 0.0],
                }
            }
        }

        // Build NodeGpu (positions + zero velocity, padded to vec4)
        let node_data: Vec<NodeGpu> = positions
            .iter()
            .map(|p| NodeGpu {
                pos: [p[0], p[1], p[2], 0.0],
                vel: [0.0, 0.0, 0.0, 0.0],
            })
            .collect();

        // Build NodeInfo (colors, flags, radius)
        let node_info: Vec<NodeInfo> = (0..n_nodes)
            .map(|i| {
                if Some(i) == win_idx {
                    return NodeInfo {
                        color: [0.2, 0.9, 0.2, 1.0],
                        flags: FLAG_WIN,
                        bfs_depth: u32::MAX,
                        radius: 1.0,
                        _pad: 0.0,
                    };
                }
                if Some(i) == dead_idx {
                    return NodeInfo {
                        color: [0.7, 0.1, 0.1, 1.0],
                        flags: FLAG_DEAD,
                        bfs_depth: u32::MAX,
                        radius: 1.0,
                        _pad: 0.0,
                    };
                }

                let state = idx_to_state[i].unwrap();
                let is_start = state == graph.start;
                let win_prob = self
                    .state_win_probs
                    .get(&state)
                    .copied()
                    .unwrap_or(0.0);
                let is_winning = win_prob > 0.0;

                let color = if is_start {
                    [0.3, 0.55, 1.0, 1.0]
                } else if !is_winning {
                    [0.25, 0.25, 0.25, 0.8]
                } else if win_prob > 0.8 {
                    [0.2, 0.8, 0.2, 1.0]
                } else if win_prob > 0.3 {
                    [0.9, 0.8, 0.2, 1.0]
                } else {
                    [0.8, 0.2, 0.2, 1.0]
                };

                let radius = if is_start {
                    0.8
                } else if !is_winning {
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
                    bfs_depth: depths.get(&state).copied().unwrap_or(u32::MAX),
                    radius,
                    _pad: 0.0,
                }
            })
            .collect();

        // Recreate GPU buffers sized to this graph (pipelines are reused)
        let n_edges_u32 = edges.len() as u32;
        let new_buffers = GraphBuffers::new(
            &self.render_state.device,
            &self.pipelines,
            n_nodes as u32,
            n_edges_u32,
        );
        let queue = &self.render_state.queue;
        queue.write_buffer(&new_buffers.node_buf, 0, bytemuck::cast_slice(&node_data));
        queue.write_buffer(
            &new_buffers.node_info_buf,
            0,
            bytemuck::cast_slice(&node_info),
        );
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
        self.positions = positions;
        self.state_to_idx = state_to_idx;
        self.idx_to_state = idx_to_state;
        self.sim_running = self.layout_mode == LayoutMode::ForceDirected;

        // Fit the appropriate camera
        if self.layout_mode.is_3d() {
            self.cam_3d.fit(&self.positions);
        } else {
            self.cam_2d.fit(&self.positions);
        }
    }

    fn camera_uniform(&self) -> CameraUniform {
        if self.layout_mode.is_3d() {
            self.cam_3d.to_uniform()
        } else {
            self.cam_2d.to_uniform()
        }
    }

    /// Main draw function, called from the Graph tab.
    pub fn draw(&mut self, ui: &mut egui::Ui, selected_level_idx: Option<usize>) {
        self.draw_toolbar(ui, selected_level_idx);
        ui.separator();

        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, egui::Sense::click_and_drag());
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
                let callback = GraphPaintCallback {
                    pipelines: Arc::clone(&self.pipelines),
                    buffers: Arc::clone(buffers),
                    camera: self.camera_uniform(),
                    run_compute: self.sim_running,
                    iterations_per_frame: self.iterations_per_frame,
                };

                painter.add(Callback::new_paint_callback(rect, callback));

                if self.sim_running {
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
        });
    }

    fn handle_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let rect = response.rect;

        if self.layout_mode.is_3d() {
            self.handle_3d_interaction(ui, response, rect);
        } else {
            self.handle_2d_interaction(ui, response, rect);
        }
    }

    fn handle_3d_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
    ) {
        // Orbit: left-drag
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.cam_3d.yaw -= delta.x * 0.005;
            self.cam_3d.pitch += delta.y * 0.005;
            let limit = std::f32::consts::FRAC_PI_2 - 0.01;
            self.cam_3d.pitch = self.cam_3d.pitch.clamp(-limit, limit);
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
        }

        // Dolly: scroll wheel
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            self.cam_3d.distance *= (-scroll * 0.003).exp();
            self.cam_3d.distance = self.cam_3d.distance.clamp(0.1, 10000.0);
        }

        // Hover hit test
        self.hovered_node = None;
        if let Some(cursor) = response.hover_pos() {
            let uniform = self.cam_3d.to_uniform();
            let vp = Mat4::from_cols_array_2d(&uniform.view_proj);
            let cx = rect.center().x;
            let cy = rect.center().y;
            let half_w = rect.width() / 2.0;
            let half_h = rect.height() / 2.0;

            let mut best_dist_sq = 100.0f32;
            for (i, pos) in self.positions.iter().enumerate() {
                if let Some(screen) =
                    project_to_screen(Vec3::from_array(*pos), &vp, [cx, cy], [half_w, half_h])
                {
                    let dx = screen[0] - cursor.x;
                    let dy = screen[1] - cursor.y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        self.hovered_node = Some(i);
                    }
                }
            }
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
            for (i, pos) in self.positions.iter().enumerate() {
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

        let state = match self.idx_to_state.get(node_idx) {
            Some(Some(s)) => *s,
            Some(None) => {
                let label = if Some(node_idx) == self.win_idx {
                    "WIN terminal"
                } else {
                    "DEAD terminal"
                };
                response.clone().on_hover_ui(|ui: &mut egui::Ui| {
                    ui.label(label);
                });
                return;
            }
            None => return,
        };

        let win_prob = self.state_win_probs.get(&state).copied().unwrap_or(0.0);
        let level = Arc::clone(level);

        response.clone().on_hover_ui(move |ui: &mut egui::Ui| {
            ui.label(format!("Win prob: {:.1}%", win_prob * 100.0));

            let size = egui::Vec2::new(200.0, 200.0);
            let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
            crate::render::draw_maze_state(&painter, resp.rect, &level, &state);
        });
    }
}
