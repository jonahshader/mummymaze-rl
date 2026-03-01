//! State graph visualization with GPU-accelerated rendering and force-directed layout.

mod gpu;
mod layout;
mod shaders;
pub mod types;

use eframe::egui;
use egui_wgpu::{Callback, RenderState};
use gpu::{GraphBuffers, GraphPaintCallback, GraphPipelines};
use mummymaze::game::State;
use mummymaze::graph::{StateGraph, StateKey};
use mummymaze::parse::Level;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use types::*;

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
}

/// CPU-side state for the graph view.
pub struct GraphView {
    render_state: RenderState,
    pipelines: Arc<GraphPipelines>,
    buffers: Option<Arc<GraphBuffers>>,

    // CPU mirror of node positions (for hit testing)
    positions: Vec<[f32; 2]>,
    /// Map from graph State to node index
    state_to_idx: FxHashMap<State, usize>,
    /// Map from node index to graph State (None for WIN/DEAD virtual nodes)
    idx_to_state: Vec<Option<State>>,
    /// Per-state win probabilities (from analyze_full)
    state_win_probs: FxHashMap<State, f64>,
    /// Level associated with current graph
    level: Option<Level>,
    /// Index of WIN virtual node (if present)
    win_idx: Option<usize>,
    /// Index of DEAD virtual node (if present)
    dead_idx: Option<usize>,

    // Camera
    camera: CameraUniform,
    // Simulation state
    layout_mode: LayoutMode,
    sim_running: bool,
    max_velocity: f32,
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
            camera: CameraUniform {
                pan: [0.0, 0.0],
                zoom: 0.1,
                aspect: 1.0,
            },
            layout_mode: LayoutMode::ForceDirected,
            sim_running: false,
            max_velocity: 0.0,
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
        self.level = Some(level.clone());
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

        // Compute BFS depths
        let depths = layout::bfs_depths(graph);

        // Compute positions based on layout mode
        let positions = match self.layout_mode {
            LayoutMode::ForceDirected => layout::random_positions(n_nodes, 42),
            LayoutMode::BfsLayers => {
                let mut pos =
                    layout::bfs_layer_positions(&state_to_idx, &depths, n_nodes);
                // Place virtual nodes below
                let max_depth = depths.values().copied().max().unwrap_or(0);
                if let Some(wi) = win_idx {
                    pos[wi] = [-3.0, -((max_depth + 2) as f32 * 3.0)];
                }
                if let Some(di) = dead_idx {
                    pos[di] = [3.0, -((max_depth + 2) as f32 * 3.0)];
                }
                pos
            }
            LayoutMode::RadialTree => {
                let mut pos =
                    layout::radial_tree_positions(&state_to_idx, &depths, n_nodes);
                let max_depth = depths.values().copied().max().unwrap_or(0);
                let outer = (max_depth + 2) as f32 * 3.0;
                if let Some(wi) = win_idx {
                    pos[wi] = [outer, 0.0];
                }
                if let Some(di) = dead_idx {
                    pos[di] = [-outer, 0.0];
                }
                pos
            }
        };

        // Build NodeGpu (positions + zero velocity)
        let node_data: Vec<NodeGpu> = positions
            .iter()
            .map(|p| NodeGpu {
                pos: *p,
                vel: [0.0, 0.0],
            })
            .collect();

        // Build NodeInfo (colors, flags, radius)
        let node_info: Vec<NodeInfo> = (0..n_nodes)
            .map(|i| {
                if Some(i) == win_idx {
                    return NodeInfo {
                        color: [0.2, 0.9, 0.2, 1.0], // bright green
                        flags: FLAG_WIN,
                        bfs_depth: u32::MAX,
                        radius: 1.0,
                        _pad: 0.0,
                    };
                }
                if Some(i) == dead_idx {
                    return NodeInfo {
                        color: [0.7, 0.1, 0.1, 1.0], // dark red
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
                    [0.3, 0.55, 1.0, 1.0] // blue
                } else if !is_winning {
                    [0.25, 0.25, 0.25, 0.8] // dark gray (dead-end)
                } else if win_prob > 0.8 {
                    [0.2, 0.8, 0.2, 1.0] // green
                } else if win_prob > 0.3 {
                    [0.9, 0.8, 0.2, 1.0] // yellow
                } else {
                    [0.8, 0.2, 0.2, 1.0] // red
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

        self.buffers = Some(Arc::new(new_buffers));
        self.positions = positions;
        self.state_to_idx = state_to_idx;
        self.idx_to_state = idx_to_state;
        self.sim_running = self.layout_mode == LayoutMode::ForceDirected;
        self.max_velocity = f32::MAX;

        // Auto-fit camera
        self.fit_camera();
    }

    fn fit_camera(&mut self) {
        if self.positions.is_empty() {
            return;
        }
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for p in &self.positions {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        self.camera.pan = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0];
        let range_x = (max_x - min_x).max(1.0);
        let range_y = (max_y - min_y).max(1.0);
        let range = range_x.max(range_y) * 1.2; // 20% margin
        self.camera.zoom = 2.0 / range;
    }

    /// Main draw function, called from the Graph tab.
    pub fn draw(&mut self, ui: &mut egui::Ui, selected_level_idx: Option<usize>) {
        // Toolbar
        self.draw_toolbar(ui, selected_level_idx);
        ui.separator();

        // Graph canvas
        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, egui::Sense::click_and_drag());
        let rect = response.rect;

        // Update aspect ratio
        if rect.width() > 0.0 && rect.height() > 0.0 {
            self.camera.aspect = rect.width() / rect.height();
        }

        // Handle interaction
        self.handle_interaction(ui, &response);

        // Paint background
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(25, 25, 35));

        // Issue GPU paint callback
        if let Some(buffers) = &self.buffers {
            if buffers.n_nodes > 0 {
                let sim_params = SimParams {
                    n_nodes: buffers.n_nodes,
                    n_edges: buffers.n_edges,
                    repel: 1.0,
                    attract: 1.0,
                    decay: 0.92,
                    speed_limit: 20.0,
                    _pad: [0.0; 2],
                };

                let callback = GraphPaintCallback {
                    pipelines: Arc::clone(&self.pipelines),
                    buffers: Arc::clone(buffers),
                    camera: self.camera,
                    sim_params,
                    run_compute: self.sim_running,
                    iterations_per_frame: self.iterations_per_frame,
                };

                painter.add(Callback::new_paint_callback(rect, callback));

                // Request repaint while sim is running
                if self.sim_running {
                    ui.ctx().request_repaint();
                }
            }
        }

        // Draw hover tooltip
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
                // Re-layout with new mode
                self.loaded_level_idx = None; // force rebuild
            }

            ui.separator();

            if self.layout_mode == LayoutMode::ForceDirected {
                let label = if self.sim_running { "Pause" } else { "Resume" };
                if ui.button(label).clicked() {
                    self.sim_running = !self.sim_running;
                }
                if ui.button("Reset").clicked() {
                    self.loaded_level_idx = None; // force rebuild
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

    /// Convert a screen position to normalized clip coordinates (Y-up).
    fn screen_to_clip(pos: egui::Pos2, rect: egui::Rect) -> [f32; 2] {
        [
            (pos.x - rect.center().x) / (rect.width() / 2.0),
            -(pos.y - rect.center().y) / (rect.height() / 2.0),
        ]
    }

    /// Convert normalized clip coordinates to world coordinates.
    fn clip_to_world(&self, clip: [f32; 2]) -> [f32; 2] {
        [
            clip[0] * self.camera.aspect / self.camera.zoom + self.camera.pan[0],
            clip[1] / self.camera.zoom + self.camera.pan[1],
        ]
    }

    fn handle_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let rect = response.rect;

        // Pan: right-drag or middle-drag
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = response.drag_delta();
            // Convert screen delta to world delta (negate Y: screen Y-down, clip Y-up)
            self.camera.pan[0] -=
                delta.x * self.camera.aspect / (self.camera.zoom * rect.width() / 2.0);
            self.camera.pan[1] += delta.y / (self.camera.zoom * rect.height() / 2.0);
        }

        // Zoom: scroll wheel, zoom toward cursor
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let factor = (scroll * 0.005).exp();
            let old_zoom = self.camera.zoom;
            self.camera.zoom *= factor;
            self.camera.zoom = self.camera.zoom.clamp(0.001, 100.0);

            // Zoom toward cursor: adjust pan so the world point under the cursor stays fixed
            if let Some(cursor) = response.hover_pos() {
                let clip = Self::screen_to_clip(cursor, rect);
                let world_before = [
                    clip[0] * self.camera.aspect / old_zoom + self.camera.pan[0],
                    clip[1] / old_zoom + self.camera.pan[1],
                ];
                let world_after = self.clip_to_world(clip);
                self.camera.pan[0] += world_before[0] - world_after[0];
                self.camera.pan[1] += world_before[1] - world_after[1];
            }
        }

        // Hover hit test
        self.hovered_node = None;
        if let Some(cursor) = response.hover_pos() {
            let clip = Self::screen_to_clip(cursor, rect);
            let world = self.clip_to_world(clip);

            let mut best_dist = f32::MAX;
            for (i, pos) in self.positions.iter().enumerate() {
                let dx = pos[0] - world[0];
                let dy = pos[1] - world[1];
                let dist = (dx * dx + dy * dy).sqrt();
                let hit_r = 1.0;
                if dist < hit_r && dist < best_dist {
                    best_dist = dist;
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

        // Get the state for this node
        let state = match self.idx_to_state.get(node_idx) {
            Some(Some(s)) => *s,
            Some(None) => {
                // Virtual node (WIN/DEAD) — identify using stored indices
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
        let level = level.clone();

        response.clone().on_hover_ui(|ui: &mut egui::Ui| {
            ui.label(format!("Win prob: {:.1}%", win_prob * 100.0));

            // Draw maze state preview
            let size = egui::Vec2::new(200.0, 200.0);
            let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
            crate::render::draw_maze_state(&painter, resp.rect, &level, &state);
        });
    }

}
