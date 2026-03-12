mod adversarial;
mod adversarial_tab;
mod level_gen_tab;
mod data;
mod gameplay;
mod graph_view;
mod render;
mod table;
mod training_metrics;
mod training_tab;
mod ws_client;

use data::DataStore;
use eframe::egui;
use gameplay::GameplayState;
use graph_view::GraphView;
use mummymaze::game::{Action, State};
use mummymaze::graph::StateGraph;
use rustc_hash::FxHashMap;
use ws_client::{ServerEvent, WsClient};

#[derive(PartialEq, Clone, Copy)]
enum RightTab {
    Graph,
    Training,
    LevelGen,
    Adversarial,
    Logs,
}

struct App {
    store: DataStore,
    gameplay: Option<GameplayState>,
    graph_view: Option<GraphView>,
    /// Stored graph for click-to-navigate BFS lookups.
    graph: Option<StateGraph>,
    /// Precomputed BFS optimal action bitmask per state (from graph).
    bfs_optimal: FxHashMap<State, u8>,
    /// Analysis for GA-generated levels (not in the dataset).
    generated_analysis: Option<mummymaze::batch::LevelAnalysis>,
    /// Cached agent action probabilities from WS evaluate.
    /// `None` = no request in flight, `Some((key, None))` = pending,
    /// `Some((key, Some(map)))` = ready.
    cached_agent_probs: Option<(String, Option<FxHashMap<State, [f32; 5]>>)>,
    level_gen: level_gen_tab::LevelGenState,
    adversarial: adversarial::AdversarialState,
    /// WebSocket client for the Python model server.
    ws_client: Option<WsClient>,
    show_bfs_overlay: bool,
    show_agent_overlay: bool,
    right_tab: RightTab,
}

impl App {
    fn new(maze_dir: &std::path::Path, cc: &eframe::CreationContext<'_>) -> Self {
        let mut store = DataStore::load_levels(maze_dir);
        store.start_analysis();

        let graph_view = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| GraphView::new(rs.clone()));

        // Connect to WebSocket model server
        let ws_url = std::env::var("MODEL_SERVER_URL")
            .unwrap_or_else(|_| "ws://localhost:8765".to_string());
        let ws_client = match WsClient::connect(&ws_url) {
            Ok(ws) => {
                eprintln!("Connected to model server at {ws_url}");
                Some(ws)
            }
            Err(e) => {
                eprintln!("Failed to connect to model server at {ws_url}: {e}");
                None
            }
        };

        App {
            store,
            gameplay: None,
            graph_view,
            graph: None,
            bfs_optimal: FxHashMap::default(),
            generated_analysis: None,
            cached_agent_probs: None,
            level_gen: level_gen_tab::LevelGenState::new(),
            adversarial: adversarial::AdversarialState::new(),
            ws_client,
            show_bfs_overlay: false,
            show_agent_overlay: false,
            right_tab: RightTab::Graph,
        }
    }

    fn select_level(&mut self, idx: usize) {
        self.store.selected = Some(idx);
        self.generated_analysis = None;
        let row = &self.store.rows[idx];
        self.gameplay = Some(GameplayState::new(row.level.clone()));

        // Build and load graph for the newly selected level
        if let Some(ref mut gv) = self.graph_view {
            if !gv.is_loaded(idx) {
                let graph = mummymaze::graph::build_graph(&row.level);
                let chain = mummymaze::markov::MarkovChain::from_graph(&graph);
                gv.load_level(&row.level, idx, &graph, &chain);
                self.set_graph(graph);
            }
        }

        // Set initial state on graph view
        if let (Some(gv), Some(gs)) = (&mut self.graph_view, &self.gameplay) {
            gv.set_current_state(gs.current_state);
            gv.update_walk_highlight(&gs.history, gs.current_state);
        }
    }

    /// Ensure the graph is loaded for the currently selected level (lazy load).
    fn ensure_graph_loaded(&mut self) {
        let Some(sel) = self.store.selected else {
            return;
        };
        let Some(ref mut gv) = self.graph_view else {
            return;
        };
        if gv.is_loaded(sel) {
            return;
        }
        let row = &self.store.rows[sel];
        let graph = mummymaze::graph::build_graph(&row.level);
        let chain = mummymaze::markov::MarkovChain::from_graph(&graph);
        gv.load_level(&row.level, sel, &graph, &chain);
        self.set_graph(graph);
    }

    /// Store graph and precompute BFS optimal action masks.
    fn set_graph(&mut self, graph: StateGraph) {
        self.bfs_optimal = graph.best_actions_per_state().into_iter().collect();
        self.graph = Some(graph);
    }

    fn draw_maze_panel(&mut self, ui: &mut egui::Ui) {
        let selected = self.store.selected;
        let has_gameplay = self.gameplay.is_some();

        if selected.is_none() && !has_gameplay {
            ui.centered_and_justified(|ui: &mut egui::Ui| {
                ui.heading("Select a level");
            });
            return;
        }

        // Heading + stats
        let analysis = if let Some(sel) = selected {
            let row = &self.store.rows[sel];
            ui.heading(format!("{} sub {}", row.file_stem, row.sublevel));
            row.analysis.as_ref()
        } else {
            ui.heading("Generated Level");
            self.generated_analysis.as_ref()
        };
        if let Some(analysis) = analysis {
            ui.horizontal_wrapped(|ui: &mut egui::Ui| {
                ui.label(format!("Grid: {}", analysis.grid_size));
                ui.separator();
                if let Some(bfs) = analysis.bfs_moves {
                    ui.label(format!("BFS: {bfs}"));
                    ui.separator();
                }
                ui.label(format!("States: {}", analysis.n_states));
                ui.separator();
                let wp = if analysis.win_prob != 0.0 && analysis.win_prob.abs() < 0.0001 {
                    format!("{:.2e}", analysis.win_prob)
                } else {
                    format!("{:.4}", analysis.win_prob)
                };
                ui.label(format!("Win%: {wp}"));
                ui.separator();
                ui.label(format!("E[steps]: {:.1}", analysis.expected_steps));
                if let Some(safety) = analysis.difficulty.path_safety {
                    ui.separator();
                    ui.label(format!("Safety: {safety:.2}"));
                }
            });
        }

        ui.separator();

        if let Some(ref mut gs) = self.gameplay {
            // Maze display
            let available = ui.available_size();
            let size = render::maze_preferred_size(available);
            let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
            render::draw_maze_state(&painter, response.rect, &gs.level, &gs.current_state);

            // Action probability + BFS optimal overlay
            let agent_probs = if self.show_agent_overlay {
                selected.and_then(|sel| {
                    let row = &self.store.rows[sel];
                    let level_key = format!("{}:{}", row.file_stem, row.sublevel);
                    // Fire off a non-blocking evaluate if not cached/pending
                    let needs_fetch = match &self.cached_agent_probs {
                        Some((k, _)) => k != &level_key,
                        None => true,
                    };
                    if needs_fetch {
                        if let Some(ref ws) = self.ws_client {
                            if let Err(e) = ws.send_evaluate(&level_key) {
                                self.store.log_messages.push(
                                    format!("Evaluate request failed: {e}"),
                                );
                            }
                        }
                        // Mark as pending (result arrives via poll_events)
                        self.cached_agent_probs = Some((level_key.clone(), None));
                    }
                    self.cached_agent_probs
                        .as_ref()
                        .and_then(|(k, data)| {
                            if k == &level_key {
                                data.as_ref()?.get(&gs.current_state).copied()
                            } else {
                                None
                            }
                        })
                })
            } else {
                None
            };
            let bfs_mask = if self.show_bfs_overlay {
                self.bfs_optimal.get(&gs.current_state).copied()
            } else {
                None
            };
            render::draw_action_bars(
                &painter,
                response.rect,
                gs.level.grid_size,
                gs.current_state.player_row,
                gs.current_state.player_col,
                agent_probs.as_ref(),
                bfs_mask,
            );

            // Status + controls below the maze
            ui.add_space(4.0);
            let status = gs.status_text();
            let status_color = match gs.result {
                Some(mummymaze::game::StepResult::Win) => egui::Color32::GREEN,
                Some(mummymaze::game::StepResult::Dead) => egui::Color32::RED,
                _ => ui.visuals().text_color(),
            };

            // Measure for centering
            let mut measure_ui = ui.new_child(egui::UiBuilder::new().invisible());
            measure_ui.horizontal(|ui: &mut egui::Ui| {
                ui.colored_label(status_color, &status);
                ui.separator();
                let _ = ui.button("Undo (Z)");
                let _ = ui.button("Redo (Y)");
                let _ = ui.button("Reset (R)");
            });
            let row_width = measure_ui.min_rect().width();
            let indent = (ui.available_width() - row_width) / 2.0;

            ui.horizontal(|ui: &mut egui::Ui| {
                ui.add_space(indent.max(0.0));
                ui.colored_label(status_color, &status);
                ui.separator();
                if ui.button("Undo (Z)").clicked() {
                    gs.undo();
                }
                if ui.button("Redo (Y)").clicked() {
                    gs.redo();
                }
                if ui.button("Reset (R)").clicked() {
                    gs.reset();
                }
            });

            // Overlay checkboxes
            ui.horizontal(|ui: &mut egui::Ui| {
                ui.checkbox(&mut self.show_bfs_overlay, "BFS optimal");
                ui.checkbox(&mut self.show_agent_overlay, "Agent policy");
            });
        }
    }

    fn draw_graph_panel(&mut self, ui: &mut egui::Ui) {
        self.ensure_graph_loaded();
        let selected = self.store.selected;
        let has_gameplay = self.gameplay.is_some();
        if let Some(ref mut gv) = self.graph_view {
            if selected.is_some() || has_gameplay {
                gv.draw(ui, selected);
            } else {
                ui.centered_and_justified(|ui: &mut egui::Ui| {
                    ui.heading("Select a level");
                });
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll background analysis
        if self.store.poll_analysis() {
            ctx.request_repaint();
        }
        if self.store.is_analyzing() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        // Poll WebSocket events and dispatch
        if let Some(ref ws) = self.ws_client {
            let events = ws.poll_events();
            if !events.is_empty() {
                let mut training_events = Vec::new();
                let mut adversarial_events = Vec::new();
                let mut ga_events = Vec::new();
                for event in events {
                    match event {
                        ServerEvent::Training(te) => training_events.push(te),
                        ServerEvent::Adversarial(ae) => adversarial_events.push(ae),
                        ServerEvent::Ga(ge) => ga_events.push(ge),
                        ServerEvent::Evaluate(result) => {
                            // Populate cache if still pending
                            if let Some((_, ref mut data)) = self.cached_agent_probs {
                                if data.is_none() {
                                    *data = Some(
                                        result
                                            .probs_by_state
                                            .into_iter()
                                            .collect(),
                                    );
                                }
                            }
                        }
                        ServerEvent::Error(msg) => {
                            self.store.log_messages.push(format!("Server error: {msg}"));
                        }
                    }
                }
                if self.store.handle_training_events(&training_events) {
                    ctx.request_repaint();
                }
                if self.adversarial.handle_events(&adversarial_events) {
                    ctx.request_repaint();
                }
                if self.level_gen.handle_events(ga_events) {
                    ctx.request_repaint();
                }
            }
        }
        // Fast repaint while training, adversarial, or GA is active
        if self.store.is_training() || self.adversarial.is_running() || self.level_gen.is_running() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        // Consume keyboard input for gameplay BEFORE panels process it
        let mut gameplay_action = None;
        let mut undo_pressed = false;
        let mut redo_pressed = false;
        let mut reset_pressed = false;
        if let Some(ref gs) = self.gameplay {
            ctx.input(|i: &egui::InputState| {
                if i.key_pressed(egui::Key::Z) {
                    undo_pressed = true;
                } else if i.key_pressed(egui::Key::Y) {
                    redo_pressed = true;
                } else if i.key_pressed(egui::Key::R) {
                    reset_pressed = true;
                } else if !gs.is_over() {
                    if i.key_pressed(egui::Key::ArrowUp) {
                        gameplay_action = Some(Action::North);
                    } else if i.key_pressed(egui::Key::ArrowDown) {
                        gameplay_action = Some(Action::South);
                    } else if i.key_pressed(egui::Key::ArrowRight) {
                        gameplay_action = Some(Action::East);
                    } else if i.key_pressed(egui::Key::ArrowLeft) {
                        gameplay_action = Some(Action::West);
                    } else if i.key_pressed(egui::Key::Space) {
                        gameplay_action = Some(Action::Wait);
                    }
                }
            });
        }

        // Track whether state changed this frame (for graph sync)
        let mut state_changed = false;

        if let Some(action) = gameplay_action {
            if let Some(ref mut gs) = self.gameplay {
                gs.apply_action(action);
                state_changed = true;
                // Only player moves re-engage auto-follow
                if let Some(ref mut gv) = self.graph_view {
                    gv.reengage_auto_follow();
                }
            }
        } else if undo_pressed {
            if let Some(ref mut gs) = self.gameplay {
                gs.undo();
                state_changed = true;
            }
        } else if redo_pressed {
            if let Some(ref mut gs) = self.gameplay {
                gs.redo();
                state_changed = true;
            }
        } else if reset_pressed {
            if let Some(ref mut gs) = self.gameplay {
                gs.reset();
                state_changed = true;
            }
        }

        // Handle undo/reset from button clicks (tracked via state_changed flag below)
        // Undo/Reset clicks are handled inside draw_maze_panel; we detect state change
        // by snapshotting state before and after drawing.
        let state_before = self.gameplay.as_ref().map(|gs| (gs.current_state, gs.turn));

        // Left side panel: level table
        egui::SidePanel::left("level_panel")
            .default_width(420.0)
            .min_width(300.0)
            .show(ctx, |ui: &mut egui::Ui| {
                ui.heading("Mummy Maze Levels");
                ui.separator();

                table::draw_filters(ui, &mut self.store);
                ui.separator();
                table::draw_progress(ui, &self.store);

                ui.separator();
                ui.horizontal(|ui: &mut egui::Ui| {
                    ui.label(format!(
                        "{} / {} levels shown",
                        self.store.sorted_indices.len(),
                        self.store.rows.len()
                    ));
                    if let Some((done, total)) = self.store.analysis_progress {
                        ui.separator();
                        ui.label(format!("Analysis: {done}/{total}"));
                    }
                });
                ui.separator();

                if let Some(clicked) = table::draw_table(ui, &mut self.store) {
                    self.select_level(clicked);
                }
            });

        // Central panel: combined maze (left) + graph (right) split view
        egui::CentralPanel::default().show(ctx, |ui: &mut egui::Ui| {
            // Use a left sub-panel for the maze, graph fills remaining space
            egui::SidePanel::left("maze_panel")
                .default_width(ui.available_width() * 0.45)
                .min_width(200.0)
                .resizable(true)
                .show_inside(ui, |ui: &mut egui::Ui| {
                    self.draw_maze_panel(ui);
                });

            // Remaining space: tabbed graph/training view
            ui.horizontal(|ui: &mut egui::Ui| {
                ui.selectable_value(&mut self.right_tab, RightTab::Graph, "Graph");
                ui.selectable_value(&mut self.right_tab, RightTab::Training, "Training");
                ui.selectable_value(&mut self.right_tab, RightTab::LevelGen, "Level Gen");
                ui.selectable_value(&mut self.right_tab, RightTab::Adversarial, "Adversarial");
                let log_label = if self.store.log_messages.is_empty() {
                    "Logs".to_string()
                } else {
                    format!("Logs ({})", self.store.log_messages.len())
                };
                ui.selectable_value(&mut self.right_tab, RightTab::Logs, log_label);
            });
            ui.separator();

            match self.right_tab {
                RightTab::Graph => self.draw_graph_panel(ui),
                RightTab::Training => {
                    if let Some(clicked) =
                        training_tab::draw_training_panel(ui, &mut self.store, self.ws_client.as_ref())
                    {
                        self.select_level(clicked);
                    }
                }
                RightTab::LevelGen => {
                    if let Some(level) =
                        level_gen_tab::draw_level_gen_panel(ui, &mut self.level_gen, &self.store.rows, self.ws_client.as_ref())
                    {
                        // Load GA-generated level into maze panel
                        self.gameplay = Some(GameplayState::new(level.clone()));
                        if let Ok(full) =
                            mummymaze::batch::analyze_level_full("generated", 0, &level)
                        {
                            self.generated_analysis = Some(full.analysis);
                            if let Some(ref mut gv) = self.graph_view {
                                gv.load_level(&level, usize::MAX, &full.graph, &full.chain);
                            }
                            self.set_graph(full.graph);
                        }
                        self.store.selected = None;
                    }
                }
                RightTab::Adversarial => {
                    adversarial_tab::draw_panel(
                        ui,
                        &mut self.adversarial,
                        self.ws_client.as_ref(),
                    );
                }
                RightTab::Logs => {
                    ui.horizontal(|ui: &mut egui::Ui| {
                        ui.label(format!("{} messages", self.store.log_messages.len()));
                        if ui.button("Clear").clicked() {
                            self.store.log_messages.clear();
                        }
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .stick_to_bottom(true)
                        .show(ui, |ui: &mut egui::Ui| {
                            for msg in &self.store.log_messages {
                                ui.label(egui::RichText::new(msg).monospace().size(12.0));
                            }
                        });
                }
            }
        });

        // Detect state changes from undo/reset button clicks
        let state_after = self.gameplay.as_ref().map(|gs| (gs.current_state, gs.turn));
        if state_before != state_after {
            state_changed = true;
        }

        // Handle node click navigation
        if let Some(ref mut gv) = self.graph_view {
            if let Some(clicked_idx) = gv.take_clicked_node() {
                if let Some(clicked_state) = gv.node_state(clicked_idx) {
                    // Skip terminal nodes
                    if !gv.is_terminal(clicked_idx) {
                        if let Some(ref mut gs) = self.gameplay {
                            // Case 1: visited node — undo to it
                            let visited_pos = gs.history.iter().position(|(_a, s)| *s == clicked_state);
                            if clicked_state == gs.initial_state {
                                gs.reset();
                                state_changed = true;
                            } else if let Some(pos) = visited_pos {
                                // Undo until we're at that state
                                while gs.history.len() > pos + 1 {
                                    gs.undo();
                                }
                                // The state at history[pos] is the state BEFORE the action,
                                // so after undoing to pos+1 entries, current_state should be
                                // the state we reached after the action at `pos`.
                                // Actually we want to be AT clicked_state, so undo until current == clicked
                                while gs.current_state != clicked_state && !gs.history.is_empty() {
                                    gs.undo();
                                }
                                state_changed = true;
                            } else if let Some(ref graph) = self.graph {
                                // Case 2: non-visited reachable node — BFS path forward
                                if let Some(actions) = bfs_path(graph, gs.current_state, clicked_state) {
                                    for action in actions {
                                        gs.apply_action(action);
                                    }
                                    state_changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sync graph view with gameplay state
        if state_changed {
            if let (Some(gv), Some(gs)) = (&mut self.graph_view, &self.gameplay) {
                gv.set_current_state(gs.current_state);
                gv.update_walk_highlight(&gs.history, gs.current_state);
            }
        }
    }
}

/// BFS from `from` to `to` using the state graph, returning the action sequence.
fn bfs_path(graph: &StateGraph, from: mummymaze::game::State, to: mummymaze::game::State) -> Option<Vec<Action>> {
    use mummymaze::graph::StateKey;
    use rustc_hash::FxHashMap;
    use std::collections::VecDeque;

    let mut came_from: FxHashMap<mummymaze::game::State, (mummymaze::game::State, Action)> = FxHashMap::default();
    let mut queue = VecDeque::new();
    queue.push_back(from);

    while let Some(cur) = queue.pop_front() {
        if cur == to {
            // Reconstruct path
            let mut actions = Vec::new();
            let mut s = to;
            while s != from {
                let (prev, action) = came_from[&s];
                actions.push(action);
                s = prev;
            }
            actions.reverse();
            return Some(actions);
        }
        if let Some(transitions) = graph.transitions.get(&cur) {
            for &(action, dest) in transitions {
                if let StateKey::Transient(ns) = dest {
                    if ns != from && !came_from.contains_key(&ns) {
                        came_from.insert(ns, (cur, action));
                        queue.push_back(ns);
                    }
                }
            }
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let maze_dir = if args.len() > 1 {
        std::path::PathBuf::from(&args[1])
    } else {
        std::path::PathBuf::from("mazes")
    };

    if !maze_dir.exists() {
        eprintln!("Maze directory not found: {}", maze_dir.display());
        eprintln!("Usage: mummymaze-viewer [MAZE_DIR]");
        std::process::exit(1);
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 800.0])
            .with_title("Mummy Maze Viewer"),
        depth_buffer: 24,
        ..Default::default()
    };

    eframe::run_native(
        "Mummy Maze Viewer",
        native_options,
        Box::new(move |cc: &eframe::CreationContext<'_>| Ok(Box::new(App::new(&maze_dir, cc)))),
    )
    .expect("Failed to launch eframe");
}
