mod data;
mod gameplay;
mod graph_view;
mod render;
mod table;

use data::DataStore;
use eframe::egui;
use gameplay::GameplayState;
use graph_view::GraphView;
use mummymaze::game::Action;
use mummymaze::graph::StateGraph;

struct App {
    store: DataStore,
    gameplay: Option<GameplayState>,
    graph_view: Option<GraphView>,
    /// Stored graph for click-to-navigate BFS lookups.
    graph: Option<StateGraph>,
}

impl App {
    fn new(maze_dir: &std::path::Path, cc: &eframe::CreationContext<'_>) -> Self {
        let mut store = DataStore::load_levels(maze_dir);
        store.start_analysis();

        let graph_view = cc
            .wgpu_render_state
            .as_ref()
            .map(|rs| GraphView::new(rs.clone()));

        App {
            store,
            gameplay: None,
            graph_view,
            graph: None,
        }
    }

    fn select_level(&mut self, idx: usize) {
        self.store.selected = Some(idx);
        let row = &self.store.rows[idx];
        self.gameplay = Some(GameplayState::new(row.level.clone()));

        // Build and load graph for the newly selected level
        if let Some(ref mut gv) = self.graph_view {
            if !gv.is_loaded(idx) {
                let graph = mummymaze::graph::build_graph(&row.level);
                let chain = mummymaze::markov::MarkovChain::from_graph(&graph);
                let state_win_probs = match chain.solve_win_probs() {
                    Ok(probs) => chain.per_state_map(&probs),
                    Err(_) => Default::default(),
                };
                gv.load_level(&row.level, idx, &graph, state_win_probs);
                self.graph = Some(graph);
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
        let state_win_probs = match chain.solve_win_probs() {
            Ok(probs) => chain.per_state_map(&probs),
            Err(_) => Default::default(),
        };
        gv.load_level(&row.level, sel, &graph, state_win_probs);
        self.graph = Some(graph);
    }

    fn draw_maze_panel(&mut self, ui: &mut egui::Ui) {
        if let Some(sel) = self.store.selected {
            let row = &self.store.rows[sel];
            ui.horizontal(|ui: &mut egui::Ui| {
                ui.label(format!(
                    "{} sub {} | grid {} | moves: {}",
                    row.file_stem,
                    row.sublevel,
                    row.level.grid_size,
                    row.bfs_moves
                        .map(|m: u32| m.to_string())
                        .unwrap_or("-".into()),
                ));
                if let Some(a) = &row.analysis {
                    ui.label(format!("| {} states", a.n_states));
                    ui.label(format!("| win {:.1}%", a.win_prob * 100.0));
                    ui.label(format!("| E[steps] {:.1}", a.expected_steps));
                }
            });

            if let Some(ref mut gs) = self.gameplay {
                ui.horizontal(|ui: &mut egui::Ui| {
                    let status = gs.status_text();
                    let color = match gs.result {
                        Some(mummymaze::game::StepResult::Win) => egui::Color32::GREEN,
                        Some(mummymaze::game::StepResult::Dead) => egui::Color32::RED,
                        _ => ui.visuals().text_color(),
                    };
                    ui.colored_label(color, egui::RichText::new(&status).size(16.0));
                    ui.separator();
                    if ui.button("Undo").clicked() {
                        gs.undo();
                    }
                    if ui.button("Reset").clicked() {
                        gs.reset();
                    }
                });
            }

            ui.separator();

            if let Some(ref gs) = self.gameplay {
                let available = ui.available_size();
                let size = render::maze_preferred_size(available);
                let (response, painter) =
                    ui.allocate_painter(size, egui::Sense::hover());
                render::draw_maze_state(
                    &painter,
                    response.rect,
                    &gs.level,
                    &gs.current_state,
                );
            }
        } else {
            ui.centered_and_justified(|ui: &mut egui::Ui| {
                ui.heading("Select a level");
            });
        }
    }

    fn draw_graph_panel(&mut self, ui: &mut egui::Ui) {
        self.ensure_graph_loaded();
        let selected = self.store.selected;
        if let Some(ref mut gv) = self.graph_view {
            if selected.is_some() {
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

        // Consume keyboard input for gameplay BEFORE panels process it
        let mut gameplay_action = None;
        if let Some(ref gs) = self.gameplay {
            if !gs.is_over() {
                ctx.input(|i: &egui::InputState| {
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
                });
            }
        }

        // Track whether state changed this frame (for graph sync)
        let mut state_changed = false;

        if let Some(action) = gameplay_action {
            if let Some(ref mut gs) = self.gameplay {
                gs.apply_action(action);
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

            // Remaining space: graph view
            self.draw_graph_panel(ui);
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
