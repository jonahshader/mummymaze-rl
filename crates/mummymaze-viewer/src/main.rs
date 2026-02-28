mod data;
mod gameplay;
mod render;
mod table;

use data::DataStore;
use eframe::egui;
use gameplay::GameplayState;
use mummymaze::game::Action;

enum Mode {
    Preview,
    Playing(GameplayState),
}

struct App {
    store: DataStore,
    mode: Mode,
}

impl App {
    fn new(maze_dir: &std::path::Path) -> Self {
        let mut store = DataStore::load_levels(maze_dir);
        store.start_analysis();
        App {
            store,
            mode: Mode::Preview,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll background analysis
        if self.store.poll_analysis() {
            ctx.request_repaint();
        }

        // If analysis is still running, request repaint to keep polling
        if self.store.is_analyzing() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Consume keyboard input for gameplay BEFORE panels process it
        let mut gameplay_action = None;
        if let Mode::Playing(ref gs) = self.mode {
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

        // Apply action
        if let Some(action) = gameplay_action {
            if let Mode::Playing(ref mut gs) = self.mode {
                gs.apply_action(action);
            }
        }

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
                    self.store.selected = Some(clicked);
                    self.mode = Mode::Preview;
                }
            });

        // Central panel: preview or gameplay
        egui::CentralPanel::default().show(ctx, |ui: &mut egui::Ui| {
            // Buttons and info at the TOP, before the maze
            ui.horizontal(|ui: &mut egui::Ui| {
                match &self.mode {
                    Mode::Preview => {
                        if let Some(sel) = self.store.selected {
                            if ui.button("Play").clicked() {
                                let row = &self.store.rows[sel];
                                self.mode =
                                    Mode::Playing(GameplayState::new(row.level.clone()));
                            }
                        }
                    }
                    Mode::Playing(_) => {
                        if ui.button("Back").clicked() {
                            self.mode = Mode::Preview;
                        }
                    }
                }
            });

            // Level info
            if let Some(sel) = self.store.selected {
                let row = &self.store.rows[sel];
                ui.horizontal_wrapped(|ui: &mut egui::Ui| {
                    ui.label(format!(
                        "{} sub {} | grid {} | BFS: {}",
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
            }

            ui.separator();

            // Maze rendering takes remaining space
            match &mut self.mode {
                Mode::Preview => {
                    draw_preview_panel(ui, &self.store);
                }
                Mode::Playing(gs) => {
                    draw_gameplay_panel(ui, gs);
                }
            }
        });
    }
}

fn draw_preview_panel(ui: &mut egui::Ui, store: &DataStore) {
    if let Some(sel) = store.selected {
        let row = &store.rows[sel];
        let available = ui.available_size();
        let size = render::maze_preferred_size(available);
        let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
        render::draw_maze_state(&painter, response.rect, &row.level, &row.initial_state);
    } else {
        ui.centered_and_justified(|ui: &mut egui::Ui| {
            ui.heading("Select a level from the table");
        });
    }
}

fn draw_gameplay_panel(ui: &mut egui::Ui, gs: &mut GameplayState) {
    // Status + controls
    ui.horizontal(|ui: &mut egui::Ui| {
        let status = gs.status_text();
        let color = match gs.result {
            Some(mummymaze::game::StepResult::Win) => egui::Color32::GREEN,
            Some(mummymaze::game::StepResult::Dead) => egui::Color32::RED,
            _ => egui::Color32::WHITE,
        };
        ui.colored_label(color, egui::RichText::new(&status).size(18.0));

        ui.separator();
        if ui.button("Undo").clicked() {
            gs.undo();
        }
        if ui.button("Reset").clicked() {
            gs.reset();
        }
    });

    ui.separator();

    // Maze
    let available = ui.available_size();
    let size = render::maze_preferred_size(available);
    let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
    render::draw_maze_state(&painter, response.rect, &gs.level, &gs.current_state);
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
            .with_inner_size([1200.0, 800.0])
            .with_title("Mummy Maze Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "Mummy Maze Viewer",
        native_options,
        Box::new(move |_cc: &eframe::CreationContext<'_>| Ok(Box::new(App::new(&maze_dir)))),
    )
    .expect("Failed to launch eframe");
}
