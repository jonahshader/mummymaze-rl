use crate::data::LevelRow;
use crate::render;
use eframe::egui;
use mummymaze::ga::fitness::{FitnessExpr, PRESETS};
use mummymaze::ga::{CrossoverMode, GaConfig, GaMessage, Individual};
use mummymaze::game::State;
use mummymaze::parse::Level;
use mummymaze::policy_client::PolicyClient;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum GaStatus {
    Idle,
    Starting(String),
    Running {
        generation: usize,
        total_generations: usize,
    },
    Done,
    Error(String),
}

pub struct AdversarialState {
    pub config: GaConfig,
    pub show_config: bool,
    pub status: GaStatus,
    pub best: Option<Individual>,
    pub history: Vec<(usize, f64, f64)>, // (generation, best_fitness, avg_fitness)
    rx: Option<Receiver<GaMessage>>,
    stop_flag: Option<Arc<AtomicBool>>,
    /// Validation error for the fitness expression (empty = valid).
    fitness_error: String,
    /// Optional policy checkpoint path for policy_win_prob evaluation.
    pub policy_checkpoint: String,
    /// Whether to use the policy net for fitness evaluation.
    pub use_policy: bool,
}

impl AdversarialState {
    pub fn new() -> Self {
        AdversarialState {
            config: GaConfig {
                seed: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                ..GaConfig::default()
            },
            show_config: false,
            status: GaStatus::Idle,
            best: None,
            history: Vec::new(),
            rx: None,
            stop_flag: None,
            fitness_error: String::new(),
            policy_checkpoint: latest_checkpoint().unwrap_or_default(),
            use_policy: false,
        }
    }

    pub fn start(&mut self, seed_levels: Vec<Level>) {
        let (tx, rx) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let config = self.config.clone();
        let flag = stop_flag.clone();
        let use_policy = self.use_policy;
        let checkpoint_path = self.policy_checkpoint.clone();

        self.history.clear();
        self.best = None;
        self.status = if use_policy && !checkpoint_path.is_empty() {
            GaStatus::Starting("Starting policy server...".to_string())
        } else {
            GaStatus::Starting("Evaluating seeds...".to_string())
        };
        self.rx = Some(rx);
        self.stop_flag = Some(stop_flag);

        std::thread::spawn(move || {
            if use_policy && !checkpoint_path.is_empty() {
                let path = PathBuf::from(&checkpoint_path);
                let _ = tx.send(GaMessage::Status(
                    "Starting policy server...".to_string(),
                ));
                match PolicyClient::spawn(&path) {
                    Ok(policy_client) => {
                        mummymaze::ga::run_ga_with_policy(
                            &config,
                            seed_levels,
                            tx,
                            flag,
                            policy_client,
                        );
                    }
                    Err(e) => {
                        let _ = tx.send(GaMessage::Error(format!(
                            "Failed to start policy server: {e}"
                        )));
                    }
                }
            } else {
                mummymaze::ga::run_ga(&config, seed_levels, tx, flag);
            }
        });
    }

    pub fn stop(&mut self) {
        if let Some(ref flag) = self.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Drain channel messages. Returns true if anything changed.
    pub fn poll(&mut self) -> bool {
        if self.rx.is_none() {
            return false;
        }
        let mut changed = false;
        let mut finished = false;
        // Drain messages while receiver exists
        while let Some(msg) = self.rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            changed = true;
            match msg {
                GaMessage::Status(s) => {
                    self.status = GaStatus::Starting(s);
                }
                GaMessage::SeedsDone { .. } => {}
                GaMessage::ArchiveUpdate { .. } => {}
                GaMessage::Generation(result) => {
                    self.history
                        .push((result.generation, result.best.fitness, result.avg_fitness));
                    self.best = Some(result.best);
                    self.status = GaStatus::Running {
                        generation: result.generation,
                        total_generations: self.config.generations,
                    };
                }
                GaMessage::Done => {
                    self.status = GaStatus::Done;
                    finished = true;
                }
                GaMessage::Error(e) => {
                    self.status = GaStatus::Error(e);
                    finished = true;
                }
            }
        }
        if finished {
            self.rx = None;
            self.stop_flag = None;
        }
        changed
    }

    pub fn is_running(&self) -> bool {
        matches!(
            self.status,
            GaStatus::Starting(_) | GaStatus::Running { .. }
        )
    }
}

/// Draw the adversarial GA panel. Returns Some(level) when "Load into Maze" is clicked.
pub fn draw_adversarial_panel(
    ui: &mut egui::Ui,
    state: &mut AdversarialState,
    rows: &[LevelRow],
) -> Option<Level> {
    let mut load_level = None;

    // Controls
    ui.horizontal(|ui: &mut egui::Ui| {
        if state.is_running() {
            if ui.button("Stop").clicked() {
                state.stop();
            }
        } else {
            if ui.button("Configure").clicked() {
                state.show_config = !state.show_config;
            }
            if ui.button("Start").clicked() {
                let seeds: Vec<Level> = rows
                    .iter()
                    .filter(|r| r.level.grid_size == state.config.grid_size)
                    .map(|r| r.level.clone())
                    .collect();
                state.start(seeds);
            }
        }

        // Status line
        match &state.status {
            GaStatus::Idle => {
                ui.label("Idle");
            }
            GaStatus::Starting(msg) => {
                // Try to parse "Evaluating seeds: 123/456" for a progress bar
                if let Some(frac) = parse_progress_frac(msg) {
                    ui.add(
                        egui::ProgressBar::new(frac)
                            .text(msg.as_str())
                            .desired_width(200.0),
                    );
                } else {
                    ui.spinner();
                    ui.label(msg);
                }
            }
            GaStatus::Running {
                generation,
                total_generations,
            } => {
                ui.label(format!("Gen {generation}/{total_generations}"));
            }
            GaStatus::Done => {
                ui.colored_label(egui::Color32::GREEN, "Done");
            }
            GaStatus::Error(e) => {
                ui.colored_label(egui::Color32::RED, format!("Error: {e}"));
            }
        }
    });

    // Config window
    egui::Window::new("GA Config")
        .open(&mut state.show_config)
        .resizable(false)
        .default_width(250.0)
        .show(ui.ctx(), |ui| {
            egui::Grid::new("ga_config_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Grid size:");
                    egui::ComboBox::from_id_salt("ga_grid_size")
                        .selected_text(format!("{}", state.config.grid_size))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.config.grid_size, 6, "6");
                            ui.selectable_value(&mut state.config.grid_size, 8, "8");
                            ui.selectable_value(&mut state.config.grid_size, 10, "10");
                        });
                    ui.end_row();

                    ui.label("Population:");
                    ui.add(egui::DragValue::new(&mut state.config.pop_size).range(4..=512));
                    ui.end_row();

                    ui.label("Generations:");
                    ui.add(egui::DragValue::new(&mut state.config.generations).range(1..=1000));
                    ui.end_row();

                    ui.label("Elite fraction:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.elite_frac)
                            .range(0.0..=0.5)
                            .speed(0.01)
                            .max_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Crossover rate:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.crossover_rate)
                            .range(0.0..=1.0)
                            .speed(0.01)
                            .max_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Crossover mode:");
                    egui::ComboBox::from_id_salt("ga_crossover_mode")
                        .selected_text(state.config.crossover_mode.label())
                        .show_ui(ui, |ui| {
                            for mode in CrossoverMode::ALL {
                                ui.selectable_value(
                                    &mut state.config.crossover_mode,
                                    mode,
                                    mode.label(),
                                );
                            }
                        });
                    ui.end_row();

                    ui.label("Seed:");
                    ui.add(egui::DragValue::new(&mut state.config.seed));
                    ui.end_row();
                });

            ui.separator();
            ui.label("Mutation weights:");
            egui::Grid::new("ga_mutation_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Toggle wall:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.w_wall)
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .max_decimals(1),
                    );
                    ui.end_row();

                    ui.label("Move entity:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.w_move_entity)
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .max_decimals(1),
                    );
                    ui.end_row();

                    ui.label("Move player:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.w_move_player)
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .max_decimals(1),
                    );
                    ui.end_row();

                    ui.label("Add entity:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.w_add_entity)
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .max_decimals(1),
                    );
                    ui.end_row();

                    ui.label("Remove entity:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.w_remove_entity)
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .max_decimals(1),
                    );
                    ui.end_row();

                    ui.label("Move exit:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.w_move_exit)
                            .range(0.0..=20.0)
                            .speed(0.1)
                            .max_decimals(1),
                    );
                    ui.end_row();

                    ui.label("Extra wall prob:");
                    ui.add(
                        egui::DragValue::new(&mut state.config.extra_wall_prob)
                            .range(0.0..=1.0)
                            .speed(0.01)
                            .max_decimals(2),
                    );
                    ui.end_row();
                });

            ui.separator();

            // Policy net section
            ui.checkbox(&mut state.use_policy, "Use policy net");
            if state.use_policy {
                ui.horizontal(|ui| {
                    ui.label("Checkpoint:");
                    ui.add(
                        egui::TextEdit::singleline(&mut state.policy_checkpoint)
                            .desired_width(ui.available_width() - 4.0)
                            .hint_text("path/to/model.eqx")
                            .font(egui::TextStyle::Monospace),
                    );
                });
                if state.policy_checkpoint.is_empty() {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "Set checkpoint path to use policy_win_prob",
                    );
                }
            }

            ui.separator();
            ui.label("Fitness expression:");

            // Preset dropdown
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("fitness_preset")
                    .selected_text("Presets")
                    .show_ui(ui, |ui| {
                        for (name, expr) in PRESETS {
                            if ui
                                .selectable_label(
                                    state.config.fitness_expr == *expr,
                                    format!("{name}: {expr}"),
                                )
                                .clicked()
                            {
                                state.config.fitness_expr = expr.to_string();
                                state.fitness_error.clear();
                            }
                        }
                    });
            });

            // Expression text field
            let response = ui.add(
                egui::TextEdit::singleline(&mut state.config.fitness_expr)
                    .desired_width(ui.available_width())
                    .font(egui::TextStyle::Monospace),
            );
            if response.changed() {
                // Revalidate on every change
                match FitnessExpr::parse(&state.config.fitness_expr) {
                    Ok(_) => state.fitness_error.clear(),
                    Err(e) => state.fitness_error = e,
                }
            }
            if !state.fitness_error.is_empty() {
                ui.colored_label(egui::Color32::RED, &state.fitness_error);
            }

            // Variable reference
            egui::CollapsingHeader::new("Available variables")
                .default_open(false)
                .show(ui, |ui| {
                    for (name, desc) in mummymaze::ga::fitness::VARIABLES {
                        ui.horizontal(|ui| {
                            ui.monospace(*name);
                            ui.label("—");
                            ui.label(*desc);
                        });
                    }
                });
        });

    ui.separator();

    // Mini maze preview + stats
    if let Some(ref best) = state.best {
        let best_state = State::from_level(&best.level);

        // Maze preview
        let available = ui.available_width();
        let side = available.min(300.0);
        let size = egui::Vec2::new(side, side);
        let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
        render::draw_maze_state(&painter, response.rect, &best.level, &best_state);

        // Stats below maze
        ui.horizontal(|ui: &mut egui::Ui| {
            let wp = if best.win_prob != 0.0 && best.win_prob.abs() < 0.0001 {
                format!("{:.2e}", best.win_prob)
            } else {
                format!("{:.4}", best.win_prob)
            };
            ui.label(format!(
                "BFS: {}  States: {}  Win%: {}  Fitness: {:.4}",
                best.bfs_moves, best.n_states, wp, best.fitness
            ));
        });

        if ui.button("Load into Maze").clicked() {
            load_level = Some(best.level.clone());
        }

        ui.separator();
    }

    // Fitness chart
    if !state.history.is_empty() {
        ui.label("Fitness over generations");
        let best_line: egui_plot::PlotPoints = state
            .history
            .iter()
            .map(|(g, best, _avg)| [*g as f64, *best])
            .collect();
        let avg_line: egui_plot::PlotPoints = state
            .history
            .iter()
            .map(|(g, _best, avg)| [*g as f64, *avg])
            .collect();

        let plot = egui_plot::Plot::new("fitness_chart")
            .height(200.0)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .x_axis_label("Generation")
            .y_axis_label("Fitness")
            .legend(egui_plot::Legend::default());

        plot.show(ui, |plot_ui| {
            plot_ui.line(
                egui_plot::Line::new(best_line)
                    .name("Best")
                    .color(egui::Color32::from_rgb(80, 140, 255)),
            );
            plot_ui.line(
                egui_plot::Line::new(avg_line)
                    .name("Avg")
                    .color(egui::Color32::from_rgb(230, 160, 40)),
            );
        });
    }

    load_level
}

/// Try to extract a progress fraction from "... N/M" strings.
fn parse_progress_frac(msg: &str) -> Option<f32> {
    let slash_part = msg.rsplit_once(' ')?.1;
    let (done_s, total_s) = slash_part.split_once('/')?;
    let done: f32 = done_s.parse().ok()?;
    let total: f32 = total_s.parse().ok()?;
    if total > 0.0 {
        Some(done / total)
    } else {
        None
    }
}

/// Find the newest .eqx checkpoint in `checkpoints/`.
fn latest_checkpoint() -> Option<String> {
    let dir = PathBuf::from("checkpoints");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "eqx") {
            continue;
        }
        let mtime = entry.metadata().ok()?.modified().ok()?;
        if best.as_ref().map_or(true, |(t, _)| mtime > *t) {
            best = Some((mtime, path));
        }
    }
    best.map(|(_, p)| p.to_string_lossy().into_owned())
}
