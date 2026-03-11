//! UI for the adversarial training loop tab.

use crate::adversarial::{AdversarialConfig, AdversarialPhase, AdversarialState, AdversarialStatus};
use crate::training_tab::draw_epoch_curves;
use crate::ws_client::WsClient;
use eframe::egui;
use egui::Ui;

/// Draw the adversarial loop panel.
pub fn draw_panel(
    ui: &mut Ui,
    state: &mut AdversarialState,
    ws_client: Option<&WsClient>,
) {
    draw_controls(ui, state, ws_client);
    ui.separator();

    // Training curves (cumulative across rounds)
    let has_curves = !state.epoch_history.is_empty() || !state.batch_loss_history.is_empty();
    if has_curves {
        draw_epoch_curves(
            ui,
            &state.epoch_history,
            &state.batch_loss_history,
            state.curve_plot_height,
            &state.round_boundaries,
        );

        // Draggable separator
        let sep_id = ui.id().with("adv_curve_sep");
        let sep_rect = ui.allocate_space(egui::vec2(ui.available_width(), 6.0)).1;
        let sep_response = ui.interact(sep_rect, sep_id, egui::Sense::drag());
        let active = sep_response.hovered() || sep_response.dragged();
        ui.painter().hline(
            sep_rect.x_range(),
            sep_rect.center().y,
            egui::Stroke::new(
                if active { 2.0 } else { 1.0 },
                if active {
                    ui.visuals().widgets.active.fg_stroke.color
                } else {
                    ui.visuals().widgets.noninteractive.bg_stroke.color
                },
            ),
        );
        if active {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        }
        if sep_response.dragged() {
            state.curve_plot_height = (state.curve_plot_height + sep_response.drag_delta().y)
                .clamp(60.0, 500.0);
        }
    }

    // Bottom section: GA fitness chart + archive stats
    draw_ga_charts(ui, &state.ga_history);
}

/// Snapshot of running status fields for UI rendering (avoids borrow issues).
struct RunningSnapshot {
    round: u32,
    n_rounds: u32,
    phase: AdversarialPhase,
    training_epoch: u32,
    training_total_epochs: u32,
    training_step: u32,
    training_steps_in_epoch: u32,
    training_loss: f64,
    training_acc: f64,
    training_gs: i32,
    training_phase: crate::data::TrainingPhase,
    ga_generation: u32,
    ga_best_fitness: f64,
    ga_solvable_rate: f64,
    archive_occupancy: u32,
    archive_total_cells: u32,
}

/// Draw controls: status bar + config button + start/stop.
fn draw_controls(
    ui: &mut Ui,
    state: &mut AdversarialState,
    ws_client: Option<&WsClient>,
) {
    // Snapshot status to avoid borrow conflicts
    enum StatusKind {
        Idle,
        Running(RunningSnapshot),
        Done,
        Error(String),
    }

    let kind = match &state.status {
        AdversarialStatus::Idle => StatusKind::Idle,
        AdversarialStatus::Running {
            round,
            n_rounds,
            phase,
            training_epoch,
            training_total_epochs,
            training_step,
            training_steps_in_epoch,
            training_loss,
            training_acc,
            training_gs,
            training_phase,
            ga_generation,
            ga_best_fitness,
            ga_solvable_rate,
            archive_occupancy,
            archive_total_cells,
            ..
        } => StatusKind::Running(RunningSnapshot {
            round: *round,
            n_rounds: *n_rounds,
            phase: phase.clone(),
            training_epoch: *training_epoch,
            training_total_epochs: *training_total_epochs,
            training_step: *training_step,
            training_steps_in_epoch: *training_steps_in_epoch,
            training_loss: *training_loss,
            training_acc: *training_acc,
            training_gs: *training_gs,
            training_phase: training_phase.clone(),
            ga_generation: *ga_generation,
            ga_best_fitness: *ga_best_fitness,
            ga_solvable_rate: *ga_solvable_rate,
            archive_occupancy: *archive_occupancy,
            archive_total_cells: *archive_total_cells,
        }),
        AdversarialStatus::Done => StatusKind::Done,
        AdversarialStatus::Error(msg) => StatusKind::Error(msg.clone()),
    };

    match kind {
        StatusKind::Idle => {
            ui.horizontal(|ui: &mut Ui| {
                if ui.button("Configure").clicked() {
                    state.show_config = !state.show_config;
                }
                let can_start = ws_client.is_some();
                if ui.add_enabled(can_start, egui::Button::new("Start")).clicked() {
                    if let Some(ws) = ws_client {
                        state.start(ws);
                    }
                }
            });

            draw_config_window(ui, &mut state.config, &mut state.show_config);
        }
        StatusKind::Running(snap) => {
            ui.horizontal(|ui: &mut Ui| {
                ui.strong(format!("Round {}/{}", snap.round + 1, snap.n_rounds));
                ui.separator();

                match &snap.phase {
                    AdversarialPhase::Training => {
                        let progress = if snap.training_total_epochs > 0 {
                            snap.training_epoch as f32 / snap.training_total_epochs as f32
                        } else {
                            0.0
                        };
                        ui.add(
                            egui::ProgressBar::new(progress)
                                .text(format!(
                                    "Epoch {}/{}",
                                    snap.training_epoch, snap.training_total_epochs
                                ))
                                .desired_width(120.0),
                        );

                        match &snap.training_phase {
                            crate::data::TrainingPhase::Training => {
                                let step_progress = if snap.training_steps_in_epoch > 0 {
                                    snap.training_step as f32
                                        / snap.training_steps_in_epoch as f32
                                } else {
                                    0.0
                                };
                                ui.separator();
                                ui.add(
                                    egui::ProgressBar::new(step_progress)
                                        .text(format!(
                                            "Step {}/{}",
                                            snap.training_step, snap.training_steps_in_epoch
                                        ))
                                        .desired_width(120.0),
                                );
                                ui.separator();
                                ui.label(format!("Loss: {:.3}", snap.training_loss));
                                ui.separator();
                                ui.label(format!("Acc: {:.3}", snap.training_acc));
                                if snap.training_gs > 0 {
                                    ui.separator();
                                    ui.label(format!("GS: {}", snap.training_gs));
                                }
                            }
                            crate::data::TrainingPhase::Status(text) => {
                                ui.separator();
                                ui.spinner();
                                ui.label(text);
                            }
                        }
                    }
                    AdversarialPhase::GA { grid_size } => {
                        ui.label(format!("GA gs={grid_size}"));
                        ui.separator();
                        ui.label(format!("Gen {}", snap.ga_generation));
                        ui.separator();
                        ui.label(format!("Best: {:.3}", snap.ga_best_fitness));
                        ui.separator();
                        ui.label(format!("Solvable: {:.0}%", snap.ga_solvable_rate * 100.0));

                        if snap.archive_total_cells > 0 {
                            ui.separator();
                            ui.label(format!(
                                "Archive: {}/{}",
                                snap.archive_occupancy, snap.archive_total_cells
                            ));
                        }
                    }
                }

                // Right-align Stop button
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Stop").clicked() {
                        if let Some(ws) = ws_client {
                            state.stop(ws);
                        }
                    }
                });
            });
        }
        StatusKind::Done => {
            ui.horizontal(|ui: &mut Ui| {
                ui.colored_label(egui::Color32::GREEN, "Adversarial loop complete");
                if ui.button("Reset").clicked() {
                    state.status = AdversarialStatus::Idle;
                }
            });
        }
        StatusKind::Error(msg) => {
            ui.horizontal(|ui: &mut Ui| {
                ui.colored_label(egui::Color32::RED, format!("Error: {msg}"));
                if ui.button("Dismiss").clicked() {
                    state.status = AdversarialStatus::Idle;
                }
            });
        }
    }
}

/// Draw the configuration window.
fn draw_config_window(ui: &mut Ui, config: &mut AdversarialConfig, show: &mut bool) {
    egui::Window::new("Adversarial Loop Config")
        .open(show)
        .resizable(false)
        .default_width(280.0)
        .show(ui.ctx(), |ui| {
            egui::Grid::new("adv_config_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Rounds:");
                    ui.add(egui::DragValue::new(&mut config.n_rounds).range(1..=20));
                    ui.end_row();

                    ui.label("Epochs/round:");
                    ui.add(egui::DragValue::new(&mut config.epochs_per_round).range(1..=100));
                    ui.end_row();

                    ui.label("Batch size:");
                    ui.add(egui::DragValue::new(&mut config.batch_size).range(64..=8192));
                    ui.end_row();

                    ui.label("Learning rate:");
                    ui.add(
                        egui::DragValue::new(&mut config.lr)
                            .range(1e-6..=1e-1)
                            .speed(1e-5)
                            .max_decimals(6),
                    );
                    ui.end_row();

                    ui.label("Seed:");
                    ui.add(egui::DragValue::new(&mut config.seed).range(0..=9999));
                    ui.end_row();
                });

            ui.separator();
            ui.label("GA settings:");
            egui::Grid::new("adv_ga_config_grid")
                .num_columns(2)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Population:");
                    ui.add(egui::DragValue::new(&mut config.ga_config.pop_size).range(4..=512));
                    ui.end_row();

                    ui.label("Generations:");
                    ui.add(
                        egui::DragValue::new(&mut config.ga_config.generations).range(1..=1000),
                    );
                    ui.end_row();

                    ui.label("Target log WP:");
                    ui.add(
                        egui::DragValue::new(&mut config.target_log_wp)
                            .range(-10.0..=0.0)
                            .speed(0.1)
                            .max_decimals(1),
                    );
                    ui.end_row();

                    ui.label("BFS bins:");
                    ui.add(
                        egui::DragValue::new(&mut config.archive_bfs_bins).range(5..=50),
                    );
                    ui.end_row();

                    ui.label("States bins:");
                    ui.add(
                        egui::DragValue::new(&mut config.archive_states_bins).range(5..=50),
                    );
                    ui.end_row();
                });
        });
}

/// Draw GA fitness history.
fn draw_ga_charts(ui: &mut Ui, ga_history: &[(u32, f64, f64)]) {
    if ga_history.is_empty() {
        return;
    }

    ui.label("GA Fitness");
    let best_line: egui_plot::PlotPoints = ga_history
        .iter()
        .map(|(g, best, _)| [*g as f64, *best])
        .collect();
    let avg_line: egui_plot::PlotPoints = ga_history
        .iter()
        .map(|(g, _, avg)| [*g as f64, *avg])
        .collect();

    let plot = egui_plot::Plot::new("adv_ga_fitness")
        .height(180.0)
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
