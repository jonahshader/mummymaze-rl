//! UI for the adversarial training loop tab.

use crate::adversarial::{AdversarialConfig, AdversarialPhase, AdversarialState, AdversarialStatus};
use crate::data::LevelRow;
use crate::training_tab::draw_epoch_curves;
use eframe::egui;
use egui::Ui;
use mummymaze::ga::archive::ArchiveSnapshot;

/// Draw the adversarial loop panel.
pub fn draw_panel(ui: &mut Ui, state: &mut AdversarialState, rows: &[LevelRow], maze_dir: &std::path::Path) {
    draw_controls(ui, state, rows, maze_dir);
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

    // Bottom section: heatmap (left) + GA fitness chart (right)
    ui.columns(2, |cols| {
        // Left: Archive heatmap
        draw_archive_heatmap(&mut cols[0], state.archive_snapshot.as_ref(), state.config.target_log_wp);

        // Right: GA fitness chart + archive occupancy
        draw_ga_charts(&mut cols[1], &state.ga_history, &state.archive_occupancy_history);
    });
}

/// Snapshot of running status fields for UI rendering (avoids borrow issues).
struct RunningSnapshot {
    round: usize,
    n_rounds: usize,
    phase: AdversarialPhase,
    training_epoch: u32,
    training_total_epochs: u32,
    training_step: u32,
    training_steps_in_epoch: u32,
    training_loss: f64,
    training_acc: f64,
    training_gs: i32,
    training_phase: crate::data::TrainingPhase,
    ga_generation: usize,
    ga_total_generations: usize,
    archive_occ: Option<(usize, usize)>,
}

/// Draw controls: status bar + config button + start/stop.
fn draw_controls(ui: &mut Ui, state: &mut AdversarialState, rows: &[LevelRow], maze_dir: &std::path::Path) {
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
            ga_total_generations,
        } => StatusKind::Running(RunningSnapshot {
            round: *round,
            n_rounds: state.config.n_rounds,
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
            ga_total_generations: *ga_total_generations,
            archive_occ: state.archive_snapshot.as_ref().map(|snap| {
                let occupied = snap.cells.iter().filter(|c| c.is_some()).count();
                (occupied, snap.cells.len())
            }),
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
                if ui.button("Start").clicked() {
                    state.start(maze_dir, rows);
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
                    AdversarialPhase::GA { grid_size, .. } => {
                        let progress = if snap.ga_total_generations > 0 {
                            snap.ga_generation as f32 / snap.ga_total_generations as f32
                        } else {
                            0.0
                        };
                        ui.label(format!("GA gs={grid_size}"));
                        ui.separator();
                        ui.add(
                            egui::ProgressBar::new(progress)
                                .text(format!(
                                    "Gen {}/{}",
                                    snap.ga_generation, snap.ga_total_generations
                                ))
                                .desired_width(150.0),
                        );

                        if let Some((occupied, total)) = snap.archive_occ {
                            ui.separator();
                            ui.label(format!("Archive: {occupied}/{total}"));
                        }
                    }
                }

                // Right-align Stop button
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Stop").clicked() {
                        state.stop();
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

/// Draw the MAP-Elites archive heatmap.
fn draw_archive_heatmap(ui: &mut Ui, snapshot: Option<&ArchiveSnapshot>, target_log_wp: f64) {
    ui.label("MAP-Elites Archive");

    let Some(snap) = snapshot else {
        ui.colored_label(egui::Color32::GRAY, "No archive data yet");
        return;
    };

    let occupied = snap.cells.iter().filter(|c| c.is_some()).count();
    let total = snap.cells.len();
    ui.label(format!("{occupied}/{total} cells occupied"));

    let available = ui.available_size();
    let side = available.x.min(available.y).min(300.0);
    let size = egui::Vec2::new(side, side);
    let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
    let rect = response.rect;

    let cell_w = rect.width() / snap.bfs_bins as f32;
    let cell_h = rect.height() / snap.states_bins as f32;

    for bi in 0..snap.bfs_bins {
        for si in 0..snap.states_bins {
            let idx = bi * snap.states_bins + si;
            let x = rect.left() + bi as f32 * cell_w;
            // Invert Y so states_bins increases upward
            let y = rect.bottom() - (si + 1) as f32 * cell_h;
            let cell_rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell_w, cell_h));

            let color = match &snap.cells[idx] {
                None => egui::Color32::from_gray(40),
                Some(cell) => {
                    let dist = (cell.log_policy_wp - target_log_wp).abs();
                    // Map distance to color: 0 = green, >2 = red
                    let t = (dist / 2.0).min(1.0) as f32;
                    let r = (t * 220.0) as u8 + 30;
                    let g = ((1.0 - t) * 200.0) as u8 + 30;
                    egui::Color32::from_rgb(r, g, 50)
                }
            };

            painter.rect_filled(cell_rect, 0.0, color);
        }
    }

    // Grid lines
    let grid_color = egui::Color32::from_gray(60);
    for i in 0..=snap.bfs_bins {
        let x = rect.left() + i as f32 * cell_w;
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(0.5, grid_color),
        );
    }
    for i in 0..=snap.states_bins {
        let y = rect.top() + i as f32 * cell_h;
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(0.5, grid_color),
        );
    }

    // Tooltip on hover
    if let Some(hover_pos) = response.hover_pos() {
        let bi = ((hover_pos.x - rect.left()) / cell_w) as usize;
        let si = snap.states_bins.saturating_sub(1)
            - ((hover_pos.y - rect.top()) / cell_h).min(snap.states_bins as f32 - 1.0) as usize;
        if bi < snap.bfs_bins && si < snap.states_bins {
            let idx = bi * snap.states_bins + si;
            if let Some(ref cell) = snap.cells[idx] {
                response.clone().on_hover_ui_at_pointer(|ui: &mut Ui| {
                    ui.label(format!("BFS moves: {}", cell.bfs_moves));
                    ui.label(format!("States: {}", cell.n_states));
                    ui.label(format!("log(policy WP): {:.2}", cell.log_policy_wp));
                    ui.label(format!(
                        "|dist to target|: {:.2}",
                        (cell.log_policy_wp - target_log_wp).abs()
                    ));
                });
            }
        }
    }

    // Axis labels
    ui.horizontal(|ui| {
        ui.label(format!("BFS: {}..{}", snap.bfs_range.0, snap.bfs_range.1));
        ui.separator();
        ui.label(format!("States: {}..{}", snap.states_range.0, snap.states_range.1));
    });
}

/// Draw GA fitness history and archive occupancy.
fn draw_ga_charts(
    ui: &mut Ui,
    ga_history: &[(usize, f64, f64)],
    occupancy_history: &[(usize, usize)],
) {
    if !ga_history.is_empty() {
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

    if !occupancy_history.is_empty() {
        ui.add_space(8.0);
        ui.label("Archive Occupancy");
        let occ_points: egui_plot::PlotPoints = occupancy_history
            .iter()
            .enumerate()
            .map(|(i, (_, occ))| [i as f64, *occ as f64])
            .collect();

        let plot = egui_plot::Plot::new("adv_archive_occ")
            .height(100.0)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .x_axis_label("Update")
            .y_axis_label("Occupied");

        plot.show(ui, |plot_ui| {
            plot_ui.line(
                egui_plot::Line::new(occ_points)
                    .color(egui::Color32::from_rgb(100, 200, 100)),
            );
        });
    }

    if ga_history.is_empty() && occupancy_history.is_empty() {
        ui.colored_label(egui::Color32::GRAY, "No GA data yet");
    }
}
