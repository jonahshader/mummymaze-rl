//! UI for the adversarial training loop tab.

use crate::adversarial::{
    AdversarialConfig, AdversarialPhase, AdversarialState, AdversarialStatus, LoadedArchive,
    load_archives_from_dir,
};
use crate::training_tab::draw_epoch_curves;
use crate::ws_client::WsClient;
use eframe::egui;
use egui::Ui;
use mummymaze::parse::Level;
use std::path::Path;

/// Draw the adversarial loop panel. Returns a level if the user clicked an archive cell.
pub fn draw_panel(
    ui: &mut Ui,
    state: &mut AdversarialState,
    ws_client: Option<&WsClient>,
) -> Option<Level> {
    draw_controls(ui, state, ws_client);
    ui.separator();

    // Archive loader + heatmap
    let clicked_level = draw_archive_section(ui, state);

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

    // Bottom section: GA fitness chart
    draw_ga_charts(ui, &state.ga_history);

    clicked_level
}

/// Archive loader bar + selector + heatmap. Returns clicked level if any.
fn draw_archive_section(ui: &mut Ui, state: &mut AdversarialState) -> Option<Level> {
    // Load from disk controls
    ui.horizontal(|ui: &mut Ui| {
        ui.label("Archive dir:");
        ui.add(
            egui::TextEdit::singleline(&mut state.archive_dir)
                .desired_width(200.0),
        );
        if ui.button("Load").clicked() {
            let dir = Path::new(&state.archive_dir);
            let archives = load_archives_from_dir(dir);
            if archives.is_empty() {
                state.log_messages.push(format!(
                    "No archive_gs*.json files found in {}",
                    dir.display()
                ));
            } else {
                state.log_messages.push(format!(
                    "Loaded {} archives from {}",
                    archives.len(),
                    dir.display()
                ));
                state.loaded_archives = archives;
                state.selected_archive_idx = 0;
            }
        }
    });

    if state.loaded_archives.is_empty() {
        ui.colored_label(egui::Color32::GRAY, "No archive data");
        return None;
    }

    // Build unique grid sizes and rounds for selectors
    let grid_sizes: Vec<i32> = {
        let mut gs: Vec<i32> = state
            .loaded_archives
            .iter()
            .map(|a| a.grid_size)
            .collect();
        gs.sort();
        gs.dedup();
        gs
    };
    let rounds: Vec<usize> = {
        let mut rs: Vec<usize> = state
            .loaded_archives
            .iter()
            .map(|a| a.round)
            .collect();
        rs.sort();
        rs.dedup();
        rs
    };

    // Current selection
    let cur_gs = state
        .loaded_archives
        .get(state.selected_archive_idx)
        .map(|a| a.grid_size)
        .unwrap_or(0);
    let cur_round = state
        .loaded_archives
        .get(state.selected_archive_idx)
        .map(|a| a.round)
        .unwrap_or(0);

    let mut selected_gs = cur_gs;
    let mut selected_round = cur_round;

    ui.horizontal(|ui: &mut Ui| {
        // Grid size tabs
        for &gs in &grid_sizes {
            if ui
                .selectable_label(gs == cur_gs, format!("gs={gs}"))
                .clicked()
            {
                selected_gs = gs;
            }
        }
        ui.separator();
        // Round selector
        ui.label("Round:");
        egui::ComboBox::from_id_salt("archive_round_select")
            .selected_text(format!("{selected_round}"))
            .width(60.0)
            .show_ui(ui, |ui| {
                for &r in &rounds {
                    if ui
                        .selectable_value(&mut selected_round, r, format!("{r}"))
                        .clicked()
                    {
                        selected_round = r;
                    }
                }
            });
    });

    // Update selected index if user changed gs or round
    if selected_gs != cur_gs || selected_round != cur_round {
        // Find best match: exact (round, gs), or fallback to any with matching gs
        if let Some(pos) = state
            .loaded_archives
            .iter()
            .position(|a| a.round == selected_round && a.grid_size == selected_gs)
        {
            state.selected_archive_idx = pos;
        } else if let Some(pos) = state
            .loaded_archives
            .iter()
            .position(|a| a.grid_size == selected_gs)
        {
            state.selected_archive_idx = pos;
        }
    }

    // Draw the heatmap for the selected archive
    let idx = state.selected_archive_idx;
    if idx < state.loaded_archives.len() {
        draw_archive_heatmap(ui, &state.loaded_archives[idx])
    } else {
        None
    }
}

/// Draw the MAP-Elites archive heatmap. Returns the clicked cell's level if any.
fn draw_archive_heatmap(ui: &mut Ui, archive: &LoadedArchive) -> Option<Level> {
    let occupied = archive.cells.iter().filter(|c| c.is_some()).count();
    let total = archive.cells.len();
    ui.label(format!(
        "Archive: {occupied}/{total} cells (gs={}, round {})",
        archive.grid_size, archive.round
    ));

    let available = ui.available_size();
    let side = available.x.min(available.y).min(350.0);
    let size = egui::Vec2::new(side, side);
    let (response, painter) = ui.allocate_painter(size, egui::Sense::click());
    let rect = response.rect;

    let cell_w = rect.width() / archive.bfs_bins as f32;
    let cell_h = rect.height() / archive.states_bins as f32;

    for bi in 0..archive.bfs_bins {
        for si in 0..archive.states_bins {
            let idx = bi * archive.states_bins + si;
            let x = rect.left() + bi as f32 * cell_w;
            // Invert Y so states increases upward
            let y = rect.bottom() - (si + 1) as f32 * cell_h;
            let cell_rect =
                egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell_w, cell_h));

            let color = match &archive.cells[idx] {
                None => egui::Color32::from_gray(40),
                Some(cell) => {
                    let dist = (cell.log_policy_win_prob - archive.target_log_wp).abs();
                    // 0 = green (close to target), >2 = red (far)
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
    for i in 0..=archive.bfs_bins {
        let x = rect.left() + i as f32 * cell_w;
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(0.5, grid_color),
        );
    }
    for i in 0..=archive.states_bins {
        let y = rect.top() + i as f32 * cell_h;
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(0.5, grid_color),
        );
    }

    // Axis labels
    ui.horizontal(|ui| {
        ui.label(format!(
            "BFS: {}..{}",
            archive.bfs_range.0, archive.bfs_range.1
        ));
        ui.separator();
        ui.label(format!(
            "States: {}..{}",
            archive.states_range.0, archive.states_range.1
        ));
    });

    // Hover/click handling
    let mut clicked_level = None;

    if let Some(pos) = response.hover_pos() {
        let bi = ((pos.x - rect.left()) / cell_w) as usize;
        let si = archive
            .states_bins
            .saturating_sub(1)
            .min(((rect.bottom() - pos.y) / cell_h) as usize);
        if bi < archive.bfs_bins && si < archive.states_bins {
            let idx = bi * archive.states_bins + si;
            if let Some(ref cell) = archive.cells[idx] {
                // Tooltip
                response.clone().on_hover_ui_at_pointer(|ui: &mut Ui| {
                    ui.label(format!("BFS moves: {}", cell.bfs_moves));
                    ui.label(format!("States: {}", cell.n_states));
                    ui.label(format!(
                        "log(policy WP): {:.2}",
                        cell.log_policy_win_prob
                    ));
                    ui.label(format!("Fitness: {:.3}", cell.fitness));
                    ui.label(format!(
                        "|dist to target|: {:.2}",
                        (cell.log_policy_win_prob - archive.target_log_wp).abs()
                    ));
                });

                // Click to load
                if response.clicked() {
                    clicked_level = Some(cell.level.clone());
                }
            }
        }
    }

    clicked_level
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
