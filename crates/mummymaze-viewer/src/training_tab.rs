use crate::data::{DataStore, EpochRecord, TrainingPhase, TrainingStatus};
use eframe::egui;
use egui::Ui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use crate::ws_client::WsClient;

/// Precomputed point data for the scatter plot.
struct ScatterPoint {
    row_idx: usize,
    win_prob: f64,
    agent_win_prob: f64,
    grid_size: i32,
}

/// Draw the training tab content. Returns Some(row_idx) if a point was clicked.
pub fn draw_training_panel(
    ui: &mut Ui,
    store: &mut DataStore,
    model_server: Option<&WsClient>,
) -> Option<usize> {
    // Training controls section
    draw_training_controls(ui, store, model_server);
    ui.separator();

    // Epoch history curves with draggable divider
    let has_curves = !store.epoch_history.is_empty() || !store.batch_loss_history.is_empty();
    if has_curves {
        draw_epoch_curves(ui, &store.epoch_history, &store.batch_loss_history, store.curve_plot_height, &[]);

        // Draggable separator
        let sep_id = ui.id().with("curve_sep");
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
            store.curve_plot_height = (store.curve_plot_height + sep_response.drag_delta().y)
                .clamp(60.0, 500.0);
        }
    }

    let has_training = store
        .training_metrics
        .as_ref()
        .is_some_and(|tm| !tm.levels.is_empty());

    if !has_training {
        ui.centered_and_justified(|ui: &mut Ui| {
            ui.heading("No training data yet");
        });
        return None;
    }

    let tm = store.training_metrics.as_ref().unwrap();

    // Header
    ui.horizontal(|ui: &mut Ui| {
        ui.label(format!("Run: {}", tm.run_id));
        ui.separator();
        ui.label(format!("Step: {}", tm.step));
        ui.separator();
        ui.label(format!("{} levels", tm.levels.len()));
    });
    ui.separator();

    // Build scatter data
    let mut points: Vec<ScatterPoint> = Vec::new();
    for (i, row) in store.rows.iter().enumerate() {
        let Some(analysis) = &row.analysis else {
            continue;
        };
        let Some(metric) = tm.get(&row.file_stem, row.sublevel) else {
            continue;
        };
        let Some(agent_wp) = metric.agent_win_prob else {
            continue;
        };
        points.push(ScatterPoint {
            row_idx: i,
            win_prob: analysis.win_prob,
            agent_win_prob: agent_wp,
            grid_size: row.level.grid_size,
        });
    }

    let mut clicked_row = None;

    // Scatter plot
    let plot = Plot::new("training_scatter")
        .x_axis_label("Random Win%")
        .y_axis_label("Agent Win%")
        .data_aspect(1.0)
        .allow_boxed_zoom(true)
        .allow_drag(true)
        .allow_scroll(true);

    let plot_response = plot.show(ui, |plot_ui| {
        // Diagonal reference line y = x
        let line_points: PlotPoints = (0..=100)
            .map(|i| {
                let v = i as f64 / 100.0;
                [v, v]
            })
            .collect();
        plot_ui.line(
            Line::new(line_points)
                .color(egui::Color32::from_gray(100))
                .width(1.0)
                .name("y = x"),
        );

        // Separate points by grid size
        let colors = [
            (6, egui::Color32::from_rgb(70, 130, 230)),  // blue
            (8, egui::Color32::from_rgb(50, 180, 80)),    // green
            (10, egui::Color32::from_rgb(220, 70, 70)),   // red
        ];

        for (gs, color) in colors {
            let pts: Vec<[f64; 2]> = points
                .iter()
                .filter(|p| p.grid_size == gs)
                .map(|p| [p.win_prob, p.agent_win_prob])
                .collect();

            if !pts.is_empty() {
                plot_ui.points(
                    Points::new(pts)
                        .color(color)
                        .radius(3.0)
                        .name(format!("Grid {gs}")),
                );
            }
        }

        // Highlight selected level
        if let Some(sel) = store.selected
            && let Some(p) = points.iter().find(|p| p.row_idx == sel)
        {
            plot_ui.points(
                Points::new(vec![[p.win_prob, p.agent_win_prob]])
                    .color(egui::Color32::YELLOW)
                    .radius(6.0)
                    .name("Selected"),
            );
        }
    });

    // Handle click-to-select
    if plot_response.response.clicked()
        && let Some(hover_pos) = plot_response.response.hover_pos()
    {
        let plot_pos = plot_response.transform.value_from_position(hover_pos);
        // Find nearest point
        let mut best_dist = f64::MAX;
        let mut best_idx = None;
        for p in &points {
            let dx = p.win_prob - plot_pos.x;
            let dy = p.agent_win_prob - plot_pos.y;
            let dist = dx * dx + dy * dy;
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(p.row_idx);
            }
        }
        // Only select if reasonably close (within 5% of axis range)
        if best_dist < 0.05 * 0.05 {
            clicked_row = best_idx;
        }
    }

    // Selected level stats panel
    ui.separator();
    if let Some(sel) = store.selected {
        let row = &store.rows[sel];
        if let Some(metric) = tm.get(&row.file_stem, row.sublevel) {
            ui.horizontal(|ui: &mut Ui| {
                if let Some(wp) = metric.agent_win_prob {
                    ui.label(format!("Win% (agent): {:.1}%", wp * 100.0));
                    ui.separator();
                }
                ui.label(format!("Accuracy: {:.1}%", metric.accuracy * 100.0));
                ui.separator();
                ui.label(format!("Mean Loss: {:.3}", metric.mean_loss));
            });
            ui.horizontal(|ui: &mut Ui| {
                if let Some(a) = &row.analysis {
                    ui.label(format!("States: {}", a.n_states));
                    ui.separator();
                    ui.label(format!("BFS Moves: {}", row.bfs_moves.map(|m| m.to_string()).unwrap_or("-".into())));
                    ui.separator();
                    let wp_pct = a.win_prob * 100.0;
                    let wp = if a.win_prob != 0.0 && wp_pct.abs() < 0.01 {
                        format!("{:.2e}%", wp_pct)
                    } else {
                        format!("{:.1}%", wp_pct)
                    };
                    ui.label(format!("Win% (random): {wp}"));
                    ui.separator();
                    ui.label(format!("Dead-end%: {:.1}%", a.difficulty.dead_end_ratio * 100.0));
                }
            });
        } else {
            ui.label("No training data for selected level");
        }
    } else {
        ui.label("Select a level to see stats");
    }

    clicked_row
}

/// Draw training controls: config form when idle, status + stop when running.
fn draw_training_controls(ui: &mut Ui, store: &mut DataStore, model_server: Option<&WsClient>) {
    match &store.training_status {
        TrainingStatus::Idle => {
            ui.horizontal(|ui: &mut Ui| {
                if ui.button("Configure").clicked() {
                    store.show_training_config = !store.show_training_config;
                }
                let can_start = model_server.is_some();
                if ui.add_enabled(can_start, egui::Button::new("Start")).clicked() {
                    if let Some(ms) = model_server {
                        store.start_training(ms);
                    }
                }
            });

            egui::Window::new("Training Config")
                .open(&mut store.show_training_config)
                .resizable(false)
                .default_width(200.0)
                .show(ui.ctx(), |ui| {
                    egui::Grid::new("training_config_grid")
                        .num_columns(2)
                        .spacing([8.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Epochs:");
                            let mut epochs = store.training_config.epochs;
                            if ui
                                .add(egui::DragValue::new(&mut epochs).range(1..=100))
                                .changed()
                            {
                                store.training_config.epochs = epochs;
                            }
                            ui.end_row();

                            ui.label("Batch size:");
                            let mut bs = store.training_config.batch_size;
                            if ui
                                .add(egui::DragValue::new(&mut bs).range(64..=8192))
                                .changed()
                            {
                                store.training_config.batch_size = bs;
                            }
                            ui.end_row();

                            ui.label("Learning rate:");
                            let mut lr = store.training_config.lr;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut lr)
                                        .range(1e-6..=1e-1)
                                        .speed(1e-5)
                                        .max_decimals(6),
                                )
                                .changed()
                            {
                                store.training_config.lr = lr;
                            }
                            ui.end_row();

                            ui.label("Seed:");
                            let mut seed = store.training_config.seed;
                            if ui
                                .add(egui::DragValue::new(&mut seed).range(0..=9999))
                                .changed()
                            {
                                store.training_config.seed = seed;
                            }
                            ui.end_row();

                            ui.label("W&B:");
                            ui.checkbox(&mut store.training_config.wandb, "");
                            ui.end_row();
                        });
                });
        }
        TrainingStatus::Running {
            epoch,
            total_epochs,
            epoch_step,
            steps_in_epoch,
            loss,
            acc,
            gs,
            phase,
            ..
        } => {
            let epoch = *epoch;
            let total_epochs = *total_epochs;
            let epoch_step = *epoch_step;
            let steps_in_epoch = *steps_in_epoch;
            let loss = *loss;
            let acc = *acc;
            let gs = *gs;
            let phase = phase.clone();

            ui.horizontal(|ui: &mut Ui| {
                // Epoch progress bar
                let progress = if total_epochs > 0 {
                    epoch as f32 / total_epochs as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(progress)
                        .text(format!("Epoch {epoch}/{total_epochs}"))
                        .desired_width(150.0),
                );

                match &phase {
                    TrainingPhase::Training => {
                        ui.separator();
                        let step_progress = if steps_in_epoch > 0 {
                            epoch_step as f32 / steps_in_epoch as f32
                        } else {
                            0.0
                        };
                        ui.add(
                            egui::ProgressBar::new(step_progress)
                                .text(format!("Step {epoch_step}/{steps_in_epoch}"))
                                .desired_width(150.0),
                        );
                        ui.separator();
                        ui.label(format!("Loss: {loss:.3}"));
                        ui.separator();
                        ui.label(format!("Acc: {acc:.3}"));
                        if gs > 0 {
                            ui.separator();
                            ui.label(format!("GS: {gs}"));
                        }
                    }
                    TrainingPhase::Status(text) => {
                        ui.separator();
                        ui.spinner();
                        ui.label(text);
                    }
                }

                // Right-align Stop button so it doesn't shift when content changes
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Stop").clicked() {
                        if let Some(ms) = model_server {
                            store.stop_training(ms);
                        }
                    }
                });
            });
        }
        TrainingStatus::Done => {
            ui.horizontal(|ui: &mut Ui| {
                ui.colored_label(egui::Color32::GREEN, "Training complete");
                if ui.button("Reset").clicked() {
                    store.training_status = TrainingStatus::Idle;
                }
            });
        }
        TrainingStatus::Error(msg) => {
            let msg = msg.clone();
            ui.horizontal(|ui: &mut Ui| {
                ui.colored_label(egui::Color32::RED, format!("Error: {msg}"));
                if ui.button("Dismiss").clicked() {
                    store.training_status = TrainingStatus::Idle;
                }
            });
        }
    }
}

/// Draw loss curves, accuracy curves, and batch loss over epochs.
/// `round_boundaries` contains epoch numbers where new rounds start (vertical lines).
pub fn draw_epoch_curves(
    ui: &mut Ui,
    history: &[EpochRecord],
    batch_loss: &[[f64; 2]],
    height: f32,
    round_boundaries: &[usize],
) {
    let train_color = egui::Color32::from_rgb(70, 130, 230); // blue
    let val_color = egui::Color32::from_rgb(230, 150, 50); // orange

    ui.columns(3, |cols| {
        // Left: Epoch Loss
        let loss_plot = Plot::new("epoch_loss")
            .x_axis_label("Epoch")
            .y_axis_label("Loss")
            .height(height)
            .allow_drag(true)
            .allow_scroll(true)
            .legend(egui_plot::Legend::default());

        loss_plot.show(&mut cols[0], |plot_ui| {
            let train: PlotPoints = history
                .iter()
                .map(|r| [r.epoch as f64, r.train_loss])
                .collect();
            let val: PlotPoints = history
                .iter()
                .map(|r| [r.epoch as f64, r.val_loss])
                .collect();
            plot_ui.line(Line::new(train).color(train_color).name("Train"));
            plot_ui.line(Line::new(val).color(val_color).name("Val"));
            draw_round_vlines(plot_ui, round_boundaries);
        });

        // Middle: Epoch Accuracy
        let acc_plot = Plot::new("epoch_acc")
            .x_axis_label("Epoch")
            .y_axis_label("Accuracy")
            .height(height)
            .allow_drag(true)
            .allow_scroll(true)
            .legend(egui_plot::Legend::default());

        acc_plot.show(&mut cols[1], |plot_ui| {
            let train: PlotPoints = history
                .iter()
                .map(|r| [r.epoch as f64, r.train_acc])
                .collect();
            let val: PlotPoints = history
                .iter()
                .map(|r| [r.epoch as f64, r.val_acc])
                .collect();
            plot_ui.line(Line::new(train).color(train_color).name("Train"));
            plot_ui.line(Line::new(val).color(val_color).name("Val"));
            draw_round_vlines(plot_ui, round_boundaries);
        });

        // Right: Batch Loss
        let batch_plot = Plot::new("batch_loss")
            .x_axis_label("Step")
            .y_axis_label("Loss")
            .height(height)
            .allow_drag(true)
            .allow_scroll(true);

        batch_plot.show(&mut cols[2], |plot_ui| {
            let pts: PlotPoints = batch_loss.iter().copied().collect();
            plot_ui.line(Line::new(pts).color(train_color).name("Batch Loss"));
        });
    });
}

/// Draw dashed vertical lines at round boundaries on epoch-axis plots.
fn draw_round_vlines(plot_ui: &mut egui_plot::PlotUi, boundaries: &[usize]) {
    let boundary_color = egui::Color32::from_rgba_premultiplied(180, 180, 180, 120);
    for &epoch in boundaries {
        let x = epoch as f64;
        let vline = egui_plot::VLine::new(x)
            .color(boundary_color)
            .width(1.0)
            .style(egui_plot::LineStyle::dashed_dense());
        plot_ui.vline(vline);
    }
}
