use crate::data::{DataStore, TrainingStatus};
use eframe::egui;
use egui::Ui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use std::path::Path;

/// Precomputed point data for the scatter plot.
struct ScatterPoint {
    row_idx: usize,
    win_prob: f64,
    accuracy: f64,
    grid_size: i32,
}

/// Draw the training tab content. Returns Some(row_idx) if a point was clicked.
pub fn draw_training_panel(ui: &mut Ui, store: &mut DataStore, maze_dir: &Path) -> Option<usize> {
    // Training controls section
    draw_training_controls(ui, store, maze_dir);
    ui.separator();

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
        points.push(ScatterPoint {
            row_idx: i,
            win_prob: analysis.win_prob,
            accuracy: metric.accuracy,
            grid_size: row.level.grid_size,
        });
    }

    let mut clicked_row = None;

    // Scatter plot
    let plot = Plot::new("training_scatter")
        .x_axis_label("Random Win%")
        .y_axis_label("Agent Accuracy")
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
                .map(|p| [p.win_prob, p.accuracy])
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
                Points::new(vec![[p.win_prob, p.accuracy]])
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
            let dy = p.accuracy - plot_pos.y;
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
                ui.label(format!("Agent Accuracy: {:.1}%", metric.accuracy * 100.0));
                ui.separator();
                ui.label(format!("Mean Loss: {:.3}", metric.mean_loss));
            });
            ui.horizontal(|ui: &mut Ui| {
                if let Some(a) = &row.analysis {
                    ui.label(format!("States: {}", a.n_states));
                    ui.separator();
                    ui.label(format!("BFS Moves: {}", row.bfs_moves.map(|m| m.to_string()).unwrap_or("-".into())));
                    ui.separator();
                    ui.label(format!("Win% (random): {:.1}%", a.win_prob * 100.0));
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
fn draw_training_controls(ui: &mut Ui, store: &mut DataStore, maze_dir: &Path) {
    match &store.training_status {
        TrainingStatus::Idle => {
            ui.horizontal(|ui: &mut Ui| {
                ui.label("Epochs:");
                let mut epochs = store.training_config.epochs;
                if ui
                    .add(egui::DragValue::new(&mut epochs).range(1..=100))
                    .changed()
                {
                    store.training_config.epochs = epochs;
                }

                ui.separator();
                ui.label("Batch:");
                let mut bs = store.training_config.batch_size;
                if ui
                    .add(egui::DragValue::new(&mut bs).range(64..=8192))
                    .changed()
                {
                    store.training_config.batch_size = bs;
                }

                ui.separator();
                ui.label("LR:");
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

                ui.separator();
                ui.label("Seed:");
                let mut seed = store.training_config.seed;
                if ui
                    .add(egui::DragValue::new(&mut seed).range(0..=9999))
                    .changed()
                {
                    store.training_config.seed = seed;
                }

                ui.separator();
                ui.checkbox(&mut store.training_config.wandb, "W&B");

                ui.separator();
                if ui.button("Start Training").clicked() {
                    store.start_training(maze_dir);
                }
            });
        }
        TrainingStatus::Running {
            epoch,
            total_epochs,
            step,
            loss,
            acc,
            gs,
        } => {
            let epoch = *epoch;
            let total_epochs = *total_epochs;
            let step = *step;
            let loss = *loss;
            let acc = *acc;
            let gs = *gs;

            ui.horizontal(|ui: &mut Ui| {
                // Progress bar
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

                ui.separator();
                ui.label(format!("Step: {step}"));
                ui.separator();
                ui.label(format!("Loss: {loss:.3}"));
                ui.separator();
                ui.label(format!("Acc: {acc:.3}"));
                if gs > 0 {
                    ui.separator();
                    ui.label(format!("GS: {gs}"));
                }

                ui.separator();
                if ui.button("Stop").clicked() {
                    store.stop_training();
                }
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
