//! Adversarial training loop state — thin wrapper over server-side loop.
//!
//! The Python WS server runs the full adversarial loop (training + GA).
//! This module sends start/stop commands and displays progress from events.

use crate::data::{EpochRecord, TrainingPhase};
use crate::ws_client::{AdversarialEvent, WsClient};
use mummymaze::event_types::TrainingEvent;
use mummymaze::ga::GaConfig;

/// Configuration for the adversarial training loop.
#[derive(Clone)]
pub struct AdversarialConfig {
    pub n_rounds: usize,
    pub epochs_per_round: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub seed: u32,
    pub ga_config: GaConfig,
    pub target_log_wp: f64,
    pub archive_bfs_bins: usize,
    pub archive_states_bins: usize,
    pub grid_sizes: Vec<i32>,
}

impl AdversarialConfig {
    /// Build the fitness expression from the target log win probability.
    pub fn fitness_expr(&self) -> String {
        format!("-abs(log_policy_win_prob - ({}))", self.target_log_wp)
    }

    /// Serialize to JSON for the WS server.
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "n_rounds": self.n_rounds,
            "epochs_per_round": self.epochs_per_round,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "seed": self.seed,
            "target_log_wp": self.target_log_wp,
            "ga_pop_size": self.ga_config.pop_size,
            "ga_generations": self.ga_config.generations,
            "grid_sizes": self.grid_sizes,
            "fitness_expr": self.fitness_expr(),
            "archive_bfs_bins": self.archive_bfs_bins,
            "archive_states_bins": self.archive_states_bins,
        })
    }
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        let target_log_wp = -1.0;
        AdversarialConfig {
            n_rounds: 3,
            epochs_per_round: 5,
            batch_size: 1024,
            lr: 3e-4,
            seed: 0,
            ga_config: GaConfig {
                fitness_expr: format!("-abs(log_policy_win_prob - ({target_log_wp}))"),
                ..GaConfig::default()
            },
            target_log_wp,
            archive_bfs_bins: 20,
            archive_states_bins: 20,
            grid_sizes: vec![6, 8, 10],
        }
    }
}

#[derive(Clone)]
pub enum AdversarialPhase {
    Training,
    GA { grid_size: i32 },
}

#[derive(Default)]
pub enum AdversarialStatus {
    #[default]
    Idle,
    Running {
        round: u32,
        n_rounds: u32,
        phase: AdversarialPhase,
        // Training sub-state
        training_epoch: u32,
        training_total_epochs: u32,
        training_step: u32,
        training_steps_in_epoch: u32,
        training_loss: f64,
        training_acc: f64,
        training_gs: i32,
        training_phase: TrainingPhase,
        // GA sub-state
        ga_generation: u32,
        ga_best_fitness: f64,
        ga_solvable_rate: f64,
        ga_pop_size: u32,
        // Archive sub-state
        archive_n_levels: u32,
        archive_occupancy: u32,
        archive_total_cells: u32,
    },
    Done,
    Error(String),
}

pub struct AdversarialState {
    pub config: AdversarialConfig,
    pub show_config: bool,
    pub status: AdversarialStatus,
    // Training data (cumulative across rounds)
    pub epoch_history: Vec<EpochRecord>,
    pub batch_loss_history: Vec<[f64; 2]>,
    // GA data (current round)
    pub ga_history: Vec<(u32, f64, f64)>,
    // Round boundaries for vertical lines on training curves
    pub round_boundaries: Vec<usize>,
    pub curve_plot_height: f32,
    pub log_messages: Vec<String>,
    global_epoch: usize,
}

impl AdversarialState {
    pub fn new() -> Self {
        AdversarialState {
            config: AdversarialConfig::default(),
            show_config: false,
            status: AdversarialStatus::Idle,
            epoch_history: Vec::new(),
            batch_loss_history: Vec::new(),
            ga_history: Vec::new(),
            round_boundaries: Vec::new(),
            curve_plot_height: 150.0,
            log_messages: Vec::new(),
            global_epoch: 0,
        }
    }

    /// Send start command to the server.
    pub fn start(&mut self, ws: &WsClient) {
        self.epoch_history.clear();
        self.batch_loss_history.clear();
        self.ga_history.clear();
        self.round_boundaries.clear();
        self.global_epoch = 0;
        self.log_messages.clear();

        match ws.send_adversarial(&self.config.to_json()) {
            Ok(()) => {
                self.status = AdversarialStatus::Running {
                    round: 0,
                    n_rounds: self.config.n_rounds as u32,
                    phase: AdversarialPhase::Training,
                    training_epoch: 0,
                    training_total_epochs: self.config.epochs_per_round,
                    training_step: 0,
                    training_steps_in_epoch: 0,
                    training_loss: 0.0,
                    training_acc: 0.0,
                    training_gs: 0,
                    training_phase: TrainingPhase::Status("Starting...".into()),
                    ga_generation: 0,
                    ga_best_fitness: 0.0,
                    ga_solvable_rate: 0.0,
                    ga_pop_size: 0,
                    archive_n_levels: 0,
                    archive_occupancy: 0,
                    archive_total_cells: 0,
                };
            }
            Err(e) => {
                self.status = AdversarialStatus::Error(format!("Failed to start: {e}"));
            }
        }
    }

    /// Send stop command to the server.
    pub fn stop(&mut self, ws: &WsClient) {
        let _ = ws.send_stop_adversarial();
    }

    /// Handle adversarial events from the server. Returns true if any processed.
    pub fn handle_events(&mut self, events: &[AdversarialEvent]) -> bool {
        if !matches!(self.status, AdversarialStatus::Running { .. }) {
            return false;
        }
        if events.is_empty() {
            return false;
        }

        for event in events {
            match event {
                AdversarialEvent::RoundStart { round, n_rounds } => {
                    if let AdversarialStatus::Running {
                        round: ref mut r,
                        n_rounds: ref mut nr,
                        phase: ref mut ph,
                        training_phase: ref mut tph,
                        ..
                    } = self.status
                    {
                        *r = *round;
                        *nr = *n_rounds;
                        *ph = AdversarialPhase::Training;
                        *tph = TrainingPhase::Status("Loading dataset...".into());
                    }
                    self.round_boundaries.push(self.global_epoch);
                    self.log_messages
                        .push(format!("Round {}/{n_rounds} started", round + 1));
                }
                AdversarialEvent::GaGeneration {
                    round: _,
                    grid_size,
                    generation,
                    best_fitness,
                    avg_fitness,
                    solvable_rate,
                    pop_size,
                } => {
                    if let AdversarialStatus::Running {
                        phase: ref mut ph,
                        ga_generation: ref mut g,
                        ga_best_fitness: ref mut bf,
                        ga_solvable_rate: ref mut sr,
                        ga_pop_size: ref mut ps,
                        ..
                    } = self.status
                    {
                        *ph = AdversarialPhase::GA {
                            grid_size: *grid_size as i32,
                        };
                        *g = *generation;
                        *bf = *best_fitness;
                        *sr = *solvable_rate;
                        *ps = *pop_size;
                    }
                    self.ga_history
                        .push((*generation, *best_fitness, *avg_fitness));
                }
                AdversarialEvent::ArchiveUpdate {
                    n_levels,
                    occupancy,
                    total_cells,
                    ..
                } => {
                    if let AdversarialStatus::Running {
                        archive_n_levels: ref mut nl,
                        archive_occupancy: ref mut occ,
                        archive_total_cells: ref mut tc,
                        ..
                    } = self.status
                    {
                        *nl = *n_levels;
                        *occ = *occupancy;
                        *tc = *total_cells;
                    }
                }
                AdversarialEvent::RoundEnd {
                    round,
                    time,
                    ga_levels,
                } => {
                    self.log_messages.push(format!(
                        "Round {} complete: {ga_levels} GA levels in {time:.1}s",
                        round + 1
                    ));
                }
                AdversarialEvent::Done { total_ga_levels } => {
                    self.log_messages.push(format!(
                        "Adversarial loop complete: {total_ga_levels} total GA levels"
                    ));
                    self.status = AdversarialStatus::Done;
                }
                AdversarialEvent::Training(training_event) => {
                    self.handle_training_event(training_event);
                }
            }
        }

        true
    }

    /// Process an embedded training event from the adversarial loop.
    fn handle_training_event(&mut self, event: &TrainingEvent) {
        match event {
            TrainingEvent::Init { .. } => {}
            TrainingEvent::EpochStart {
                epoch,
                total_epochs,
                steps_in_epoch,
            } => {
                if let AdversarialStatus::Running {
                    training_epoch: ref mut e,
                    training_total_epochs: ref mut te,
                    training_steps_in_epoch: ref mut sie,
                    training_step: ref mut es,
                    ..
                } = self.status
                {
                    *e = *epoch;
                    *te = *total_epochs;
                    *sie = *steps_in_epoch;
                    *es = 0;
                }
            }
            TrainingEvent::Batch {
                epoch_step,
                loss,
                acc,
                gs,
                ..
            } => {
                if let AdversarialStatus::Running {
                    training_step: ref mut es,
                    training_loss: ref mut l,
                    training_acc: ref mut a,
                    training_gs: ref mut g,
                    training_phase: ref mut ph,
                    ..
                } = self.status
                {
                    *es = *epoch_step;
                    *l = *loss;
                    *a = *acc;
                    *g = *gs;
                    *ph = TrainingPhase::Training;
                }
                self.batch_loss_history
                    .push([self.batch_loss_history.len() as f64, *loss]);
            }
            TrainingEvent::Status(text) => {
                if let AdversarialStatus::Running {
                    training_phase: ref mut ph,
                    ..
                } = self.status
                {
                    *ph = TrainingPhase::Status(text.clone());
                }
            }
            TrainingEvent::EpochEnd {
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                ..
            } => {
                self.epoch_history.push(EpochRecord {
                    epoch: *epoch,
                    train_loss: *train_loss,
                    train_acc: *train_acc,
                    val_loss: *val_loss,
                    val_acc: *val_acc,
                });
                self.global_epoch = *epoch as usize;
            }
            TrainingEvent::LevelMetrics { .. } => {}
            TrainingEvent::Log(msg) => {
                self.log_messages.push(msg.clone());
            }
            TrainingEvent::Done => {
                // Training phase within a round finished — server handles transition
            }
            TrainingEvent::Error(msg) => {
                self.status = AdversarialStatus::Error(msg.clone());
            }
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, AdversarialStatus::Running { .. })
    }
}
