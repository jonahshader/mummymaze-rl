use crate::training_metrics::TrainingMetrics;
use mummymaze::model_server::{ModelServer, TrainingEvent};
use std::sync::Arc;

use super::DataStore;

#[derive(Default, Clone)]
pub enum TrainingPhase {
    #[default]
    Training,
    Status(String),
}

#[derive(Default)]
pub enum TrainingStatus {
    #[default]
    Idle,
    Running {
        epoch: u32,
        total_epochs: u32,
        epoch_step: u32,
        steps_in_epoch: u32,
        loss: f64,
        acc: f64,
        gs: i32,
        phase: TrainingPhase,
    },
    Done,
    Error(String),
}

/// Config for launching training from the UI.
pub struct TrainingConfig {
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub seed: u32,
    pub wandb: bool,
    pub wandb_project: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            epochs: 10,
            batch_size: 1024,
            lr: 3e-4,
            seed: 0,
            wandb: false,
            wandb_project: "mummy-maze-rl".into(),
        }
    }
}

pub struct EpochRecord {
    pub epoch: u32,
    pub train_loss: f64,
    pub train_acc: f64,
    pub val_loss: f64,
    pub val_acc: f64,
}

impl DataStore {
    /// Start training via the model server.
    pub fn start_training(&mut self, model_server: &Arc<ModelServer>) {
        let config = &self.training_config;
        self.epoch_history.clear();
        self.batch_loss_history.clear();

        let train_config = serde_json::json!({
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "seed": config.seed,
        });
        match model_server.send_train(&train_config) {
            Ok(()) => {
                self.training_status = TrainingStatus::Running {
                    epoch: 0,
                    total_epochs: config.epochs,
                    epoch_step: 0,
                    steps_in_epoch: 0,
                    loss: 0.0,
                    acc: 0.0,
                    gs: 0,
                    phase: TrainingPhase::Status("Loading dataset...".into()),
                };
            }
            Err(e) => {
                self.training_status = TrainingStatus::Error(e);
            }
        }
    }

    /// Send stop command to training via model server.
    pub fn stop_training(&mut self, model_server: &Arc<ModelServer>) {
        let _ = model_server.send_stop_train();
    }

    /// Poll model server for training events. Returns true if any received.
    pub fn poll_training(&mut self, model_server: &Arc<ModelServer>) -> bool {
        if !matches!(self.training_status, TrainingStatus::Running { .. }) {
            return false;
        }

        let events = model_server.poll_events();
        if events.is_empty() {
            return false;
        }

        let mut needs_resort = false;

        for event in events {
            match event {
                TrainingEvent::Init { .. } => {}
                TrainingEvent::EpochStart {
                    epoch,
                    total_epochs,
                    steps_in_epoch: sie,
                } => {
                    if let TrainingStatus::Running {
                        epoch: ref mut e,
                        total_epochs: ref mut te,
                        steps_in_epoch: ref mut sie_ref,
                        epoch_step: ref mut es,
                        ..
                    } = self.training_status
                    {
                        *e = epoch;
                        *te = total_epochs;
                        *sie_ref = sie;
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
                    if let TrainingStatus::Running {
                        epoch_step: ref mut es,
                        loss: ref mut l,
                        acc: ref mut a,
                        gs: ref mut g,
                        phase: ref mut ph,
                        ..
                    } = self.training_status
                    {
                        *es = epoch_step;
                        *l = loss;
                        *a = acc;
                        *g = gs;
                        *ph = TrainingPhase::Training;
                    }
                    self.batch_loss_history
                        .push([self.batch_loss_history.len() as f64, loss]);
                }
                TrainingEvent::Status(text) => {
                    if let TrainingStatus::Running {
                        phase: ref mut ph, ..
                    } = self.training_status
                    {
                        *ph = TrainingPhase::Status(text);
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
                        epoch,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                    });
                }
                TrainingEvent::LevelMetrics {
                    step,
                    run_id,
                    levels,
                } => {
                    let tm = self
                        .training_metrics
                        .get_or_insert_with(|| TrainingMetrics::new(std::path::PathBuf::new()));
                    tm.update_from_event(run_id, step, levels);
                    needs_resort = true;
                }
                TrainingEvent::Log(msg) => {
                    self.log_messages.push(msg);
                }
                TrainingEvent::Done => {
                    self.training_status = TrainingStatus::Done;
                }
                TrainingEvent::Error(msg) => {
                    self.training_status = TrainingStatus::Error(msg);
                }
            }
        }

        if needs_resort && self.sort_col.is_tier2() {
            self.refresh_sort_filter();
        }

        true
    }

    /// Check if training is actively running.
    pub fn is_training(&self) -> bool {
        matches!(self.training_status, TrainingStatus::Running { .. })
    }
}
