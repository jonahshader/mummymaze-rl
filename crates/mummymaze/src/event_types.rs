//! Shared event types for training and adversarial communication.
//!
//! Used by both the WebSocket client (viewer) and any other consumer
//! of training/adversarial progress events.

use serde::Deserialize;
use std::collections::HashMap;

/// Training event from the Python server.
#[derive(Debug)]
pub enum TrainingEvent {
    Init {
        n_params: u64,
        epochs: u32,
        batch_size: u32,
        lr: f64,
        datasets: HashMap<String, DatasetInfo>,
    },
    EpochStart {
        epoch: u32,
        total_epochs: u32,
        steps_in_epoch: u32,
    },
    Batch {
        step: u64,
        epoch_step: u32,
        loss: f64,
        acc: f64,
        gs: i32,
    },
    EpochEnd {
        epoch: u32,
        train_loss: f64,
        train_acc: f64,
        val_loss: f64,
        val_acc: f64,
        time: f64,
    },
    LevelMetrics {
        step: u64,
        run_id: String,
        levels: HashMap<String, LevelMetric>,
    },
    Status(String),
    Log(String),
    Done,
    Error(String),
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct DatasetInfo {
    pub n_states: u64,
    pub n_levels: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LevelMetric {
    #[allow(dead_code)]
    pub grid_size: i32,
    #[allow(dead_code)]
    pub n_states: usize,
    pub accuracy: f64,
    pub mean_loss: f64,
    pub agent_win_prob: Option<f64>,
}

/// Raw JSON event for serde deserialization.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum RawTrainingEvent {
    Init {
        #[serde(default)]
        n_params: u64,
        #[serde(default)]
        epochs: u32,
        #[serde(default)]
        batch_size: u32,
        #[serde(default)]
        lr: f64,
        #[allow(dead_code)]
        #[serde(default)]
        seed: u32,
        #[serde(default)]
        datasets: HashMap<String, DatasetInfo>,
    },
    EpochStart {
        epoch: u32,
        total_epochs: u32,
        steps_in_epoch: u32,
    },
    Batch {
        step: u64,
        epoch_step: u32,
        loss: f64,
        acc: f64,
        gs: i32,
    },
    EpochEnd {
        epoch: u32,
        train_loss: f64,
        train_acc: f64,
        val_loss: f64,
        val_acc: f64,
        time: f64,
    },
    LevelMetrics {
        step: u64,
        run_id: String,
        #[allow(dead_code)]
        timestamp: String,
        levels: HashMap<String, LevelMetric>,
    },
    Status {
        status: String,
    },
    Log {
        message: String,
    },
    Done,
    Error {
        message: String,
    },
}

/// Convert a deserialized raw event to the public `TrainingEvent` enum.
pub fn raw_to_training_event(raw: RawTrainingEvent) -> TrainingEvent {
    match raw {
        RawTrainingEvent::Init {
            n_params,
            epochs,
            batch_size,
            lr,
            seed: _,
            datasets,
        } => TrainingEvent::Init {
            n_params,
            epochs,
            batch_size,
            lr,
            datasets,
        },
        RawTrainingEvent::EpochStart {
            epoch,
            total_epochs,
            steps_in_epoch,
        } => TrainingEvent::EpochStart {
            epoch,
            total_epochs,
            steps_in_epoch,
        },
        RawTrainingEvent::Batch {
            step,
            epoch_step,
            loss,
            acc,
            gs,
        } => TrainingEvent::Batch {
            step,
            epoch_step,
            loss,
            acc,
            gs,
        },
        RawTrainingEvent::EpochEnd {
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            time,
        } => TrainingEvent::EpochEnd {
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            time,
        },
        RawTrainingEvent::LevelMetrics {
            step,
            run_id,
            timestamp: _,
            levels,
        } => TrainingEvent::LevelMetrics {
            step,
            run_id,
            levels,
        },
        RawTrainingEvent::Status { status } => TrainingEvent::Status(status),
        RawTrainingEvent::Log { message } => TrainingEvent::Log(message),
        RawTrainingEvent::Done => TrainingEvent::Done,
        RawTrainingEvent::Error { message } => TrainingEvent::Error(message),
    }
}
