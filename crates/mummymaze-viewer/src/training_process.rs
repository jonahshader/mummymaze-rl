use crate::data::TrainingConfig;
use crate::training_metrics::LevelMetric;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum RawEvent {
    Init {
        n_params: u64,
        epochs: u32,
        batch_size: u32,
        lr: f64,
        #[allow(dead_code)]
        seed: u32,
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
    Done,
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct DatasetInfo {
    pub n_states: u64,
    pub n_levels: u64,
}

#[derive(Debug)]
#[allow(dead_code)]
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
    Done,
    Error(String),
}

pub struct TrainingProcess {
    event_rx: Receiver<TrainingEvent>,
    stdin_tx: Sender<String>,
    child: Option<Child>,
}

impl TrainingProcess {
    /// Spawn the Python training subprocess.
    pub fn spawn(maze_dir: &Path, config: &TrainingConfig) -> Result<Self, String> {
        let mut cmd = Command::new("uv");
        cmd.args([
            "run",
            "python",
            "-m",
            "src.train.train_bc",
            "--mode",
            "subprocess",
            "--mazes",
        ])
        .arg(maze_dir.as_os_str())
        .args(["--epochs", &config.epochs.to_string()])
        .args(["--batch-size", &config.batch_size.to_string()])
        .args(["--lr", &config.lr.to_string()])
        .args(["--seed", &config.seed.to_string()]);
        if config.wandb {
            cmd.args(["--wandb-project", &config.wandb_project]);
        }
        let mut child = cmd
            .stdout(Stdio::piped())
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn training process: {e}"))?;

        let stdout = child.stdout.take().unwrap();
        let stdin = child.stdin.take().unwrap();

        let (event_tx, event_rx) = mpsc::channel();
        let (stdin_tx, stdin_rx) = mpsc::channel::<String>();

        // Background reader thread: stdout -> events
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => break,
                };
                if line.is_empty() {
                    continue;
                }
                let event = match serde_json::from_str::<RawEvent>(&line) {
                    Ok(raw) => match raw {
                        RawEvent::Init {
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
                        RawEvent::EpochStart {
                            epoch,
                            total_epochs,
                            steps_in_epoch,
                        } => TrainingEvent::EpochStart {
                            epoch,
                            total_epochs,
                            steps_in_epoch,
                        },
                        RawEvent::Batch {
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
                        RawEvent::EpochEnd {
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
                        RawEvent::LevelMetrics {
                            step,
                            run_id,
                            timestamp: _,
                            levels,
                        } => TrainingEvent::LevelMetrics {
                            step,
                            run_id,
                            levels,
                        },
                        RawEvent::Status { status } => TrainingEvent::Status(status),
                        RawEvent::Done => TrainingEvent::Done,
                        RawEvent::Error { message } => TrainingEvent::Error(message),
                    },
                    Err(e) => {
                        eprintln!("Failed to parse training JSON: {e}: {line}");
                        continue;
                    }
                };
                if event_tx.send(event).is_err() {
                    break;
                }
            }
        });

        // Background writer thread: stdin channel -> child stdin
        std::thread::spawn(move || {
            let mut stdin = stdin;
            while let Ok(msg) = stdin_rx.recv() {
                if writeln!(stdin, "{msg}").is_err() {
                    break;
                }
                if stdin.flush().is_err() {
                    break;
                }
            }
        });

        Ok(TrainingProcess {
            event_rx,
            stdin_tx,
            child: Some(child),
        })
    }

    /// Drain available events without blocking.
    /// Consecutive Batch events are collapsed to only the last one.
    pub fn poll(&mut self) -> Vec<TrainingEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.event_rx.try_recv() {
            // Collapse consecutive batch events — only the latest matters
            if matches!(event, TrainingEvent::Batch { .. })
                && matches!(events.last(), Some(TrainingEvent::Batch { .. }))
            {
                *events.last_mut().unwrap() = event;
            } else {
                events.push(event);
            }
        }
        events
    }

    /// Send a stop command to the training process.
    pub fn send_stop(&self) {
        let _ = self.stdin_tx.send(r#"{"cmd":"stop"}"#.to_string());
    }

    /// Force kill the child process.
    pub fn kill(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
    }

    /// Check if the child process is still running.
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => match child.try_wait() {
                Ok(Some(_)) => {
                    self.child = None;
                    false
                }
                Ok(None) => true,
                Err(_) => false,
            },
            None => false,
        }
    }
}

impl Drop for TrainingProcess {
    fn drop(&mut self) {
        self.kill();
    }
}
