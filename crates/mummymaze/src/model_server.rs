//! Unified model server client — communicates with a single Python subprocess
//! for both neural network inference and training.
//!
//! Replaces both `PolicyClient` (inference) and `TrainingProcess` (training).
//!
//! ## Frame protocol
//!
//! All messages are length-prefixed: `[u32 length][u8 type][payload]`
//! where length includes the type byte. Little-endian throughout.
//!
//! ### Request types (Rust → Python)
//! - `0x01` Evaluate — binary level data + state tuples
//! - `0x02` Train — UTF-8 JSON config
//! - `0x03` StopTrain — empty
//! - `0x04` ReloadCheckpoint — UTF-8 path to .eqx file
//! - `0x05` Shutdown — empty
//!
//! ### Response types (Python → Rust)
//! - `0x81` EvaluateResult — raw f32 action probabilities
//! - `0x82` TrainingEvent — UTF-8 JSON line
//! - `0x83` Error — UTF-8 error message

use crate::event_types::{self, RawTrainingEvent};
use crate::game::State;
use crate::graph::StateGraph;
use crate::parse::Level;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};

// Re-export shared types for backwards compatibility
pub use crate::event_types::{DatasetInfo, LevelMetric, TrainingEvent};

// Request types
const REQ_EVALUATE: u8 = 0x01;
const REQ_TRAIN: u8 = 0x02;
const REQ_STOP_TRAIN: u8 = 0x03;
const REQ_RELOAD_CHECKPOINT: u8 = 0x04;
const REQ_SHUTDOWN: u8 = 0x05;

// Response types
const RESP_EVALUATE_RESULT: u8 = 0x81;
const RESP_TRAINING_EVENT: u8 = 0x82;
const RESP_ERROR: u8 = 0x83;

/// Unified model server client.
///
/// Owns a Python subprocess and provides synchronous `query()` for inference
/// and async `send_train()` / `poll_events()` for training.
pub struct ModelServer {
    writer: Arc<Mutex<BufWriter<std::process::ChildStdin>>>,
    /// Sender to register a oneshot for the next evaluate result.
    eval_register_tx: Mutex<Sender<Sender<Result<Vec<u8>, String>>>>,
    /// Training events channel.
    training_event_rx: Mutex<Receiver<TrainingEvent>>,
    /// Whether training is currently active.
    training_active: Mutex<bool>,
    child: Mutex<Option<Child>>,
    /// Whether the first query has been made (for JIT tracking).
    query_count: AtomicU32,
    /// Max batch size for evaluate chunking (sent per-request).
    /// 0 means no cap. Set via `set_max_batch_size()`.
    max_batch_size: AtomicU32,
}

impl ModelServer {
    /// Spawn the Python model server subprocess.
    pub fn spawn(
        maze_dir: &Path,
        checkpoint: Option<&Path>,
    ) -> Result<Self, String> {
        let mut cmd = Command::new("uv");
        cmd.args([
            "run",
            "python",
            "-m",
            "src.train.model_server",
            "--mazes",
        ])
        .arg(maze_dir.as_os_str());
        if let Some(ckpt) = checkpoint {
            cmd.args(["--checkpoint"]).arg(ckpt.as_os_str());
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn model server: {e}"))?;

        let stdout = child.stdout.take().unwrap();
        let stdin = child.stdin.take().unwrap();
        let writer = Arc::new(Mutex::new(BufWriter::new(stdin)));

        // Channels for evaluate results (oneshot pattern)
        let (eval_register_tx, eval_register_rx) = mpsc::channel::<Sender<Result<Vec<u8>, String>>>();

        // Channel for training events
        let (training_tx, training_rx) = mpsc::channel::<TrainingEvent>();

        // Background reader thread
        std::thread::spawn(move || {
            reader_loop(stdout, eval_register_rx, training_tx);
        });

        Ok(ModelServer {
            writer,
            eval_register_tx: Mutex::new(eval_register_tx),
            training_event_rx: Mutex::new(training_rx),
            training_active: Mutex::new(false),
            child: Mutex::new(Some(child)),
            query_count: AtomicU32::new(0),
            max_batch_size: AtomicU32::new(0),
        })
    }

    /// Set the max batch size for evaluate chunking.
    /// Levels with more states than this are processed in chunks on the Python side.
    /// 0 means no cap.
    pub fn set_max_batch_size(&self, size: u32) {
        self.max_batch_size.store(size, Ordering::Relaxed);
    }

    /// Whether the next query will trigger JIT compilation (first query).
    pub fn needs_jit(&self) -> bool {
        self.query_count.load(Ordering::Relaxed) == 0
    }

    /// Synchronous evaluate: send level data, block until response arrives.
    pub fn query(
        &self,
        levels_and_graphs: &[(&Level, &StateGraph)],
    ) -> Result<Vec<Vec<(State, [f32; 5])>>, String> {
        if levels_and_graphs.is_empty() {
            return Ok(vec![]);
        }

        let grid_size = levels_and_graphs[0].0.grid_size;
        let n_levels = levels_and_graphs.len() as u32;

        // Register a oneshot channel for the response
        let (result_tx, result_rx) = mpsc::channel();
        self.eval_register_tx
            .lock()
            .unwrap()
            .send(result_tx)
            .map_err(|_| "Reader thread dead")?;

        // Build and send the evaluate frame
        let mut payload = Vec::new();
        let mut all_states: Vec<Vec<State>> = Vec::with_capacity(levels_and_graphs.len());

        // Grid size + n_levels + max_batch_size
        payload.extend_from_slice(&(grid_size as u32).to_le_bytes());
        payload.extend_from_slice(&n_levels.to_le_bytes());
        payload.extend_from_slice(&self.max_batch_size.load(Ordering::Relaxed).to_le_bytes());

        for (level, graph) in levels_and_graphs {
            let obs_data = extract_level_obs_data(level);
            payload.extend_from_slice(&obs_data.h_walls);
            payload.extend_from_slice(&obs_data.v_walls);
            payload.push(obs_data.is_red as u8);
            payload.push(obs_data.has_key_gate as u8);
            payload.extend_from_slice(&obs_data.gate_row.to_le_bytes());
            payload.extend_from_slice(&obs_data.gate_col.to_le_bytes());
            for &v in &obs_data.trap_pos {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            payload.push(obs_data.trap_active[0] as u8);
            payload.push(obs_data.trap_active[1] as u8);
            for &v in &obs_data.key_pos {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            for &v in &obs_data.exit_cell {
                payload.extend_from_slice(&v.to_le_bytes());
            }

            let states: Vec<State> = graph.transitions.keys().copied().collect();
            let n_states = states.len() as u32;
            payload.extend_from_slice(&n_states.to_le_bytes());
            for state in &states {
                let arr = state.to_i32_array();
                for &v in &arr {
                    payload.extend_from_slice(&v.to_le_bytes());
                }
            }
            all_states.push(states);
        }

        // Write the frame
        write_frame(&self.writer, REQ_EVALUATE, &payload)?;

        // Block until response arrives
        let result_bytes = result_rx
            .recv()
            .map_err(|_| "Reader thread dead while waiting for evaluate result".to_string())?
            .map_err(|e| format!("Evaluate error: {e}"))?;

        // Parse response
        let mut offset = 0;
        let mut results = Vec::with_capacity(all_states.len());
        for states in &all_states {
            let n_states = states.len();
            let n_bytes = n_states * 5 * 4;
            if offset + n_bytes > result_bytes.len() {
                return Err(format!(
                    "Response too short: expected {} bytes at offset {}, got {} total",
                    n_bytes,
                    offset,
                    result_bytes.len()
                ));
            }
            let mut level_results = Vec::with_capacity(n_states);
            for (i, state) in states.iter().enumerate() {
                let base = offset + i * 5 * 4;
                let mut probs = [0f32; 5];
                for j in 0..5 {
                    let start = base + j * 4;
                    probs[j] = f32::from_le_bytes(
                        result_bytes[start..start + 4].try_into().unwrap(),
                    );
                }
                level_results.push((*state, probs));
            }
            offset += n_bytes;
            results.push(level_results);
        }

        self.query_count.fetch_add(1, Ordering::Relaxed);
        Ok(results)
    }

    /// Send a Train request. Returns immediately; events arrive via `poll_events()`.
    pub fn send_train(&self, config: &serde_json::Value) -> Result<(), String> {
        let payload = serde_json::to_vec(config)
            .map_err(|e| format!("Failed to serialize train config: {e}"))?;
        write_frame(&self.writer, REQ_TRAIN, &payload)?;
        *self.training_active.lock().unwrap() = true;
        Ok(())
    }

    /// Send a StopTrain request.
    pub fn send_stop_train(&self) -> Result<(), String> {
        write_frame(&self.writer, REQ_STOP_TRAIN, &[])
    }

    /// Send a ReloadCheckpoint request. Does not block for acknowledgement.
    pub fn send_reload_checkpoint(&self, path: &Path) -> Result<(), String> {
        let lossy = path.to_string_lossy();
        write_frame(&self.writer, REQ_RELOAD_CHECKPOINT, lossy.as_bytes())
    }

    /// Non-blocking drain of training event channel.
    /// Consecutive Batch events are collapsed to only the last one.
    pub fn poll_events(&self) -> Vec<TrainingEvent> {
        let rx = self.training_event_rx.lock().unwrap();
        let mut events = Vec::new();
        loop {
            match rx.try_recv() {
                Ok(event) => {
                    // Track training completion
                    if matches!(event, TrainingEvent::Done | TrainingEvent::Error(_)) {
                        *self.training_active.lock().unwrap() = false;
                    }
                    // Collapse consecutive batch events
                    if matches!(event, TrainingEvent::Batch { .. })
                        && matches!(events.last(), Some(TrainingEvent::Batch { .. }))
                    {
                        *events.last_mut().unwrap() = event;
                    } else {
                        events.push(event);
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    *self.training_active.lock().unwrap() = false;
                    break;
                }
            }
        }
        events
    }

    /// Check if training is currently active.
    pub fn is_training(&self) -> bool {
        *self.training_active.lock().unwrap()
    }

    /// Send shutdown signal and wait for process to exit.
    pub fn shutdown(&self) {
        // Send shutdown frame (ignore errors if pipe already closed)
        let _ = write_frame(&self.writer, REQ_SHUTDOWN, &[]);

        let mut child_guard = self.child.lock().unwrap();
        if let Some(ref mut child) = *child_guard {
            // Wait briefly for graceful exit
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        *child_guard = None;
                        return;
                    }
                    Ok(None) => {
                        if std::time::Instant::now() >= deadline {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    Err(_) => {
                        *child_guard = None;
                        return;
                    }
                }
            }
            eprintln!("model_server: graceful shutdown timed out, killing process");
            let _ = child.kill();
            let _ = child.wait();
        }
        *child_guard = None;
    }

    /// Check if the subprocess is still running.
    pub fn is_running(&self) -> bool {
        let mut child_guard = self.child.lock().unwrap();
        match *child_guard {
            Some(ref mut child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }
}

impl Drop for ModelServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Write a length-prefixed frame to the writer.
fn write_frame(
    writer: &Arc<Mutex<BufWriter<std::process::ChildStdin>>>,
    frame_type: u8,
    payload: &[u8],
) -> Result<(), String> {
    let length = 1u32 + payload.len() as u32;
    let mut w = writer.lock().unwrap();
    w.write_all(&length.to_le_bytes())
        .map_err(|e| format!("write error: {e}"))?;
    w.write_all(&[frame_type])
        .map_err(|e| format!("write error: {e}"))?;
    w.write_all(payload)
        .map_err(|e| format!("write error: {e}"))?;
    w.flush().map_err(|e| format!("flush error: {e}"))?;
    Ok(())
}

/// Read a length-prefixed frame from the reader.
fn read_frame(reader: &mut impl Read) -> Result<(u8, Vec<u8>), std::io::Error> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let length = u32::from_le_bytes(len_buf) as usize;
    if length < 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid frame length",
        ));
    }
    let mut type_buf = [0u8; 1];
    reader.read_exact(&mut type_buf)?;
    let mut payload = vec![0u8; length - 1];
    if !payload.is_empty() {
        reader.read_exact(&mut payload)?;
    }
    Ok((type_buf[0], payload))
}

/// Background reader loop: reads frames from stdout and dispatches them.
fn reader_loop(
    mut stdout: std::process::ChildStdout,
    eval_register_rx: Receiver<Sender<Result<Vec<u8>, String>>>,
    training_tx: Sender<TrainingEvent>,
) {
    // The current pending evaluate oneshot sender, if any.
    let mut pending_eval: Option<Sender<Result<Vec<u8>, String>>> = None;

    loop {
        let (frame_type, payload) = match read_frame(&mut stdout) {
            Ok(frame) => frame,
            Err(e) => {
                // EOF or read error — process died
                if let Some(eval_tx) = pending_eval.take() {
                    let _ = eval_tx.send(Err(format!("Reader died: {e}")));
                }
                let _ = training_tx.send(TrainingEvent::Error(format!("Process died: {e}")));
                return;
            }
        };

        match frame_type {
            RESP_EVALUATE_RESULT => {
                // Check for pending eval registration first
                if pending_eval.is_none() {
                    // Try to receive a registered oneshot
                    if let Ok(tx) = eval_register_rx.try_recv() {
                        pending_eval = Some(tx);
                    }
                }
                if let Some(eval_tx) = pending_eval.take() {
                    let _ = eval_tx.send(Ok(payload));
                } else {
                    eprintln!("model_server: received EvaluateResult with no pending request");
                }
            }
            RESP_TRAINING_EVENT => {
                let json_str = match std::str::from_utf8(&payload) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("model_server: invalid UTF-8 in training event: {e}");
                        continue;
                    }
                };
                match serde_json::from_str::<RawTrainingEvent>(json_str) {
                    Ok(raw) => {
                        let event = event_types::raw_to_training_event(raw);
                        if training_tx.send(event).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!("model_server: failed to parse training event: {e}: {json_str}");
                    }
                }
            }
            RESP_ERROR => {
                let msg = String::from_utf8_lossy(&payload).to_string();
                // Route error to appropriate channel
                if let Some(eval_tx) = pending_eval.take() {
                    let _ = eval_tx.send(Err(msg));
                } else {
                    let _ = training_tx.send(TrainingEvent::Error(msg));
                }
            }
            _ => {
                eprintln!("model_server: unknown response type: {frame_type:#x}");
            }
        }

        // Check for new eval registrations between frames
        while let Ok(tx) = eval_register_rx.try_recv() {
            if pending_eval.is_some() {
                // Shouldn't happen (only one eval at a time), but handle gracefully
                let _ = tx.send(Err("Another evaluate already pending".to_string()));
            } else {
                pending_eval = Some(tx);
            }
        }
    }
}

/// Level observation data needed by the model server to build CNN inputs.
pub(crate) struct LevelObsData {
    pub(crate) h_walls: Vec<u8>,
    pub(crate) v_walls: Vec<u8>,
    pub(crate) is_red: bool,
    pub(crate) has_key_gate: bool,
    pub(crate) gate_row: i32,
    pub(crate) gate_col: i32,
    pub(crate) trap_pos: [i32; 4],
    pub(crate) trap_active: [bool; 2],
    pub(crate) key_pos: [i32; 2],
    pub(crate) exit_cell: [i32; 2],
}

/// Extract the observation data from a Level for the model server protocol.
pub(crate) fn extract_level_obs_data(level: &Level) -> LevelObsData {
    let (h_bools, v_bools) = level.to_edges();
    let h_walls: Vec<u8> = h_bools.into_iter().map(|b| b as u8).collect();
    let v_walls: Vec<u8> = v_bools.into_iter().map(|b| b as u8).collect();

    let has_key_gate = level.has_gate;

    let trap_pos = [
        level.trap1_row,
        level.trap1_col,
        level.trap2_row,
        level.trap2_col,
    ];
    let trap_active = [level.trap_count >= 1, level.trap_count >= 2];

    let gate_row = if has_key_gate { level.gate_row } else { 0 };
    let gate_col = if has_key_gate { level.gate_col } else { 0 };
    let key_pos = if has_key_gate {
        [level.key_row, level.key_col]
    } else {
        [0, 0]
    };

    LevelObsData {
        h_walls,
        v_walls,
        is_red: level.flip,
        has_key_gate,
        gate_row,
        gate_col,
        trap_pos,
        trap_active,
        key_pos,
        exit_cell: [level.exit_row, level.exit_col],
    }
}
