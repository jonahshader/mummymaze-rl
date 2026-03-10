//! Policy server client — communicates with a Python subprocess for neural network inference.
//!
//! The client spawns `uv run python -m src.train.policy_server --checkpoint <path>`
//! and communicates via stdin/stdout binary protocol.
//!
//! ## Protocol
//!
//! Each message starts with a `u8` message type:
//!   - `0` = Evaluate: followed by level data + state tuples, expects response
//!   - `1` = Shutdown: server should exit cleanly

use crate::game::State;
use crate::graph::StateGraph;
use crate::parse::Level;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

const MSG_EVALUATE: u8 = 0;
const MSG_SHUTDOWN: u8 = 1;

/// Client for a Python policy server subprocess.
pub struct PolicyClient {
    child: Child,
    query_count: u32,
}

/// Level observation data needed by the policy server to build CNN inputs.
struct LevelObsData {
    h_walls: Vec<u8>,
    v_walls: Vec<u8>,
    is_red: bool,
    has_key_gate: bool,
    gate_row: i32,
    gate_col: i32,
    trap_pos: [i32; 4], // r0,c0,r1,c1
    trap_active: [bool; 2],
    key_pos: [i32; 2],
    exit_cell: [i32; 2],
}

impl PolicyClient {
    /// Spawn the policy server subprocess.
    pub fn spawn(checkpoint: &Path) -> Result<Self, String> {
        Self::spawn_with_max_batch(checkpoint, 0)
    }

    /// Spawn the policy server with an optional max batch size cap.
    ///
    /// If `max_batch_size > 0`, levels with more states than this are processed
    /// in chunks to avoid GPU OOM. Use the training batch size as a good default.
    pub fn spawn_with_max_batch(
        checkpoint: &Path,
        max_batch_size: u32,
    ) -> Result<Self, String> {
        let mut cmd = Command::new("uv");
        cmd.args([
            "run",
            "python",
            "-m",
            "src.train.policy_server",
            "--checkpoint",
        ])
        .arg(checkpoint.as_os_str());
        if max_batch_size > 0 {
            cmd.args(["--max-batch-size", &max_batch_size.to_string()]);
        }
        let child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("Failed to spawn policy server: {e}"))?;

        Ok(PolicyClient {
            child,
            query_count: 0,
        })
    }

    /// Whether the next query will trigger JIT compilation (first query).
    pub fn needs_jit(&self) -> bool {
        self.query_count == 0
    }

    /// Query the policy server for action probabilities.
    ///
    /// Takes a list of (Level, StateGraph) pairs and returns per-state action
    /// probabilities for each level: `Vec<Vec<(State, [f32; 5])>>`.
    pub fn query(
        &mut self,
        levels_and_graphs: &[(&Level, &StateGraph)],
    ) -> Result<Vec<Vec<(State, [f32; 5])>>, String> {
        if levels_and_graphs.is_empty() {
            return Ok(vec![]);
        }

        let grid_size = levels_and_graphs[0].0.grid_size;
        let n_levels = levels_and_graphs.len() as u32;

        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or("stdin not available")?;

        // Message type
        stdin
            .write_all(&[MSG_EVALUATE])
            .map_err(|e| format!("write error: {e}"))?;
        // Header
        stdin
            .write_all(&(grid_size as u32).to_le_bytes())
            .map_err(|e| format!("write error: {e}"))?;
        stdin
            .write_all(&n_levels.to_le_bytes())
            .map_err(|e| format!("write error: {e}"))?;

        // Collect state lists for reading response later
        let mut all_states: Vec<Vec<State>> = Vec::with_capacity(levels_and_graphs.len());

        for (level, graph) in levels_and_graphs {
            let obs_data = extract_level_obs_data(level);

            // Write level observation data
            stdin
                .write_all(&obs_data.h_walls)
                .map_err(|e| format!("write error: {e}"))?;
            stdin
                .write_all(&obs_data.v_walls)
                .map_err(|e| format!("write error: {e}"))?;
            stdin
                .write_all(&[obs_data.is_red as u8, obs_data.has_key_gate as u8])
                .map_err(|e| format!("write error: {e}"))?;
            stdin
                .write_all(&obs_data.gate_row.to_le_bytes())
                .map_err(|e| format!("write error: {e}"))?;
            stdin
                .write_all(&obs_data.gate_col.to_le_bytes())
                .map_err(|e| format!("write error: {e}"))?;
            for &v in &obs_data.trap_pos {
                stdin
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write error: {e}"))?;
            }
            stdin
                .write_all(&[obs_data.trap_active[0] as u8, obs_data.trap_active[1] as u8])
                .map_err(|e| format!("write error: {e}"))?;
            for &v in &obs_data.key_pos {
                stdin
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write error: {e}"))?;
            }
            for &v in &obs_data.exit_cell {
                stdin
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| format!("write error: {e}"))?;
            }

            // Collect states from graph
            let states: Vec<State> = graph
                .transitions
                .keys()
                .copied()
                .collect();
            let n_states = states.len() as u32;

            // Write state count + tuples
            stdin
                .write_all(&n_states.to_le_bytes())
                .map_err(|e| format!("write error: {e}"))?;
            for state in &states {
                let arr = state.to_i32_array();
                for &v in &arr {
                    stdin
                        .write_all(&v.to_le_bytes())
                        .map_err(|e| format!("write error: {e}"))?;
                }
            }

            all_states.push(states);
        }

        stdin.flush().map_err(|e| format!("flush error: {e}"))?;

        // Read response
        let stdout = self
            .child
            .stdout
            .as_mut()
            .ok_or("stdout not available")?;

        let max_states = all_states.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut buf = vec![0u8; max_states * 5 * 4];
        let mut results = Vec::with_capacity(all_states.len());
        for states in &all_states {
            let n_states = states.len();
            let n_bytes = n_states * 5 * 4;
            stdout
                .read_exact(&mut buf[..n_bytes])
                .map_err(|e| format!("read error: {e}"))?;

            let mut level_results = Vec::with_capacity(n_states);
            for (i, state) in states.iter().enumerate() {
                let offset = i * 5 * 4;
                let mut probs = [0f32; 5];
                for j in 0..5 {
                    probs[j] =
                        f32::from_le_bytes(buf[offset + j * 4..offset + (j + 1) * 4].try_into().unwrap());
                }
                level_results.push((*state, probs));
            }
            results.push(level_results);
        }

        self.query_count += 1;
        Ok(results)
    }

    /// Send shutdown signal and wait for process to exit.
    /// Falls back to kill if graceful shutdown doesn't work within 2 seconds.
    pub fn shutdown(&mut self) {
        if let Some(ref mut stdin) = self.child.stdin {
            let _ = stdin.write_all(&[MSG_SHUTDOWN]);
            let _ = stdin.flush();
        }
        self.child.stdin.take();

        // Wait briefly for graceful exit
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(_) => return,
            }
        }

        eprintln!("policy_client: graceful shutdown timed out, killing process");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Check if the subprocess is still running.
    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for PolicyClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Extract the observation data from a Level that the policy server needs.
fn extract_level_obs_data(level: &Level) -> LevelObsData {
    let (h_bools, v_bools) = level.to_edges();
    let h_walls: Vec<u8> = h_bools.into_iter().map(|b| b as u8).collect();
    let v_walls: Vec<u8> = v_bools.into_iter().map(|b| b as u8).collect();

    let has_key_gate = level.has_gate;

    // Trap positions and active flags
    let trap_pos = [
        level.trap1_row,
        level.trap1_col,
        level.trap2_row,
        level.trap2_col,
    ];
    let trap_active = [level.trap_count >= 1, level.trap_count >= 2];

    // Key/gate positions (0,0 if absent)
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
