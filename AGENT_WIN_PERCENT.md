# Per-State Agent Data & Viewer Integration

## Background

Agent win% under a learned policy is computed via Markov chain analysis: the
agent's softmax outputs define a stochastic policy, and the existing Markov
solver computes exact win probability. This is implemented end-to-end:

- **Rust core**: `MarkovChain::from_graph_with_policy()` in `markov.rs`,
  `policy_win_prob_batch()` in `batch.rs` + `python.rs` (PyO3, numpy zero-copy,
  GIL-released, rayon-parallel).
- **Python integration**: `train_bc.py` softmaxes logits buffers after each
  epoch, calls `policy_win_prob_batch`, injects `agent_win_prob` into level
  metrics.
- **Viewer scatter plot**: Y-axis = agent win%, X-axis = random win%, stats
  panel shows both win% and accuracy.
- **Convergence handling**: non-fatal — returns NaN for failing levels, logs
  which levels failed and why.

## Current Issue: Markov Convergence Failures

Some levels fail to converge even with relaxed tolerance (1e-8, 500K
iterations). Root cause: the agent policy can concentrate probability mass on
self-loop actions (e.g., Wait when it doesn't change state), making
`diag[i] = 1 - Q[i,i]` very small. Gauss-Seidel converges slowly when the
spectral radius approaches 1.

### Part 1: Convergence Diagnostics

Enrich `policy_win_prob_batch` to return per-level diagnostic info alongside
win probabilities, so we can see exactly what's happening in failing levels.

**Return per level:**
- `win_prob: f64` (NaN if failed)
- `iterations: u32` (how many Gauss-Seidel iterations used)
- `final_residual: f64` (max diff at termination)
- `n_near_trapped: u32` (states with `diag < 1e-6`)
- `worst_diag: f64` (smallest diagonal value)
- `worst_diag_state: [i32; 12]` (state tuple of worst state)
- `worst_diag_probs: [f32; 5]` (agent's action probs on that state)

Python side: log these for failing levels so we can inspect the self-loop
concentration and decide on a fix (e.g., entropy regularization, diagonal
clamping, or SOR acceleration).

**Files:**
- `crates/mummymaze/src/markov.rs` — `solve_win_probs_tol` returns diagnostics
- `crates/mummymaze/src/batch.rs` — `policy_win_prob_batch` collects diagnostics
- `crates/mummymaze/src/python.rs` — return dicts instead of bare floats
- `src/train/train_bc.py` — log diagnostics for failing levels

## Shared Per-State Data: Mmap Architecture

The viewer (separate Rust process) needs per-state data from the training
process (Python + Rust via PyO3). Current IPC (stdio JSON + file polling)
handles scalar per-level metrics but doesn't scale to per-state data (4.4M
states × 5 floats = ~88MB as f32).

### Part 2: Mmap Shared State File

Use a memory-mapped binary file as the shared data store. Python writes
per-state action probabilities after each epoch; the viewer mmaps the same
file read-only and loads per-level data on demand.

**Why mmap:**
- Zero-copy: both processes see the same data, no serialization
- Random access: viewer reads ~50KB for the selected level, not 286MB
- Naturally extends to GA data (candidate levels + evaluations)
- Trivial in both Python (`numpy.memmap` / `mmap`) and Rust (`memmap2`)

**File format** (`agent_state_data.bin` in checkpoint dir):

```
Header (fixed):
  magic: b"MMSD" (4 bytes)
  version: u32
  n_levels: u32
  index_offset: u64

Data section (bulk, sequential per level):
  Per state: state_tuple (12 × i32) + probs (5 × f32) = 68 bytes

Index section (at index_offset, after all data):
  Per level:
    key_len: u16
    key_bytes: [u8; key_len]
    data_offset: u64
    n_states: u32
```

Index at the end so data can be written sequentially. Viewer reads the small
index (~100KB for 6K levels), then seeks to the selected level's slice.

**Files:**
- `src/train/train_bc.py` — write mmap file after softmax (data already exists)
- NEW `crates/mummymaze-viewer/src/agent_probs.rs` — mmap reader, lazy per-level
- `crates/mummymaze-viewer/src/main.rs` — poll + wire up

### Part 3: Viewer Action Probability Overlay

Draw the agent's action distribution on the maze while playing.

**Rendering:**
- 4 directional arrows from player center (N/S/E/W), length and opacity ∝
  probability
- Wait indicator: ring around player, opacity ∝ wait probability
- Drawn after entities so arrows are on top

**Files:**
- `crates/mummymaze-viewer/src/render.rs` — new `draw_action_probs()`, extract
  `maze_geometry()` helper from `draw_maze_state()`
- `crates/mummymaze-viewer/src/main.rs` — call overlay in `draw_maze_panel()`

## Future: Adversarial Level Generation (GA)

Use agent win% as a fitness function for a genetic algorithm that generates
levels the current agent struggles with, creating a curriculum of increasingly
difficult training data.

**Approach:** Python owns the GA loop (JAX for batched wall/entity mutations,
selection, tournament). Rust evaluates fitness via `policy_win_prob_batch`.
The mmap file extends naturally to include GA candidate levels and their
evaluations, visible in the viewer for analysis.

This is a longer-term direction — documenting here to ensure the mmap
architecture supports it.
