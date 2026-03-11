# Refactor Plan — Infrastructure & Architecture

This document tracks a set of interconnected refactors to the training
infrastructure, communication protocols, checkpoint management, and Rust
codebase organization.

## Dependency Graph

```
                    ┌─────────────────────┐
                    │ 1. Model registry / │
                    │    architecture      │
                    │    selection         │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
   ┌──────────────┐  ┌────────────┐  ┌───────────────┐
   │ 2. Rich      │  │ 3. Retire  │  │ 4. Python-    │
   │ checkpoints  │  │ legacy     │  │ owned loops   │
   │              │  │ policy_    │  │ + WebSocket   │
   │              │  │ server     │  │ API           │
   └──────┬───────┘  └─────┬──────┘  └──────┬────────┘
          │                │               │
          ▼                ▼               ▼
   ┌─────────────────────────────────────────────┐
   │ 5. Unified state management                 │
   └─────────────────────────────────────────────┘
```

---

## Task 1 — Model Registry & Architecture Selection

**Status:** Done

Currently `MazeCNN` is hardcoded and instantiated identically in 5 places.
No way to select an architecture — every component assumes `MazeCNN`.

### What was done

- Model factory: `make_model(arch, key)` in `src/train/model.py`
- `MODEL_REGISTRY` with `"cnn"` (MazeCNN, ~244K) and `"resnet"` (MazeResNet, ~340K)
- `--arch` CLI flag on `train_bc.py`, `model_server.py`, `adversarial_loop.py`
- Architecture name stored in checkpoint metadata (used by Task 2)

---

## Task 2 — Rich Checkpoints (Training State Persistence)

**Status:** Done

### What was done

- Checkpoint is now a directory: `model.eqx`, `opt_state.eqx`, `training_state.json`
- `save_checkpoint()` / `load_checkpoint()` in `src/train/checkpoint.py`
- Optimizer state (Adam moments) restored on resume
- RNG key, epoch, global_step, arch, lr, batch_size all persisted

---

## Task 3 — Retire Legacy Policy Server & Unify Protocols

**Status:** Done

### What was done

- Deleted `src/train/policy_server.py`
- Deleted `crates/mummymaze/src/policy_client.rs` and `PolicyClientWrapper`
- Unified on `ModelServer` protocol
- Single `PolicyQuery` implementation via `ModelServer`

---

## Task 4 — Python-Owned Loops + WebSocket API

**Status:** Not started
**Depends on:** Tasks 1-3

### Motivation

The viewer and CLI currently have separate implementations of the same loops
(training, GA, adversarial). The viewer orchestrates everything in Rust via a
binary frame protocol to a Python subprocess, while the CLI runs everything
in Python. This means two diverging codepaths for the same logic.

**New architecture:** Python owns all loops (training, GA, adversarial).
The Rust viewer becomes a pure visualization layer that sends configuration
and displays progress. A WebSocket API replaces the binary frame protocol,
enabling both the Rust viewer and a future web frontend to connect.

### 4a — PyO3 bindings for GA primitives

**Status:** Not started

Expose the CPU-intensive GA building blocks so Python can orchestrate the
GA loop while Rust handles the hot paths (BFS, Markov, mutation).

New PyO3 functions:
- `mutate(level, config) → Level` — single mutation
- `mutate_batch(levels, config, seed) → list[Level]` — batch mutation
- `crossover(a, b, mode, seed) → Level` — single crossover
- `evaluate_batch(levels) → list[dict]` — parallel BFS + graph + Markov
  (no policy). Returns `{level, bfs_moves, n_states, win_prob, ...}` per
  solvable level. Releases GIL, uses rayon.

Existing bindings already cover the rest:
- `policy_win_prob_batch()` — Markov under policy
- `analyze()` / `solve()` — single-level analysis

### 4b — Rewrite GA loop in Python

**Status:** Not started
**Depends on:** 4a

Rewrite `run_ga_round` as a Python function using the new PyO3 primitives
plus in-process JAX inference:

```python
for generation in range(n_generations):
    # Rust (release GIL): mutate/crossover offspring
    offspring = mummymaze_rust.mutate_batch(parents, config)
    # Rust (release GIL): parallel BFS + Markov
    evals = mummymaze_rust.evaluate_batch(offspring)
    # Python (JAX): neural net inference on solvable levels
    probs = model(observations)
    # Rust (release GIL): Markov under policy
    policy_wps = mummymaze_rust.policy_win_prob_batch(...)
    # Python: fitness scoring, tournament selection, archive
    population = select(evals, policy_wps, ...)
```

Benefits:
- No ModelServer subprocess spawn per GA round (currently re-JITs every time)
- Model stays in Python process memory — zero serialization overhead
- Per-generation progress available to any caller
- Single implementation for both CLI and viewer

Delete from Rust:
- `run_ga_round` PyO3 function (replaced by Python loop)
- `PolicyQuery` trait and `ModelServer` impl in `ga/mod.rs`
- `run_ga_inner`, `run_ga_with_model_server*` entry points
- Keep: `mutation.rs`, `crossover.rs`, `fitness.rs`, `archive.rs` (used via PyO3)

### 4c — WebSocket API

**Status:** Not started
**Depends on:** 4b

Replace the binary frame protocol (`model_server.py`) with a WebSocket
server. The Python process becomes a long-lived service that any client
can connect to.

Endpoints / message types:
- **Training:** start, stop, progress events (epoch, batch, done)
- **GA / Adversarial:** start, per-generation progress, archive updates
- **Inference:** query agent action probs for a single level on demand
  (replaces `agent_probs.bin` mmap file)
- **Checkpoint:** reload, list available

Delete:
- `src/train/model_server.py` (binary frame protocol)
- `src/train/wire.py` (binary I/O helpers for frames)
- `crates/mummymaze/src/model_server.rs` (Rust subprocess client)
- `agent_probs.bin` writing in `train_bc.py`

### 4d — Viewer as WebSocket client

**Status:** Not started
**Depends on:** 4c

Refactor the viewer to connect to the Python WebSocket server instead of
spawning and managing a subprocess.

Delete from viewer:
- `adversarial.rs` — state machine (Python owns the loop now)
- `agent_probs.rs` — mmap reader (query WS on demand)
- `data/training.rs` — ModelServer polling (receive WS events)
- `training_metrics.rs` — JSON file watcher (receive WS events)

Keep in viewer:
- All rendering (`render.rs`, `graph_view/`)
- UI widgets (`table.rs`, `training_tab.rs`, `adversarial_tab.rs`, `level_gen_tab.rs`)
- Level loading + analysis (`data/mod.rs`) — still Rust, still fast

---

## Task 5 — Unified State Management

**Status:** Not started
**Depends on:** Tasks 1-4 being settled

Currently 10+ distinct file I/O patterns with ad-hoc paths. After Task 4
removes `agent_probs.bin` and the binary frame protocol, the remaining I/O
simplifies to:

| File | Format | Lifecycle |
|------|--------|-----------|
| `checkpoints/epoch*/` | Directory (Task 2) | Persistent |
| `checkpoints/adversarial/round*/archive.json` | JSON | Persistent |
| `level_metrics.json` | JSON | Ephemeral (may move to WS-only) |
| WebSocket messages | JSON | Transient |

### Goals

- Define a coherent project directory layout
- All components discover paths from a single config/root
- Cleanup policy for ephemeral files
