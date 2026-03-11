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

**Status:** Done

Exposed CPU-intensive GA building blocks via PyO3:
- `mutate(level, seed, w_wall=5.0, ...)` — single weighted mutation
- `mutate_batch(levels, base_seed, ...)` — batch mutation (GIL-released)
- `ga_crossover(a, b, mode, seed)` — 4 crossover modes
- `ga_evaluate_batch(levels, fitness_expr)` — parallel BFS + graph + Markov (rayon, GIL-released)
- `eval_fitness(expr, metrics)` — evaluate fitness expression

### 4b — Rewrite GA loop in Python

**Status:** Done

- `src/train/ga.py`: Python GA loop with in-process JAX inference
  - `run_ga()` — full GA loop (tournament selection, elitism, mutation/crossover)
  - `compute_policy_win_probs()` — batched inference + Markov under policy
  - `MapElitesArchive` — 2D grid indexed by (bfs_moves, n_states)
  - `level_to_level_data()` — Rust Level → JAX LevelData conversion
- `adversarial_loop.py` updated to use `run_ga()` instead of `mummymaze_rust.run_ga_round()`
- Model stays in-process — no subprocess spawn, no re-JIT per GA round

Cleanup: `run_ga_round` PyO3 function deleted from `python.rs`.
`PolicyQuery`/`ModelServer` impl kept in `ga/mod.rs` — still used by viewer (Task 4d).

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

---

## Backlog

Small improvements not blocking any task, to be addressed opportunistically.

- **Redundant graph rebuilds in policy evaluation**: `compute_policy_win_probs()`
  calls `build_graph()` per level to extract state tuples, then
  `policy_win_prob_batch()` rebuilds the graph internally. Could expose a
  `policy_win_prob_from_graph()` variant or return graphs from
  `ga_evaluate_batch()` to avoid the double build.
