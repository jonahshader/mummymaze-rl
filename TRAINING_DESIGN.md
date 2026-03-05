# Training System Design

## Overview

Two-part system: a Python/JAX training pipeline and a Rust viewer training tab,
connected by a shared `level_metrics.json` file. Standard ML metrics (loss curves,
learning rate, run comparison) go to wandb. The viewer handles domain-specific
visualizations that need maze/graph context.

## Part 1: Behavioral Cloning Pipeline (Python/JAX)

### Data: Per-State Optimal Actions from Rust

New Rust export function: for every winnable state in a level, return the
observation (or raw state) and the set of optimal BFS actions.

Algorithm: BFS backward from WIN to compute depth per state, then for each
transient state, collect actions whose successor has depth = current_depth - 1.

PyO3 binding: `best_actions_all(maze_dir) -> list[dict]`,
returning per-state data across all solvable levels.

### Dataset Construction

For each state in the dataset:
- Build the 10-channel (N+1)x(N+1) observation using the JAX `observe()` function
- Target: soft label vector of length 5 (N/S/E/W/Wait), with 1/k for each of k
  equally-optimal actions, 0 elsewhere

Group by grid_size (6/8/10) since observation dimensions differ.
Train/val split by level (not by state) to test generalization.

### CNN Model

Simple conv net (equinox):
- 3-4 conv layers with GroupNorm/LayerNorm, ReLU
- Global average pooling
- Linear head -> 5 logits
- Loss: cross-entropy against soft targets

### Training Loop

- Standard supervised training with optax (adam)
- Batch by grid_size (all states in a batch share grid dimensions)
- wandb logging: loss, top-1 accuracy, top-k accuracy, learning rate
- Periodically compute per-level accuracy and write `level_metrics.json`

### Output: `level_metrics.json`

Written periodically during training (and at end). Schema:

```json
{
  "run_id": "bc-cnn-001",
  "step": 50000,
  "timestamp": "2026-03-02T12:34:56Z",
  "levels": {
    "B-0:0": {
      "grid_size": 6,
      "n_states": 42,
      "accuracy": 0.85,
      "top2_accuracy": 0.95,
      "mean_loss": 0.32
    },
    "B-0:1": { ... },
    ...
  }
}
```

Key format: `"{dat_stem}:{sublevel}"` matching the viewer's level identification.

## Part 2: Viewer Training Tab (Rust/egui)

### UI Layout

The existing right panel (state graph) gains a tab bar: `[Graph] [Training]`.
Table and maze panels are unchanged.

```
+----------+----------+-------------------------+
|          |          | [Graph] [Training]       |
|  Level   |          +-------------------------+
|  Table   |  Maze    |                         |
|          |          |  (graph OR training)    |
| +agent   | +action  |                         |
| columns  |  arrows  |                         |
|          |          |                         |
+----------+----------+-------------------------+
```

### Training Tab Contents

Top: Difficulty scatter plot
- X axis: random-policy win probability (from Markov analysis, already computed)
- Y axis: agent accuracy or solve rate (from level_metrics.json)
- One dot per level, colored by grid size
- Interactive: click a dot to select that level in the table/maze
- Diagonal reference line: points above = agent beats random

Bottom: Selected level stats panel
- Agent accuracy, top-2 accuracy, mean loss for the selected level
- Comparison with graph metrics (win%, dead-end ratio, safety)

### Level Table Changes

New sortable columns populated from level_metrics.json:
- Agent accuracy (%)
- Agent loss

### File Watching

Poll `level_metrics.json` mtime every ~2 seconds. On change, re-read and
update table columns + scatter plot. No networking needed.

### Future: Checkpoint Evaluation

Later addition (not in initial implementation):
- Long-running Python eval sidecar communicating over Unix socket
- Viewer sends a game state, receives action probabilities
- Action probabilities rendered as directional arrows on the maze
- Per-state correctness overlay on the state graph (green=correct, red=wrong)

## Interop Summary

```
Python training          level_metrics.json          Rust viewer
+-----------------+      +------------------+      +----------------+
| JAX BC training | ---> | periodic write   | ---> | file poll/read |
| wandb logging   |      | per-level stats  |      | table columns  |
+-----------------+      +------------------+      | scatter plot   |
                                                    +----------------+

Standard ML metrics (loss curves, etc.) -> wandb (not viewer)
```
