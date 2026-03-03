# mummymaze-rl

PopCap's Mummy Maze Deluxe as a JAX RL testbed.

101 pyramid files (10,100 sublevels, 6,090 unique after dedup) with a verified game engine,
BFS solver, Markov analysis, and a gymnax-compatible RL environment.

## Setup

Requires Python 3.13+, uv, and a Rust toolchain (for the maturin-built native crate).

```bash
uv sync
```

This installs all Python dependencies and builds the Rust PyO3 crate (`mummymaze-rust`) via maturin.

## Entry Points

### Python

**`src/train/train_bc.py`** — Behavioral cloning training. Trains a CNN to predict
optimal BFS actions from observations. Writes `level_metrics.json` (per-level
accuracy/loss) after each epoch for the viewer's Training tab.

```bash
# Basic training (10 epochs, writes level_metrics.json to CWD)
uv run python -m src.train.train_bc

# With wandb logging
uv run python -m src.train.train_bc --wandb-project mummymaze

# Custom settings
uv run python -m src.train.train_bc --epochs 20 --batch-size 2048 --lr 1e-3
```

**`solve_all.py`** — Batch BFS solver. Solves all 10,100 sublevels via the Rust engine
and prints summary stats. Used to populate solutions for the replay test.

```bash
uv run python solve_all.py
```

### Rust CLI

**`mummymaze-cli`** — Batch solver and Markov analyzer. Builds state graphs, computes
win probabilities, expected steps, and difficulty metrics for every level.

```bash
# Build (from repo root)
cargo build --manifest-path crates/mummymaze/Cargo.toml \
    --no-default-features --bin mummymaze-cli --release

# Analyze all levels, write CSV
./target/release/mummymaze-cli mazes/ -o results.csv

# Analyze a single sublevel
./target/release/mummymaze-cli mazes/ --file B-0.dat --sublevel 3

# BFS-only (skip Markov analysis)
./target/release/mummymaze-cli mazes/ --file B-0.dat --sublevel 3 --bfs-only
```

### Viewer

**`mummymaze-viewer`** — egui GUI for browsing levels, viewing difficulty metrics,
and playing levels with arrow keys.

```bash
# Build and run (from repo root)
cargo run --manifest-path crates/mummymaze-viewer/Cargo.toml --release -- mazes/
```

## Tests

```bash
# Full solver replay — verifies all 9,814 solvable levels through the JAX step function
uv run pytest tests/test_jax_solutions.py -v

# JAX vs Python engine agreement on action sequences
uv run pytest tests/test_step.py -v

# JIT/vmap smoke tests for the RL environment
uv run pytest tests/test_vmap.py -v

# Solution matching against human walkthroughs
uv run pytest tests/test_solutions.py -v

# All tests
uv run pytest -v
```

## Linting

```bash
uv run ruff check --fix && uv run ruff format && uv run ty check
```
