# src/ — Python/JAX

## Layout

- `game.py` — Pure Python game engine (mutable, reference implementation)
- `env/` — JAX port (immutable pytrees, jit/vmap compatible)
  - `types.py` — `LevelData` and `EnvState` pytree definitions
  - `mechanics.py` — wall checks, gate blocking, enemy movement
  - `step.py` — full turn pipeline (the core game logic)
  - `obs.py` — CNN observation encoder (10-channel grid)
  - `env.py` — gymnax-style functional RL wrapper
  - `level_load.py` / `level_bank.py` — level loading and batched storage
- `train/` — behavioral cloning pipeline
  - `train_bc.py` — training loop (supervised, BFS-optimal targets)
  - `model.py` — CNN model (equinox)
  - `dataset.py` — dataset construction from Rust solver
  - `reporter.py` — metrics reporting (level_metrics.json, wandb)
- `baselines/random_agent.py` — random policy baseline
- `tui.py` — terminal UI

## Stack

Python 3.13, uv, ruff, ty. Deep learning: jax[cuda13], equinox, optax.

Runtime shape validation via jaxtyping + beartype. The import hook applies beartype automatically to all modules — no per-function decorators needed.

Avoid `__init__.py` re-exports — vulture can't trace them, causing false negatives. Import directly from submodules.

## Workflow

After any code changes:
```
uv run ruff check --fix && uv run ruff format && uv run ty check
```

After moderate/significant refactors:
```
uv run vulture
```

After changes to game logic (`game.py`, `env/step.py`, `env/mechanics.py`):
```
uv run pytest tests/test_jax_solutions.py -v
```
