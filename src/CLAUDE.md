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
  - `train_bc.py` — training loop (`train_epochs`, `train`, CLI entry point)
  - `config.py` — `TrainConfig` (hyperparams) and `TrainState` (mutable model/optimizer state)
  - `callbacks.py` — `LogFn`, `CheckpointFn` types and factory functions (wandb, directory checkpoints)
  - `loss.py` — `cross_entropy_loss`, `top1_accuracy`
  - `optim.py` — `make_optimizer`, `count_params`
  - `eval.py` — `compute_level_metrics`, `compute_markov_win_probs`, `parse_rust_levels`
  - `model.py` — model architectures (equinox, `@register_model` decorator)
  - `dataset.py` — dataset construction from Rust solver
  - `augment.py` — dataset augmentation (GA levels, dihedral variants, `load_augment_levels`)
  - `reporter.py` — metrics reporting (level_metrics.json, WebSocket, stdio)
  - `wire.py` — shared state conversion helpers (`state_tuples_to_env_states`)
  - `checkpoint.py` — checkpoint save/load
  - `adversarial_loop.py` — MAP-Elites adversarial training loop
  - `model_server.py` — in-process model server for inference + training
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
