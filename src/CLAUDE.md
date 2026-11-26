# src/ — Python/JAX

## Layout

- `game.py` — pure Python game engine (mutable, reference implementation)
- `env/` — JAX port: immutable pytrees, jit/vmap compatible. `step.py` is the full turn pipeline (the core game logic), `mechanics.py` the wall, gate and enemy-movement rules, `obs.py` the 10-channel CNN observation, `env.py` the gymnax-style functional wrapper, and `level_load.py` / `level_bank.py` level loading and batched storage.
- `train/` — training. What holds it together:
  - One entry point, `python -m src.train config/foo.py --key=val`. A config is a `.py` file that is `exec()`'d, plus `--key=value` overrides (`configurator.py`, nanoGPT-style).
  - Every mode runs `setup_training()` (`session.py`) then `training_loop(inner_stop, outer_stop, on_round_end)` (`loop.py`). BC is one round; adversarial training is several rounds with a GA callback as `on_round_end` (`adversarial_loop.py`). Stop conditions are combinators in `stopping.py`.
  - `ga.py` is the GA loop with in-process batched policy inference (`run_ga`, `run_multi_ga`, MAP-Elites archive). Rust does the mutation, crossover, BFS and Markov solves.
  - `markov_jax.py` is a differentiable Markov solver (surrogate gradient trick).
  - `ws_server.py` is the JSON-over-WebSocket server the viewer connects to, wrapping `model_server.py`.
  - Models are equinox modules registered with `@register_model` in `model.py`.
- `baselines/random_agent.py` — random policy baseline
- `tui.py` — terminal UI

## Stack

Python 3.13, uv, ruff, ty. Deep learning: jax[cuda13], equinox, optax.

Array shapes are annotated with jaxtyping. Nothing checks them at runtime: beartype is a dependency, but no import hook or decorator installs it.

`mummymaze_rust` is typed by a hand-written stub, `crates/mummymaze/mummymaze_rust.pyi` (`crates/CLAUDE.md`). Its dict returns are `TypedDict`s that exist only in the stub, so a module that names one in an annotation needs `from __future__ import annotations`.

Avoid `__init__.py` re-exports — vulture can't trace them, causing false negatives. Import directly from submodules.

## Workflow

After any code changes:
```
uv run ruff check --fix && uv run ruff format && uv run ty check
```
All three pass on a clean tree, so any output is yours. Notebooks are outside the format and type checks (`pyproject.toml`).

After moderate/significant refactors:
```
uv run vulture
```

After changes to game logic (`game.py`, `env/step.py`, `env/mechanics.py`):
```
uv run pytest tests/test_jax_solutions.py -v
```
