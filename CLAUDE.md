# CLAUDE.md

Training an RL agent to play Mummy Maze Deluxe using JAX. See `IDEAS.md` for goals and research directions.

## Project Structure

- `src/game.py` — Pure Python game engine (mutable, reference implementation)
- `src/solver.py` — BFS solver using the Python engine
- `src/env/` — JAX port (immutable pytrees, jit/vmap compatible)
  - `types.py` — `LevelData` and `EnvState` pytree definitions
  - `mechanics.py` — wall checks, gate blocking, enemy movement
  - `step.py` — full turn pipeline (the core game logic)
  - `obs.py` — CNN observation encoder (11-channel grid)
  - `env.py` — gymnax-style functional RL wrapper
  - `level_load.py` / `level_bank.py` — level loading and batched storage
- `csolver/` — C reference port of the binary's exact logic
- `mazes/` — `.dat` level files (101 pyramids, 10,100 sublevels)
- `solve_all.py` — batch solver, populates cache for replay tests
- `GAME_RULES.md` — verified game rules (turn order, collisions, gate mechanics)

## Key Design Constraints

- `grid_size` is the only trace-time constant (Python int). Different grid sizes (6/8/10) produce different compiled functions.
- Everything else (is_red, has_key_gate, entity counts) is a runtime value with masks, enabling vmap across heterogeneous levels.
- Both engines (Python and JAX) must agree exactly. The JAX env is verified against 9,814 BFS solutions via `tests/test_jax_solutions.py`.
- Game rules are subtle — see `GAME_RULES.md` for the full verified spec. Do not change game logic without re-running the solution replay test.

## Development Practices

We are using some cutting edge python stuff: uv, ruff, ty, python 3.13.

The deep learning stack is jax[cuda13], diffrax, equinox, optax.

For type checking and debugging, we use jaxtyping + beartype for runtime shape validation. Concrete dimensions (e.g., `Float[Array, "64 64 16"]`) validate exact sizes, while symbolic dimensions (e.g., `Float[Array, "height width channels"]`) validate consistency - all uses of "height" must have the same size within a function call, but that size can vary between calls.

We use the import hook pattern to apply beartype checking automatically to all modules without needing decorators on every function. The runtime overhead is negligible since beartype is O(1) and JAX JIT compiles away the checks after the first trace.

Avoid `__init__.py` re-exports - vulture can't trace them properly, causing false negatives. Import directly from submodules instead.

## Workflow

Run after completing any code changes:
```
uv run ruff check --fix && uv run ruff format && uv run ty check
```

Run after moderate/significant refactors:
```
uv run vulture
```

Run after any changes to game logic (`src/game.py`, `src/env/step.py`, `src/env/mechanics.py`):
```
uv run pytest tests/test_jax_solutions.py -v
```
