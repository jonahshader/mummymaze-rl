# Mummy Maze RL — Ideas & Goals

## Project Goal

Use Mummy Maze Deluxe as a testbed for RL architectures, particularly those
targeting long-horizon planning and partial observability. Parse official levels
via `mummy-maze-parser` and convert them to JAX-friendly representations.

## Why Mummy Maze?

- Fully deterministic (enemies follow fixed movement rules) — isolates the planning problem
- Difficulty scales smoothly: trivial levels to 66+ move optimal solutions
- Partial observability in "dark pyramid" levels (only player cell + 8 neighbors visible)
- State space is small enough to train quickly, but planning problem is hard
- Ground truth optimal solutions available for evaluation

## Environment Design

- **Framework:** Gymnax-style functional API (`step`/`reset`, pytree state, `jax.vmap` for batching)
- **State:** Carries both true game state and agent's observation history (for partial observability)
- **Grid sizes:** 6, 8, or 10 (from .dat file header)
- **Entities:** Player, mummy (white/red), scorpion, trap, key, gate

## Architecture Ideas

### Observation Encoding

Per-cell token: `position_embed(row, col) + entity_embed(type) + wall_embed(bitmask)`
with a mask for unobserved cells in dark levels.

### CNN Baseline

- 3-4 conv layers on a multi-channel grid encoding (one channel per wall direction, one per entity type)
- Stateless — sufficient for fully observable levels
- Simple, fast, good baseline to measure against

### Transformer

- Entity embeddings + spatial embeddings summed per cell (ViT-style)
- Established pattern: ViT, Gato, AlphaFold pair representations, board game agents
- Memory component for partial observability (context window of recent observations)

### Models to Test

- **Deep residual networks** — "1000 Layer Networks for Self-Supervised RL" (scaling depth as implicit planning/computation)
- **Continuous Thought Machines** — Sakana AI CTM (adaptive computation time, "think longer" on harder states)
- **Standard baselines** — PPO with CNN, PPO with transformer

### Stateless vs Stateful

- Fully observable levels: stateless (reactive) policy should suffice
- Dark pyramid levels: stateful architecture required (RNN, memory-augmented, or observation history window)
- Design architecture to support both modes

## Open Questions

- Reward shaping: sparse (win/lose) vs dense (distance to exit, progress)?
- Curriculum: train on easy levels first, or mixed difficulty?
- How to represent enemy movement rules to the agent (implicit via experience, or explicit)?
- Evaluation metric: % levels solved, move optimality gap, generalization to unseen levels?
