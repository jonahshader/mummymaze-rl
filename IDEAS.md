# Mummy Maze RL — Ideas & Goals

## Project Goal

Use Mummy Maze Deluxe as a testbed for RL architectures, particularly those
targeting long-horizon planning and partial observability. The game is fully
parsed, solved, and analyzed — 10,100 levels with exact ground-truth metrics
(optimal solutions, state graphs, win probabilities, difficulty features).
The next phase is training agents and using their performance to understand
both the architectures and the levels.

## Why Mummy Maze?

- Fully deterministic (enemies follow fixed movement rules) — isolates the planning problem
- Difficulty scales smoothly: trivial levels to 66+ move optimal solutions
- Partial observability in "dark pyramid" levels (only player cell + 8 neighbors visible)
- State space is small enough to train quickly, but planning problem is hard
- Ground truth optimal solutions and exact Markov analysis available for evaluation
- Rich per-level analytics: win probability under random play, dead-end ratio, path safety, etc.

## What's Built

### JAX Environment (`src/env/`)
- Gymnax-style functional API (`step`/`reset`, pytree state, `jax.vmap` for batching)
- 11-channel CNN grid observation encoder
- Grid sizes 6/8/10, runtime masking for heterogeneous levels
- 100% verified against 9,814 BFS solutions

### Rust Crate (`crates/mummymaze/`)
- Game engine, BFS solver, full state graph builder, Markov chain solver
- Per-level metrics: n_states, win_prob, expected_steps, dead-end ratio, branching factor, optimal solution count, greedy deviation, path safety
- PyO3 bindings for Python access; all 10,100 levels analyzed in ~25s

### Interactive Viewer (`crates/mummymaze-viewer/`)
- Playable maze with undo/redo, filterable level table
- 3D force-directed state graph visualization (GPU-accelerated)
- Node coloring by win%, expected steps, BFS depth, safety
- Click-to-navigate: click a graph node to jump gameplay to that state

## Next Up

### Behavioral Cloning from BFS (CNN)
- Supervised learning: CNN predicts optimal action from the 11-channel grid observation
- Train on all winnable states (not just the optimal path) — BFS backward from WIN gives optimal action(s) at every reachable winnable state
- Soft labels: if k actions are equally optimal at a state, target is 1/k for each (cross-entropy against soft targets)
- Needs a new Rust export: for each winnable state in a level, return the observation + set of optimal actions
- Serves as architecture sanity check (if CNN can't clone BFS, it can't learn PPO) and gives a first difficulty signal (per-level cloning accuracy)

### PPO + CNN Baseline
- Train PPO with the existing 11-channel CNN observation on fully observable levels
- Log per-level episode success rate at regular training checkpoints
- Start with a mixed set of levels across difficulty range

### Empirical Difficulty Metric
- Use trained agent pass rates as ground truth for "level difficulty"
- Random-policy win probability (already computed via Markov) is the zero-intelligence baseline
- Regress agent pass rates against existing graph features to find which matter
- Multiple training snapshots give a learning curve per level — the inflection point (when the agent starts reliably solving a level) may be the best single difficulty signal
- Compare architectures: if CNN finds a level hard but transformer doesn't, that isolates planning depth as the difficulty factor

### Per-Cell Token Observation Encoder
- `position_embed(row, col) + entity_embed(type) + wall_embed(bitmask)`
- Mask for unobserved cells in dark pyramid levels
- Needed for transformer and attention-based architectures

## Future Directions

### Architectures
- **Transformer** — ViT-style over per-cell tokens, memory component for partial observability
- **Deep residual networks** — "1000 Layer Networks for Self-Supervised RL" (depth as implicit planning)
- **Continuous Thought Machines** — Sakana AI CTM (adaptive computation, "think longer" on harder states)

### Partial Observability
- Dark pyramid levels: only player cell + 8 neighbors visible
- Requires stateful architecture (RNN, memory-augmented, or observation history window)
- Fully observable levels should work with stateless (reactive) policies

### State Graph Analysis
- Deception metric: states where apparent progress is high but win probability is low
- Critical decision analysis: win-prob gap between best and second-best action per state
- Winning corridor width: how narrow is the set of winnable states at each BFS depth?
- Trap depth: how far into a losing basin can you go before getting stuck?

## Open Questions

- Reward shaping: sparse (win/lose) vs dense (distance to exit, progress)?
- Curriculum: train on easy levels first, or mixed difficulty?
- How to represent enemy movement rules to the agent (implicit via experience, or explicit)?
- Evaluation metric: % levels solved, move optimality gap, generalization to unseen levels?

## Backlog / Nice-to-have

- **Agent expected_steps**: same Markov solver with `(I-Q)t = 1` using agent
  probs. Cheap to add once win% is working. Gives "how many moves does the
  agent typically need" — arguably more intuitive than win%.
- **Async Markov computation**: fire off Rust call in a background thread
  (Rust releases the GIL), continue training next epoch, collect results
  before next level_metrics report. Would need to snapshot/copy the logits
  buffer (~50MB per gs) since it gets overwritten during training. Not worth
  it unless the synchronous ~5-6s overhead becomes a bottleneck.
- **Self-loop handling in game engines**: `build_graph()` now skips self-loops
  (Wait that doesn't change state), but the game engines still allow them. If we
  add agent playback/replay in the viewer, agents could get stuck in infinite
  Wait loops. May need a max-consecutive-wait limit or filter self-loop actions
  from the policy during playback.
