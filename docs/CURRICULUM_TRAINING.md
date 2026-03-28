# Curriculum Training for Human-Like Difficulty Estimation

## Goal

Train a policy network that plays Mummy Maze with human-like reasoning, so that GA-minimized win probability under this policy correlates with human-perceived difficulty.

## Why not just use a uniform random policy?

With a uniformly random policy, GA-minimizing win probability produces levels that exploit a specific trick: the player must cart a scorpion through long traces of the maze while staying adjacent to it. Under a random policy this is nearly impossible (one wrong move = death), so these levels have extremely low win probability. But they aren't genuinely hard for humans — a human can easily see "walk the scorpion to the exit." The GA is exploiting a quirk of the random policy, not discovering true difficulty.

## Why not just use BC?

Behavioral cloning on the real levels (BFS-optimal actions) plateaus at ~65% mean win probability on a held-out validation set. The model memorizes the training levels but doesn't generalize — it hasn't learned the underlying reasoning needed for novel positions.

## The curriculum approach

Train on GA-generated levels of increasing difficulty. The GA targets levels where the current model has low win probability (~10%), so the training data is always at the frontier of the model's ability. As the model improves, the GA generates harder levels, pushing the frontier further.

The key insight: if we can get the model's validation win probability extremely high (close to 100%) on real levels — purely from training on unseen GA-generated levels — then the model has learned general maze-solving reasoning, not just memorized patterns. At that point, GA-minimizing win probability under this model should produce levels that are genuinely hard, not just exploit policy quirks.

## Current setup

### Architecture
- `resnet-attn` with width=4, ~4.5M parameters
- Starting from a BC-pretrained checkpoint (50 epochs, 65% mean val win prob)

### Curriculum loop
```
Outer loop (rounds):
  1. GA generates N hard levels (targeting ~10% policy win prob)
  2. Augment training set with new levels (accumulates across rounds)

  Inner loop (epochs):
    a. Train one epoch on ALL accumulated GA levels (BC loss)
    b. Eval win prob on THIS ROUND'S new levels only
    c. If model masters current round's levels → break to outer
    d. If max_epochs reached → break to outer

  Log real validation win prob for monitoring (not used for stopping)
```

### Key design decisions
- **Train only on GA levels** — real levels are held out entirely for validation
- **Constant learning rate** — dataset size changes each round, cosine schedule doesn't apply
- **Accumulate levels** — model trains on all past rounds to prevent catastrophic forgetting
- **Threshold on current round only** — measures mastery of the new hard levels, not overall performance
- **Batched GA inference** — temp LevelBank + chunked obs/NN forward for 1024-level populations

### Infrastructure
- `experiments/curriculum_gs6.py` — the experiment script
- `src/train/ga.py` — GA with batched inference (`compute_policy_win_probs_batched`)
- `src/train/augment.py` — dataset augmentation from GA levels
- Rust crate handles: level mutation/crossover, BFS solving, Markov win prob, state graph building

## Open questions

- **Level diversity**: a single GA cycle may produce similar levels that all exploit the same model weakness. Smaller populations taken more frequently might provide better coverage.
- **Target win probability**: -1.0 (10%) might be too hard. -0.3 (50%) could give more learnable signal.
- **Dataset growth**: levels accumulate forever. May need a cap or pruning strategy (drop levels the model has mastered).
- **Evaluation frequency**: computing win prob on GA levels every epoch is expensive. Could check every N epochs.
