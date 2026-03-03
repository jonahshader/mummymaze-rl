"""Behavioral cloning dataset: per-state optimal actions from Rust BFS."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

import mummymaze_rust
from src.env.level_bank import LevelBank, get_level, load_all_levels
from src.env.obs import observe
from src.env.types import EnvState


@dataclass(frozen=True)
class BCDataset:
  """Behavioral cloning dataset for one grid_size."""

  grid_size: int
  n_levels: int
  n_states: int

  # Level data bank (shared with RL env)
  bank: LevelBank

  # Per-state arrays
  state_tuples: Int[Array, "n_states 12"]  # raw state fields
  action_targets: Float[Array, "n_states 5"]  # soft labels
  level_idx: Int[Array, "n_states"]  # maps states -> bank index

  # Train/val masks over states (derived from bank's level-based split)
  train_mask: Bool[Array, "n_states"]
  val_mask: Bool[Array, "n_states"]


def _state_tuple_to_env_state(
  tuples: Int[Array, "B 12"],
) -> EnvState:
  """Convert raw state tuples to EnvState batch.

  Gate polarity is inverted: Rust gate_open=True means blocking,
  JAX gate_open=True means open/not blocking.
  Dead entities (99,99) are clamped to (0,0).
  """
  pr, pc = tuples[:, 0], tuples[:, 1]
  m1r, m1c = tuples[:, 2], tuples[:, 3]
  m1_alive = tuples[:, 4].astype(jnp.bool_)
  m2r, m2c = tuples[:, 5], tuples[:, 6]
  m2_alive = tuples[:, 7].astype(jnp.bool_)
  sr, sc = tuples[:, 8], tuples[:, 9]
  s_alive = tuples[:, 10].astype(jnp.bool_)
  gate_open_rust = tuples[:, 11].astype(jnp.bool_)

  # Clamp dead entity positions to (0,0)
  m1r = jnp.where(m1_alive, m1r, 0)
  m1c = jnp.where(m1_alive, m1c, 0)
  m2r = jnp.where(m2_alive, m2r, 0)
  m2c = jnp.where(m2_alive, m2c, 0)
  sr = jnp.where(s_alive, sr, 0)
  sc = jnp.where(s_alive, sc, 0)

  b = tuples.shape[0]
  return EnvState(
    player=jnp.stack([pr, pc], axis=-1),
    mummy_pos=jnp.stack(
      [
        jnp.stack([m1r, m1c], axis=-1),
        jnp.stack([m2r, m2c], axis=-1),
      ],
      axis=1,
    ),
    mummy_alive=jnp.stack([m1_alive, m2_alive], axis=-1),
    scorpion_pos=jnp.stack([sr, sc], axis=-1)[:, None, :],
    scorpion_alive=s_alive[:, None],
    gate_open=~gate_open_rust,  # invert polarity
    done=jnp.zeros(b, dtype=jnp.bool_),
    won=jnp.zeros(b, dtype=jnp.bool_),
    turn=jnp.zeros(b, dtype=jnp.int32),
  )


def make_batch_obs(
  grid_size: int,
  bank: LevelBank,
  state_tuples: Int[Array, "B 12"],
  level_idx: Int[Array, "B"],
) -> Float[Array, "B 10 Np1 Np1"]:
  """Build observations for a batch of states on-the-fly.

  JIT'd per grid_size. Uses vmap over observe().
  """
  # Get per-state LevelData by indexing into the bank
  level_data = jax.vmap(lambda i: get_level(bank, i))(level_idx)

  # Convert state tuples to EnvState
  env_states = _state_tuple_to_env_state(state_tuples)

  # vmap observe over the batch
  return jax.vmap(lambda ld, es: observe(grid_size, ld, es))(level_data, env_states)


def load_bc_dataset(
  maze_dir: Path,
  val_fraction: float = 0.1,
  seed: int = 42,
) -> dict[int, BCDataset]:
  """Load the full behavioral cloning dataset.

  1. Calls Rust optimal_actions_all for per-state optimal action data
  2. Calls load_all_levels for dedup set + LevelBank data
  3. Filters to canonical (deduplicated) levels
  4. Builds BCDataset per grid_size

  Returns dict mapping grid_size -> BCDataset.
  """
  # Load Rust optimal actions
  rust_data = mummymaze_rust.optimal_actions_all(str(maze_dir))

  # Load level banks (with dedup) for LevelData and train/val split
  banks, sources = load_all_levels(maze_dir, val_fraction=val_fraction, seed=seed)

  # Build dedup lookup: (file_stem, sublevel) -> bank_index per grid_size
  source_to_bank_idx: dict[int, dict[tuple[str, int], int]] = {}
  for gs, src_list in sources.items():
    source_to_bank_idx[gs] = {(stem, sub): i for i, (stem, sub) in enumerate(src_list)}

  # Build train/val index sets per grid_size for level-based split
  train_idx_sets: dict[int, set[int]] = {}
  val_idx_sets: dict[int, set[int]] = {}
  for gs, bank in banks.items():
    train_idx_sets[gs] = set(np.array(bank.train_indices).tolist())
    val_idx_sets[gs] = set(np.array(bank.val_indices).tolist())

  # Accumulate per-grid_size data
  gs_tuples: dict[int, list[np.ndarray]] = {}
  gs_masks: dict[int, list[np.ndarray]] = {}
  gs_level_idx: dict[int, list[int]] = {}
  gs_is_train: dict[int, list[bool]] = {}

  for entry in rust_data:
    file_stem: str = entry["file"]
    sublevel: int = entry["sublevel"]
    grid_size: int = entry["grid_size"]

    if grid_size not in source_to_bank_idx:
      continue
    lookup = source_to_bank_idx[grid_size]
    key = (file_stem, sublevel)
    if key not in lookup:
      continue  # not in canonical deduped set

    bank_idx = lookup[key]
    states = entry["states"]
    action_masks = entry["action_masks"]
    n = len(states)

    # Convert state tuples to numpy array
    tuples_np = np.zeros((n, 12), dtype=np.int32)
    for i, s in enumerate(states):
      # (pr, pc, m1r, m1c, m1_alive, m2r, m2c, m2_alive, sr, sc, s_alive, gate_open)
      tuples_np[i] = [
        s[0],
        s[1],
        s[2],
        s[3],
        int(s[4]),
        s[5],
        s[6],
        int(s[7]),
        s[8],
        s[9],
        int(s[10]),
        int(s[11]),
      ]

    # Convert action bitmasks to soft label vectors
    targets_np = np.zeros((n, 5), dtype=np.float32)
    for i, mask in enumerate(action_masks):
      bits = [(mask >> j) & 1 for j in range(5)]
      k = sum(bits)
      if k > 0:
        for j in range(5):
          if bits[j]:
            targets_np[i, j] = 1.0 / k

    if grid_size not in gs_tuples:
      gs_tuples[grid_size] = []
      gs_masks[grid_size] = []
      gs_level_idx[grid_size] = []
      gs_is_train[grid_size] = []

    gs_tuples[grid_size].append(tuples_np)
    gs_masks[grid_size].append(targets_np)
    gs_level_idx[grid_size].extend([bank_idx] * n)
    is_train = bank_idx in train_idx_sets[grid_size]
    gs_is_train[grid_size].extend([is_train] * n)

  # Build BCDataset per grid_size
  datasets: dict[int, BCDataset] = {}
  for gs in sorted(gs_tuples):
    all_tuples = np.concatenate(gs_tuples[gs], axis=0)
    all_targets = np.concatenate(gs_masks[gs], axis=0)
    all_level_idx = np.array(gs_level_idx[gs], dtype=np.int32)
    all_is_train = np.array(gs_is_train[gs], dtype=np.bool_)

    n_states = all_tuples.shape[0]
    n_levels = banks[gs].n_levels

    datasets[gs] = BCDataset(
      grid_size=gs,
      n_levels=n_levels,
      n_states=n_states,
      bank=banks[gs],
      state_tuples=jnp.array(all_tuples),
      action_targets=jnp.array(all_targets),
      level_idx=jnp.array(all_level_idx),
      train_mask=jnp.array(all_is_train),
      val_mask=jnp.array(~all_is_train),
    )

  return datasets
