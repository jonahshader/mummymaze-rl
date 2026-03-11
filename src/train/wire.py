"""Shared helpers for state conversion and padding.

Used by model_server.py and ga.py for JAX inference batching.
"""

import jax.numpy as jnp
import numpy as np

from src.env.types import EnvState


def state_tuples_to_env_states(tuples: np.ndarray) -> EnvState:
  """Convert (N, 12) i32 state tuples to batched EnvState.

  Gate polarity inverted: Rust gate_open=True means blocking,
  JAX gate_open=True means open.
  Dead entities (99,99) clamped to (0,0).
  """
  t = jnp.array(tuples)
  pr, pc = t[:, 0], t[:, 1]
  m1r, m1c = t[:, 2], t[:, 3]
  m1_alive = t[:, 4].astype(jnp.bool_)
  m2r, m2c = t[:, 5], t[:, 6]
  m2_alive = t[:, 7].astype(jnp.bool_)
  sr, sc = t[:, 8], t[:, 9]
  s_alive = t[:, 10].astype(jnp.bool_)
  gate_open_rust = t[:, 11].astype(jnp.bool_)

  m1r = jnp.where(m1_alive, m1r, 0)
  m1c = jnp.where(m1_alive, m1c, 0)
  m2r = jnp.where(m2_alive, m2r, 0)
  m2c = jnp.where(m2_alive, m2c, 0)
  sr = jnp.where(s_alive, sr, 0)
  sc = jnp.where(s_alive, sc, 0)

  b = t.shape[0]
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
    gate_open=~gate_open_rust,
    done=jnp.zeros(b, dtype=jnp.bool_),
    won=jnp.zeros(b, dtype=jnp.bool_),
    turn=jnp.zeros(b, dtype=jnp.int32),
  )


def next_power_of_2(n: int) -> int:
  """Round up to next power of 2 (minimum 1)."""
  if n <= 1:
    return 1
  return 1 << (n - 1).bit_length()
