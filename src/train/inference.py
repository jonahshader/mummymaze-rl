"""Shared inference closure: observation + forward pass."""

import functools
from collections.abc import Callable

import equinox as eqx
import jax

from src.env.obs import observe
from src.env.types import EnvState, LevelData


def make_obs_and_forward(model: eqx.Module) -> Callable:
  """Build a JIT'd function: (grid_size, LevelData, EnvState) -> logits.

  The closure captures model weights. After weight updates, call again
  to get a new closure (JIT cache keys on pytree structure, not leaf values).
  """

  @functools.partial(jax.jit, static_argnums=(0,))
  def obs_and_forward(
    grid_size: int,
    level_data: LevelData,
    env_states: EnvState,
  ) -> jax.Array:
    obs = jax.vmap(lambda es: observe(grid_size, level_data, es))(env_states)
    return jax.vmap(model)(obs)

  return obs_and_forward
