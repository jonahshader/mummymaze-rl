"""MummyMazeEnv — gymnax-style functional RL environment wrapper."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from src.env.obs import observe
from src.env.step import step as step_fn
from src.env.types import EnvState, LevelData


@dataclass(frozen=True)
class EnvOut:
  """Output from a single environment step."""

  state: EnvState
  obs: Float[Array, "11 N N"]
  reward: Float[Array, ""]
  done: Bool[Array, ""]


def _flatten_envout(
  out: "EnvOut",
) -> tuple[list[object], None]:
  return ([out.state, out.obs, out.reward, out.done], None)


def _unflatten_envout(_aux: None, children: list[object]) -> "EnvOut":
  return EnvOut(
    state=children[0],  # type: ignore[arg-type]
    obs=children[1],  # type: ignore[arg-type]
    reward=children[2],  # type: ignore[arg-type]
    done=children[3],  # type: ignore[arg-type]
  )


jax.tree_util.register_pytree_node(EnvOut, _flatten_envout, _unflatten_envout)


@dataclass(frozen=True)
class MummyMazeEnv:
  """Functional RL environment for Mummy Maze.

  grid_size is the only trace-time constant. Different grid sizes produce
  different compiled functions. All levels in a vmap batch must share the
  same grid_size, but can differ in walls, entities, is_red, etc.

  Usage:
    grid_size, level, initial_state = load_level(sublevel, header)
    env = MummyMazeEnv(grid_size)
    state, obs = env.reset(level)
    out = env.step(level, state, action)

  Batched:
    batch_out = jax.vmap(env.step)(batch_levels, batch_states, batch_actions)
  """

  grid_size: int

  def reset(self, level: LevelData) -> tuple[EnvState, Float[Array, "11 N N"]]:
    """Return initial state and observation from level data."""
    state = EnvState(
      player=level.initial_player,
      mummy_pos=level.initial_mummy_pos,
      mummy_alive=level.initial_mummy_alive,
      scorpion_pos=level.initial_scorpion_pos,
      scorpion_alive=level.initial_scorpion_alive,
      gate_open=jnp.bool_(False),
      done=jnp.bool_(False),
      won=jnp.bool_(False),
      turn=jnp.int32(0),
    )
    obs = observe(self.grid_size, level, state)
    return state, obs

  def step(
    self,
    level: LevelData,
    state: EnvState,
    action: Int[Array, ""],
  ) -> EnvOut:
    """Execute one turn and return new state, obs, reward, done."""
    new_state = step_fn(self.grid_size, level, state, action)
    obs = observe(self.grid_size, level, new_state)
    reward = jnp.where(
      new_state.won,
      jnp.float32(1.0),
      jnp.where(
        new_state.done & ~new_state.won,
        jnp.float32(-1.0),
        jnp.float32(0.0),
      ),
    )
    return EnvOut(
      state=new_state,
      obs=obs,
      reward=reward,
      done=new_state.done | new_state.won,
    )
