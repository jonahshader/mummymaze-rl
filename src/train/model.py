"""CNN model for behavioral cloning on Mummy Maze observations."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class MazeCNN(eqx.Module):
  """Grid-size-agnostic CNN for action prediction.

  Architecture:
    Conv2d(10->32, 3x3) -> GroupNorm(8) -> GELU
    Conv2d(32->64, 3x3) -> GroupNorm(8) -> GELU
    Conv2d(64->128, 3x3) -> GroupNorm(8) -> GELU
    Conv2d(128->128, 3x3) -> GroupNorm(8) -> GELU
    GlobalAvgPool -> Linear(128->5)

  Same weights work for all grid sizes (6/8/10) via global average pooling.
  """

  conv1: eqx.nn.Conv2d
  norm1: eqx.nn.GroupNorm
  conv2: eqx.nn.Conv2d
  norm2: eqx.nn.GroupNorm
  conv3: eqx.nn.Conv2d
  norm3: eqx.nn.GroupNorm
  conv4: eqx.nn.Conv2d
  norm4: eqx.nn.GroupNorm
  head: eqx.nn.Linear

  def __init__(self, key: PRNGKeyArray) -> None:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    self.conv1 = eqx.nn.Conv2d(10, 32, 3, padding=1, key=k1)
    self.norm1 = eqx.nn.GroupNorm(8, 32)
    self.conv2 = eqx.nn.Conv2d(32, 64, 3, padding=1, key=k2)
    self.norm2 = eqx.nn.GroupNorm(8, 64)
    self.conv3 = eqx.nn.Conv2d(64, 128, 3, padding=1, key=k3)
    self.norm3 = eqx.nn.GroupNorm(8, 128)
    self.conv4 = eqx.nn.Conv2d(128, 128, 3, padding=1, key=k4)
    self.norm4 = eqx.nn.GroupNorm(8, 128)
    self.head = eqx.nn.Linear(128, 5, key=k5)

  def __call__(self, x: Float[Array, "10 H W"]) -> Float[Array, "5"]:
    x = jax.nn.gelu(self.norm1(self.conv1(x)))
    x = jax.nn.gelu(self.norm2(self.conv2(x)))
    x = jax.nn.gelu(self.norm3(self.conv3(x)))
    x = jax.nn.gelu(self.norm4(self.conv4(x)))
    # Global average pool: (C, H, W) -> (C,)
    x = jnp.mean(x, axis=(1, 2))
    return self.head(x)
