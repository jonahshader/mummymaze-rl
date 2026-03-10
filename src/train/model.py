"""Model architectures for behavioral cloning on Mummy Maze observations."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

# All model classes must:
#   - Be eqx.Module subclasses
#   - Accept (key: PRNGKeyArray) as sole __init__ arg
#   - Accept (x: Float[Array, "10 H W"]) and return Float[Array, "5"]
#   - Work for all grid sizes (6/8/10) via global average pooling


class MazeCNN(eqx.Module):
  """Grid-size-agnostic CNN for action prediction.

  Architecture:
    Conv2d(10->32, 3x3) -> GroupNorm(8) -> GELU
    Conv2d(32->64, 3x3) -> GroupNorm(8) -> GELU
    Conv2d(64->128, 3x3) -> GroupNorm(8) -> GELU
    Conv2d(128->128, 3x3) -> GroupNorm(8) -> GELU
    GlobalAvgPool -> Linear(128->5)

  Same weights work for all grid sizes (6/8/10) via global average pooling.
  ~244K parameters.
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


class MazeResNet(eqx.Module):
  """Deeper residual CNN. Same I/O contract as MazeCNN.

  Architecture:
    Conv2d(10->64, 3x3) -> GroupNorm -> GELU
    4x ResBlock(64->64): Conv->GN->GELU->Conv->GN + skip -> GELU
    Conv2d(64->128, 3x3) -> GroupNorm -> GELU
    GlobalAvgPool -> Linear(128->5)

  ~340K parameters.
  """

  stem: eqx.nn.Conv2d
  stem_norm: eqx.nn.GroupNorm
  res_conv1: list[eqx.nn.Conv2d]
  res_norm1: list[eqx.nn.GroupNorm]
  res_conv2: list[eqx.nn.Conv2d]
  res_norm2: list[eqx.nn.GroupNorm]
  proj: eqx.nn.Conv2d
  proj_norm: eqx.nn.GroupNorm
  head: eqx.nn.Linear

  def __init__(self, key: PRNGKeyArray) -> None:
    n_blocks = 4
    keys = jax.random.split(key, 2 * n_blocks + 3)

    self.stem = eqx.nn.Conv2d(10, 64, 3, padding=1, key=keys[0])
    self.stem_norm = eqx.nn.GroupNorm(8, 64)

    self.res_conv1 = []
    self.res_norm1 = []
    self.res_conv2 = []
    self.res_norm2 = []
    for i in range(n_blocks):
      self.res_conv1.append(eqx.nn.Conv2d(64, 64, 3, padding=1, key=keys[1 + 2 * i]))
      self.res_norm1.append(eqx.nn.GroupNorm(8, 64))
      self.res_conv2.append(eqx.nn.Conv2d(64, 64, 3, padding=1, key=keys[2 + 2 * i]))
      self.res_norm2.append(eqx.nn.GroupNorm(8, 64))

    self.proj = eqx.nn.Conv2d(64, 128, 3, padding=1, key=keys[-2])
    self.proj_norm = eqx.nn.GroupNorm(8, 128)
    self.head = eqx.nn.Linear(128, 5, key=keys[-1])

  def __call__(self, x: Float[Array, "10 H W"]) -> Float[Array, "5"]:
    x = jax.nn.gelu(self.stem_norm(self.stem(x)))
    for c1, n1, c2, n2 in zip(
      self.res_conv1, self.res_norm1, self.res_conv2, self.res_norm2
    ):
      residual = x
      x = jax.nn.gelu(n1(c1(x)))
      x = n2(c2(x))
      x = jax.nn.gelu(x + residual)
    x = jax.nn.gelu(self.proj_norm(self.proj(x)))
    x = jnp.mean(x, axis=(1, 2))
    return self.head(x)


# --- Model registry ---

ModelFactory = Callable[[PRNGKeyArray], eqx.Module]

MODEL_REGISTRY: dict[str, ModelFactory] = {
  "cnn": MazeCNN,
  "resnet": MazeResNet,
}

DEFAULT_ARCH = "cnn"


def make_model(arch: str, key: PRNGKeyArray) -> eqx.Module:
  """Create a model by architecture name.

  Raises KeyError with available architectures if name is unknown.
  """
  if arch not in MODEL_REGISTRY:
    available = ", ".join(sorted(MODEL_REGISTRY))
    raise KeyError(f"Unknown architecture {arch!r}. Available: {available}")
  return MODEL_REGISTRY[arch](key)
