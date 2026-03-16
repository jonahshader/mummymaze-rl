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

# --- Model registry ---

ModelFactory = Callable[[PRNGKeyArray], eqx.Module]

MODEL_REGISTRY: dict[str, ModelFactory] = {}

DEFAULT_ARCH = "cnn"


def register_model(name: str) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
  """Decorator that registers a model class in MODEL_REGISTRY."""

  def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
    MODEL_REGISTRY[name] = cls
    return cls

  return decorator


def make_model(arch: str, key: PRNGKeyArray) -> eqx.Module:
  """Create a model by architecture name.

  Raises KeyError with available architectures if name is unknown.
  """
  if arch not in MODEL_REGISTRY:
    available = ", ".join(sorted(MODEL_REGISTRY))
    raise KeyError(f"Unknown architecture {arch!r}. Available: {available}")
  return MODEL_REGISTRY[arch](key)


# --- Shared building blocks ---


class ResNetStem(eqx.Module):
  """Conv stem + N residual blocks. Shared by ResNet and ResNet+Attention."""

  stem: eqx.nn.Conv2d
  stem_norm: eqx.nn.GroupNorm
  res_conv1: list[eqx.nn.Conv2d]
  res_norm1: list[eqx.nn.GroupNorm]
  res_conv2: list[eqx.nn.Conv2d]
  res_norm2: list[eqx.nn.GroupNorm]

  def __init__(self, n_blocks: int, keys: list) -> None:
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

  def __call__(self, x: Float[Array, "10 H W"]) -> Float[Array, "64 H W"]:
    x = jax.nn.gelu(self.stem_norm(self.stem(x)))
    for c1, n1, c2, n2 in zip(
      self.res_conv1, self.res_norm1, self.res_conv2, self.res_norm2
    ):
      residual = x
      x = jax.nn.gelu(n1(c1(x)))
      x = n2(c2(x))
      x = jax.nn.gelu(x + residual)
    return x


# --- Architectures ---


@register_model("cnn")
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


@register_model("resnet")
class MazeResNet(eqx.Module):
  """Deeper residual CNN. Same I/O contract as MazeCNN.

  Architecture:
    Conv2d(10->64, 3x3) -> GroupNorm -> GELU
    4x ResBlock(64->64): Conv->GN->GELU->Conv->GN + skip -> GELU
    Conv2d(64->128, 3x3) -> GroupNorm -> GELU
    GlobalAvgPool -> Linear(128->5)

  ~340K parameters.
  """

  body: ResNetStem
  proj: eqx.nn.Conv2d
  proj_norm: eqx.nn.GroupNorm
  head: eqx.nn.Linear

  def __init__(self, key: PRNGKeyArray) -> None:
    n_blocks = 4
    keys = jax.random.split(key, 2 * n_blocks + 3)

    self.body = ResNetStem(n_blocks, keys)
    self.proj = eqx.nn.Conv2d(64, 128, 3, padding=1, key=keys[-2])
    self.proj_norm = eqx.nn.GroupNorm(8, 128)
    self.head = eqx.nn.Linear(128, 5, key=keys[-1])

  def __call__(self, x: Float[Array, "10 H W"]) -> Float[Array, "5"]:
    x = self.body(x)
    x = jax.nn.gelu(self.proj_norm(self.proj(x)))
    x = jnp.mean(x, axis=(1, 2))
    return self.head(x)


@register_model("resnet-attn")
class MazeResNetAttn(eqx.Module):
  """ResNet stem + multi-head self-attention for global reasoning.

  Architecture:
    Conv2d(10->64, 3x3) -> GroupNorm -> GELU
    4x ResBlock(64->64): Conv->GN->GELU->Conv->GN + skip -> GELU
    Conv2d(64->128, 1x1) -> GroupNorm -> GELU
    2x SelfAttention(128, 4 heads) with pre-norm residual
    GlobalAvgPool -> Linear(128->5)

  The ResNet stem extracts local features (walls, entities). Self-attention
  layers then reason globally over the spatial feature map — every cell can
  attend to every other cell in a single layer, enabling relational reasoning
  about connectivity, threats, and paths that CNNs require many layers for.

  Grid-size-agnostic: attention operates on (H*W) tokens, no fixed positional
  encoding needed since the conv stem already encodes spatial structure.
  """

  body: ResNetStem
  proj: eqx.nn.Conv2d
  proj_norm: eqx.nn.GroupNorm
  attn_norm: list[eqx.nn.LayerNorm]
  attn_qkv: list[eqx.nn.Linear]
  attn_out: list[eqx.nn.Linear]
  ffn_norm: list[eqx.nn.LayerNorm]
  ffn_up: list[eqx.nn.Linear]
  ffn_down: list[eqx.nn.Linear]
  head: eqx.nn.Linear

  n_heads: int = eqx.field(static=True)
  d_model: int = eqx.field(static=True)

  def __init__(self, key: PRNGKeyArray) -> None:
    n_res_blocks = 4
    n_attn_blocks = 2
    d_model = 128
    n_heads = 4

    self.d_model = d_model
    self.n_heads = n_heads

    n_keys = 2 * n_res_blocks + 4 * n_attn_blocks + 3
    keys = jax.random.split(key, n_keys)

    # ResNet stem (uses first 2*n_res_blocks + 1 keys)
    self.body = ResNetStem(n_res_blocks, keys)
    ki = 2 * n_res_blocks + 1

    # 1x1 projection to attention dimension
    self.proj = eqx.nn.Conv2d(64, d_model, 1, key=keys[ki])
    ki += 1
    self.proj_norm = eqx.nn.GroupNorm(8, d_model)

    # Self-attention blocks (pre-norm residual)
    self.attn_norm = []
    self.attn_qkv = []
    self.attn_out = []
    self.ffn_norm = []
    self.ffn_up = []
    self.ffn_down = []
    for _ in range(n_attn_blocks):
      self.attn_norm.append(eqx.nn.LayerNorm(d_model))
      self.attn_qkv.append(eqx.nn.Linear(d_model, 3 * d_model, key=keys[ki]))
      ki += 1
      self.attn_out.append(eqx.nn.Linear(d_model, d_model, key=keys[ki]))
      ki += 1
      self.ffn_norm.append(eqx.nn.LayerNorm(d_model))
      self.ffn_up.append(eqx.nn.Linear(d_model, d_model * 2, key=keys[ki]))
      ki += 1
      self.ffn_down.append(eqx.nn.Linear(d_model * 2, d_model, key=keys[ki]))
      ki += 1

    self.head = eqx.nn.Linear(d_model, 5, key=keys[ki])

  def _attention(
    self,
    x: Float[Array, "S D"],
    qkv: eqx.nn.Linear,
    out: eqx.nn.Linear,
  ) -> Float[Array, "S D"]:
    """Multi-head self-attention over sequence of tokens."""
    seq_len = x.shape[0]
    d_head = self.d_model // self.n_heads

    # Project to Q, K, V
    qkv_out = jax.vmap(qkv)(x)  # (S, 3*D)
    q, k, v = jnp.split(qkv_out, 3, axis=-1)  # each (S, D)

    # Reshape to (H, S, d_head)
    q = q.reshape(seq_len, self.n_heads, d_head).transpose(1, 0, 2)
    k = k.reshape(seq_len, self.n_heads, d_head).transpose(1, 0, 2)
    v = v.reshape(seq_len, self.n_heads, d_head).transpose(1, 0, 2)

    # Scaled dot-product attention
    scale = jnp.sqrt(jnp.float32(d_head))
    attn = jnp.matmul(q, k.transpose(0, 2, 1)) / scale  # (H, S, S)
    attn = jax.nn.softmax(attn, axis=-1)
    out_val = jnp.matmul(attn, v)  # (H, S, d_head)

    # Reshape back to (S, D)
    out_val = out_val.transpose(1, 0, 2).reshape(seq_len, self.d_model)
    return jax.vmap(out)(out_val)

  def __call__(self, x: Float[Array, "10 H W"]) -> Float[Array, "5"]:
    # ResNet stem: local feature extraction
    x = self.body(x)

    # Project to attention dim
    x = jax.nn.gelu(self.proj_norm(self.proj(x)))  # (D, H, W)

    # Flatten spatial dims to token sequence: (D, H, W) -> (H*W, D)
    _d, h, w = x.shape
    x = x.reshape(self.d_model, h * w).T

    # Self-attention blocks with pre-norm residual
    for a_norm, qkv, out, f_norm, f_up, f_down in zip(
      self.attn_norm,
      self.attn_qkv,
      self.attn_out,
      self.ffn_norm,
      self.ffn_up,
      self.ffn_down,
    ):
      # Attention
      normed = jax.vmap(a_norm)(x)
      x = x + self._attention(normed, qkv, out)
      # FFN
      normed = jax.vmap(f_norm)(x)
      x = x + jax.vmap(f_down)(jax.nn.gelu(jax.vmap(f_up)(normed)))

    # Global average pool over tokens: (S, D) -> (D,)
    x = jnp.mean(x, axis=0)
    return self.head(x)
