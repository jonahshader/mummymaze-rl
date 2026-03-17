"""Model architectures for behavioral cloning on Mummy Maze observations."""

import inspect
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

# All model classes must:
#   - Be eqx.Module subclasses
#   - Accept (key: PRNGKeyArray) as sole __init__ arg (plus optional **hparams)
#   - Accept (x: Float[Array, "10 H W"]) and return Float[Array, "5"]
#   - Work for all grid sizes (6/8/10) via global average pooling

# --- Model registry ---

ModelFactory = Callable[..., eqx.Module]

MODEL_REGISTRY: dict[str, ModelFactory] = {}

DEFAULT_ARCH = "cnn"


def register_model(name: str) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
  """Decorator that registers a model class in MODEL_REGISTRY."""

  def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
    MODEL_REGISTRY[name] = cls
    return cls

  return decorator


def make_model(arch: str, key: PRNGKeyArray, **hparams: object) -> eqx.Module:
  """Create a model by architecture name, forwarding hparams to __init__.

  Raises KeyError with available architectures if name is unknown.
  """
  if arch not in MODEL_REGISTRY:
    available = ", ".join(sorted(MODEL_REGISTRY))
    raise KeyError(f"Unknown architecture {arch!r}. Available: {available}")
  return MODEL_REGISTRY[arch](key, **hparams)


def model_hparams(arch: str) -> dict[str, inspect.Parameter]:
  """Return the optional hparams (with defaults) for an architecture.

  Excludes `self` and `key` — returns only keyword-only params that
  the model's __init__ accepts beyond the required PRNG key.
  """
  if arch not in MODEL_REGISTRY:
    return {}
  sig = inspect.signature(MODEL_REGISTRY[arch].__init__)
  return {
    name: param
    for name, param in sig.parameters.items()
    if name not in ("self", "key") and param.default is not inspect.Parameter.empty
  }


def parse_hparams(arch: str, raw: list[str]) -> dict[str, object]:
  """Parse CLI key=value strings into typed hparams for an architecture.

  Uses the model's __init__ type hints to coerce string values.
  """
  available = model_hparams(arch)
  result: dict[str, object] = {}
  for item in raw:
    if "=" not in item:
      raise ValueError(f"Invalid hparam format {item!r}, expected key=value")
    k, v = item.split("=", 1)
    if k not in available:
      valid = ", ".join(sorted(available)) or "(none)"
      raise ValueError(f"Unknown hparam {k!r} for {arch!r}. Available: {valid}")
    hint = available[k].annotation
    if (
      hint is bool
      or hint is inspect.Parameter.empty
      and isinstance(available[k].default, bool)
    ):
      result[k] = v.lower() in ("true", "1", "yes")
    elif hint is int:
      result[k] = int(v)
    elif hint is float:
      result[k] = float(v)
    else:
      result[k] = v
  return result


# --- Shared building blocks ---


class ResNetStem(eqx.Module):
  """Conv stem + N residual blocks. Shared by ResNet and ResNet+Attention.

  Args:
    n_blocks: Number of residual blocks.
    channels: Channel width for all conv layers (must be divisible by 8).
    keys: Pre-split PRNG keys (needs 2*n_blocks + 1).
    attn_res: When True, replaces standard residual connections with Attention
      Residuals (Kimi Team, 2025): each block's input is a learned softmax-
      weighted sum of all previous block outputs.
  """

  stem: eqx.nn.Conv2d
  stem_norm: eqx.nn.GroupNorm
  res_conv1: list[eqx.nn.Conv2d]
  res_norm1: list[eqx.nn.GroupNorm]
  res_conv2: list[eqx.nn.Conv2d]
  res_norm2: list[eqx.nn.GroupNorm]
  attn_res: bool = eqx.field(static=True)
  n_blocks: int = eqx.field(static=True)
  channels: int = eqx.field(static=True)
  ar_queries: list[Array] | None

  def __init__(
    self,
    n_blocks: int,
    keys: list,
    *,
    channels: int = 64,
    attn_res: bool = False,
  ) -> None:
    self.n_blocks = n_blocks
    self.channels = channels
    self.attn_res = attn_res

    self.stem = eqx.nn.Conv2d(10, channels, 3, padding=1, key=keys[0])
    self.stem_norm = eqx.nn.GroupNorm(8, channels)

    self.res_conv1 = []
    self.res_norm1 = []
    self.res_conv2 = []
    self.res_norm2 = []
    for i in range(n_blocks):
      self.res_conv1.append(
        eqx.nn.Conv2d(channels, channels, 3, padding=1, key=keys[1 + 2 * i])
      )
      self.res_norm1.append(eqx.nn.GroupNorm(8, channels))
      self.res_conv2.append(
        eqx.nn.Conv2d(channels, channels, 3, padding=1, key=keys[2 + 2 * i])
      )
      self.res_norm2.append(eqx.nn.GroupNorm(8, channels))

    if attn_res:
      # Queries for blocks 1..n_blocks-1 (block 0 trivially gets v_0)
      # plus one final query for output aggregation. Init to zero so
      # initial attention weights are uniform (paper Section 5).
      self.ar_queries = [jnp.zeros(channels) for _ in range(n_blocks)]
    else:
      self.ar_queries = None

  def _ar_aggregate(
    self,
    query: Float[Array, "C"],
    values: list[Float[Array, "C H W"]],
  ) -> Float[Array, "C H W"]:
    """Depth-wise softmax attention over previous block outputs."""
    v_stack = jnp.stack(values)  # (L, C, H, W)
    # RMSNorm keys over channel dim
    rms = jnp.sqrt(jnp.mean(v_stack**2, axis=1, keepdims=True) + 1e-6)  # (L, 1, H, W)
    keys = v_stack / rms  # (L, C, H, W)
    # Dot product: query (C,) against each key (C, H, W) -> (L, H, W)
    logits = jnp.einsum("d, l d h w -> l h w", query, keys)
    weights = jax.nn.softmax(logits, axis=0)  # (L, H, W)
    return jnp.einsum("l h w, l d h w -> d h w", weights, v_stack)

  def __call__(self, x: Float[Array, "10 H W"]) -> Float[Array, "C H W"]:
    x = jax.nn.gelu(self.stem_norm(self.stem(x)))

    if not self.attn_res:
      # Standard residual connections
      for c1, n1, c2, n2 in zip(
        self.res_conv1, self.res_norm1, self.res_conv2, self.res_norm2
      ):
        residual = x
        x = jax.nn.gelu(n1(c1(x)))
        x = n2(c2(x))
        x = jax.nn.gelu(x + residual)
      return x

    # Attention Residuals: each block's input is a learned weighted
    # sum of all previous block outputs (Full AttnRes, L=n_blocks).
    assert self.ar_queries is not None
    values: list[Float[Array, "C H W"]] = [x]  # v_0 = stem output

    for i, (c1, n1, c2, n2) in enumerate(
      zip(
        self.res_conv1,
        self.res_norm1,
        self.res_conv2,
        self.res_norm2,
      )
    ):
      # Block 0 trivially gets v_0; blocks 1+ aggregate over history
      if i > 0:
        x = self._ar_aggregate(self.ar_queries[i - 1], values)

      # Block transformation (no skip connection — AttnRes replaces it)
      v = jax.nn.gelu(n1(c1(x)))
      v = n2(c2(v))
      values.append(v)

    # Final aggregation over all values, then activate
    return jax.nn.gelu(self._ar_aggregate(self.ar_queries[self.n_blocks - 1], values))


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

  Controlled by a single `width` multiplier (default 2):
    res_ch = 32 * width     (ResNet channel width)
    res_blocks = 2 * width  (number of residual blocks)
    d_model = 64 * width    (attention dimension)
    attn_blocks = width     (number of self-attention blocks)
    n_heads = 4             (fixed)

  width=1: ~80K params, width=2: ~577K (default), width=3: ~1.8M, width=4: ~4.2M

  Grid-size-agnostic: conv stem uses global-average-pool-compatible spatial
  ops, attention operates on (H*W) tokens with no fixed positional encoding.
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

  def __init__(
    self,
    key: PRNGKeyArray,
    *,
    width: int = 2,
    attn_res: bool = False,
  ) -> None:
    res_ch = 32 * width
    n_res_blocks = 2 * width
    d_model = 64 * width
    n_attn_blocks = width
    n_heads = 4

    self.d_model = d_model
    self.n_heads = n_heads

    n_keys = 2 * n_res_blocks + 4 * n_attn_blocks + 3
    keys = jax.random.split(key, n_keys)

    # ResNet stem (optionally with Attention Residuals)
    self.body = ResNetStem(n_res_blocks, keys, channels=res_ch, attn_res=attn_res)
    ki = 2 * n_res_blocks + 1

    # 1x1 projection to attention dimension
    self.proj = eqx.nn.Conv2d(res_ch, d_model, 1, key=keys[ki])
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
