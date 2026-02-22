"""Example module demonstrating jaxtyping + beartype + equinox patterns."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
  """Simple MLP with typed layers."""

  layers: list[eqx.nn.Linear]

  def __init__(self, key: PRNGKeyArray, dims: list[int]) -> None:
    keys = jax.random.split(key, len(dims) - 1)
    self.layers = [
      eqx.nn.Linear(d_in, d_out, key=k)
      for k, d_in, d_out in zip(keys, dims[:-1], dims[1:], strict=True)
    ]

  def __call__(self, x: Float[Array, " dim"]) -> Float[Array, " dim"]:
    for layer in self.layers[:-1]:
      x = jax.nn.relu(layer(x))
    return self.layers[-1](x)


def batched_forward(
  model: MLP, x: Float[Array, "batch dim"]
) -> Float[Array, "batch dim"]:
  """Apply model to a batch - jaxtyping ensures batch dimension consistency."""
  return jax.vmap(model)(x)


def main() -> None:
  key = jax.random.key(42)
  model = MLP(key, dims=[8, 32, 32, 8])

  # Single input
  x = jnp.ones(8)
  y = model(x)
  print(f"Single: {x.shape} -> {y.shape}")

  # Batched input - jaxtyping validates "batch" is consistent
  x_batch = jnp.ones((4, 8))
  y_batch = batched_forward(model, x_batch)
  print(f"Batched: {x_batch.shape} -> {y_batch.shape}")
