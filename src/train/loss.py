"""Loss functions and metrics for behavioral cloning."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def cross_entropy_loss(
  logits: Float[Array, "B 5"],
  targets: Float[Array, "B 5"],
) -> Float[Array, ""]:
  """Cross-entropy against soft targets: -sum(targets * log_softmax(logits))."""
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))


def top1_accuracy(
  logits: Float[Array, "B 5"],
  targets: Float[Array, "B 5"],
) -> Float[Array, ""]:
  """Fraction where argmax(logits) is an optimal action."""
  preds = jnp.argmax(logits, axis=-1)
  # An action is correct if target > 0 for that action
  correct = jnp.take_along_axis(targets, preds[:, None], axis=1).squeeze(-1)
  return jnp.mean(correct > 0)
