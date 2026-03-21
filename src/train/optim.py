"""Shared optimizer construction and utilities."""

import equinox as eqx
import jax
import optax


def make_optimizer(lr: float, total_steps: int) -> optax.GradientTransformation:
  """Warmup cosine decay schedule with gradient clipping and Adam."""
  schedule = optax.warmup_cosine_decay_schedule(
    init_value=lr * 0.1,
    peak_value=lr,
    warmup_steps=min(500, total_steps // 10),
    decay_steps=total_steps,
    end_value=lr * 0.01,
  )
  return optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(schedule),
  )


def count_params(model: eqx.Module) -> int:
  """Count total trainable parameters in an equinox model."""
  return sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
