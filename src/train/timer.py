"""Lightweight section timer for profiling training loops."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator


class SectionTimer:
  """Accumulates wall-clock time by named section.

  Usage::

      timer = SectionTimer()
      with timer("ga"):
          run_ga(...)
      with timer("train"):
          train_epochs(...)

      # Per-round deltas (resets snapshot):
      deltas = timer.round_times()
      # {"ga": 12.3, "train": 45.6}
  """

  def __init__(self) -> None:
    self._cumulative: dict[str, float] = defaultdict(float)
    self._snapshot: dict[str, float] = defaultdict(float)

  @contextmanager
  def __call__(self, label: str) -> Generator[None]:
    t0 = time.perf_counter()
    try:
      yield
    finally:
      self._cumulative[label] += time.perf_counter() - t0

  def round_times(self) -> dict[str, float]:
    """Return per-section time since last call, then reset snapshot."""
    deltas = {k: self._cumulative[k] - self._snapshot[k] for k in self._cumulative}
    self._snapshot = defaultdict(float, self._cumulative)
    return deltas

  def summary(self) -> str:
    """Format cumulative times as a table with percentages."""
    total = sum(self._cumulative.values())
    if total == 0:
      return "SectionTimer: no data"
    lines = ["Section          Time (s)    %"]
    for label in sorted(
      self._cumulative,
      key=lambda k: self._cumulative[k],
      reverse=True,
    ):
      t = self._cumulative[label]
      pct = 100.0 * t / total
      lines.append(f"  {label:<14s} {t:>8.1f}  {pct:>5.1f}%")
    lines.append(f"  {'TOTAL':<14s} {total:>8.1f}")
    return "\n".join(lines)
