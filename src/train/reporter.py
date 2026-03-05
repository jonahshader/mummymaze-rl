"""Metrics reporter protocol and implementations for training."""

import json
import select
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class MetricsReporter(Protocol):
  """Protocol for reporting training progress and metrics."""

  def report_init(self, config: dict) -> None: ...
  def report_epoch_start(self, epoch: int, total_epochs: int) -> None: ...
  def report_batch(self, step: int, loss: float, acc: float, gs: int) -> None: ...
  def report_epoch_end(
    self,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch_time: float,
  ) -> None: ...
  def report_level_metrics(
    self,
    step: int,
    run_id: str,
    metrics: dict,
    sources: dict,
  ) -> None: ...
  def report_done(self) -> None: ...
  def check_command(self) -> str | None: ...


class FileReporter:
  """File-based reporter. Writes level_metrics.json; tqdm handles display."""

  def __init__(self, metrics_path: Path) -> None:
    self.metrics_path = metrics_path

  def report_init(self, config: dict) -> None:
    pass

  def report_epoch_start(self, epoch: int, total_epochs: int) -> None:
    pass

  def report_batch(self, step: int, loss: float, acc: float, gs: int) -> None:
    pass

  def report_epoch_end(
    self,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch_time: float,
  ) -> None:
    pass

  def report_level_metrics(
    self,
    step: int,
    run_id: str,
    metrics: dict,
    sources: dict,
  ) -> None:
    from src.train.train_bc import write_level_metrics

    write_level_metrics(metrics, sources, step, run_id, self.metrics_path)

  def report_done(self) -> None:
    pass

  def check_command(self) -> str | None:
    return None


class StdioReporter:
  """Reporter that writes JSON lines to stdout for subprocess mode."""

  def __init__(self) -> None:
    pass

  def _emit(self, msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()

  def report_init(self, config: dict) -> None:
    self._emit({"type": "init", **config})

  def report_epoch_start(self, epoch: int, total_epochs: int) -> None:
    self._emit({"type": "epoch_start", "epoch": epoch, "total_epochs": total_epochs})

  def report_batch(self, step: int, loss: float, acc: float, gs: int) -> None:
    self._emit({"type": "batch", "step": step, "loss": loss, "acc": acc, "gs": gs})

  def report_epoch_end(
    self,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch_time: float,
  ) -> None:
    self._emit(
      {
        "type": "epoch_end",
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "time": epoch_time,
      }
    )

  def report_level_metrics(
    self,
    step: int,
    run_id: str,
    metrics: dict,
    sources: dict,
  ) -> None:
    # Build levels dict for JSON line
    levels: dict[str, object] = {}
    for gs, gs_metrics in metrics.items():
      src_list = sources.get(gs, [])
      for bank_idx, stats in gs_metrics.items():
        if bank_idx < len(src_list):
          file_stem, sublevel = src_list[bank_idx]
          key = f"{file_stem}:{sublevel}"
        else:
          key = f"gs{gs}:idx{bank_idx}"
        levels[key] = {"grid_size": gs, **stats}

    self._emit(
      {
        "type": "level_metrics",
        "step": step,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "levels": levels,
      }
    )

  def report_done(self) -> None:
    self._emit({"type": "done"})

  def check_command(self) -> str | None:
    """Non-blocking read from stdin for commands from parent process."""
    if select.select([sys.stdin], [], [], 0)[0]:
      line = sys.stdin.readline().strip()
      if line:
        try:
          msg = json.loads(line)
          return msg.get("cmd")
        except json.JSONDecodeError:
          return None
    return None
