# Refactor Ideas

Ideas and improvements discovered during the `src/train/` refactoring.
Discuss after the initial refactor is complete.

## Discovered During Refactoring

### `adversarial_loop.py` is still a monolith (430+ lines)
The GA phase, archive management, seed selection, and wandb logging are all interleaved in one function. The GA phase could take a callback for the training step rather than inlining the `train_epochs` call. See Step 3 of `TRAINING_REFACTOR_PLAN.md`.

### `reporter` protocol could absorb `log_fn`
The `MetricsReporter` protocol already has `report_batch`, `report_epoch_end`, etc. The `log_fn` callback runs alongside it with similar data. Consider whether `reporter` should be the single sink for all metrics, with wandb as a reporter implementation rather than a separate callback.
