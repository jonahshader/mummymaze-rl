# BC training with custom components — example config
# Usage: uv run python -m src.train config/bc_custom.py
#
# Demonstrates swappable loss, optimizer, and stop conditions via lambdas.

import optax

from src.train.loss import cross_entropy_loss, top1_accuracy
from src.train.stopping import any_of, stop_after

mode = "bc"
epochs = 20
batch_size = 1024
arch = "cnn"
seed = 0

# Custom optimizer (no warmup, just adam)
optimizer = optax.adam(3e-4)

# Custom stopping: stop after 20 epochs OR if manually interrupted
inner_stop = any_of(stop_after(20))

# Standard loss/metric (override these to experiment)
loss_fn = cross_entropy_loss
metric_fn = top1_accuracy
