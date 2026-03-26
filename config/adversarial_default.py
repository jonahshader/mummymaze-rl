# MAP-Elites adversarial training — default config
# Usage: uv run python -m src.train config/adversarial_default.py
#
# n_rounds > 1 triggers the outer loop with GA level generation.

n_rounds = 3
epochs = 5
batch_size = 1024
lr = 3e-4
arch = "cnn"
seed = 0
ga_generations = 50
ga_pop_size = 64
target_log_policy_wp = -1.0
checkpoint_dir = "checkpoints/adversarial"
