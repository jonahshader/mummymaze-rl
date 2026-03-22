"""nanoGPT-style configurator: exec() Python config files + CLI overrides.

Config files are plain .py files that set variables. They can use lambdas,
imports, computed values — anything Python can produce. CLI overrides are
--key=value strings that coerce types to match existing defaults.

Usage:
  config = load_config("config/bc_default.py", ["--lr=1e-4", "--seed=42"])
"""

import ast
import sys


def load_config(
  config_file: str | None = None,
  overrides: list[str] | None = None,
  defaults: dict[str, object] | None = None,
) -> dict[str, object]:
  """Load config from a .py file + CLI --key=value overrides.

  1. Start with defaults (if provided)
  2. exec() config file into namespace (if provided)
  3. Apply --key=value overrides, coercing types to match existing values

  Unknown keys in overrides raise ValueError. Config files can introduce
  new keys freely.
  """
  config: dict[str, object] = dict(defaults) if defaults else {}

  # Step 2: exec config file
  if config_file is not None:
    with open(config_file) as f:
      source = f.read()
    # exec into a clean namespace so config files can't pollute globals
    ns: dict[str, object] = {}
    exec(compile(source, config_file, "exec"), ns)  # noqa: S102
    # Merge non-dunder, non-module keys
    for k, v in ns.items():
      if not k.startswith("_") and not hasattr(v, "__module__"):
        config[k] = v

  # Step 3: apply CLI overrides
  for item in overrides or []:
    item = item.lstrip("-")
    if "=" not in item:
      raise ValueError(f"Invalid override {item!r}, expected --key=value")
    k, v_str = item.split("=", 1)
    k = k.replace("-", "_")

    if k not in config:
      raise ValueError(
        f"Unknown config key {k!r}. Available: {', '.join(sorted(config))}"
      )

    # Coerce type to match existing value
    existing = config[k]
    config[k] = _coerce(v_str, existing)

  return config


def _coerce(v_str: str, existing: object) -> object:
  """Coerce a string value to match the type of an existing value."""
  if existing is None:
    # No type hint — try literal_eval, fall back to string
    return _try_literal(v_str)

  if isinstance(existing, bool):
    return v_str.lower() in ("true", "1", "yes")

  if isinstance(existing, int):
    return int(v_str)

  if isinstance(existing, float):
    return float(v_str)

  if isinstance(existing, str):
    return v_str

  if isinstance(existing, list):
    return ast.literal_eval(v_str)

  if isinstance(existing, dict):
    return ast.literal_eval(v_str)

  # Fallback: try literal_eval
  return _try_literal(v_str)


def _try_literal(v_str: str) -> object:
  """Try ast.literal_eval, fall back to string."""
  try:
    return ast.literal_eval(v_str)
  except (ValueError, SyntaxError):
    return v_str


def parse_argv(
  argv: list[str] | None = None,
) -> tuple[str | None, list[str]]:
  """Split argv into (config_file, overrides).

  The first positional argument (not starting with --) that ends in .py
  is the config file. Everything else starting with -- is an override.
  """
  if argv is None:
    argv = sys.argv[1:]

  config_file: str | None = None
  overrides: list[str] = []

  for arg in argv:
    if arg.startswith("--"):
      overrides.append(arg)
    elif config_file is None and arg.endswith(".py"):
      config_file = arg
    else:
      raise ValueError(f"Unexpected argument: {arg!r}")

  return config_file, overrides
