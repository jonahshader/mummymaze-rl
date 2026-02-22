# CLAUDE.md

This document contains useful information for new Claude Code instances regarding the mummymaze-rl project.

## Development Practices

We are using some cutting edge python stuff: uv, ruff, ty, python 3.13.

The deep learning stack is jax[cuda13], diffrax, equinox, optax.

For type checking and debugging, we use jaxtyping + beartype for runtime shape validation. Concrete dimensions (e.g., `Float[Array, "64 64 16"]`) validate exact sizes, while symbolic dimensions (e.g., `Float[Array, "height width channels"]`) validate consistency - all uses of "height" must have the same size within a function call, but that size can vary between calls.

We use the import hook pattern to apply beartype checking automatically to all modules without needing decorators on every function. The runtime overhead is negligible since beartype is O(1) and JAX JIT compiles away the checks after the first trace.

Avoid `__init__.py` re-exports - vulture can't trace them properly, causing false negatives. Import directly from submodules instead.

## Workflow

Run after completing any code changes:
```
uv run ruff check --fix && uv run ruff format && uv run ty check
```

Run after moderate/significant refactors:
```
uv run vulture
```
