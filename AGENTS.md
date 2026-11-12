# Indicator package notes

This folder is a standalone Pixi project for Numba compiled versions of the
numerical indicators in the adjacent *Picture Perfect Trading* manual.

- Keep indicator kernels in NumPy arrays and decorate public numerical
  functions with `numba.njit`.
- Do not use pandas or Polars inside compiled kernels.
- Commands are run from the Backtesting workspace root. Set `PYTHONPATH` to
  this project's `src` directory and pass this folder's `pixi.toml` to Pixi.
- The adjacent `research/` toolkit remains a separate project.

See the workspace `AGENTS.md` for cross-project conventions.
