"""Path helpers for experiment notebooks and scripts."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start=None) -> Path:
    """Find the repository root by walking upward from ``start`` or CWD."""
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "experiments").exists():
            return candidate
    raise RuntimeError("Could not find stochastic_simulations project root.")
