from __future__ import annotations

import math

import numpy as np

from packplot.types import InitializationConfig


def _best_grid_dims(n_items: int, target_aspect_ratio: float) -> tuple[int, int]:
    best_rows = 1
    best_cols = n_items
    best_score = float("inf")
    for rows in range(1, n_items + 1):
        cols = int(math.ceil(n_items / rows))
        aspect = cols / max(1, rows)
        blanks = cols * rows - n_items
        score = abs(aspect - target_aspect_ratio) + 0.01 * blanks
        if score < best_score:
            best_score = score
            best_rows = rows
            best_cols = cols
    return best_rows, best_cols


def _grid_centers(n_items: int, target_aspect_ratio: float, spacing: float) -> np.ndarray:
    rows, cols = _best_grid_dims(n_items, target_aspect_ratio)
    centers = np.zeros((n_items, 2), dtype=float)
    for idx in range(n_items):
        row = idx // cols
        col = idx % cols
        centers[idx, 0] = col * spacing
        centers[idx, 1] = row * spacing
    centers -= np.mean(centers, axis=0, keepdims=True)
    return centers


def make_initial_center_layouts(
    *,
    n_items: int,
    target_aspect_ratio: float,
    config: InitializationConfig,
    seed: int | None,
) -> list[np.ndarray]:
    """Generate one or more initial center layouts for pack-phase solving."""
    if n_items <= 0:
        return [np.zeros((0, 2), dtype=float)]
    spacing = max(1e-3, float(config.grid_spacing))
    base = _grid_centers(n_items, target_aspect_ratio, spacing)
    method = config.method.lower().strip()
    if method == "grid":
        return [base]
    if method == "randomized_grid":
        rng = np.random.default_rng(seed if seed is not None else None)
        count = max(1, int(config.randomized_layout_count))
        layouts = [base]
        for _ in range(max(0, count - 1)):
            perm = rng.permutation(n_items)
            layouts.append(base[perm])
        return layouts
    raise ValueError("Unknown initialization method. Expected 'grid' or 'randomized_grid'.")
