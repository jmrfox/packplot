from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image
    from shapely.geometry import Polygon


@dataclass(frozen=True)
class PackOptions:
    """User-facing options for extraction, packing, and optimization behavior."""

    solver: str = "optimize"
    target_aspect_ratio: float = 1.0
    padding: int = 2
    edge_buffer: float = 1.0
    jacobi_inflation: float = 1.1
    allow_flip: bool = True
    rotation_step_degrees: int = 15
    white_threshold: int = 245
    alpha_threshold: int = 1
    fill_ratio: float = 0.72
    max_grow_steps: int = 12
    grow_factor: float = 1.15
    optimizer_maxiter: int = 800
    optimizer_method: str = "lbfgsb"
    optimizer_workers: int = 1
    optimizer_de_maxiter: int = 35
    optimizer_de_popsize: int = 10
    optimizer_restarts: int = 12
    optimizer_alternating_cycles: int = 4
    optimizer_progress_interval: int = 4000
    optimizer_overlap_weight: float = 1e5
    optimizer_aspect_weight: float = 1e3
    optimizer_regularization: float = 0.1
    enable_spread_phase: bool = True
    spread_maxiter: int = 300
    spread_method: str = "lbfgsb"
    spread_workers: int = 1
    spread_de_maxiter: int = 35
    spread_de_popsize: int = 10
    spread_restarts: int = 6
    spread_progress_interval: int = 2000
    spread_smoothness: float = 20.0
    spread_quantile_weight: float = 0.35
    spread_mean_weight: float = 0.1
    spread_overlap_weight: float = 2e5
    spread_outside_weight: float = 2e5
    spread_regularization: float = 1e-2
    random_seed: int | None = 0


@dataclass
class SourceObject:
    """Extracted object artifact from one source image."""

    source_path: Path
    cropped_image: Image.Image
    mask: np.ndarray
    hull: Polygon
    background_color: tuple[int, int, int]


@dataclass
class PackedPlacement:
    """Final placement metadata for one packed object."""

    source_path: Path
    polygon: Polygon
    angle_degrees: float
    flipped: bool
    top_left: tuple[int, int]
    image: Image.Image


@dataclass
class PackResult:
    """Result returned by `pack_images`, including image and placement metadata."""

    image: Image.Image
    placements: list[PackedPlacement]
    canvas_size: tuple[int, int]
    target_aspect_ratio: float
    fill_ratio: float
    background_color: tuple[int, int, int]
