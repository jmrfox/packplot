from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image
    from shapely.geometry import Polygon


@dataclass(frozen=True)
class LbfgsbConfig:
    """Configuration for the L-BFGS-B local optimizer."""

    max_iterations: int = 800
    random_restart_count: int = 12
    alternating_refinement_cycles: int = 4


@dataclass(frozen=True)
class DifferentialEvolutionConfig:
    """Configuration for differential evolution (genetic-style search)."""

    max_generations: int = 35
    population_size: int = 10
    worker_count: int = 1


@dataclass(frozen=True)
class OptimizationPhaseConfig:
    """Optimizer settings for one optimization stage."""

    method: str = "lbfgsb"  # "lbfgsb", "de", "hybrid", "nsga2"
    progress_log_every_evaluations: int = 500
    progress_log_heartbeat_seconds: float = 5.0
    lbfgsb: LbfgsbConfig = field(default_factory=LbfgsbConfig)
    differential_evolution: DifferentialEvolutionConfig = field(default_factory=DifferentialEvolutionConfig)


@dataclass(frozen=True)
class CompactLayoutObjectiveConfig:
    """Objective weights used during compact-layout optimization."""

    overlap_penalty_weight: float = 1e5
    aspect_ratio_penalty_weight: float = 1e3
    jacobi_regularization_weight: float = 0.1


@dataclass(frozen=True)
class ClearanceRefinementObjectiveConfig:
    """Objective weights used during fixed-canvas clearance refinement."""

    softmin_smoothness: float = 20.0
    lower_quartile_spacing_weight: float = 0.35
    mean_spacing_weight: float = 0.1
    overlap_penalty_weight: float = 2e5
    outside_canvas_penalty_weight: float = 2e5
    center_shift_regularization_weight: float = 1e-2


@dataclass(frozen=True)
class OptimizeConfig:
    """Complete optimization configuration grouped by phase and objective."""

    compact_layout_backend: str = "optimize"  # "optimize" or "pymoo"
    compact_layout: OptimizationPhaseConfig = field(default_factory=OptimizationPhaseConfig)
    compact_layout_best_count: int = 1
    compact_to_clearance_beam_width: int = 1
    enable_clearance_refinement_phase: bool = True
    clearance_refinement: OptimizationPhaseConfig = field(default_factory=OptimizationPhaseConfig)
    compact_layout_objective: CompactLayoutObjectiveConfig = field(default_factory=CompactLayoutObjectiveConfig)
    clearance_refinement_objective: ClearanceRefinementObjectiveConfig = field(
        default_factory=ClearanceRefinementObjectiveConfig
    )


@dataclass(frozen=True)
class PymooConfig:
    """Configuration for the multi-objective pymoo solver."""

    algorithm: str = "nsga2"
    generations: int = 60
    population_size: int = 48
    offspring_count: int | None = None
    eliminate_duplicates: bool = True
    best_layout_count: int = 1
    verbose: bool = True


@dataclass(frozen=True)
class InitializationConfig:
    """Configuration for compact-layout initialization before optimization."""

    method: str = "grid"  # "grid" or "randomized_grid"
    grid_spacing: float = 1.25
    randomized_layout_count: int = 8


@dataclass(frozen=True)
class PackOptions:
    """User-facing options for extraction, packing, and optimization behavior."""

    target_aspect_ratio: float = 1.0
    padding: int = 2
    edge_buffer: float = 1.0
    jacobi_inflation: float = 1.1
    allow_flip: bool = False
    rotation_step_degrees: int = 15
    white_threshold: int = 245
    alpha_threshold: int = 1
    fill_ratio: float = 0.72
    max_grow_steps: int = 12
    grow_factor: float = 1.15
    initialization_config: InitializationConfig = field(default_factory=InitializationConfig)
    optimize_config: OptimizeConfig = field(default_factory=OptimizeConfig)
    pymoo_config: PymooConfig = field(default_factory=PymooConfig)
    random_seed: int | None = 0

    def resolved_optimize_config(self) -> OptimizeConfig:
        """Return optimization config currently attached to this options object."""
        return self.optimize_config


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
    total_overlap_area: float = 0.0
    out_of_bounds_count: int = 0
    sanity_passed: bool = True
    minimum_clearance: float | None = None
    outside_violation_magnitude: float | None = None
    solver_method: str | None = None
    solver_iterations: int | None = None
    solver_success: bool | None = None
    objective_values: tuple[float, float, float] | None = None
    rank: int = 1


