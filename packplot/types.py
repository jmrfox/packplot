from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    """Optimizer settings for one phase (`phase1` or `spread`)."""

    method: str = "lbfgsb"  # "lbfgsb", "de", "hybrid"
    progress_log_every_evaluations: int = 500
    progress_log_heartbeat_seconds: float = 5.0
    lbfgsb: LbfgsbConfig = field(default_factory=LbfgsbConfig)
    differential_evolution: DifferentialEvolutionConfig = field(default_factory=DifferentialEvolutionConfig)


@dataclass(frozen=True)
class Phase1ObjectiveConfig:
    """Objective weights used during initial compact-layout optimization."""

    overlap_penalty_weight: float = 1e5
    aspect_ratio_penalty_weight: float = 1e3
    jacobi_regularization_weight: float = 0.1


@dataclass(frozen=True)
class SpreadObjectiveConfig:
    """Objective weights used during fixed-canvas spread refinement."""

    softmin_smoothness: float = 20.0
    lower_quartile_spacing_weight: float = 0.35
    mean_spacing_weight: float = 0.1
    overlap_penalty_weight: float = 2e5
    outside_canvas_penalty_weight: float = 2e5
    center_shift_regularization_weight: float = 1e-2


@dataclass(frozen=True)
class OptimizeConfig:
    """Complete optimization configuration grouped by phase and objective."""

    phase1: OptimizationPhaseConfig = field(default_factory=OptimizationPhaseConfig)
    enable_spread_phase: bool = True
    spread: OptimizationPhaseConfig = field(default_factory=OptimizationPhaseConfig)
    phase1_objective: Phase1ObjectiveConfig = field(default_factory=Phase1ObjectiveConfig)
    spread_objective: SpreadObjectiveConfig = field(default_factory=SpreadObjectiveConfig)


@dataclass(frozen=True)
class CircPackerConfig:
    """Configuration for circle-slot packing via the `circpacker` library."""

    initial_depth: int = 4
    max_depth: int = 8
    max_canvas_growth_steps: int = 8
    canvas_growth_factor: float = 1.2


@dataclass(frozen=True)
class CellPackingConfig:
    """Configuration for polygon relaxation via the `cell-packing` library."""

    iterations: int = 300
    attraction_step: float = 0.7
    repulsion_step: float = 2.0
    local_source_checkout_path: str = "third_party/cell-packing"


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
    optimize_config: OptimizeConfig = field(default_factory=OptimizeConfig)
    circpacker_config: CircPackerConfig = field(default_factory=CircPackerConfig)
    cell_packing_config: CellPackingConfig = field(default_factory=CellPackingConfig)

    # Legacy compatibility aliases (deprecated in favor of optimize_config).
    optimizer_maxiter: int | None = None
    optimizer_method: str | None = None
    optimizer_workers: int | None = None
    optimizer_de_maxiter: int | None = None
    optimizer_de_popsize: int | None = None
    optimizer_restarts: int | None = None
    optimizer_alternating_cycles: int | None = None
    optimizer_progress_interval: int | None = None
    optimizer_overlap_weight: float | None = None
    optimizer_aspect_weight: float | None = None
    optimizer_regularization: float | None = None
    enable_spread_phase: bool | None = None
    spread_maxiter: int | None = None
    spread_method: str | None = None
    spread_workers: int | None = None
    spread_de_maxiter: int | None = None
    spread_de_popsize: int | None = None
    spread_restarts: int | None = None
    spread_progress_interval: int | None = None
    spread_smoothness: float | None = None
    spread_quantile_weight: float | None = None
    spread_mean_weight: float | None = None
    spread_overlap_weight: float | None = None
    spread_outside_weight: float | None = None
    spread_regularization: float | None = None
    random_seed: int | None = 0

    def resolved_optimize_config(self) -> OptimizeConfig:
        """Return optimization config after applying legacy alias overrides."""
        cfg = self.optimize_config
        phase1 = cfg.phase1
        spread = cfg.spread
        phase1_lbfgsb = phase1.lbfgsb
        spread_lbfgsb = spread.lbfgsb
        phase1_de = phase1.differential_evolution
        spread_de = spread.differential_evolution
        phase1_obj = cfg.phase1_objective
        spread_obj = cfg.spread_objective

        if self.optimizer_method is not None:
            phase1 = replace(phase1, method=self.optimizer_method)
        if self.optimizer_progress_interval is not None:
            phase1 = replace(phase1, progress_log_every_evaluations=self.optimizer_progress_interval)
        if self.optimizer_maxiter is not None:
            phase1_lbfgsb = replace(phase1_lbfgsb, max_iterations=self.optimizer_maxiter)
        if self.optimizer_restarts is not None:
            phase1_lbfgsb = replace(phase1_lbfgsb, random_restart_count=self.optimizer_restarts)
        if self.optimizer_alternating_cycles is not None:
            phase1_lbfgsb = replace(
                phase1_lbfgsb,
                alternating_refinement_cycles=self.optimizer_alternating_cycles,
            )
        if self.optimizer_de_maxiter is not None:
            phase1_de = replace(phase1_de, max_generations=self.optimizer_de_maxiter)
        if self.optimizer_de_popsize is not None:
            phase1_de = replace(phase1_de, population_size=self.optimizer_de_popsize)
        if self.optimizer_workers is not None:
            phase1_de = replace(phase1_de, worker_count=self.optimizer_workers)

        if self.optimizer_overlap_weight is not None:
            phase1_obj = replace(phase1_obj, overlap_penalty_weight=self.optimizer_overlap_weight)
        if self.optimizer_aspect_weight is not None:
            phase1_obj = replace(phase1_obj, aspect_ratio_penalty_weight=self.optimizer_aspect_weight)
        if self.optimizer_regularization is not None:
            phase1_obj = replace(phase1_obj, jacobi_regularization_weight=self.optimizer_regularization)

        if self.enable_spread_phase is not None:
            cfg = replace(cfg, enable_spread_phase=self.enable_spread_phase)
        if self.spread_method is not None:
            spread = replace(spread, method=self.spread_method)
        if self.spread_progress_interval is not None:
            spread = replace(spread, progress_log_every_evaluations=self.spread_progress_interval)
        if self.spread_maxiter is not None:
            spread_lbfgsb = replace(spread_lbfgsb, max_iterations=self.spread_maxiter)
        if self.spread_restarts is not None:
            spread_lbfgsb = replace(spread_lbfgsb, random_restart_count=self.spread_restarts)
        if self.spread_de_maxiter is not None:
            spread_de = replace(spread_de, max_generations=self.spread_de_maxiter)
        if self.spread_de_popsize is not None:
            spread_de = replace(spread_de, population_size=self.spread_de_popsize)
        if self.spread_workers is not None:
            spread_de = replace(spread_de, worker_count=self.spread_workers)

        if self.spread_smoothness is not None:
            spread_obj = replace(spread_obj, softmin_smoothness=self.spread_smoothness)
        if self.spread_quantile_weight is not None:
            spread_obj = replace(spread_obj, lower_quartile_spacing_weight=self.spread_quantile_weight)
        if self.spread_mean_weight is not None:
            spread_obj = replace(spread_obj, mean_spacing_weight=self.spread_mean_weight)
        if self.spread_overlap_weight is not None:
            spread_obj = replace(spread_obj, overlap_penalty_weight=self.spread_overlap_weight)
        if self.spread_outside_weight is not None:
            spread_obj = replace(spread_obj, outside_canvas_penalty_weight=self.spread_outside_weight)
        if self.spread_regularization is not None:
            spread_obj = replace(spread_obj, center_shift_regularization_weight=self.spread_regularization)

        phase1 = replace(phase1, lbfgsb=phase1_lbfgsb, differential_evolution=phase1_de)
        spread = replace(spread, lbfgsb=spread_lbfgsb, differential_evolution=spread_de)
        return replace(
            cfg,
            phase1=phase1,
            spread=spread,
            phase1_objective=phase1_obj,
            spread_objective=spread_obj,
        )


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
