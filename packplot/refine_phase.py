"""Refine phase: improve spacing between shapes on a fixed canvas.

Contains the refine-phase objective, center-based variable mapping, the
numerical solver, overlap repair, integer-grid cleanup, and a quality
gate that rejects refinements that make the layout worse.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import OptimizeResult
from shapely import affinity
from shapely.geometry import Polygon

from packplot.layout_metrics import (
    buffered,
    clearance_stats,
    clearance_values,
    outside_violation,
    softmin,
    total_overlap,
)
from packplot.optimizer import ProgressTracker, run_with_method
from packplot.types import PipelineConfig, PackOptions, PackedPlacement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClearanceAsset:
    source_path: Path
    image: Image.Image
    local_polygon: Polygon
    angle_degrees: float
    flipped: bool


@dataclass
class ClearanceRefinementRunResult:
    success: bool
    nit: int | None
    message: str


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def project_centers_inside_canvas(
    centers: np.ndarray,
    assets: list[ClearanceAsset],
    width: int,
    height: int,
) -> np.ndarray:
    projected = centers.copy()
    for idx, asset in enumerate(assets):
        local_cx = asset.local_polygon.centroid.x
        local_cy = asset.local_polygon.centroid.y
        min_cx = local_cx
        max_cx = max(min_cx, width - (asset.image.width - local_cx))
        min_cy = local_cy
        max_cy = max(min_cy, height - (asset.image.height - local_cy))
        projected[idx, 0] = float(np.clip(projected[idx, 0], min_cx, max_cx))
        projected[idx, 1] = float(np.clip(projected[idx, 1], min_cy, max_cy))
    return projected


def polygon_at_center(local_polygon: Polygon, center_xy: np.ndarray) -> Polygon:
    cx, cy = local_polygon.centroid.x, local_polygon.centroid.y
    return affinity.translate(local_polygon, xoff=float(center_xy[0] - cx), yoff=float(center_xy[1] - cy))


def polygons_from_centers(assets: list[ClearanceAsset], centers: np.ndarray) -> list[Polygon]:
    return [polygon_at_center(assets[i].local_polygon, centers[i]) for i in range(len(assets))]


def placements_from_centers(
    assets: list[ClearanceAsset],
    centers: np.ndarray,
    width: int,
    height: int,
) -> list[PackedPlacement]:
    placements: list[PackedPlacement] = []
    for idx, asset in enumerate(assets):
        local_cx, local_cy = asset.local_polygon.centroid.x, asset.local_polygon.centroid.y
        x = int(round(centers[idx, 0] - local_cx))
        y = int(round(centers[idx, 1] - local_cy))
        x = min(max(0, x), max(0, width - asset.image.width))
        y = min(max(0, y), max(0, height - asset.image.height))
        polygon = affinity.translate(asset.local_polygon, xoff=x, yoff=y)
        placements.append(
            PackedPlacement(
                source_path=asset.source_path,
                polygon=polygon,
                angle_degrees=asset.angle_degrees,
                flipped=asset.flipped,
                top_left=(x, y),
                image=asset.image,
            )
        )
    return placements


# ---------------------------------------------------------------------------
# Overlap repair and integer-grid cleanup
# ---------------------------------------------------------------------------

def repair_centers_for_overlap(
    assets: list[ClearanceAsset],
    centers: np.ndarray,
    width: int,
    height: int,
    options: PackOptions,
    *,
    iterations: int = 60,
) -> np.ndarray:
    repaired = project_centers_inside_canvas(centers, assets, width, height)
    n = len(assets)
    for _ in range(iterations):
        polygons = polygons_from_centers(assets, repaired)
        buffered_polys = buffered(polygons, options)
        overlap = total_overlap(buffered_polys)
        if overlap <= 1e-4:
            break

        displacements = np.zeros_like(repaired)
        for i in range(n):
            for j in range(i + 1, n):
                area = buffered_polys[i].intersection(buffered_polys[j]).area
                if area <= 1e-6:
                    continue
                direction = repaired[i] - repaired[j]
                norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    direction = np.array([1.0 if (i + j) % 2 == 0 else -1.0, 0.0], dtype=float)
                    norm = 1.0
                push = min(8.0, 0.2 + 0.05 * math.sqrt(area))
                unit = direction / norm
                displacements[i] += unit * push
                displacements[j] -= unit * push

        repaired = repaired + 0.5 * displacements
        repaired = project_centers_inside_canvas(repaired, assets, width, height)

    return repaired


def cleanup_integer_overlaps(
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
    options: PackOptions,
    *,
    max_passes: int = 20,
) -> list[PackedPlacement]:
    width, height = canvas_size
    current = placements[:]
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    def score(items: list[PackedPlacement]) -> float:
        return total_overlap(buffered([item.polygon for item in items], options))

    for _ in range(max_passes):
        base_score = score(current)
        if base_score <= 1e-4:
            break
        improved = False

        for idx, placement in enumerate(current):
            best_candidate = current
            best_score = base_score
            for step in (1, 2, 3, 4):
                for dx, dy in directions:
                    new_x = min(max(0, placement.top_left[0] + dx * step), max(0, width - placement.image.width))
                    new_y = min(max(0, placement.top_left[1] + dy * step), max(0, height - placement.image.height))
                    if (new_x, new_y) == placement.top_left:
                        continue
                    moved_polygon = affinity.translate(
                        placement.polygon,
                        xoff=new_x - placement.top_left[0],
                        yoff=new_y - placement.top_left[1],
                    )
                    candidate = current[:]
                    candidate[idx] = PackedPlacement(
                        source_path=placement.source_path,
                        polygon=moved_polygon,
                        angle_degrees=placement.angle_degrees,
                        flipped=placement.flipped,
                        top_left=(new_x, new_y),
                        image=placement.image,
                    )
                    candidate_score = score(candidate)
                    if candidate_score + 1e-6 < best_score:
                        best_score = candidate_score
                        best_candidate = candidate
            if best_score + 1e-6 < base_score:
                current = best_candidate
                base_score = best_score
                improved = True

        if not improved:
            break

    return current


# ---------------------------------------------------------------------------
# Core refine-phase numerical solver
# ---------------------------------------------------------------------------

def _run_refine_optimization(
    *,
    assets: list[ClearanceAsset],
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
    options: PackOptions,
    pipeline_cfg: PipelineConfig,
) -> tuple[list[PackedPlacement], float, float, ClearanceRefinementRunResult]:
    width, height = canvas_size
    n = len(assets)
    if n <= 1:
        overlap = total_overlap(buffered([item.polygon for item in placements], options))
        min_clearance = float(np.min(clearance_values([item.polygon for item in placements], width, height)))
        return (
            placements, min_clearance, overlap,
            ClearanceRefinementRunResult(success=True, nit=0, message="Skipped (<=1 shape)."),
        )

    init_centers = np.array([[p.polygon.centroid.x, p.polygon.centroid.y] for p in placements], dtype=float)
    x0 = np.empty(2 * n, dtype=float)
    x0[0::2] = np.clip(init_centers[:, 0] / max(1.0, width), 0.0, 1.0)
    x0[1::2] = np.clip(init_centers[:, 1] / max(1.0, height), 0.0, 1.0)
    bounds = [(0.0, 1.0)] * (2 * n)

    def polygons_from_vector(vec: np.ndarray) -> tuple[list[Polygon], np.ndarray]:
        centers = np.column_stack((vec[0::2] * width, vec[1::2] * height))
        centers = project_centers_inside_canvas(centers, assets, width, height)
        polys = [polygon_at_center(assets[i].local_polygon, centers[i]) for i in range(n)]
        return polys, centers

    init_polys, _ = polygons_from_vector(x0)
    init_clearance, init_q25, init_mean = clearance_stats(init_polys, width, height)
    refine_cfg = pipeline_cfg.refine_phase
    obj_cfg = pipeline_cfg.refine_objective

    refine_progress = ProgressTracker(
        phase="refine",
        interval=max(0, int(refine_cfg.progress_log_every_evaluations)),
        heartbeat_seconds=max(0.0, float(refine_cfg.progress_log_heartbeat_seconds)),
    )

    def objective(vec: np.ndarray) -> float:
        polys, _ = polygons_from_vector(vec)
        values = clearance_values(polys, width, height)
        soft_min_clearance = softmin(values, obj_cfg.softmin_smoothness)
        q25_clearance = float(np.quantile(values, 0.25))
        mean_log_clearance = float(np.mean(np.log1p(values)))
        overlap = total_overlap(buffered(polys, options))
        outside = outside_violation(polys, width, height)
        regularization = float(np.sum((vec - x0) ** 2))
        score = float(
            -soft_min_clearance
            - obj_cfg.lower_quartile_spacing_weight * q25_clearance
            - obj_cfg.mean_spacing_weight * mean_log_clearance
            + obj_cfg.overlap_penalty_weight * overlap
            + obj_cfg.outside_canvas_penalty_weight * outside
            + obj_cfg.center_shift_regularization_weight * regularization
        )
        refine_progress.update(score=score, overlap=overlap)
        return score

    optimizer = refine_cfg.optimizer.lower().strip()
    if optimizer.startswith("scipy-"):
        method = optimizer.replace("scipy-", "", 1)
    elif optimizer == "pymoo-nsga2":
        method = "nsga2"
    else:
        raise ValueError(
            "Unsupported refine phase optimizer. Expected one of: "
            "'scipy-lbfgsb', 'scipy-de', 'scipy-hybrid', 'scipy-nsga2', 'pymoo-nsga2'."
        )

    best_result: OptimizeResult | None = None
    best_score = float("inf")
    if method in {"lbfgsb", "hybrid"}:
        starts = [x0]
        rng = np.random.default_rng(options.random_seed if options.random_seed is not None else None)
        for _ in range(max(0, refine_cfg.lbfgsb.random_restart_count - 1)):
            jitter = rng.uniform(-0.18, 0.18, size=x0.shape)
            starts.append(np.clip(x0 + jitter, 0.0, 1.0))
    else:
        starts = [x0]

    logger.info("Refine phase starting with %d restart(s).", len(starts))
    for idx, start in enumerate(starts):
        logger.info("Refine phase restart %d/%d...", idx + 1, len(starts))
        result = run_with_method(
            phase="refine", method=method, objective=objective, x0=start,
            bounds=bounds, maxiter=refine_cfg.lbfgsb.max_iterations,
            de_maxiter=refine_cfg.differential_evolution.max_generations,
            de_popsize=refine_cfg.differential_evolution.population_size,
            workers=refine_cfg.differential_evolution.worker_count,
            seed=options.random_seed,
        )
        score = float(result.fun)
        if score < best_score:
            best_score = score
            best_result = result

    assert best_result is not None
    if not best_result.success:
        logger.warning("Refine phase did not fully converge: %s", best_result.message)
    else:
        logger.info("Refine phase converged in %d iterations.", best_result.nit)

    final_polys, final_centers = polygons_from_vector(best_result.x)
    final_centers = repair_centers_for_overlap(assets, final_centers, width, height, options)
    final_polys = polygons_from_centers(assets, final_centers)
    final_clearance, final_q25, final_mean = clearance_stats(final_polys, width, height)
    logger.info(
        "Refine phase stats: min %.3f->%.3f q25 %.3f->%.3f mean %.3f->%.3f",
        init_clearance, final_clearance, init_q25, final_q25, init_mean, final_mean,
    )

    refined_placements = placements_from_centers(assets, final_centers, width, height)
    refined_overlap = total_overlap(buffered([item.polygon for item in refined_placements], options))
    run_result = ClearanceRefinementRunResult(
        success=bool(best_result.success),
        nit=int(best_result.nit) if best_result.nit is not None else None,
        message=str(best_result.message),
    )
    return refined_placements, final_clearance, refined_overlap, run_result


# ---------------------------------------------------------------------------
# Quality gate: accept refinement only if it actually improves the layout
# ---------------------------------------------------------------------------

def refine_with_quality_gate(
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
    options: PackOptions,
    pipeline_cfg: PipelineConfig,
) -> tuple[list[PackedPlacement], int | None, bool | None]:
    """Run the refine phase and accept results only if quality improves."""
    width, height = canvas_size
    if not pipeline_cfg.enable_refine_phase:
        return placements, 0, True
    if len(placements) <= 1:
        return placements, 0, True

    base_polygons = [item.polygon for item in placements]
    base_overlap = total_overlap(buffered(base_polygons, options))
    base_min_clearance, base_q25, base_mean = clearance_stats(base_polygons, width, height)
    base_quality = (
        base_min_clearance
        + pipeline_cfg.refine_objective.lower_quartile_spacing_weight * base_q25
        + pipeline_cfg.refine_objective.mean_spacing_weight * base_mean
    )

    assets: list[ClearanceAsset] = []
    for placement in placements:
        local_polygon = affinity.translate(
            placement.polygon,
            xoff=-placement.top_left[0],
            yoff=-placement.top_left[1],
        )
        assets.append(
            ClearanceAsset(
                source_path=placement.source_path,
                image=placement.image,
                local_polygon=local_polygon,
                angle_degrees=placement.angle_degrees,
                flipped=placement.flipped,
            )
        )

    refined_placements, refined_min_clearance, refined_overlap, refinement_result = _run_refine_optimization(
        assets=assets, placements=placements, canvas_size=canvas_size,
        options=options, pipeline_cfg=pipeline_cfg,
    )
    refined_polygons = [item.polygon for item in refined_placements]
    _, refined_q25, refined_mean = clearance_stats(refined_polygons, width, height)
    refined_quality = (
        refined_min_clearance
        + pipeline_cfg.refine_objective.lower_quartile_spacing_weight * refined_q25
        + pipeline_cfg.refine_objective.mean_spacing_weight * refined_mean
    )

    improves_overlap = refined_overlap <= base_overlap + 1e-3
    improves_quality = refined_quality >= base_quality + 0.05
    fixes_meaningful_overlap = base_overlap > 1e-3 and refined_overlap < 0.5 * base_overlap
    if improves_overlap and (improves_quality or fixes_meaningful_overlap):
        logger.info(
            "Accepting refine phase: overlap %.6f -> %.6f, quality %.3f -> %.3f",
            base_overlap, refined_overlap, base_quality, refined_quality,
        )
        return (
            refined_placements,
            int(refinement_result.nit) if refinement_result.nit is not None else None,
            bool(refinement_result.success),
        )
    logger.warning(
        "Rejecting refine phase: overlap %.6f -> %.6f, quality %.3f -> %.3f",
        base_overlap, refined_overlap, base_quality, refined_quality,
    )
    return (
        placements,
        int(refinement_result.nit) if refinement_result.nit is not None else None,
        bool(refinement_result.success),
    )
