from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from shapely import affinity

from packplot.optimize_clearance_refinement import (
    ClearanceAsset as _ClearanceAsset,
    cleanup_integer_overlaps as _cleanup_integer_overlaps,
    run_clearance_refinement_with_fixed_canvas,
)
from packplot.optimize_compact_layout import (
    CompactLayoutSolveResult,
    centers_from_jacobi as _centers_from_jacobi,
    run_compact_layout,
)
from packplot.optimize_objectives import (
    buffered as _buffered,
    clearance_stats as _clearance_stats,
    total_overlap as _total_overlap,
)
from packplot.problem import PackingProblem, build_packing_problem
from packplot.types import OptimizeConfig, PackOptions, PackedPlacement, SolverMetadata, SourceObject

logger = logging.getLogger(__name__)


@dataclass
class _ProgressTracker:
    phase: str
    interval: int
    count: int = 0
    best_score: float = float("inf")
    best_overlap: float = float("inf")
    best_aspect_error: float = float("inf")
    heartbeat_seconds: float = 5.0
    _last_log_ts: float = 0.0

    def update(self, *, score: float, overlap: float, aspect_error: float | None = None) -> None:
        self.count += 1
        improved = score < self.best_score
        if improved:
            self.best_score = score
            self.best_overlap = overlap
            if aspect_error is not None:
                self.best_aspect_error = aspect_error

        now = time.monotonic()
        should_log = False
        if self.count == 1:
            should_log = True
        elif self.interval > 0 and self.count % self.interval == 0:
            should_log = True
        elif self.heartbeat_seconds > 0 and (now - self._last_log_ts) >= self.heartbeat_seconds:
            should_log = True

        if should_log:
            self._last_log_ts = now
            if self.best_aspect_error != float("inf"):
                logger.info(
                    "%s progress: eval=%d best_score=%.3f best_overlap=%.6f best_aspect_error=%.4f",
                    self.phase,
                    self.count,
                    self.best_score,
                    self.best_overlap,
                    self.best_aspect_error,
                )
            else:
                logger.info(
                    "%s progress: eval=%d best_score=%.3f best_overlap=%.6f",
                    self.phase,
                    self.count,
                    self.best_score,
                    self.best_overlap,
                )


def _run_with_method(
    *,
    phase: str,
    method: str,
    objective,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    maxiter: int,
    de_maxiter: int,
    de_popsize: int,
    workers: int,
    seed: int | None,
) -> OptimizeResult:
    chosen = method.lower().strip()

    def _run_lbfgsb(start: np.ndarray) -> OptimizeResult:
        return minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )

    def _run_de() -> OptimizeResult:
        if workers > 1:
            # Use threads instead of process-based workers to avoid Windows
            # pickling issues with closure-defined objectives.
            def _thread_map(func, iterable):
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    return list(executor.map(func, iterable))

            de_workers = _thread_map
        else:
            de_workers = 1
        return differential_evolution(
            objective,
            bounds=bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            seed=seed,
            workers=de_workers,
            updating="deferred" if workers != 1 else "immediate",
            polish=True,
        )

    if chosen == "lbfgsb":
        return _run_lbfgsb(x0)
    if chosen in {"de", "differential_evolution"}:
        return _run_de()
    if chosen == "hybrid":
        first = _run_lbfgsb(x0)
        message = str(first.message).upper()
        if first.success and "ABNORMAL" not in message:
            return first
        logger.warning("%s L-BFGS-B ended with '%s'; falling back to differential evolution.", phase, first.message)
        second = _run_de()
        # Keep whichever objective value is better.
        if float(second.fun) <= float(first.fun):
            return second
        return first
    raise ValueError(f"Unknown optimizer method '{method}' for phase {phase}.")


def _render_oriented_image(source: SourceObject, angle_degrees: float):
    image = source.cropped_image.rotate(
        angle_degrees,
        expand=True,
        resample=Image.Resampling.BICUBIC,
    )
    polygon = affinity.rotate(
        source.hull,
        angle_degrees,
        origin=(source.hull.centroid.x, source.hull.centroid.y),
        use_radians=False,
    )
    min_x, min_y, _, _ = polygon.bounds
    polygon = affinity.translate(polygon, xoff=-min_x, yoff=-min_y)
    return image, polygon


def optimize_pack(
    source_objects: list[SourceObject],
    options: PackOptions,
    problem: PackingProblem | None = None,
) -> tuple[list[PackedPlacement], tuple[int, int], SolverMetadata]:
    """Pack polygons with normalized optimization and optional clearance refinement."""
    optimize_cfg = options.resolved_optimize_config()
    n = len(source_objects)
    if n == 0:
        raise ValueError("No source objects to pack.")
    if options.jacobi_inflation <= 0:
        raise ValueError("jacobi_inflation must be > 0.")
    if options.allow_flip:
        logger.warning(
            "optimizer solver currently ignores allow_flip=True; optimization is rotation-only "
            "and will return flipped=False placements. If mirroring is required, use "
            "solver='heuristic' for now."
        )

    # Compact-layout phase runs in normalized geometry so optimization variables stay bounded.
    if problem is None:
        problem = build_packing_problem(source_objects, options)
    compact_layout_result: CompactLayoutSolveResult = run_compact_layout(
        source_objects=source_objects,
        options=options,
        problem=problem,
        optimize_cfg=optimize_cfg,
        run_with_method=_run_with_method,
        progress_tracker_cls=_ProgressTracker,
        logger=logger,
    )

    solution = compact_layout_result.result.x
    jacobi_raw = solution[: 2 * (n - 1)].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
    rotations = solution[2 * (n - 1) :]
    # Map optimized normalized centers back to full-resolution coordinates.
    centers = _centers_from_jacobi(jacobi_raw, options.jacobi_inflation) * compact_layout_result.norm_scale

    placements: list[PackedPlacement] = []
    polygons: list[Polygon] = []
    for idx, source in enumerate(source_objects):
        angle = float(rotations[idx] % 1.0) * 360.0
        image, local_polygon = _render_oriented_image(source, angle)
        cx, cy = local_polygon.centroid.x, local_polygon.centroid.y
        top_left_x = int(round(centers[idx, 0] - cx))
        top_left_y = int(round(centers[idx, 1] - cy))
        polygon = affinity.translate(local_polygon, xoff=top_left_x, yoff=top_left_y)
        placements.append(
            PackedPlacement(
                source_path=source.source_path,
                polygon=polygon,
                angle_degrees=angle,
                flipped=False,
                top_left=(top_left_x, top_left_y),
                image=image,
            )
        )
        polygons.append(polygon)

    buffered_final = _buffered(polygons, options)
    min_x = min(poly.bounds[0] for poly in buffered_final)
    min_y = min(poly.bounds[1] for poly in buffered_final)
    max_x = max(poly.bounds[2] for poly in buffered_final)
    max_y = max(poly.bounds[3] for poly in buffered_final)

    img_min_x = min(float(item.top_left[0]) for item in placements)
    img_min_y = min(float(item.top_left[1]) for item in placements)
    img_max_x = max(float(item.top_left[0] + item.image.width) for item in placements)
    img_max_y = max(float(item.top_left[1] + item.image.height) for item in placements)

    min_x = min(min_x, img_min_x)
    min_y = min(min_y, img_min_y)
    max_x = max(max_x, img_max_x)
    max_y = max(max_y, img_max_y)

    shift_x = -int(math.floor(min_x))
    shift_y = -int(math.floor(min_y))

    shifted: list[PackedPlacement] = []
    for placement in placements:
        shifted.append(
            PackedPlacement(
                source_path=placement.source_path,
                polygon=affinity.translate(placement.polygon, xoff=shift_x, yoff=shift_y),
                angle_degrees=placement.angle_degrees,
                flipped=placement.flipped,
                top_left=(placement.top_left[0] + shift_x, placement.top_left[1] + shift_y),
                image=placement.image,
            )
        )

    width = max(1, int(math.ceil(max_x + shift_x)))
    height = max(1, int(math.ceil(max_y + shift_y)))
    refinement_success: bool | None = None
    refinement_iterations: int | None = None
    if optimize_cfg.enable_clearance_refinement_phase:
        base_polygons = [item.polygon for item in shifted]
        base_overlap = _total_overlap(_buffered(base_polygons, options))
        base_min_clearance, base_q25_clearance, base_mean_clearance = _clearance_stats(base_polygons, width, height)
        base_quality = (
            base_min_clearance
            + optimize_cfg.clearance_refinement_objective.lower_quartile_spacing_weight * base_q25_clearance
            + optimize_cfg.clearance_refinement_objective.mean_spacing_weight * base_mean_clearance
        )
        clearance_assets: list[_ClearanceAsset] = []
        for placement in shifted:
            local_polygon = affinity.translate(
                placement.polygon,
                xoff=-placement.top_left[0],
                yoff=-placement.top_left[1],
            )
            clearance_assets.append(
                _ClearanceAsset(
                    source_path=placement.source_path,
                    image=placement.image,
                    local_polygon=local_polygon,
                    angle_degrees=placement.angle_degrees,
                    flipped=placement.flipped,
                )
            )
        refined_placements, refined_min_clearance, refined_overlap, refinement_result = run_clearance_refinement_with_fixed_canvas(
            assets=clearance_assets,
            placements=shifted,
            canvas_size=(width, height),
            options=options,
            optimize_cfg=optimize_cfg,
            run_with_method=_run_with_method,
            progress_tracker_cls=_ProgressTracker,
            logger=logger,
        )
        refinement_success = bool(refinement_result.success)
        refinement_iterations = int(refinement_result.nit) if refinement_result.nit is not None else None
        refined_polygons = [item.polygon for item in refined_placements]
        _, refined_q25_clearance, refined_mean_clearance = _clearance_stats(refined_polygons, width, height)
        refined_quality = (
            refined_min_clearance
            + optimize_cfg.clearance_refinement_objective.lower_quartile_spacing_weight * refined_q25_clearance
            + optimize_cfg.clearance_refinement_objective.mean_spacing_weight * refined_mean_clearance
        )
        improves_overlap = refined_overlap <= base_overlap + 1e-3
        improves_quality = refined_quality >= base_quality + 0.05
        fixes_meaningful_overlap = base_overlap > 1e-3 and refined_overlap < 0.5 * base_overlap
        if improves_overlap and (improves_quality or fixes_meaningful_overlap):
            logger.info(
                "Accepting clearance-refinement phase: overlap %.6f -> %.6f, quality %.3f -> %.3f",
                base_overlap,
                refined_overlap,
                base_quality,
                refined_quality,
            )
            shifted = refined_placements
        else:
            logger.warning(
                "Rejecting clearance-refinement phase: overlap %.6f -> %.6f, quality %.3f -> %.3f",
                base_overlap,
                refined_overlap,
                base_quality,
                refined_quality,
            )

    shifted = _cleanup_integer_overlaps(shifted, (width, height), options)
    overlap_final = _total_overlap(_buffered([item.polygon for item in shifted], options))
    if overlap_final > 1e-3:
        logger.warning(
            "Final optimized layout has non-zero overlap area: %.4f. "
            "Consider increasing edge_buffer/padding or optimization iterations.",
            overlap_final,
        )
    logger.info("Optimization pack complete: canvas=%dx%d overlap=%.6f", width, height, overlap_final)
    compact_success = bool(compact_layout_result.result.success)
    compact_iterations = (
        int(compact_layout_result.result.nit) if compact_layout_result.result.nit is not None else 0
    )
    total_iterations = compact_iterations + (refinement_iterations or 0)
    overall_success = compact_success and (True if refinement_success is None else refinement_success)
    method = (
        f"optimize(compact_layout={optimize_cfg.compact_layout.method},"
        f"clearance_refinement={optimize_cfg.clearance_refinement.method if optimize_cfg.enable_clearance_refinement_phase else 'disabled'})"
    )
    return shifted, (width, height), SolverMetadata(
        method=method,
        iterations=total_iterations,
        success=overall_success,
    )
