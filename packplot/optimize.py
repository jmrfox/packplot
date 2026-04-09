from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from shapely import affinity
from shapely.geometry import Polygon

from packplot.optimize_clearance_refinement import cleanup_integer_overlaps as _cleanup_integer_overlaps
from packplot.optimize_compact_layout import (
    CompactLayoutSolveResult,
    centers_from_jacobi as _centers_from_jacobi,
    run_compact_layout,
)
from packplot.optimize_objectives import buffered as _buffered, total_overlap as _total_overlap
from packplot.problem import PackingProblem, build_packing_problem
from packplot.types import PackOptions, PackedPlacement, SourceObject

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
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


def run_with_method(
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

    def _run_nsga2() -> OptimizeResult:
        class _ScalarProblem(ElementwiseProblem):
            def __init__(self):
                xl = np.asarray([bound[0] for bound in bounds], dtype=float)
                xu = np.asarray([bound[1] for bound in bounds], dtype=float)
                super().__init__(n_var=len(bounds), n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = [float(objective(np.asarray(x, dtype=float)))]

        nsga2_pop = max(8, int(de_popsize))
        nsga2_gen = max(1, int(de_maxiter))
        problem = _ScalarProblem()
        algorithm = NSGA2(
            pop_size=nsga2_pop,
            n_offsprings=None,
            eliminate_duplicates=True,
        )
        result = pymoo_minimize(
            problem,
            algorithm,
            ("n_gen", nsga2_gen),
            seed=seed,
            verbose=False,
        )
        if result.X is None:
            raise RuntimeError("NSGA-II phase optimization produced no solution.")
        x = np.asarray(result.X, dtype=float)
        f = np.asarray(result.F, dtype=float)
        if x.ndim == 1:
            best_x = x
            best_f = float(f[0]) if f.ndim > 0 else float(f)
        else:
            best_idx = int(np.argmin(f[:, 0]))
            best_x = x[best_idx]
            best_f = float(f[best_idx, 0])
        return OptimizeResult(
            x=best_x,
            fun=best_f,
            success=True,
            nit=int(getattr(result.algorithm, "n_gen", nsga2_gen)),
            message="NSGA-II optimization completed.",
        )

    if chosen == "lbfgsb":
        return _run_lbfgsb(x0)
    if chosen in {"de", "differential_evolution"}:
        return _run_de()
    if chosen == "nsga2":
        return _run_nsga2()
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
) -> tuple[list[tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]], str, int | None, bool | None]:
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
            "and will return flipped=False placements. Set allow_flip=False for consistency."
        )

    # Compact-layout phase runs in normalized geometry so optimization variables stay bounded.
    if problem is None:
        problem = build_packing_problem(source_objects, options)
    compact_layout_result: CompactLayoutSolveResult = run_compact_layout(
        source_objects=source_objects,
        options=options,
        problem=problem,
        optimize_cfg=optimize_cfg,
        run_with_method=run_with_method,
        progress_tracker_cls=ProgressTracker,
        logger=logger,
    )

    ranked_compact = compact_layout_result.candidates[: max(1, int(optimize_cfg.compact_layout_best_count))]
    compact_candidates: list[tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]] = []
    iteration_sum = 0
    success = True
    for rank, compact in enumerate(ranked_compact, start=1):
        result = compact.result
        success = success and bool(result.success)
        iteration_sum += int(result.nit) if result.nit is not None else 0
        solution = result.x
        jacobi_raw = solution[: 2 * (n - 1)].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
        rotations = solution[2 * (n - 1) :]
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
        shifted = _cleanup_integer_overlaps(shifted, (width, height), options)
        overlap_final = _total_overlap(_buffered([item.polygon for item in shifted], options))
        if overlap_final > 1e-3:
            logger.warning(
                "Compact candidate %d has non-zero overlap area: %.4f.",
                rank,
                overlap_final,
            )
        compact_candidates.append((shifted, (width, height), None))

    method = f"optimize(compact_layout={optimize_cfg.compact_layout.method})"
    return compact_candidates, method, iteration_sum, success
