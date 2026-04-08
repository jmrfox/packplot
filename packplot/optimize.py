from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from shapely import affinity
from shapely.geometry import Polygon

from packplot.types import PackOptions, PackedPlacement, SourceObject

logger = logging.getLogger(__name__)


@dataclass
class _SpreadAsset:
    source_path: Path
    image: Image.Image
    local_polygon: Polygon
    angle_degrees: float
    flipped: bool


@dataclass
class _Phase1Eval:
    result: OptimizeResult
    overlap: float
    aspect_error: float
    area: float
    score: float


@dataclass
class _ProgressTracker:
    phase: str
    interval: int
    count: int = 0
    best_score: float = float("inf")
    best_overlap: float = float("inf")
    best_aspect_error: float = float("inf")

    def update(self, *, score: float, overlap: float, aspect_error: float | None = None) -> None:
        self.count += 1
        improved = score < self.best_score
        if improved:
            self.best_score = score
            self.best_overlap = overlap
            if aspect_error is not None:
                self.best_aspect_error = aspect_error

        if self.interval > 0 and self.count % self.interval == 0:
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


def _pick_best_phase1(candidates: list[_Phase1Eval]) -> _Phase1Eval:
    # Feasibility-first ordering:
    # 1) minimize overlap, 2) minimize aspect error, 3) minimize area, 4) objective.
    return min(
        candidates,
        key=lambda item: (
            item.overlap > 1e-6,
            item.overlap,
            item.aspect_error,
            item.area,
            item.score,
        ),
    )


def _jacobi_from_centers(centers: np.ndarray) -> np.ndarray:
    n = centers.shape[0]
    if n <= 1:
        return np.zeros((0, 2), dtype=float)
    jacobi = np.zeros((n - 1, 2), dtype=float)
    for k in range(1, n):
        jacobi[k - 1] = centers[k] - np.mean(centers[:k], axis=0)
    return jacobi


def _centers_from_jacobi(jacobi: np.ndarray, inflation: float) -> np.ndarray:
    n = jacobi.shape[0] + 1
    centers = np.zeros((n, 2), dtype=float)
    running_mean = centers[0].copy()
    for k in range(1, n):
        next_center = running_mean + jacobi[k - 1] * inflation
        centers[k] = next_center
        running_mean = (running_mean * k + next_center) / (k + 1)
    centers -= np.mean(centers, axis=0, keepdims=True)
    return centers


def _rotate_polygon_to_center(polygon: Polygon, angle_degrees: float, target_center: np.ndarray) -> Polygon:
    rotated = affinity.rotate(
        polygon,
        angle_degrees,
        origin=(polygon.centroid.x, polygon.centroid.y),
        use_radians=False,
    )
    cx, cy = rotated.centroid.x, rotated.centroid.y
    return affinity.translate(rotated, xoff=target_center[0] - cx, yoff=target_center[1] - cy)


def _buffered(polygons: list[Polygon], options: PackOptions) -> list[Polygon]:
    total_buffer = max(0.0, float(options.padding) + float(options.edge_buffer))
    if total_buffer <= 0:
        return polygons
    return [polygon.buffer(total_buffer, join_style=2) for polygon in polygons]


def _buffered_with_radius(polygons: list[Polygon], radius: float) -> list[Polygon]:
    if radius <= 0:
        return polygons
    return [polygon.buffer(radius, join_style=2) for polygon in polygons]


def _bbox_metrics(polygons: list[Polygon]) -> tuple[float, float, float]:
    min_x = min(poly.bounds[0] for poly in polygons)
    min_y = min(poly.bounds[1] for poly in polygons)
    max_x = max(poly.bounds[2] for poly in polygons)
    max_y = max(poly.bounds[3] for poly in polygons)
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    area = width * height
    return width, height, area


def _total_overlap(polygons: list[Polygon]) -> float:
    overlap = 0.0
    for i, left in enumerate(polygons):
        for right in polygons[i + 1 :]:
            overlap += left.intersection(right).area
    return overlap


def _make_initial_centers(source_objects: list[SourceObject], scale: float) -> np.ndarray:
    n = len(source_objects)
    side = max(1, math.ceil(math.sqrt(n)))
    centers = np.zeros((n, 2), dtype=float)
    for idx in range(n):
        row = idx // side
        col = idx % side
        centers[idx, 0] = col * scale
        centers[idx, 1] = row * scale
    centers -= np.mean(centers, axis=0, keepdims=True)
    return centers


def _render_oriented_image(source: SourceObject, angle_degrees: float) -> tuple[Image.Image, Polygon]:
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


def _normalize_polygons(source_objects: list[SourceObject]) -> tuple[list[Polygon], float]:
    diagonals = []
    for source in source_objects:
        min_x, min_y, max_x, max_y = source.hull.bounds
        diagonals.append(math.hypot(max_x - min_x, max_y - min_y))
    scale = max(1.0, float(np.mean(diagonals)))
    normalized = [affinity.scale(source.hull, xfact=1.0 / scale, yfact=1.0 / scale, origin=(0, 0)) for source in source_objects]
    return normalized, scale


def _polygon_at_center(local_polygon: Polygon, center_xy: np.ndarray) -> Polygon:
    cx, cy = local_polygon.centroid.x, local_polygon.centroid.y
    return affinity.translate(local_polygon, xoff=float(center_xy[0] - cx), yoff=float(center_xy[1] - cy))


def _clearance_values(polygons: list[Polygon], width: int, height: int) -> np.ndarray:
    values: list[float] = []
    for polygon in polygons:
        min_x, min_y, max_x, max_y = polygon.bounds
        values.extend([min_x, min_y, width - max_x, height - max_y])
    for i, left in enumerate(polygons):
        for right in polygons[i + 1 :]:
            values.append(left.distance(right))
    if not values:
        return np.array([0.0], dtype=float)
    return np.asarray(values, dtype=float)


def _clearance_stats(polygons: list[Polygon], width: int, height: int) -> tuple[float, float, float]:
    values = _clearance_values(polygons, width, height)
    min_clearance = float(np.min(values))
    q25_clearance = float(np.quantile(values, 0.25))
    mean_clearance = float(np.mean(values))
    return min_clearance, q25_clearance, mean_clearance


def _softmin(values: np.ndarray, smoothness: float) -> float:
    # Differentiable approximation of min(values), used so L-BFGS can optimize
    # a "maximize minimum clearance" objective.
    k = max(1.0, float(smoothness))
    scaled = -k * values
    max_scaled = float(np.max(scaled))
    logsum = max_scaled + float(np.log(np.sum(np.exp(scaled - max_scaled))))
    return float(-logsum / k)


def _outside_violation(polygons: list[Polygon], width: int, height: int) -> float:
    violation = 0.0
    for polygon in polygons:
        min_x, min_y, max_x, max_y = polygon.bounds
        violation += max(0.0, -min_x)
        violation += max(0.0, -min_y)
        violation += max(0.0, max_x - width)
        violation += max(0.0, max_y - height)
    return violation


def _project_centers_inside_canvas(centers: np.ndarray, assets: list[_SpreadAsset], width: int, height: int) -> np.ndarray:
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


def _polygons_from_centers(assets: list[_SpreadAsset], centers: np.ndarray) -> list[Polygon]:
    return [_polygon_at_center(assets[i].local_polygon, centers[i]) for i in range(len(assets))]


def _placements_from_centers(
    assets: list[_SpreadAsset],
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


def _repair_centers_for_overlap(
    assets: list[_SpreadAsset],
    centers: np.ndarray,
    width: int,
    height: int,
    options: PackOptions,
    *,
    iterations: int = 60,
) -> np.ndarray:
    # Local geometric relaxation that pushes overlapping buffered polygons apart.
    # This is a post-optimizer safety step and keeps centers inside canvas bounds.
    repaired = _project_centers_inside_canvas(centers, assets, width, height)
    n = len(assets)
    for _ in range(iterations):
        polygons = _polygons_from_centers(assets, repaired)
        buffered = _buffered(polygons, options)
        overlap = _total_overlap(buffered)
        if overlap <= 1e-4:
            break

        displacements = np.zeros_like(repaired)
        for i in range(n):
            for j in range(i + 1, n):
                area = buffered[i].intersection(buffered[j]).area
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
        repaired = _project_centers_inside_canvas(repaired, assets, width, height)

    return repaired


def _cleanup_integer_overlaps(
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
    options: PackOptions,
    *,
    max_passes: int = 20,
) -> list[PackedPlacement]:
    # Final discrete cleanup pass after float centers are rounded to integer pixels.
    # Small grid moves can remove tiny overlaps introduced by rounding.
    width, height = canvas_size
    current = placements[:]
    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]

    def _score(items: list[PackedPlacement]) -> float:
        return _total_overlap(_buffered([item.polygon for item in items], options))

    for _ in range(max_passes):
        base_score = _score(current)
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
                    candidate_score = _score(candidate)
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


def _spread_layout_with_fixed_canvas(
    assets: list[_SpreadAsset],
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
    options: PackOptions,
) -> tuple[list[PackedPlacement], float, float]:
    """Run second-phase optimization on a fixed canvas to improve spacing."""
    width, height = canvas_size
    n = len(assets)
    if n <= 1:
        overlap = _total_overlap(_buffered([item.polygon for item in placements], options))
        min_clearance = float(np.min(_clearance_values([item.polygon for item in placements], width, height)))
        return placements, min_clearance, overlap

    init_centers = np.array([[p.polygon.centroid.x, p.polygon.centroid.y] for p in placements], dtype=float)
    x0 = np.empty(2 * n, dtype=float)
    x0[0::2] = np.clip(init_centers[:, 0] / max(1.0, width), 0.0, 1.0)
    x0[1::2] = np.clip(init_centers[:, 1] / max(1.0, height), 0.0, 1.0)
    bounds = [(0.0, 1.0)] * (2 * n)

    def _polygons_from_vector(vec: np.ndarray) -> tuple[list[Polygon], np.ndarray]:
        centers = np.column_stack((vec[0::2] * width, vec[1::2] * height))
        centers = _project_centers_inside_canvas(centers, assets, width, height)
        polygons = [_polygon_at_center(assets[i].local_polygon, centers[i]) for i in range(n)]
        return polygons, centers

    init_polys, _ = _polygons_from_vector(x0)
    init_clearance, init_q25, init_mean = _clearance_stats(init_polys, width, height)
    spread_progress = _ProgressTracker(
        phase="spread",
        interval=max(0, int(options.spread_progress_interval)),
    )

    def objective(vec: np.ndarray) -> float:
        polygons, _ = _polygons_from_vector(vec)
        clearance_values = _clearance_values(polygons, width, height)
        soft_min_clearance = _softmin(clearance_values, options.spread_smoothness)
        q25_clearance = float(np.quantile(clearance_values, 0.25))
        mean_log_clearance = float(np.mean(np.log1p(clearance_values)))
        overlap = _total_overlap(_buffered(polygons, options))
        outside = _outside_violation(polygons, width, height)
        regularization = float(np.sum((vec - x0) ** 2))
        score = float(
            -soft_min_clearance
            - options.spread_quantile_weight * q25_clearance
            - options.spread_mean_weight * mean_log_clearance
            + options.spread_overlap_weight * overlap
            + options.spread_outside_weight * outside
            + options.spread_regularization * regularization
        )
        spread_progress.update(score=score, overlap=overlap)
        return score

    best_result = None
    best_score = float("inf")

    if options.spread_method.lower().strip() in {"lbfgsb", "hybrid"}:
        starts = [x0]
        rng = np.random.default_rng(options.random_seed if options.random_seed is not None else None)
        for _ in range(max(0, options.spread_restarts - 1)):
            jitter = rng.uniform(-0.18, 0.18, size=x0.shape)
            starts.append(np.clip(x0 + jitter, 0.0, 1.0))
    else:
        starts = [x0]

    logger.info("Spread phase starting with %d restart(s).", len(starts))
    for idx, start in enumerate(starts):
        logger.info("Spread restart %d/%d...", idx + 1, len(starts))
        result = _run_with_method(
            phase="spread",
            method=options.spread_method,
            objective=objective,
            x0=start,
            bounds=bounds,
            maxiter=options.spread_maxiter,
            de_maxiter=options.spread_de_maxiter,
            de_popsize=options.spread_de_popsize,
            workers=options.spread_workers,
            seed=options.random_seed,
        )
        score = float(result.fun)
        if score < best_score:
            best_score = score
            best_result = result

    assert best_result is not None
    if not best_result.success:
        logger.warning("Spread phase did not fully converge: %s", best_result.message)
    else:
        logger.info("Spread phase converged in %d iterations.", best_result.nit)

    final_vec = best_result.x
    final_polys, final_centers = _polygons_from_vector(final_vec)
    final_centers = _repair_centers_for_overlap(assets, final_centers, width, height, options)
    final_polys = _polygons_from_centers(assets, final_centers)
    final_clearance, final_q25, final_mean = _clearance_stats(final_polys, width, height)
    logger.info(
        "Spread phase clearance stats: min %.3f->%.3f q25 %.3f->%.3f mean %.3f->%.3f",
        init_clearance,
        final_clearance,
        init_q25,
        final_q25,
        init_mean,
        final_mean,
    )

    spread_placements = _placements_from_centers(assets, final_centers, width, height)
    spread_overlap = _total_overlap(_buffered([item.polygon for item in spread_placements], options))
    return spread_placements, final_clearance, spread_overlap


def optimize_pack(source_objects: list[SourceObject], options: PackOptions) -> tuple[list[PackedPlacement], tuple[int, int]]:
    """Pack polygons with normalized optimization and optional spread refinement."""
    n = len(source_objects)
    if n == 0:
        raise ValueError("No source objects to pack.")
    if options.jacobi_inflation <= 0:
        raise ValueError("jacobi_inflation must be > 0.")

    # Phase 1 runs in normalized geometry so optimization variables stay bounded.
    normalized_polygons, norm_scale = _normalize_polygons(source_objects)
    init_centers = _make_initial_centers(source_objects, scale=1.25)
    init_jacobi = _jacobi_from_centers(init_centers)
    normalized_buffer = max(0.0, float(options.padding) + float(options.edge_buffer)) / norm_scale

    x0 = np.concatenate([init_jacobi.reshape(-1), np.zeros(n, dtype=float)])
    bounds = [(-5.0, 5.0)] * (2 * (n - 1)) + [(0.0, 1.0)] * n

    logger.info(
        "Starting optimization pack: items=%d norm_scale=%.2f target_aspect=%.3f",
        n,
        norm_scale,
        options.target_aspect_ratio,
    )
    phase1_progress = _ProgressTracker(
        phase="phase1",
        interval=max(0, int(options.optimizer_progress_interval)),
    )

    jacobi_size = 2 * (n - 1)

    def _unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jacobi_raw = vec[:jacobi_size].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
        rotations = vec[jacobi_size:]
        return jacobi_raw, rotations

    def _phase1_geometry(vec: np.ndarray) -> tuple[list[Polygon], float, float, float]:
        jacobi_raw, rotations = _unpack(vec)
        centers = _centers_from_jacobi(jacobi_raw, options.jacobi_inflation)
        polygons: list[Polygon] = []
        for idx, polygon in enumerate(normalized_polygons):
            angle = float(rotations[idx] % 1.0) * 360.0
            polygons.append(_rotate_polygon_to_center(polygon, angle, centers[idx]))

        buffered = _buffered_with_radius(polygons, normalized_buffer)
        width, height, area = _bbox_metrics(buffered)
        aspect_error = abs((width / height) - options.target_aspect_ratio)
        overlap = _total_overlap(buffered)
        return buffered, overlap, aspect_error, area

    def objective(vec: np.ndarray) -> float:
        jacobi_raw, _ = _unpack(vec)
        _, overlap, aspect_error, area = _phase1_geometry(vec)
        aspect_penalty = aspect_error**2
        regularization = float(np.sum(jacobi_raw**2))

        score = float(
            area
            + options.optimizer_overlap_weight * overlap
            + options.optimizer_aspect_weight * aspect_penalty
            + options.optimizer_regularization * regularization
        )
        phase1_progress.update(score=score, overlap=overlap, aspect_error=aspect_error)
        return score

    def _eval_result(result: OptimizeResult) -> _Phase1Eval:
        _, overlap, aspect_error, area = _phase1_geometry(result.x)
        return _Phase1Eval(
            result=result,
            overlap=float(overlap),
            aspect_error=float(aspect_error),
            area=float(area),
            score=float(result.fun),
        )

    method = options.optimizer_method.lower().strip()
    candidates: list[_Phase1Eval] = []
    if method in {"lbfgsb", "hybrid"}:
        rng = np.random.default_rng(options.random_seed if options.random_seed is not None else None)
        starts = [x0]
        for _ in range(max(0, options.optimizer_restarts - 1)):
            jitter = np.zeros_like(x0)
            if jacobi_size > 0:
                jitter[:jacobi_size] = rng.uniform(-0.9, 0.9, size=jacobi_size)
            jitter[jacobi_size:] = rng.uniform(-0.25, 0.25, size=n)
            starts.append(np.clip(x0 + jitter, [-5.0] * jacobi_size + [0.0] * n, [5.0] * jacobi_size + [1.0] * n))

        phase_iter = max(20, options.optimizer_maxiter // max(1, options.optimizer_alternating_cycles + 1))
        logger.info(
            "Phase1 local optimization: restarts=%d alternating_cycles=%d phase_iter=%d",
            len(starts),
            options.optimizer_alternating_cycles,
            phase_iter,
        )
        for restart_idx, start in enumerate(starts):
            logger.info("Phase1 restart %d/%d...", restart_idx + 1, len(starts))
            current = np.asarray(start, dtype=float)
            current = _run_with_method(
                phase="phase1-full",
                method="lbfgsb",
                objective=objective,
                x0=current,
                bounds=bounds,
                maxiter=phase_iter,
                de_maxiter=options.optimizer_de_maxiter,
                de_popsize=options.optimizer_de_popsize,
                workers=options.optimizer_workers,
                seed=options.random_seed,
            ).x

            for cycle_idx in range(max(0, options.optimizer_alternating_cycles)):
                logger.debug(
                    "Phase1 restart %d cycle %d/%d",
                    restart_idx + 1,
                    cycle_idx + 1,
                    options.optimizer_alternating_cycles,
                )
                if jacobi_size > 0:
                    rot_fixed = current[jacobi_size:].copy()

                    def objective_centers(cvec: np.ndarray) -> float:
                        return objective(np.concatenate([cvec, rot_fixed]))

                    c_bounds = bounds[:jacobi_size]
                    c_res = minimize(
                        objective_centers,
                        x0=current[:jacobi_size],
                        method="L-BFGS-B",
                        bounds=c_bounds,
                        options={"maxiter": phase_iter},
                    )
                    current[:jacobi_size] = c_res.x

                centers_fixed = current[:jacobi_size].copy()

                def objective_rot(rvec: np.ndarray) -> float:
                    return objective(np.concatenate([centers_fixed, rvec]))

                r_bounds = bounds[jacobi_size:]
                r_res = minimize(
                    objective_rot,
                    x0=current[jacobi_size:],
                    method="L-BFGS-B",
                    bounds=r_bounds,
                    options={"maxiter": phase_iter},
                )
                current[jacobi_size:] = r_res.x

            final_res = minimize(
                objective,
                x0=current,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": phase_iter},
            )
            candidates.append(_eval_result(final_res))
            logger.info(
                "Phase1 restart %d result: success=%s overlap=%.6f aspect_error=%.4f area=%.3f",
                restart_idx + 1,
                final_res.success,
                candidates[-1].overlap,
                candidates[-1].aspect_error,
                candidates[-1].area,
            )

        if method == "hybrid":
            de_res = _run_with_method(
                phase="phase1-de",
                method="de",
                objective=objective,
                x0=x0,
                bounds=bounds,
                maxiter=options.optimizer_maxiter,
                de_maxiter=options.optimizer_de_maxiter,
                de_popsize=options.optimizer_de_popsize,
                workers=options.optimizer_workers,
                seed=options.random_seed,
            )
            candidates.append(_eval_result(de_res))
    else:
        de_res = _run_with_method(
            phase="phase1",
            method=method,
            objective=objective,
            x0=x0,
            bounds=bounds,
            maxiter=options.optimizer_maxiter,
            de_maxiter=options.optimizer_de_maxiter,
            de_popsize=options.optimizer_de_popsize,
            workers=options.optimizer_workers,
            seed=options.random_seed,
        )
        candidates.append(_eval_result(de_res))

    best = _pick_best_phase1(candidates)
    result = best.result
    logger.info(
        "Phase1 selected candidate: overlap=%.6f aspect_error=%.4f area=%.3f score=%.3f",
        best.overlap,
        best.aspect_error,
        best.area,
        best.score,
    )
    if not result.success:
        logger.warning("Optimizer did not fully converge: %s", result.message)
    else:
        logger.info("Optimizer converged in %d iterations.", result.nit)

    solution = result.x
    jacobi_raw = solution[: 2 * (n - 1)].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
    rotations = solution[2 * (n - 1) :]
    # Map optimized normalized centers back to full-resolution coordinates.
    centers = _centers_from_jacobi(jacobi_raw, options.jacobi_inflation) * norm_scale

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
    if options.enable_spread_phase:
        base_polygons = [item.polygon for item in shifted]
        base_overlap = _total_overlap(_buffered(base_polygons, options))
        base_min_clearance, base_q25_clearance, base_mean_clearance = _clearance_stats(base_polygons, width, height)
        base_quality = (
            base_min_clearance
            + options.spread_quantile_weight * base_q25_clearance
            + options.spread_mean_weight * base_mean_clearance
        )
        spread_assets: list[_SpreadAsset] = []
        for placement in shifted:
            local_polygon = affinity.translate(
                placement.polygon,
                xoff=-placement.top_left[0],
                yoff=-placement.top_left[1],
            )
            spread_assets.append(
                _SpreadAsset(
                    source_path=placement.source_path,
                    image=placement.image,
                    local_polygon=local_polygon,
                    angle_degrees=placement.angle_degrees,
                    flipped=placement.flipped,
                )
            )
        spread_placements, spread_min_clearance, spread_overlap = _spread_layout_with_fixed_canvas(
            spread_assets,
            shifted,
            (width, height),
            options,
        )
        spread_polygons = [item.polygon for item in spread_placements]
        _, spread_q25_clearance, spread_mean_clearance = _clearance_stats(spread_polygons, width, height)
        spread_quality = (
            spread_min_clearance
            + options.spread_quantile_weight * spread_q25_clearance
            + options.spread_mean_weight * spread_mean_clearance
        )
        improves_overlap = spread_overlap <= base_overlap + 1e-3
        improves_quality = spread_quality >= base_quality + 0.05
        fixes_meaningful_overlap = base_overlap > 1e-3 and spread_overlap < 0.5 * base_overlap
        if improves_overlap and (improves_quality or fixes_meaningful_overlap):
            logger.info(
                "Accepting spread phase: overlap %.6f -> %.6f, quality %.3f -> %.3f",
                base_overlap,
                spread_overlap,
                base_quality,
                spread_quality,
            )
            shifted = spread_placements
        else:
            logger.warning(
                "Rejecting spread phase: overlap %.6f -> %.6f, quality %.3f -> %.3f",
                base_overlap,
                spread_overlap,
                base_quality,
                spread_quality,
            )

    shifted = _cleanup_integer_overlaps(shifted, (width, height), options)
    overlap_final = _total_overlap(_buffered([item.polygon for item in shifted], options))
    if overlap_final > 1e-3:
        logger.warning("Final optimized layout has non-zero overlap area: %.4f", overlap_final)
    logger.info("Optimization pack complete: canvas=%dx%d overlap=%.6f", width, height, overlap_final)
    return shifted, (width, height)
