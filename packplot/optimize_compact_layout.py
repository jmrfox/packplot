from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from shapely import affinity
from shapely.geometry import Polygon

from packplot.optimize_objectives import bbox_metrics, buffered_with_radius, total_overlap
from packplot.problem import PackingProblem
from packplot.types import OptimizeConfig, PackOptions, SourceObject


@dataclass
class CompactLayoutEval:
    result: OptimizeResult
    overlap: float
    aspect_error: float
    area: float
    score: float


@dataclass
class CompactLayoutSolveResult:
    result: OptimizeResult
    norm_scale: float


def centers_from_jacobi(jacobi: np.ndarray, inflation: float) -> np.ndarray:
    n = jacobi.shape[0] + 1
    centers = np.zeros((n, 2), dtype=float)
    running_mean = centers[0].copy()
    for k in range(1, n):
        next_center = running_mean + jacobi[k - 1] * inflation
        centers[k] = next_center
        running_mean = (running_mean * k + next_center) / (k + 1)
    centers -= np.mean(centers, axis=0, keepdims=True)
    return centers


def jacobi_from_centers(centers: np.ndarray) -> np.ndarray:
    n = centers.shape[0]
    if n <= 1:
        return np.zeros((0, 2), dtype=float)
    jacobi = np.zeros((n - 1, 2), dtype=float)
    for k in range(1, n):
        jacobi[k - 1] = centers[k] - np.mean(centers[:k], axis=0)
    return jacobi


def rotate_polygon_to_center(polygon: Polygon, angle_degrees: float, target_center: np.ndarray) -> Polygon:
    rotated = affinity.rotate(
        polygon,
        angle_degrees,
        origin=(polygon.centroid.x, polygon.centroid.y),
        use_radians=False,
    )
    cx, cy = rotated.centroid.x, rotated.centroid.y
    return affinity.translate(rotated, xoff=target_center[0] - cx, yoff=target_center[1] - cy)


def make_initial_centers(source_objects: list[SourceObject], scale: float) -> np.ndarray:
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


def pick_best_compact_layout(candidates: list[CompactLayoutEval]) -> CompactLayoutEval:
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


def run_compact_layout(
    *,
    source_objects: list[SourceObject],
    options: PackOptions,
    problem: PackingProblem,
    optimize_cfg: OptimizeConfig,
    run_with_method,
    progress_tracker_cls,
    logger,
) -> CompactLayoutSolveResult:
    compact_layout = optimize_cfg.compact_layout
    compact_layout_lbfgsb = compact_layout.lbfgsb
    compact_layout_de = compact_layout.differential_evolution
    compact_layout_obj_cfg = optimize_cfg.compact_layout_objective
    n = len(source_objects)
    normalized_polygons = problem.normalized_hulls
    norm_scale = problem.normalization_scale

    init_centers = make_initial_centers(source_objects, scale=1.25)
    init_jacobi = jacobi_from_centers(init_centers)
    normalized_buffer = max(0.0, float(options.padding) + float(options.edge_buffer)) / norm_scale
    x0 = np.concatenate([init_jacobi.reshape(-1), np.zeros(n, dtype=float)])
    bounds = [(-5.0, 5.0)] * (2 * (n - 1)) + [(0.0, 1.0)] * n

    logger.info(
        "Starting optimization pack: items=%d norm_scale=%.2f target_aspect=%.3f",
        n,
        norm_scale,
        options.target_aspect_ratio,
    )
    compact_layout_progress = progress_tracker_cls(
        phase="compact_layout",
        interval=max(0, int(compact_layout.progress_log_every_evaluations)),
        heartbeat_seconds=max(0.0, float(compact_layout.progress_log_heartbeat_seconds)),
    )
    jacobi_size = 2 * (n - 1)

    def unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jacobi_raw = vec[:jacobi_size].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
        rotations = vec[jacobi_size:]
        return jacobi_raw, rotations

    def compact_layout_geometry(vec: np.ndarray) -> tuple[list[Polygon], float, float, float]:
        jacobi_raw, rotations = unpack(vec)
        centers = centers_from_jacobi(jacobi_raw, options.jacobi_inflation)
        polygons: list[Polygon] = []
        for idx, polygon in enumerate(normalized_polygons):
            angle = float(rotations[idx] % 1.0) * 360.0
            polygons.append(rotate_polygon_to_center(polygon, angle, centers[idx]))

        buffered = buffered_with_radius(polygons, normalized_buffer)
        width, height, area = bbox_metrics(buffered)
        aspect_error = abs((width / height) - options.target_aspect_ratio)
        overlap = total_overlap(buffered)
        return buffered, overlap, aspect_error, area

    def objective(vec: np.ndarray) -> float:
        jacobi_raw, _ = unpack(vec)
        _, overlap, aspect_error, area = compact_layout_geometry(vec)
        aspect_penalty = aspect_error**2
        regularization = float(np.sum(jacobi_raw**2))

        score = float(
            area
            + compact_layout_obj_cfg.overlap_penalty_weight * overlap
            + compact_layout_obj_cfg.aspect_ratio_penalty_weight * aspect_penalty
            + compact_layout_obj_cfg.jacobi_regularization_weight * regularization
        )
        compact_layout_progress.update(score=score, overlap=overlap, aspect_error=aspect_error)
        return score

    def eval_result(result: OptimizeResult) -> CompactLayoutEval:
        _, overlap, aspect_error, area = compact_layout_geometry(result.x)
        return CompactLayoutEval(
            result=result,
            overlap=float(overlap),
            aspect_error=float(aspect_error),
            area=float(area),
            score=float(result.fun),
        )

    method = compact_layout.method.lower().strip()
    candidates: list[CompactLayoutEval] = []
    if method in {"lbfgsb", "hybrid"}:
        rng = np.random.default_rng(options.random_seed if options.random_seed is not None else None)
        starts = [x0]
        for _ in range(max(0, compact_layout_lbfgsb.random_restart_count - 1)):
            jitter = np.zeros_like(x0)
            if jacobi_size > 0:
                jitter[:jacobi_size] = rng.uniform(-0.9, 0.9, size=jacobi_size)
            jitter[jacobi_size:] = rng.uniform(-0.25, 0.25, size=n)
            starts.append(np.clip(x0 + jitter, [-5.0] * jacobi_size + [0.0] * n, [5.0] * jacobi_size + [1.0] * n))

        phase_iter = max(
            20,
            compact_layout_lbfgsb.max_iterations // max(1, compact_layout_lbfgsb.alternating_refinement_cycles + 1),
        )
        logger.info(
            "Compact-layout local optimization: restarts=%d alternating_cycles=%d phase_iter=%d",
            len(starts),
            compact_layout_lbfgsb.alternating_refinement_cycles,
            phase_iter,
        )
        for restart_idx, start in enumerate(starts):
            logger.info("Compact-layout restart %d/%d...", restart_idx + 1, len(starts))
            current = np.asarray(start, dtype=float)
            current = run_with_method(
                phase="compact_layout-full",
                method="lbfgsb",
                objective=objective,
                x0=current,
                bounds=bounds,
                maxiter=phase_iter,
                de_maxiter=compact_layout_de.max_generations,
                de_popsize=compact_layout_de.population_size,
                workers=compact_layout_de.worker_count,
                seed=options.random_seed,
            ).x

            for cycle_idx in range(max(0, compact_layout_lbfgsb.alternating_refinement_cycles)):
                logger.debug(
                    "Compact-layout restart %d cycle %d/%d",
                    restart_idx + 1,
                    cycle_idx + 1,
                    compact_layout_lbfgsb.alternating_refinement_cycles,
                )
                if jacobi_size > 0:
                    rot_fixed = current[jacobi_size:].copy()

                    def objective_centers(cvec: np.ndarray, rot_fixed=rot_fixed) -> float:
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

                def objective_rot(rvec: np.ndarray, centers_fixed=centers_fixed) -> float:
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
            candidates.append(eval_result(final_res))
            logger.info(
                "Compact-layout restart %d result: success=%s overlap=%.6f aspect_error=%.4f area=%.3f",
                restart_idx + 1,
                final_res.success,
                candidates[-1].overlap,
                candidates[-1].aspect_error,
                candidates[-1].area,
            )

        if method == "hybrid":
            de_res = run_with_method(
                phase="compact_layout-de",
                method="de",
                objective=objective,
                x0=x0,
                bounds=bounds,
                maxiter=compact_layout_lbfgsb.max_iterations,
                de_maxiter=compact_layout_de.max_generations,
                de_popsize=compact_layout_de.population_size,
                workers=compact_layout_de.worker_count,
                seed=options.random_seed,
            )
            candidates.append(eval_result(de_res))
    else:
        de_res = run_with_method(
            phase="compact_layout",
            method=method,
            objective=objective,
            x0=x0,
            bounds=bounds,
            maxiter=compact_layout_lbfgsb.max_iterations,
            de_maxiter=compact_layout_de.max_generations,
            de_popsize=compact_layout_de.population_size,
            workers=compact_layout_de.worker_count,
            seed=options.random_seed,
        )
        candidates.append(eval_result(de_res))

    best = pick_best_compact_layout(candidates)
    result = best.result
    logger.info(
        "Compact-layout selected candidate: overlap=%.6f aspect_error=%.4f area=%.3f score=%.3f",
        best.overlap,
        best.aspect_error,
        best.area,
        best.score,
    )
    if not result.success:
        logger.warning("Optimizer did not fully converge: %s", result.message)
    else:
        logger.info("Optimizer converged in %d iterations.", result.nit)
    return CompactLayoutSolveResult(result=result, norm_scale=norm_scale)
