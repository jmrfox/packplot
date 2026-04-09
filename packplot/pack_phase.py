"""Pack phase: pack shapes into the smallest bounding box.

Contains the Jacobi coordinate mapping, pack-phase objective evaluation,
the scipy solver (L-BFGS-B / DE / hybrid / NSGA-II scalar) and pymoo
multi-objective NSGA-II solver, plus the shared solution-to-placement
conversion used by all paths.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import OptimizeResult, minimize
from shapely import affinity
from shapely.geometry import Polygon

from packplot.initialization import make_initial_center_layouts
from packplot.layout_metrics import (
    bbox_metrics,
    buffered,
    buffered_with_radius,
    min_pair_clearance,
    total_overlap,
)
from packplot.optimizer import ProgressTracker, run_with_method
from packplot.problem import PackingProblem, build_packing_problem
from packplot.types import PipelineConfig, PackOptions, PackedPlacement, SourceObject

logger = logging.getLogger(__name__)

SolverCandidate = tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]
SolverOutput = tuple[list[SolverCandidate], str, int | None, bool | None]


# ---------------------------------------------------------------------------
# Internal data classes
# ---------------------------------------------------------------------------

@dataclass
class _PackEval:
    """One evaluated pack-phase candidate (internal to ranking logic)."""
    result: OptimizeResult
    overlap: float
    aspect_error: float
    area: float
    score: float


@dataclass(frozen=True)
class PackPhaseMetrics:
    """Metrics for a single solution vector in normalized space."""
    area: float
    aspect_error: float
    overlap: float
    min_pair_clearance: float


# ---------------------------------------------------------------------------
# Jacobi coordinate helpers
# ---------------------------------------------------------------------------

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
        polygon, angle_degrees,
        origin=(polygon.centroid.x, polygon.centroid.y),
        use_radians=False,
    )
    cx, cy = rotated.centroid.x, rotated.centroid.y
    return affinity.translate(rotated, xoff=target_center[0] - cx, yoff=target_center[1] - cy)


# ---------------------------------------------------------------------------
# Evaluation and ranking
# ---------------------------------------------------------------------------

def evaluate_pack_vector(
    vec: np.ndarray,
    *,
    normalized_polygons: list[Polygon],
    target_aspect_ratio: float,
    jacobi_inflation: float,
    normalized_buffer: float,
) -> PackPhaseMetrics:
    n = len(normalized_polygons)
    jacobi_size = 2 * max(0, n - 1)
    jacobi_raw = vec[:jacobi_size].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
    rotations = vec[jacobi_size:]
    centers = centers_from_jacobi(jacobi_raw, jacobi_inflation)
    polygons: list[Polygon] = []
    for idx, polygon in enumerate(normalized_polygons):
        angle = float(rotations[idx] % 1.0) * 360.0
        polygons.append(rotate_polygon_to_center(polygon, angle, centers[idx]))
    buffered_polygons = buffered_with_radius(polygons, normalized_buffer)
    width, height, area = bbox_metrics(buffered_polygons)
    return PackPhaseMetrics(
        area=float(area),
        aspect_error=float(abs((width / height) - target_aspect_ratio)),
        overlap=float(total_overlap(buffered_polygons)),
        min_pair_clearance=float(min_pair_clearance(buffered_polygons)),
    )


_RANK_KEY = lambda item: (item.overlap > 1e-6, item.overlap, item.area, item.aspect_error, item.score)


# ---------------------------------------------------------------------------
# Shared helpers: image rotation and solution decoding
# ---------------------------------------------------------------------------

def render_oriented_image(source: SourceObject, angle_degrees: float) -> tuple[Image.Image, Polygon]:
    """Rotate a source image+hull by *angle_degrees*.

    The polygon is rotated around the image center (matching PIL's
    rotation center) and then shifted for the canvas expansion, so it
    stays aligned with the foreground pixels in the returned image.
    """
    old_w, old_h = source.cropped_image.size
    image = source.cropped_image.rotate(angle_degrees, expand=True, resample=Image.Resampling.BICUBIC)
    new_w, new_h = image.size

    polygon = affinity.rotate(
        source.hull, angle_degrees,
        origin=(old_w / 2.0, old_h / 2.0),
        use_radians=False,
    )
    polygon = affinity.translate(
        polygon,
        xoff=(new_w - old_w) / 2.0,
        yoff=(new_h - old_h) / 2.0,
    )
    return image, polygon


def solution_to_placements(
    solution: np.ndarray,
    source_objects: list[SourceObject],
    norm_scale: float,
    jacobi_inflation: float,
    options: PackOptions,
) -> tuple[list[PackedPlacement], tuple[int, int]]:
    """Decode a solution vector into full-resolution placements + canvas size."""
    n = len(source_objects)
    jacobi_size = 2 * max(0, n - 1)
    jacobi_raw = solution[:jacobi_size].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
    rotations = solution[jacobi_size:]
    centers = centers_from_jacobi(jacobi_raw, jacobi_inflation) * norm_scale

    placements: list[PackedPlacement] = []
    polygons: list[Polygon] = []
    for idx, source in enumerate(source_objects):
        angle = float(rotations[idx] % 1.0) * 360.0
        image, local_polygon = render_oriented_image(source, angle)
        cx, cy = local_polygon.centroid.x, local_polygon.centroid.y
        top_left_x = int(round(centers[idx, 0] - cx))
        top_left_y = int(round(centers[idx, 1] - cy))
        polygon = affinity.translate(local_polygon, xoff=top_left_x, yoff=top_left_y)
        placements.append(PackedPlacement(
            source_path=source.source_path, polygon=polygon, angle_degrees=angle,
            flipped=False, top_left=(top_left_x, top_left_y), image=image,
        ))
        polygons.append(polygon)

    # Compute bounding box across both buffered hulls and raw image extents,
    # then shift everything to non-negative coordinates.
    buffered_final = buffered(polygons, options)
    min_x = min(poly.bounds[0] for poly in buffered_final)
    min_y = min(poly.bounds[1] for poly in buffered_final)
    max_x = max(poly.bounds[2] for poly in buffered_final)
    max_y = max(poly.bounds[3] for poly in buffered_final)
    for p in placements:
        min_x = min(min_x, float(p.top_left[0]))
        min_y = min(min_y, float(p.top_left[1]))
        max_x = max(max_x, float(p.top_left[0] + p.image.width))
        max_y = max(max_y, float(p.top_left[1] + p.image.height))

    shift_x = -int(math.floor(min_x))
    shift_y = -int(math.floor(min_y))
    shifted = [
        PackedPlacement(
            source_path=p.source_path,
            polygon=affinity.translate(p.polygon, xoff=shift_x, yoff=shift_y),
            angle_degrees=p.angle_degrees, flipped=p.flipped,
            top_left=(p.top_left[0] + shift_x, p.top_left[1] + shift_y),
            image=p.image,
        )
        for p in placements
    ]
    width = max(1, int(math.ceil(max_x + shift_x)))
    height = max(1, int(math.ceil(max_y + shift_y)))
    return shifted, (width, height)


# ---------------------------------------------------------------------------
# Shared initialization: build variable bounds and seed vectors
# ---------------------------------------------------------------------------

def _pack_bounds(n: int) -> list[tuple[float, float]]:
    return [(-5.0, 5.0)] * (2 * max(0, n - 1)) + [(0.0, 1.0)] * n


def _seed_vectors(
    n: int,
    init_layouts: list[np.ndarray],
) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for centers in init_layouts:
        jac = jacobi_from_centers(centers) if n > 1 else np.zeros((0, 2), dtype=float)
        vectors.append(np.concatenate([jac.reshape(-1), np.zeros(n, dtype=float)]))
    return vectors


# ---------------------------------------------------------------------------
# Scipy pack-phase solver
# ---------------------------------------------------------------------------

def _solve_scipy_pack(
    source_objects: list[SourceObject],
    options: PackOptions,
    problem: PackingProblem,
    pipeline_cfg: PipelineConfig,
) -> SolverOutput:
    from packplot.refine_phase import cleanup_integer_overlaps

    pack_cfg = pipeline_cfg.pack_phase
    lbfgsb_cfg = pack_cfg.lbfgsb
    de_cfg = pack_cfg.differential_evolution
    obj_cfg = pipeline_cfg.pack_objective
    n = len(source_objects)
    norm_scale = problem.normalization_scale
    normalized_buffer = max(0.0, float(options.padding) + float(options.edge_buffer)) / norm_scale

    init_layouts = make_initial_center_layouts(
        n_items=n, target_aspect_ratio=options.target_aspect_ratio,
        config=options.initialization_config, seed=options.random_seed,
    )
    seeds = _seed_vectors(n, init_layouts)
    x0 = seeds[0]
    jacobi_size = 2 * (n - 1)
    bounds = _pack_bounds(n)

    logger.info(
        "Starting pack phase (scipy): items=%d norm_scale=%.2f target_aspect=%.3f",
        n, norm_scale, options.target_aspect_ratio,
    )
    progress = ProgressTracker(
        phase="pack", interval=max(0, int(pack_cfg.progress_log_every_evaluations)),
        heartbeat_seconds=max(0.0, float(pack_cfg.progress_log_heartbeat_seconds)),
    )

    def objective(vec: np.ndarray) -> float:
        jac_raw = vec[:jacobi_size].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
        m = evaluate_pack_vector(
            vec, normalized_polygons=problem.normalized_hulls,
            target_aspect_ratio=options.target_aspect_ratio,
            jacobi_inflation=options.jacobi_inflation, normalized_buffer=normalized_buffer,
        )
        score = float(
            m.area + obj_cfg.overlap_penalty_weight * m.overlap
            + obj_cfg.aspect_ratio_penalty_weight * m.aspect_error ** 2
            + obj_cfg.jacobi_regularization_weight * float(np.sum(jac_raw ** 2))
        )
        progress.update(score=score, overlap=m.overlap, aspect_error=m.aspect_error)
        return score

    def _eval(result: OptimizeResult) -> _PackEval:
        m = evaluate_pack_vector(
            result.x, normalized_polygons=problem.normalized_hulls,
            target_aspect_ratio=options.target_aspect_ratio,
            jacobi_inflation=options.jacobi_inflation, normalized_buffer=normalized_buffer,
        )
        return _PackEval(result=result, overlap=m.overlap, aspect_error=m.aspect_error, area=m.area, score=float(result.fun))

    optimizer_str = pack_cfg.optimizer.lower().strip()
    method = optimizer_str.replace("scipy-", "", 1)
    evals: list[_PackEval] = []

    if method in {"lbfgsb", "hybrid"}:
        rng = np.random.default_rng(options.random_seed if options.random_seed is not None else None)
        starts = list(seeds) or [x0]
        for _ in range(max(0, lbfgsb_cfg.random_restart_count - 1)):
            base = starts[rng.integers(0, len(starts))]
            jitter = np.zeros_like(base)
            if jacobi_size > 0:
                jitter[:jacobi_size] = rng.uniform(-0.9, 0.9, size=jacobi_size)
            jitter[jacobi_size:] = rng.uniform(-0.25, 0.25, size=n)
            starts.append(np.clip(base + jitter, [b[0] for b in bounds], [b[1] for b in bounds]))

        phase_iter = max(20, lbfgsb_cfg.max_iterations // max(1, lbfgsb_cfg.alternating_refinement_cycles + 1))
        logger.info("Pack L-BFGS-B: restarts=%d alt_cycles=%d iter=%d", len(starts), lbfgsb_cfg.alternating_refinement_cycles, phase_iter)

        for ri, start in enumerate(starts):
            logger.info("Pack restart %d/%d...", ri + 1, len(starts))
            cur = run_with_method(
                phase="pack-full", method="lbfgsb", objective=objective, x0=np.asarray(start, dtype=float),
                bounds=bounds, maxiter=phase_iter, de_maxiter=de_cfg.max_generations,
                de_popsize=de_cfg.population_size, workers=de_cfg.worker_count, seed=options.random_seed,
            ).x

            for ci in range(max(0, lbfgsb_cfg.alternating_refinement_cycles)):
                if jacobi_size > 0:
                    rot_fixed = cur[jacobi_size:].copy()
                    c_res = minimize(
                        lambda cv, _r=rot_fixed: objective(np.concatenate([cv, _r])),
                        x0=cur[:jacobi_size], method="L-BFGS-B", bounds=bounds[:jacobi_size], options={"maxiter": phase_iter},
                    )
                    cur[:jacobi_size] = c_res.x
                cen_fixed = cur[:jacobi_size].copy()
                r_res = minimize(
                    lambda rv, _c=cen_fixed: objective(np.concatenate([_c, rv])),
                    x0=cur[jacobi_size:], method="L-BFGS-B", bounds=bounds[jacobi_size:], options={"maxiter": phase_iter},
                )
                cur[jacobi_size:] = r_res.x

            final = minimize(objective, x0=cur, method="L-BFGS-B", bounds=bounds, options={"maxiter": phase_iter})
            evals.append(_eval(final))

        if method == "hybrid":
            evals.append(_eval(run_with_method(
                phase="pack-de", method="de", objective=objective, x0=x0, bounds=bounds,
                maxiter=lbfgsb_cfg.max_iterations, de_maxiter=de_cfg.max_generations,
                de_popsize=de_cfg.population_size, workers=de_cfg.worker_count, seed=options.random_seed,
            )))
    else:
        evals.append(_eval(run_with_method(
            phase="pack", method=method, objective=objective, x0=x0, bounds=bounds,
            maxiter=lbfgsb_cfg.max_iterations, de_maxiter=de_cfg.max_generations,
            de_popsize=de_cfg.population_size, workers=de_cfg.worker_count, seed=options.random_seed,
        )))

    ranked = sorted(evals, key=_RANK_KEY)
    best = ranked[0]
    logger.info("Pack best: overlap=%.6f aspect=%.4f area=%.3f", best.overlap, best.aspect_error, best.area)
    if not best.result.success:
        logger.warning("Pack optimizer did not fully converge: %s", best.result.message)

    top = ranked[:max(1, int(pipeline_cfg.pack_best_count))]
    candidates: list[SolverCandidate] = []
    iteration_sum = 0
    success = True
    for rank, ev in enumerate(top, start=1):
        success = success and bool(ev.result.success)
        iteration_sum += int(ev.result.nit) if ev.result.nit is not None else 0
        placements, canvas_size = solution_to_placements(ev.result.x, source_objects, norm_scale, options.jacobi_inflation, options)
        placements = cleanup_integer_overlaps(placements, canvas_size, options)
        overlap_final = total_overlap(buffered([p.polygon for p in placements], options))
        if overlap_final > 1e-3:
            logger.warning("Pack candidate %d has residual overlap: %.4f.", rank, overlap_final)
        candidates.append((placements, canvas_size, None))

    return candidates, f"scipy({pack_cfg.optimizer})", iteration_sum, success


# ---------------------------------------------------------------------------
# Pymoo multi-objective NSGA-II pack solver
# ---------------------------------------------------------------------------

class _PymooPackProblem(ElementwiseProblem):
    """Two-objective pack problem: minimize area and aspect error.

    Clearance is NOT an objective here — that's the refine phase's job.
    Overlap is enforced as an inequality constraint.
    """
    def __init__(self, *, normalized_hulls: list[Polygon], target_aspect_ratio: float, jacobi_inflation: float, normalized_buffer: float):
        self.normalized_hulls = normalized_hulls
        self.target_aspect_ratio = target_aspect_ratio
        self.jacobi_inflation = jacobi_inflation
        self.normalized_buffer = normalized_buffer
        self.n_items = len(normalized_hulls)
        n_var = 2 * max(0, self.n_items - 1) + self.n_items
        jac_bound = 2.5
        xl = np.array([-jac_bound] * (2 * max(0, self.n_items - 1)) + [0.0] * self.n_items, dtype=float)
        xu = np.array([jac_bound] * (2 * max(0, self.n_items - 1)) + [1.0] * self.n_items, dtype=float)
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        m = evaluate_pack_vector(
            np.asarray(x, dtype=float), normalized_polygons=self.normalized_hulls,
            target_aspect_ratio=self.target_aspect_ratio, jacobi_inflation=self.jacobi_inflation,
            normalized_buffer=self.normalized_buffer,
        )
        out["F"] = [m.area, m.aspect_error]
        out["G"] = [m.overlap - 1e-6]


def _solve_pymoo_pack(
    source_objects: list[SourceObject],
    options: PackOptions,
    problem: PackingProblem,
) -> SolverOutput:
    cfg = options.pymoo_config
    algo_name = cfg.algorithm.lower().strip()
    if algo_name != "nsga2":
        raise ValueError("Unsupported pymoo algorithm. Expected 'nsga2'.")

    n = len(source_objects)
    norm_scale = problem.normalization_scale
    normalized_buffer = max(0.0, float(options.padding) + float(options.edge_buffer)) / norm_scale
    pymoo_prob = _PymooPackProblem(
        normalized_hulls=problem.normalized_hulls, target_aspect_ratio=options.target_aspect_ratio,
        jacobi_inflation=options.jacobi_inflation, normalized_buffer=normalized_buffer,
    )

    seed = options.random_seed
    init_layouts = make_initial_center_layouts(
        n_items=n, target_aspect_ratio=options.target_aspect_ratio,
        config=options.initialization_config, seed=seed,
    )
    seeds = _seed_vectors(n, init_layouts)
    x0 = seeds[0]

    pop_size = max(8, int(cfg.population_size))
    n_offspring = max(2, int(cfg.offspring_count)) if cfg.offspring_count is not None else None
    sampling = np.zeros((pop_size, x0.shape[0]), dtype=float)
    fill = min(pop_size, len(seeds))
    for i in range(fill):
        sampling[i] = seeds[i]
    if fill < pop_size:
        rng = np.random.default_rng(seed)
        sampling[fill:] = rng.uniform(pymoo_prob.xl, pymoo_prob.xu, size=(pop_size - fill, x0.shape[0]))

    algorithm = NSGA2(pop_size=pop_size, n_offsprings=n_offspring, eliminate_duplicates=bool(cfg.eliminate_duplicates), sampling=sampling)
    logger.info("Starting pymoo NSGA-II pack: items=%d gen=%d pop=%d", n, cfg.generations, pop_size)
    result = pymoo_minimize(pymoo_prob, algorithm, ("n_gen", max(1, int(cfg.generations))), seed=seed, verbose=cfg.verbose)
    if result.X is None:
        raise RuntimeError("pymoo failed to produce a candidate solution.")

    raw = np.atleast_2d(np.asarray(result.X, dtype=float))

    # Rank Pareto-front solutions with the same feasibility-first key used everywhere.
    metrics_list: list[PackPhaseMetrics] = [
        evaluate_pack_vector(
            raw[i], normalized_polygons=problem.normalized_hulls,
            target_aspect_ratio=options.target_aspect_ratio,
            jacobi_inflation=options.jacobi_inflation, normalized_buffer=normalized_buffer,
        )
        for i in range(raw.shape[0])
    ]
    ranked_indices = sorted(
        range(raw.shape[0]),
        key=lambda i: (metrics_list[i].overlap > 1e-6, metrics_list[i].overlap, metrics_list[i].area, metrics_list[i].aspect_error),
    )
    selected = ranked_indices[:max(1, int(cfg.best_layout_count))]

    candidates: list[SolverCandidate] = []
    for idx in selected:
        m = metrics_list[idx]
        placements, canvas_size = solution_to_placements(raw[idx], source_objects, norm_scale, options.jacobi_inflation, options)
        candidates.append((placements, canvas_size, (m.area, m.aspect_error, m.overlap)))

    logger.info("Selected %d/%d pymoo layouts.", len(candidates), raw.shape[0])
    n_gen = int(getattr(result.algorithm, "n_gen", 0))
    return candidates, f"pymoo({algo_name})", n_gen, True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve_pack_phase(
    source_objects: list[SourceObject],
    options: PackOptions,
    problem: PackingProblem | None = None,
) -> SolverOutput:
    """Run the pack phase with the configured optimizer.

    Returns ``(candidates, method_string, total_iterations, success)``.
    Each candidate is ``(placements, canvas_size, objective_values | None)``.
    """
    n = len(source_objects)
    if n == 0:
        raise ValueError("No source objects to pack.")
    if options.jacobi_inflation <= 0:
        raise ValueError("jacobi_inflation must be > 0.")
    if options.allow_flip:
        logger.warning("Pack phase ignores allow_flip=True; optimization is rotation-only.")

    if problem is None:
        problem = build_packing_problem(source_objects, options)

    optimizer = options.pipeline_config.pack_phase.optimizer.lower().strip()
    if optimizer == "pymoo-nsga2":
        return _solve_pymoo_pack(source_objects, options, problem)
    if optimizer.startswith("scipy-"):
        return _solve_scipy_pack(source_objects, options, problem, options.pipeline_config)
    raise ValueError(
        f"Unsupported pack phase optimizer '{optimizer}'. "
        "Expected: 'scipy-lbfgsb', 'scipy-de', 'scipy-hybrid', 'scipy-nsga2', or 'pymoo-nsga2'."
    )
