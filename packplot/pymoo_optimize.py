from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from PIL import Image
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from shapely import affinity
from shapely.geometry import Polygon

from packplot.initialization import make_initial_center_layouts
from packplot.optimize_compact_layout import (
    centers_from_jacobi,
    evaluate_compact_layout_vector,
    jacobi_from_centers,
)
from packplot.optimize_objectives import total_overlap
from packplot.problem import PackingProblem, build_packing_problem
from packplot.types import PackOptions, PackedPlacement, SourceObject

logger = logging.getLogger(__name__)


class _PymooCompactProblem(ElementwiseProblem):
    def __init__(
        self,
        *,
        normalized_hulls: list[Polygon],
        target_aspect_ratio: float,
        jacobi_inflation: float,
        normalized_buffer: float,
    ) -> None:
        self.normalized_hulls = normalized_hulls
        self.target_aspect_ratio = target_aspect_ratio
        self.jacobi_inflation = jacobi_inflation
        self.normalized_buffer = normalized_buffer
        self.n_items = len(normalized_hulls)
        self.jacobi_size = 2 * max(0, self.n_items - 1)
        n_var = self.jacobi_size + self.n_items
        xl = np.array(([-5.0] * self.jacobi_size) + ([0.0] * self.n_items), dtype=float)
        xu = np.array(([5.0] * self.jacobi_size) + ([1.0] * self.n_items), dtype=float)
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        metrics = evaluate_compact_layout_vector(
            np.asarray(x, dtype=float),
            normalized_polygons=self.normalized_hulls,
            target_aspect_ratio=self.target_aspect_ratio,
            jacobi_inflation=self.jacobi_inflation,
            normalized_buffer=self.normalized_buffer,
        )
        out["F"] = [metrics.area, metrics.aspect_error, -metrics.min_pair_clearance]
        out["G"] = [metrics.overlap - 1e-6]


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


def _rank_candidate_indices(
    candidates: np.ndarray,
    problem: _PymooCompactProblem,
) -> tuple[list[int], list[Any]]:
    metrics_by_idx: list[Any] = []
    for idx in range(candidates.shape[0]):
        metrics_by_idx.append(
            evaluate_compact_layout_vector(
                candidates[idx],
                normalized_polygons=problem.normalized_hulls,
                target_aspect_ratio=problem.target_aspect_ratio,
                jacobi_inflation=problem.jacobi_inflation,
                normalized_buffer=problem.normalized_buffer,
            )
        )

    ranked = sorted(
        range(candidates.shape[0]),
        key=lambda idx: (
            metrics_by_idx[idx].overlap > 1e-6,
            metrics_by_idx[idx].overlap,
            metrics_by_idx[idx].aspect_error,
            metrics_by_idx[idx].area,
            -metrics_by_idx[idx].min_pair_clearance,
        ),
    )
    return ranked, metrics_by_idx


def _placements_from_solution(
    solution: np.ndarray,
    source_objects: list[SourceObject],
    norm_scale: float,
    jacobi_inflation: float,
    padding: int,
    edge_buffer: float,
) -> tuple[list[PackedPlacement], tuple[int, int]]:
    n = len(source_objects)
    jacobi_size = 2 * max(0, n - 1)
    jacobi_raw = solution[:jacobi_size].reshape((n - 1, 2)) if n > 1 else np.zeros((0, 2), dtype=float)
    rotations = solution[jacobi_size:]
    centers = centers_from_jacobi(jacobi_raw, jacobi_inflation) * norm_scale

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

    total_buffer = max(0.0, float(padding) + float(edge_buffer))
    buffered = [polygon.buffer(total_buffer, join_style=2) for polygon in polygons] if total_buffer > 0 else polygons
    min_x = min(poly.bounds[0] for poly in buffered)
    min_y = min(poly.bounds[1] for poly in buffered)
    max_x = max(poly.bounds[2] for poly in buffered)
    max_y = max(poly.bounds[3] for poly in buffered)

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
    return shifted, (width, height)


def pymoo_pack(
    source_objects: list[SourceObject],
    options: PackOptions,
    problem: PackingProblem | None = None,
) -> tuple[list[tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]], str, int | None, bool | None]:
    """Pack polygons using pymoo's multi-objective NSGA-II."""
    if problem is None:
        problem = build_packing_problem(source_objects, options)
    cfg = options.pymoo_config
    algo_name = cfg.algorithm.lower().strip()
    if algo_name != "nsga2":
        raise ValueError("Unsupported pymoo algorithm. Expected 'nsga2'.")
    if options.allow_flip:
        logger.warning(
            "pymoo solver currently ignores allow_flip=True; optimization is rotation-only "
            "and returns flipped=False placements."
        )

    normalized_buffer = max(0.0, float(options.padding) + float(options.edge_buffer)) / problem.normalization_scale
    pymoo_problem = _PymooCompactProblem(
        normalized_hulls=problem.normalized_hulls,
        target_aspect_ratio=options.target_aspect_ratio,
        jacobi_inflation=options.jacobi_inflation,
        normalized_buffer=normalized_buffer,
    )

    seed = options.random_seed if options.random_seed is not None else None
    n_items = len(source_objects)
    init_layouts = make_initial_center_layouts(
        n_items=n_items,
        target_aspect_ratio=options.target_aspect_ratio,
        config=options.initialization_config,
        seed=options.random_seed,
    )
    seed_vectors: list[np.ndarray] = []
    for centers in init_layouts:
        init_jacobi = jacobi_from_centers(centers) if n_items > 1 else np.zeros((0, 2), dtype=float)
        seed_vectors.append(np.concatenate([init_jacobi.reshape(-1), np.zeros(n_items, dtype=float)]))
    x0 = seed_vectors[0]

    pop_size = max(8, int(cfg.population_size))
    n_offspring = cfg.offspring_count
    if n_offspring is not None:
        n_offspring = max(2, int(n_offspring))
    sampling = np.zeros((pop_size, x0.shape[0]), dtype=float)
    fill = min(pop_size, len(seed_vectors))
    for idx in range(fill):
        sampling[idx] = seed_vectors[idx]
    if fill < pop_size:
        rng = np.random.default_rng(seed if seed is not None else None)
        xl = np.array(([-5.0] * (2 * max(0, n_items - 1))) + ([0.0] * n_items), dtype=float)
        xu = np.array(([5.0] * (2 * max(0, n_items - 1))) + ([1.0] * n_items), dtype=float)
        sampling[fill:] = rng.uniform(xl, xu, size=(pop_size - fill, x0.shape[0]))
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=n_offspring,
        eliminate_duplicates=bool(cfg.eliminate_duplicates),
        sampling=sampling,
    )
    logger.info(
        "Starting pymoo NSGA-II pack: items=%d generations=%d pop=%d",
        n_items,
        cfg.generations,
        pop_size,
    )
    result = minimize(
        pymoo_problem,
        algorithm,
        ("n_gen", max(1, int(cfg.generations))),
        seed=seed,
        verbose=cfg.verbose,
    )
    if result.X is None:
        raise RuntimeError("pymoo failed to produce a candidate solution.")

    candidates = np.asarray(result.X, dtype=float)
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    ranked_indices, metrics_by_idx = _rank_candidate_indices(candidates, pymoo_problem)
    best_count = max(1, int(cfg.best_layout_count))
    selected_indices = ranked_indices[:best_count]
    best_layouts: list[tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]] = []
    for idx in selected_indices:
        vec = candidates[idx]
        metrics = metrics_by_idx[idx]
        placements_i, canvas_size_i = _placements_from_solution(
            vec,
            source_objects=source_objects,
            norm_scale=problem.normalization_scale,
            jacobi_inflation=options.jacobi_inflation,
            padding=options.padding,
            edge_buffer=options.edge_buffer,
        )
        best_layouts.append(
            (
                placements_i,
                canvas_size_i,
                (metrics.area, metrics.aspect_error, -metrics.min_pair_clearance),
            )
        )
    first_placements, first_canvas_size, _ = best_layouts[0]
    logger.info(
        "Selected %d best pymoo layouts out of %d candidates (requested=%d).",
        len(best_layouts),
        candidates.shape[0],
        best_count,
    )
    overlap = total_overlap(
        [
            item.polygon.buffer(max(0.0, float(options.padding) + float(options.edge_buffer)), join_style=2)
            for item in first_placements
        ]
    )
    if overlap > 1e-3:
        logger.warning(
            "pymoo final layout has non-zero overlap area: %.4f. "
            "Consider increasing generations/population.",
            overlap,
        )
    n_gen = int(getattr(result.algorithm, "n_gen", 0))
    logger.info("pymoo pack complete: canvas=%s overlap=%.6f", first_canvas_size, overlap)
    return best_layouts, f"pymoo({algo_name})", n_gen, result.X is not None
