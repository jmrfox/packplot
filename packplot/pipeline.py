from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from packplot.arrangement import Arrangement, ArrangementKeyFunc, ArrangementKeyMode, apply_arrangement, load_arrangement
from packplot.optimize_objectives import clearance_values, outside_violation
from packplot.problem import build_packing_problem
from packplot.render import render_composition
from packplot.solvers import get_solver
from packplot.source_loaders import get_source_loader, infer_source_loader_name
from packplot.types import PackOptions, PackedPlacement, PackResult

logger = logging.getLogger(__name__)


def _compute_layout_sanity(
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
) -> tuple[float, int, bool, float, float]:
    width, height = canvas_size
    polygons = [placement.polygon for placement in placements]
    total_overlap = 0.0
    for idx, left in enumerate(polygons):
        for right in polygons[idx + 1 :]:
            total_overlap += left.intersection(right).area

    out_of_bounds = 0
    for placement in placements:
        min_x, min_y, max_x, max_y = placement.polygon.bounds
        image_in_bounds = (
            placement.top_left[0] >= 0
            and placement.top_left[1] >= 0
            and placement.top_left[0] + placement.image.width <= width
            and placement.top_left[1] + placement.image.height <= height
        )
        polygon_in_bounds = min_x >= 0 and min_y >= 0 and max_x <= width and max_y <= height
        if not (image_in_bounds and polygon_in_bounds):
            out_of_bounds += 1

    passed = total_overlap <= 1e-6 and out_of_bounds == 0
    if polygons:
        min_clearance = float(clearance_values(polygons, width, height).min())
        outside = float(outside_violation(polygons, width, height))
    else:
        min_clearance = 0.0
        outside = 0.0
    return float(total_overlap), int(out_of_bounds), bool(passed), min_clearance, outside


def pack_images(
    image_paths: Iterable[str | Path],
    options: PackOptions | None = None,
    *,
    target_aspect_ratio: float | None = None,
    arrangement: Arrangement | str | Path | None = None,
    arrangement_key_mode: ArrangementKeyMode = "stem",
    arrangement_key_func: ArrangementKeyFunc | None = None,
    strict_arrangement: bool = True,
) -> list[PackResult]:
    """Pack input images into a single composed figure.

    Args:
        image_paths: Paths to raster/SVG files, one primary object per file.
        options: Optional `PackOptions`; defaults are used when omitted.
        target_aspect_ratio: Optional override for `options.target_aspect_ratio`.
        arrangement: Optional arrangement object or JSON path to replay layout.
        arrangement_key_mode: Key matching mode for arrangement replay.
        arrangement_key_func: Optional callable that maps a file path to an ID key.
        strict_arrangement: If True, missing/extra keys raise errors.

    Returns:
        List of `PackResult` objects ordered best-first. Solvers that produce
        one layout return a single-element list.
    """

    paths = [Path(path) for path in image_paths]
    logger.info("pack_images called with %d image paths.", len(paths))
    if options is None:
        options = PackOptions()
    if target_aspect_ratio is not None:
        options = replace(options, target_aspect_ratio=target_aspect_ratio)
    logger.debug("Effective pack options: %s", options)

    source_loader = get_source_loader(infer_source_loader_name(paths))
    source_objects = source_loader.load(paths, options)
    logger.info("Extracted %d source objects; beginning packing.", len(source_objects))
    background_color = source_objects[0].background_color
    for source in source_objects[1:]:
        if source.background_color != background_color:
            logger.warning(
                "Input backgrounds differ (%s vs %s); using first image background.",
                background_color,
                source.background_color,
            )
            break
    arrangement_obj: Arrangement | None = None
    candidates: list[tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]] = []
    solver_method: str = "unknown"
    solver_iterations: int | None = None
    solver_success: bool | None = None
    if arrangement is not None:
        arrangement_obj = load_arrangement(arrangement) if isinstance(arrangement, (str, Path)) else arrangement
        placements, canvas_size, background_color = apply_arrangement(
            source_objects,
            arrangement_obj,
            key_mode=arrangement_key_mode,
            key_func=arrangement_key_func,
            strict=strict_arrangement,
        )
        candidates = [(placements, canvas_size, None)]
        solver_method = "arrangement_replay"
        solver_iterations = 0
        solver_success = True
    else:
        problem = build_packing_problem(source_objects, options)
        compact_backend = options.resolved_optimize_config().compact_layout_backend
        solver = get_solver(compact_backend)
        candidates, solver_method, solver_iterations, solver_success = solver.solve(problem)

    results: list[PackResult] = []
    for rank, (placements, canvas_size, objective_values) in enumerate(candidates, start=1):
        total_overlap_area, out_of_bounds_count, sanity_passed, minimum_clearance, outside_violation_magnitude = (
            _compute_layout_sanity(placements, canvas_size)
        )
        if not sanity_passed:
            logger.warning(
                "Layout sanity check failed: overlap_area=%.6f out_of_bounds_shapes=%d; rendering anyway. "
                "Try increasing `edge_buffer`/`padding`, enabling clearance refinement, or increasing optimization iterations.",
                total_overlap_area,
                out_of_bounds_count,
            )
        image = render_composition(canvas_size, placements, background_color=background_color)
        fill_ratio = sum(item.polygon.area for item in placements) / float(canvas_size[0] * canvas_size[1])
        results.append(
            PackResult(
                image=image,
                placements=placements,
                canvas_size=canvas_size,
                target_aspect_ratio=options.target_aspect_ratio,
                fill_ratio=fill_ratio,
                background_color=background_color,
                total_overlap_area=total_overlap_area,
                out_of_bounds_count=out_of_bounds_count,
                sanity_passed=sanity_passed,
                minimum_clearance=minimum_clearance,
                outside_violation_magnitude=outside_violation_magnitude,
                solver_method=solver_method,
                solver_iterations=solver_iterations,
                solver_success=solver_success,
                objective_values=objective_values,
                rank=rank,
            )
        )
    logger.info("Packing complete: returned %d solution(s).", len(results))
    return results
