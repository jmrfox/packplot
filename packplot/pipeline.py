"""Main orchestration: extract -> problem -> init -> pack -> refine -> render."""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from packplot.arrangement import Arrangement, ArrangementKeyFunc, ArrangementKeyMode, apply_arrangement, load_arrangement
from packplot.layout_metrics import bbox_metrics, buffered, clearance_stats, clearance_values, outside_violation, total_overlap
from packplot.pack_phase import SolverCandidate, solve_pack_phase
from packplot.problem import build_packing_problem
from packplot.refine_phase import refine_with_quality_gate
from packplot.render import render_composition
from packplot.source_loaders import get_source_loader, infer_source_loader_name
from packplot.types import PackOptions, PackedPlacement, PackResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate ranking (post-refine)
# ---------------------------------------------------------------------------

def _rank_refined_candidates(
    candidates: list[SolverCandidate],
    options: PackOptions,
) -> list[SolverCandidate]:
    target_aspect = max(1e-6, float(options.target_aspect_ratio))
    spacing_cfg = options.pipeline_config.refine_objective

    def _key(candidate: SolverCandidate):
        placements, (width, height), _ = candidate
        polygons = [p.polygon for p in placements]
        if not polygons:
            return (False, 0.0, False, 0.0, 0.0, abs(width / max(1.0, height) - target_aspect), float(width * height))
        bp = buffered(polygons, options)
        bbox_w, bbox_h, bbox_area = bbox_metrics(bp)
        min_cl, q25_cl, mean_cl = clearance_stats(polygons, width, height)
        quality = min_cl + spacing_cfg.lower_quartile_spacing_weight * q25_cl + spacing_cfg.mean_spacing_weight * mean_cl
        ov = total_overlap(bp)
        out = outside_violation(polygons, width, height)
        return (out > 1e-6, out, ov > 1e-6, ov, -quality, abs(bbox_w / max(1e-6, bbox_h) - target_aspect), bbox_area)

    return sorted(candidates, key=_key)


# ---------------------------------------------------------------------------
# Two-phase solver: pack -> refine
# ---------------------------------------------------------------------------

def _solve_two_phase(
    source_objects, options: PackOptions,
) -> tuple[list[SolverCandidate], str, int | None, bool | None]:
    """Run pack phase, then refine phase on each candidate."""
    problem = build_packing_problem(source_objects, options)
    candidates, method, iterations, success = solve_pack_phase(source_objects, options, problem)

    cfg = options.pipeline_config
    if not cfg.enable_refine_phase:
        return candidates, f"{method}+refine(disabled)", iterations, success

    beam = max(1, int(cfg.pack_to_refine_beam_width))
    candidates = candidates[:beam]
    refined: list[SolverCandidate] = []
    total_ref_iters = 0
    for placements, canvas_size, obj_values in candidates:
        ref_placements, ref_iters, ref_success = refine_with_quality_gate(
            placements=placements, canvas_size=canvas_size, options=options, pipeline_cfg=cfg,
        )
        if ref_iters:
            total_ref_iters += ref_iters
        if ref_success is False:
            success = False
        refined.append((ref_placements, canvas_size, obj_values))

    refined = _rank_refined_candidates(refined, options)
    total_iterations = (iterations or 0) + total_ref_iters
    return refined, f"{method}+refine({cfg.refine_phase.optimizer})", total_iterations, success


# ---------------------------------------------------------------------------
# Layout sanity check
# ---------------------------------------------------------------------------

def _compute_layout_sanity(
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
) -> tuple[float, int, bool, float, float]:
    width, height = canvas_size
    polygons = [p.polygon for p in placements]

    overlap = total_overlap(polygons)

    out_of_bounds = 0
    for p in placements:
        min_x, min_y, max_x, max_y = p.polygon.bounds
        in_bounds = (
            p.top_left[0] >= 0 and p.top_left[1] >= 0
            and p.top_left[0] + p.image.width <= width
            and p.top_left[1] + p.image.height <= height
            and min_x >= 0 and min_y >= 0 and max_x <= width and max_y <= height
        )
        if not in_bounds:
            out_of_bounds += 1

    passed = overlap <= 1e-6 and out_of_bounds == 0
    min_clearance = float(clearance_values(polygons, width, height).min()) if polygons else 0.0
    outside = float(outside_violation(polygons, width, height)) if polygons else 0.0
    return overlap, out_of_bounds, passed, min_clearance, outside


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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

    Returns a best-first list of ``PackResult`` objects.
    """
    paths = [Path(p) for p in image_paths]
    logger.info("pack_images called with %d image paths.", len(paths))
    if options is None:
        options = PackOptions()
    if target_aspect_ratio is not None:
        options = replace(options, target_aspect_ratio=target_aspect_ratio)

    source_loader = get_source_loader(infer_source_loader_name(paths))
    source_objects = source_loader.load(paths, options)
    logger.info("Extracted %d source objects.", len(source_objects))

    background_color = source_objects[0].background_color
    for s in source_objects[1:]:
        if s.background_color != background_color:
            logger.warning("Input backgrounds differ (%s vs %s); using first.", background_color, s.background_color)
            break

    candidates: list[SolverCandidate]
    solver_method: str
    solver_iterations: int | None
    solver_success: bool | None

    if arrangement is not None:
        arr = load_arrangement(arrangement) if isinstance(arrangement, (str, Path)) else arrangement
        placements, canvas_size, background_color = apply_arrangement(
            source_objects, arr, key_mode=arrangement_key_mode,
            key_func=arrangement_key_func, strict=strict_arrangement,
        )
        candidates = [(placements, canvas_size, None)]
        solver_method, solver_iterations, solver_success = "arrangement_replay", 0, True
    else:
        candidates, solver_method, solver_iterations, solver_success = _solve_two_phase(source_objects, options)

    results: list[PackResult] = []
    for rank, (placements, canvas_size, objective_values) in enumerate(candidates, start=1):
        overlap, oob, passed, min_cl, outside = _compute_layout_sanity(placements, canvas_size)
        if not passed:
            logger.warning("Layout sanity check failed: overlap_area=%.6f out_of_bounds_shapes=%d; rendering anyway.", overlap, oob)
        image = render_composition(canvas_size, placements, background_color=background_color)
        fill = sum(p.polygon.area for p in placements) / float(canvas_size[0] * canvas_size[1])
        results.append(PackResult(
            image=image, placements=placements, canvas_size=canvas_size,
            target_aspect_ratio=options.target_aspect_ratio, fill_ratio=fill,
            background_color=background_color, total_overlap_area=overlap,
            out_of_bounds_count=oob, sanity_passed=passed, minimum_clearance=min_cl,
            outside_violation_magnitude=outside, solver_method=solver_method,
            solver_iterations=solver_iterations, solver_success=solver_success,
            objective_values=objective_values, rank=rank,
        ))
    logger.info("Packing complete: %d solution(s).", len(results))
    return results
