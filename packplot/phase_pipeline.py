from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from packplot.optimize import optimize_pack
from packplot.optimize_objectives import bbox_metrics, buffered, clearance_stats, outside_violation, total_overlap
from packplot.phase_refinement import refine_with_clearance_phase
from packplot.problem import PackingProblem
from packplot.pymoo_optimize import pymoo_pack
from packplot.types import OptimizeConfig, PackOptions, PackedPlacement

SolverCandidate = tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]


@dataclass(frozen=True)
class _RefinedCandidateRank:
    outside: float
    overlap: float
    spacing_quality: float
    aspect_error: float
    area: float


def _rank_refined_candidates(
    candidates: list[SolverCandidate],
    *,
    options: PackOptions,
    optimize_cfg: OptimizeConfig,
) -> list[SolverCandidate]:
    ranked_with_metrics: list[tuple[SolverCandidate, _RefinedCandidateRank]] = []
    target_aspect = max(1e-6, float(options.target_aspect_ratio))
    spacing_cfg = optimize_cfg.clearance_refinement_objective
    for candidate in candidates:
        placements, canvas_size, _ = candidate
        width, height = canvas_size
        polygons = [item.polygon for item in placements]
        if not polygons:
            metrics = _RefinedCandidateRank(
                outside=0.0,
                overlap=0.0,
                spacing_quality=0.0,
                aspect_error=abs((float(width) / max(1.0, float(height))) - target_aspect),
                area=float(width * height),
            )
            ranked_with_metrics.append((candidate, metrics))
            continue
        buffered_polygons = buffered(polygons, options)
        bbox_w, bbox_h, bbox_area = bbox_metrics(buffered_polygons)
        min_clearance, q25_clearance, mean_clearance = clearance_stats(polygons, width, height)
        metrics = _RefinedCandidateRank(
            outside=float(outside_violation(polygons, width, height)),
            overlap=float(total_overlap(buffered_polygons)),
            spacing_quality=float(
                min_clearance
                + spacing_cfg.lower_quartile_spacing_weight * q25_clearance
                + spacing_cfg.mean_spacing_weight * mean_clearance
            ),
            aspect_error=float(abs((bbox_w / max(1e-6, bbox_h)) - target_aspect)),
            area=float(bbox_area),
        )
        ranked_with_metrics.append((candidate, metrics))
    ranked_with_metrics.sort(
        key=lambda item: (
            item[1].outside > 1e-6,
            item[1].outside,
            item[1].overlap > 1e-6,
            item[1].overlap,
            -item[1].spacing_quality,
            item[1].aspect_error,
            item[1].area,
        )
    )
    return [candidate for candidate, _ in ranked_with_metrics]


def _disable_clearance_refinement(problem: PackingProblem) -> PackingProblem:
    options = problem.options
    optimize_cfg = options.resolved_optimize_config()
    if not optimize_cfg.enable_clearance_refinement_phase:
        return problem
    disabled_cfg = replace(optimize_cfg, enable_clearance_refinement_phase=False)
    disabled_options = replace(options, optimize_config=disabled_cfg)
    return replace(problem, options=disabled_options)


def solve_two_phase(
    problem: PackingProblem,
    *,
    compact_backend: str,
) -> tuple[list[SolverCandidate], str, int | None, bool | None]:
    """Run compact-layout backend first, then shared clearance refinement."""
    compact_problem = _disable_clearance_refinement(problem)
    if compact_backend == "optimize":
        candidates, method, iterations, success = optimize_pack(
            compact_problem.source_objects,
            compact_problem.options,
            compact_problem,
        )
    elif compact_backend == "pymoo":
        candidates, method, iterations, success = pymoo_pack(
            compact_problem.source_objects,
            compact_problem.options,
            compact_problem,
        )
    else:
        raise ValueError("Unknown compact backend. Expected 'optimize' or 'pymoo'.")

    cfg = problem.options.resolved_optimize_config()
    if not cfg.enable_clearance_refinement_phase:
        return candidates, f"{method}+clearance_refinement(disabled)", iterations, success

    beam = max(1, int(cfg.compact_to_clearance_beam_width))
    candidates = candidates[:beam]
    refined_candidates: list[SolverCandidate] = []
    total_ref_iters = 0
    for placements, canvas_size, objective_values in candidates:
        refined, ref_iters, ref_success = refine_with_clearance_phase(
            placements=placements,
            canvas_size=canvas_size,
            options=problem.options,
            optimize_cfg=cfg,
        )
        if ref_iters:
            total_ref_iters += ref_iters
        if ref_success is False:
            success = False
        refined_candidates.append((refined, canvas_size, objective_values))
    refined_candidates = _rank_refined_candidates(refined_candidates, options=problem.options, optimize_cfg=cfg)
    total_iterations = (iterations or 0) + total_ref_iters
    method = f"{method}+clearance_refinement({cfg.clearance_refinement.method})"
    return refined_candidates, method, total_iterations, success
