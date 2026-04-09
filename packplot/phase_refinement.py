from __future__ import annotations

import logging

from shapely import affinity

from packplot.optimize import ProgressTracker, run_with_method
from packplot.optimize_clearance_refinement import (
    ClearanceAsset,
    run_clearance_refinement_with_fixed_canvas,
)
from packplot.optimize_objectives import buffered, clearance_stats, total_overlap
from packplot.types import OptimizeConfig, PackedPlacement, PackOptions

logger = logging.getLogger(__name__)


def refine_with_clearance_phase(
    placements: list[PackedPlacement],
    canvas_size: tuple[int, int],
    options: PackOptions,
    optimize_cfg: OptimizeConfig,
) -> tuple[list[PackedPlacement], int | None, bool | None]:
    """Apply clearance refinement on a compact layout candidate."""
    width, height = canvas_size
    if not optimize_cfg.enable_clearance_refinement_phase:
        return placements, 0, True
    if len(placements) <= 1:
        return placements, 0, True

    base_polygons = [item.polygon for item in placements]
    base_overlap = total_overlap(buffered(base_polygons, options))
    base_min_clearance, base_q25_clearance, base_mean_clearance = clearance_stats(base_polygons, width, height)
    base_quality = (
        base_min_clearance
        + optimize_cfg.clearance_refinement_objective.lower_quartile_spacing_weight * base_q25_clearance
        + optimize_cfg.clearance_refinement_objective.mean_spacing_weight * base_mean_clearance
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

    refined_placements, refined_min_clearance, refined_overlap, refinement_result = run_clearance_refinement_with_fixed_canvas(
        assets=assets,
        placements=placements,
        canvas_size=canvas_size,
        options=options,
        optimize_cfg=optimize_cfg,
        run_with_method=run_with_method,
        progress_tracker_cls=ProgressTracker,
        logger=logger,
    )
    refined_polygons = [item.polygon for item in refined_placements]
    _, refined_q25_clearance, refined_mean_clearance = clearance_stats(refined_polygons, width, height)
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
            "Accepting shared clearance-refinement: overlap %.6f -> %.6f, quality %.3f -> %.3f",
            base_overlap,
            refined_overlap,
            base_quality,
            refined_quality,
        )
        return (
            refined_placements,
            int(refinement_result.nit) if refinement_result.nit is not None else None,
            bool(refinement_result.success),
        )
    logger.warning(
        "Rejecting shared clearance-refinement: overlap %.6f -> %.6f, quality %.3f -> %.3f",
        base_overlap,
        refined_overlap,
        base_quality,
        refined_quality,
    )
    return (
        placements,
        int(refinement_result.nit) if refinement_result.nit is not None else None,
        bool(refinement_result.success),
    )
