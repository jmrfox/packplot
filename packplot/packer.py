from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import Polygon

from packplot.geometry import OrientedAsset, make_orientations, apply_orientation
from packplot.types import PackOptions, PackedPlacement, SourceObject

logger = logging.getLogger(__name__)


@dataclass
class _PlacedFootprint:
    polygon: Polygon
    placement: PackedPlacement


def _to_canvas_size(area: float, target_aspect_ratio: float) -> tuple[int, int]:
    width = max(1, math.ceil(math.sqrt(area * target_aspect_ratio)))
    height = max(1, math.ceil(area / width))
    return width, height


def _within_canvas(polygon: Polygon, width: int, height: int) -> bool:
    min_x, min_y, max_x, max_y = polygon.bounds
    return min_x >= 0 and min_y >= 0 and max_x <= width and max_y <= height


def _collides(candidate: Polygon, existing: list[Polygon]) -> bool:
    for polygon in existing:
        if candidate.intersection(polygon).area > 1e-6:
            return True
    return False


def _orient_assets(source: SourceObject, options: PackOptions) -> list[OrientedAsset]:
    orientations = make_orientations(options.rotation_step_degrees, options.allow_flip)
    logger.debug(
        "Generated %d candidate orientations for %s.",
        len(orientations),
        source.source_path.name,
    )
    return [apply_orientation(source.cropped_image, source.hull, orientation) for orientation in orientations]


def _pick_best_placement(
    source: SourceObject,
    oriented_assets: list[OrientedAsset],
    placed: list[_PlacedFootprint],
    width: int,
    height: int,
    padding: int,
    edge_buffer: float,
) -> _PlacedFootprint | None:
    occupied = [item.polygon for item in placed]
    best: tuple[float, float, int, int, _PlacedFootprint] | None = None
    evaluated = 0

    for asset in oriented_assets:
        total_buffer = float(padding) + float(edge_buffer)
        footprint = asset.polygon.buffer(total_buffer, join_style=2).buffer(0)
        if footprint.is_empty:
            continue

        _, _, local_max_x, local_max_y = footprint.bounds
        max_x_extent = max(local_max_x, float(asset.image.width))
        max_y_extent = max(local_max_y, float(asset.image.height))
        max_x_start = max(0, int(math.floor(width - max_x_extent)))
        max_y_start = max(0, int(math.floor(height - max_y_extent)))

        for y in range(max_y_start + 1):
            for x in range(max_x_start + 1):
                evaluated += 1
                translated_polygon = affinity.translate(asset.polygon, xoff=x, yoff=y)
                translated_footprint = affinity.translate(footprint, xoff=x, yoff=y)
                if x + asset.image.width > width or y + asset.image.height > height:
                    continue
                if not _within_canvas(translated_footprint, width, height):
                    continue
                if _collides(translated_footprint, occupied):
                    continue

                _, _, max_x, max_y = translated_footprint.bounds
                score = (max_y * max_x, max_y, x, y)
                placement = PackedPlacement(
                    source_path=source.source_path,
                    polygon=translated_polygon,
                    angle_degrees=asset.orientation.angle_degrees,
                    flipped=asset.orientation.flipped,
                    top_left=(x, y),
                    image=asset.image,
                )
                candidate = _PlacedFootprint(polygon=translated_footprint, placement=placement)
                if best is None or score < best[:4]:
                    best = (*score, candidate)
                    logger.debug(
                        "New best placement for %s: angle=%.1f flip=%s pos=(%d,%d) score=%.2f",
                        source.source_path.name,
                        asset.orientation.angle_degrees,
                        asset.orientation.flipped,
                        x,
                        y,
                        score[0],
                    )

    if best is None:
        logger.warning(
            "No valid placement found for %s after evaluating %d candidates on %dx%d canvas.",
            source.source_path.name,
            evaluated,
            width,
            height,
        )
        return None
    logger.info(
        "Placed %s after evaluating %d candidates.",
        source.source_path.name,
        evaluated,
    )
    return best[4]


def _try_pack(source_objects: list[SourceObject], options: PackOptions, width: int, height: int) -> list[PackedPlacement] | None:
    ordered = sorted(source_objects, key=lambda item: item.hull.area, reverse=True)
    logger.info(
        "Attempting pack run: items=%d canvas=%dx%d padding=%d edge_buffer=%.2f",
        len(ordered),
        width,
        height,
        options.padding,
        options.edge_buffer,
    )

    placed: list[_PlacedFootprint] = []
    for source in ordered:
        oriented_assets = _orient_assets(source, options)
        placed_item = _pick_best_placement(
            source,
            oriented_assets,
            placed,
            width,
            height,
            options.padding,
            options.edge_buffer,
        )
        if placed_item is None:
            logger.warning(
                "Pack run failed while placing %s on %dx%d canvas.",
                source.source_path.name,
                width,
                height,
            )
            return None
        placed.append(placed_item)

    logger.info("Pack run succeeded with %d placements.", len(placed))
    return [item.placement for item in placed]


def pack_polygons(source_objects: list[SourceObject], options: PackOptions) -> tuple[list[PackedPlacement], tuple[int, int]]:
    total_area = sum(source.hull.area for source in source_objects)
    if total_area <= 0:
        raise ValueError("Total hull area must be positive.")

    canvas_area = total_area / max(0.05, options.fill_ratio)
    logger.info(
        "Starting adaptive packing: total_hull_area=%.2f target_aspect=%.3f fill_ratio=%.2f",
        total_area,
        options.target_aspect_ratio,
        options.fill_ratio,
    )
    for step in range(options.max_grow_steps):
        width, height = _to_canvas_size(canvas_area, options.target_aspect_ratio)
        logger.info("Adaptive step %d/%d: trying canvas %dx%d", step + 1, options.max_grow_steps, width, height)
        placements = _try_pack(source_objects, options, width, height)
        if placements is not None:
            logger.info("Adaptive packing converged at step %d with canvas=%dx%d", step + 1, width, height)
            return placements, (width, height)
        canvas_area *= options.grow_factor
        logger.debug("Increasing canvas area by grow_factor=%.3f", options.grow_factor)

    logger.error("Adaptive packing failed after %d grow steps.", options.max_grow_steps)
    raise RuntimeError("Could not place all polygons. Increase max_grow_steps or grow_factor.")
