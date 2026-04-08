from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import Polygon

from packplot.types import PackOptions, SourceObject


@dataclass(frozen=True)
class PackingProblem:
    """Container for source geometry in both full-resolution and normalized space."""

    source_objects: list[SourceObject]
    options: PackOptions
    normalized_hulls: list[Polygon]
    normalization_scale: float


def _normalize_source_hulls(source_objects: list[SourceObject]) -> tuple[list[Polygon], float]:
    diagonals = []
    for source in source_objects:
        min_x, min_y, max_x, max_y = source.hull.bounds
        diagonals.append(math.hypot(max_x - min_x, max_y - min_y))
    scale = max(1.0, sum(diagonals) / max(1, len(diagonals)))
    normalized = [affinity.scale(source.hull, xfact=1.0 / scale, yfact=1.0 / scale, origin=(0, 0)) for source in source_objects]
    return normalized, scale


def build_packing_problem(source_objects: list[SourceObject], options: PackOptions) -> PackingProblem:
    """Build a packing problem with full-size and normalized hull representations."""
    normalized_hulls, scale = _normalize_source_hulls(source_objects)
    return PackingProblem(
        source_objects=source_objects,
        options=options,
        normalized_hulls=normalized_hulls,
        normalization_scale=scale,
    )
