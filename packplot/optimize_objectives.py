from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from packplot.types import PackOptions


def buffered(polygons: list[Polygon], options: PackOptions) -> list[Polygon]:
    total_buffer = max(0.0, float(options.padding) + float(options.edge_buffer))
    if total_buffer <= 0:
        return polygons
    return [polygon.buffer(total_buffer, join_style=2) for polygon in polygons]


def buffered_with_radius(polygons: list[Polygon], radius: float) -> list[Polygon]:
    if radius <= 0:
        return polygons
    return [polygon.buffer(radius, join_style=2) for polygon in polygons]


def bbox_metrics(polygons: list[Polygon]) -> tuple[float, float, float]:
    min_x = min(poly.bounds[0] for poly in polygons)
    min_y = min(poly.bounds[1] for poly in polygons)
    max_x = max(poly.bounds[2] for poly in polygons)
    max_y = max(poly.bounds[3] for poly in polygons)
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    area = width * height
    return width, height, area


def total_overlap(polygons: list[Polygon]) -> float:
    overlap = 0.0
    for i, left in enumerate(polygons):
        for right in polygons[i + 1 :]:
            overlap += left.intersection(right).area
    return overlap


def clearance_values(polygons: list[Polygon], width: int, height: int) -> np.ndarray:
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


def clearance_stats(polygons: list[Polygon], width: int, height: int) -> tuple[float, float, float]:
    values = clearance_values(polygons, width, height)
    min_clearance = float(np.min(values))
    q25_clearance = float(np.quantile(values, 0.25))
    mean_clearance = float(np.mean(values))
    return min_clearance, q25_clearance, mean_clearance


def softmin(values: np.ndarray, smoothness: float) -> float:
    # Differentiable approximation of min(values), used so L-BFGS can optimize
    # a "maximize minimum clearance" objective.
    k = max(1.0, float(smoothness))
    scaled = -k * values
    max_scaled = float(np.max(scaled))
    logsum = max_scaled + float(np.log(np.sum(np.exp(scaled - max_scaled))))
    return float(-logsum / k)


def outside_violation(polygons: list[Polygon], width: int, height: int) -> float:
    violation = 0.0
    for polygon in polygons:
        min_x, min_y, max_x, max_y = polygon.bounds
        violation += max(0.0, -min_x)
        violation += max(0.0, -min_y)
        violation += max(0.0, max_x - width)
        violation += max(0.0, max_y - height)
    return violation
