from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon, box


@dataclass(frozen=True)
class Orientation:
    angle_degrees: float
    flipped: bool


@dataclass
class OrientedAsset:
    image: Image.Image
    polygon: Polygon
    orientation: Orientation


def convex_hull_from_points(points_xy: np.ndarray) -> Polygon:
    if points_xy.size == 0:
        raise ValueError("Cannot build a convex hull from empty points.")

    hull = MultiPoint(points_xy).convex_hull
    if hull.geom_type == "Point":
        x, y = hull.coords[0]
        return box(x - 0.5, y - 0.5, x + 0.5, y + 0.5)
    if hull.geom_type == "LineString":
        min_x, min_y, max_x, max_y = hull.bounds
        if min_x == max_x:
            min_x -= 0.5
            max_x += 0.5
        if min_y == max_y:
            min_y -= 0.5
            max_y += 0.5
        return box(min_x, min_y, max_x, max_y)
    return Polygon(hull)


def apply_orientation(image: Image.Image, polygon: Polygon, orientation: Orientation) -> OrientedAsset:
    transformed_image = image
    transformed_polygon = polygon
    center = (image.width / 2.0, image.height / 2.0)

    if orientation.flipped:
        transformed_image = ImageOps.mirror(transformed_image)
        transformed_polygon = affinity.scale(
            transformed_polygon,
            xfact=-1.0,
            yfact=1.0,
            origin=center,
        )

    if orientation.angle_degrees % 360 != 0:
        transformed_image = transformed_image.rotate(
            orientation.angle_degrees,
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )
        transformed_polygon = affinity.rotate(
            transformed_polygon,
            orientation.angle_degrees,
            origin=center,
            use_radians=False,
        )

    min_x, min_y, _, _ = transformed_polygon.bounds
    transformed_polygon = affinity.translate(transformed_polygon, xoff=-min_x, yoff=-min_y)

    return OrientedAsset(
        image=transformed_image,
        polygon=transformed_polygon,
        orientation=orientation,
    )


def make_orientations(rotation_step_degrees: int, allow_flip: bool) -> list[Orientation]:
    if rotation_step_degrees <= 0:
        raise ValueError("rotation_step_degrees must be greater than 0.")
    if 360 % rotation_step_degrees != 0:
        raise ValueError("rotation_step_degrees must divide 360 exactly.")

    orientations: list[Orientation] = []
    flips = [False, True] if allow_flip else [False]
    for flipped in flips:
        for angle in range(0, 360, rotation_step_degrees):
            orientations.append(Orientation(float(angle), flipped))
    return orientations
