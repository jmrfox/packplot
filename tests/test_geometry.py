from __future__ import annotations

import numpy as np
from PIL import Image

from packplot.geometry import Orientation, apply_orientation, convex_hull_from_points


def test_convex_hull_handles_collinear_points() -> None:
    points = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    hull = convex_hull_from_points(points)
    assert hull.area > 0
    min_x, min_y, max_x, max_y = hull.bounds
    assert max_x > min_x
    assert max_y > min_y


def test_apply_orientation_keeps_polygon_on_positive_plane() -> None:
    image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
    points = np.array([[2.0, 2.0], [17.0, 2.0], [9.0, 8.0]])
    polygon = convex_hull_from_points(points)
    oriented = apply_orientation(image, polygon, Orientation(angle_degrees=90.0, flipped=True))

    min_x, min_y, max_x, max_y = oriented.polygon.bounds
    assert min_x >= 0
    assert min_y >= 0
    assert max_x > 0
    assert max_y > 0
    assert oriented.image.size[0] > 0
    assert oriented.image.size[1] > 0
