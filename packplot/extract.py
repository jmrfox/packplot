from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity

from packplot.geometry import convex_hull_from_points
from packplot.types import PackOptions, SourceObject

logger = logging.getLogger(__name__)


def _mask_from_rgba(image_rgba: Image.Image, options: PackOptions) -> np.ndarray:
    data = np.asarray(image_rgba)
    alpha = data[:, :, 3]
    has_transparency = np.any(alpha < 255)
    if has_transparency:
        logger.debug("Using alpha-channel mask extraction.")
        return alpha > options.alpha_threshold

    rgb = data[:, :, :3]
    logger.debug(
        "Using white-threshold fallback mask extraction (threshold=%d).",
        options.white_threshold,
    )
    return np.any(rgb < options.white_threshold, axis=2)


def _crop_to_mask(image_rgba: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        logger.warning("No foreground pixels detected in image of size %s.", image_rgba.size)
        raise ValueError("Image has no detectable foreground pixels.")

    top = int(rows.min())
    bottom = int(rows.max()) + 1
    left = int(cols.min())
    right = int(cols.max()) + 1

    cropped_image = image_rgba.crop((left, top, right, bottom))
    cropped_mask = mask[top:bottom, left:right]
    return cropped_image, cropped_mask


def _points_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.column_stack((xs.astype(float), ys.astype(float)))


def _estimate_background_color(image_rgba: Image.Image, mask: np.ndarray) -> tuple[int, int, int]:
    # Border pixels are usually background; median is robust to small artifacts.
    rgb = np.asarray(image_rgba)[:, :, :3]
    border = np.zeros(mask.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    background_border = np.logical_and(border, np.logical_not(mask))
    candidates = rgb[background_border]
    if candidates.size == 0:
        candidates = rgb[border]

    median = np.median(candidates, axis=0).astype(int)
    return int(median[0]), int(median[1]), int(median[2])


def _cut_image_to_hull(cropped_image: Image.Image, hull) -> Image.Image:
    mask = Image.new("L", cropped_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(list(hull.exterior.coords), fill=255)

    rgba = np.asarray(cropped_image).copy()
    hull_alpha = np.asarray(mask)
    rgba[:, :, 3] = np.minimum(rgba[:, :, 3], hull_alpha)
    return Image.fromarray(rgba, mode="RGBA")


def extract_source_object_from_image(source_path: str | Path, image_rgba: Image.Image, options: PackOptions) -> SourceObject:
    """Extract one RGBA image object into a hull-clipped object and metadata."""
    path_obj = Path(source_path)
    mask = _mask_from_rgba(image_rgba, options)
    background_color = _estimate_background_color(image_rgba, mask)
    cropped_image, cropped_mask = _crop_to_mask(image_rgba, mask)

    points = _points_from_mask(cropped_mask)
    hull = convex_hull_from_points(points)
    min_x, min_y, _, _ = hull.bounds
    hull = affinity.translate(hull, xoff=-min_x, yoff=-min_y)
    hull_image = _cut_image_to_hull(cropped_image, hull)
    logger.debug(
        "Extracted object %s -> crop=%s hull_crop=%s mask_pixels=%d hull_area=%.2f bg=%s",
        path_obj.name,
        cropped_image.size,
        hull_image.size,
        int(np.count_nonzero(cropped_mask)),
        float(hull.area),
        background_color,
    )

    return SourceObject(
        source_path=path_obj,
        cropped_image=hull_image,
        mask=cropped_mask,
        hull=hull,
        background_color=background_color,
    )


def extract_source_object(path: str | Path, options: PackOptions) -> SourceObject:
    """Extract one image into a hull-clipped object and metadata."""
    source_path = Path(path)
    logger.info("Extracting source object from %s", source_path)
    image_rgba = Image.open(source_path).convert("RGBA")
    return extract_source_object_from_image(source_path, image_rgba, options)


def extract_source_objects(paths: list[str | Path], options: PackOptions) -> list[SourceObject]:
    """Extract all input images into `SourceObject` instances."""
    if not paths:
        raise ValueError("At least one image path is required.")
    logger.info("Starting extraction for %d source images.", len(paths))
    return [extract_source_object(path, options) for path in paths]
