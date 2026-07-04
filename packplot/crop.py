"""Crop images to foreground content, optionally deskewing rotated content."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_AXIS_ALIGNED_ANGLE = 0.5


def crop_to_content(
    image: Image.Image | Path | str,
    background: tuple[int, ...] = (255, 255, 255),
    threshold: int = 0,
    *,
    output_path: Path | str | None = None,
    write: bool = True,
) -> Image.Image:
    """Crop *image* to foreground content, deskewing when content is rotated."""
    loaded = _load_image(image)
    cropped = _deskew_and_crop(loaded, background, threshold)

    if write:
        if output_path is None:
            raise ValueError("output_path is required when write=True")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output)

    return cropped


def _load_image(image: Image.Image | Path | str) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(image)


def _foreground_mask(
    image: Image.Image,
    background: tuple[int, ...],
    threshold: int,
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    bg = np.array(background[:3], dtype=np.int16)
    diff = np.abs(rgb.astype(np.int16) - bg).max(axis=-1)
    return diff > threshold


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) * 255


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        _mask_to_uint8(mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _deskew_angle(rect: tuple[tuple[float, float], tuple[float, float], float]) -> float:
    """Return the smallest rotation (degrees) that axis-aligns the min-area rect."""
    _, (width, height), angle = rect
    if width < height:
        angle += 90.0
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    return angle


def _min_area_rect_angle(mask: np.ndarray) -> float:
    contour = _largest_contour(mask)
    if contour is None or len(contour) < 3:
        raise ValueError("no foreground content found")
    return _deskew_angle(cv2.minAreaRect(contour))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("no foreground content found")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _rotate_fillcolor(
    image: Image.Image,
    background: tuple[int, ...],
) -> tuple[int, ...]:
    if image.mode == "RGBA":
        if len(background) >= 4:
            return background[:4]
        return (*background[:3], 255)
    return background[:3]


def _deskew_and_crop(
    image: Image.Image,
    background: tuple[int, ...],
    threshold: int,
) -> Image.Image:
    mask = _foreground_mask(image, background, threshold)
    angle = _min_area_rect_angle(mask)

    if abs(angle) < _AXIS_ALIGNED_ANGLE:
        return image.crop(_bbox_from_mask(mask))

    rotated = image.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=_rotate_fillcolor(image, background),
    )
    rotated_mask = _foreground_mask(rotated, background, threshold)
    bbox = _bbox_from_mask(rotated_mask)
    return rotated.crop(bbox)
