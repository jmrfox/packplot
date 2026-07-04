"""Tests for packplot.crop."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from packplot.crop import _foreground_mask, _min_area_rect_angle, crop_to_content


def _tilted_rectangle_image(
    size: int = 200,
    rect_size: tuple[int, int] = (120, 40),
    angle: float = 20.0,
) -> Image.Image:
    """White canvas with a solid red rectangle rotated on it."""
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    rect = Image.new("RGB", rect_size, (255, 0, 0))
    rotated_rect = rect.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    offset = ((size - rotated_rect.width) // 2, (size - rotated_rect.height) // 2)
    canvas.paste(rotated_rect, offset)
    return canvas


def test_crop_tilted_content_reduces_area() -> None:
    image = _tilted_rectangle_image()
    cropped = crop_to_content(image, write=False)

    assert cropped.width < image.width
    assert cropped.height < image.height
    assert cropped.width * cropped.height < image.width * image.height * 0.6


def test_crop_write_false_skips_disk(tmp_path: Path) -> None:
    image = _tilted_rectangle_image()
    output = tmp_path / "out.png"

    crop_to_content(image, write=False, output_path=output)

    assert not output.exists()


def test_crop_empty_raises() -> None:
    image = Image.new("RGB", (50, 50), (255, 255, 255))

    with pytest.raises(ValueError, match="no foreground content"):
        crop_to_content(image, write=False)


def test_crop_write_requires_output_path() -> None:
    image = _tilted_rectangle_image()

    with pytest.raises(ValueError, match="output_path is required"):
        crop_to_content(image, write=True)


def test_crop_axis_aligned_fast_path() -> None:
    canvas = Image.new("RGB", (100, 80), (255, 255, 255))
    canvas.paste(Image.new("RGB", (60, 30), (255, 0, 0)), (20, 25))

    cropped = crop_to_content(canvas, write=False)

    assert cropped.size == (60, 30)


def test_min_area_rect_angle_opencv() -> None:
    image = _tilted_rectangle_image(angle=20.0)
    mask = _foreground_mask(image, (255, 255, 255), 0)

    angle = _min_area_rect_angle(mask)

    assert abs(abs(angle) - 20.0) < 1.0
