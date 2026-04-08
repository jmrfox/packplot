from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot.extract import extract_source_objects
from packplot.packer import pack_polygons
from packplot.types import PackOptions


def _make_shape(path: Path, size: tuple[int, int], shape: str) -> Path:
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if shape == "rect":
        draw.rectangle((4, 6, size[0] - 4, size[1] - 6), fill=(20, 20, 20, 255))
    elif shape == "triangle":
        draw.polygon(((size[0] // 2, 3), (size[0] - 4, size[1] - 4), (4, size[1] - 4)), fill=(20, 20, 20, 255))
    else:
        draw.ellipse((4, 4, size[0] - 4, size[1] - 4), fill=(20, 20, 20, 255))
    image.save(path)
    return path


def test_packer_places_without_overlap(tmp_path: Path) -> None:
    paths = [
        _make_shape(tmp_path / "a.png", (40, 30), "rect"),
        _make_shape(tmp_path / "b.png", (30, 35), "triangle"),
        _make_shape(tmp_path / "c.png", (28, 28), "ellipse"),
    ]
    options = PackOptions(target_aspect_ratio=1.4, rotation_step_degrees=90, padding=2)
    sources = extract_source_objects(paths, options)

    placements, canvas = pack_polygons(sources, options)
    assert len(placements) == len(paths)
    assert canvas[0] > 0 and canvas[1] > 0

    for i, left in enumerate(placements):
        for right in placements[i + 1 :]:
            assert left.polygon.intersection(right.polygon).area <= 1e-6


def test_packer_respects_target_aspect_ratio_direction(tmp_path: Path) -> None:
    paths = [
        _make_shape(tmp_path / "wide.png", (42, 24), "rect"),
        _make_shape(tmp_path / "tall.png", (24, 42), "rect"),
    ]
    options = PackOptions(target_aspect_ratio=2.0, rotation_step_degrees=90, padding=1)
    sources = extract_source_objects(paths, options)
    _, canvas = pack_polygons(sources, options)
    ratio = canvas[0] / canvas[1]
    assert ratio > 1.0
