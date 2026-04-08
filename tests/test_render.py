from __future__ import annotations

from pathlib import Path

from PIL import Image
from shapely.geometry import box

from packplot.render import render_composition
from packplot.types import PackedPlacement


def _placement(x: int, y: int, size: int, color: tuple[int, int, int, int], name: str) -> PackedPlacement:
    image = Image.new("RGBA", (size, size), color)
    return PackedPlacement(
        source_path=Path(name),
        polygon=box(x, y, x + size, y + size),
        angle_degrees=0.0,
        flipped=False,
        top_left=(x, y),
        image=image,
    )


def test_render_handles_empty_placements() -> None:
    out = render_composition((12, 10), [], background_color=(11, 22, 33))
    assert out.size == (12, 10)
    assert out.getpixel((0, 0)) == (11, 22, 33, 255)


def test_render_uses_placement_order_for_layering() -> None:
    bottom = _placement(0, 0, 8, (255, 0, 0, 255), "bottom.png")
    top = _placement(0, 0, 8, (0, 255, 0, 255), "top.png")
    out = render_composition((8, 8), [bottom, top], background_color=(0, 0, 0))
    assert out.getpixel((4, 4)) == (0, 255, 0, 255)


def test_render_applies_background_color() -> None:
    obj = _placement(2, 2, 3, (10, 20, 30, 255), "obj.png")
    out = render_composition((8, 8), [obj], background_color=(200, 210, 220))
    assert out.getpixel((0, 0)) == (200, 210, 220, 255)
    assert out.getpixel((3, 3)) == (10, 20, 30, 255)
