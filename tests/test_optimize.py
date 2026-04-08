from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot import pack_images
from conftest import fast_opt_options


def _make_shape(path: Path, shape: str) -> None:
    image = Image.new("RGBA", (30, 30), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if shape == "rect":
        draw.rectangle((4, 7, 25, 22), fill=(20, 20, 20, 255))
    elif shape == "tri":
        draw.polygon(((15, 4), (26, 25), (4, 25)), fill=(20, 20, 20, 255))
    else:
        draw.ellipse((5, 5, 24, 24), fill=(20, 20, 20, 255))
    image.save(path)


def test_optimizer_solver_produces_layout(tmp_path: Path) -> None:
    paths = [tmp_path / "o1.png", tmp_path / "o2.png", tmp_path / "o3.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")

    result = pack_images(
        paths,
        options=fast_opt_options(
            target_aspect_ratio=1.6,
            padding=2,
            edge_buffer=1.5,
            jacobi_inflation=1.1,
        ),
    )

    assert len(result.placements) == 3
    assert result.canvas_size[0] > 0
    assert result.canvas_size[1] > 0
    ratio = result.canvas_size[0] / result.canvas_size[1]
    assert ratio > 1.0
