from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot import PackOptions, pack_images


def _make_shape(path: Path, shape: str) -> None:
    image = Image.new("RGBA", (42, 42), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if shape == "rect":
        draw.rectangle((6, 10, 34, 30), fill=(20, 20, 20, 255))
    elif shape == "tri":
        draw.polygon(((21, 6), (36, 34), (6, 34)), fill=(20, 20, 20, 255))
    else:
        draw.ellipse((7, 7, 35, 35), fill=(20, 20, 20, 255))
    image.save(path)


def test_optimizer_solver_produces_layout(tmp_path: Path) -> None:
    paths = [tmp_path / "o1.png", tmp_path / "o2.png", tmp_path / "o3.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")

    result = pack_images(
        paths,
        options=PackOptions(
            solver="optimize",
            optimizer_method="lbfgsb",
            target_aspect_ratio=1.6,
            padding=2,
            edge_buffer=1.5,
            jacobi_inflation=1.1,
            optimizer_maxiter=10,
            enable_spread_phase=False,
        ),
    )

    assert len(result.placements) == 3
    assert result.canvas_size[0] > 0
    assert result.canvas_size[1] > 0
    ratio = result.canvas_size[0] / result.canvas_size[1]
    assert ratio > 1.0
