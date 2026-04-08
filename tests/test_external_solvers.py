from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot import CellPackingConfig, CircPackerConfig, PackOptions, pack_images


def _make_shape(path: Path, kind: str) -> None:
    image = Image.new("RGBA", (52, 52), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    if kind == "rect":
        draw.rectangle((10, 16, 42, 36), fill=(20, 20, 20, 255))
    elif kind == "tri":
        draw.polygon(((26, 8), (44, 42), (8, 42)), fill=(20, 20, 20, 255))
    else:
        draw.ellipse((11, 11, 41, 41), fill=(20, 20, 20, 255))
    image.save(path)


def test_circpacker_solver_smoke(tmp_path: Path) -> None:
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")
    result = pack_images(
        paths,
        options=PackOptions(
            solver="circpacker",
            target_aspect_ratio=1.5,
            circpacker_config=CircPackerConfig(initial_depth=3, max_depth=6, max_canvas_growth_steps=5),
        ),
    )
    assert len(result.placements) == 3
    assert result.canvas_size[0] > 0 and result.canvas_size[1] > 0


def test_cell_packing_solver_smoke(tmp_path: Path) -> None:
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")
    result = pack_images(
        paths,
        options=PackOptions(
            solver="cell_packing",
            target_aspect_ratio=1.2,
            cell_packing_config=CellPackingConfig(iterations=60, attraction_step=0.5, repulsion_step=1.8),
        ),
    )
    assert len(result.placements) == 3
    assert result.canvas_size[0] > 0 and result.canvas_size[1] > 0
