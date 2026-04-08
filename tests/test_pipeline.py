from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot import PackOptions, pack_images


def _make_object(path: Path, shape: str) -> None:
    image = Image.new("RGBA", (48, 48), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if shape == "blob":
        draw.ellipse((8, 8, 38, 38), fill=(0, 120, 240, 255))
    elif shape == "diamond":
        draw.polygon(((24, 4), (44, 24), (24, 44), (4, 24)), fill=(240, 60, 60, 255))
    else:
        draw.rectangle((8, 12, 36, 34), fill=(60, 220, 80, 255))
    image.save(path)


def test_pipeline_packs_images_end_to_end(tmp_path: Path) -> None:
    image_paths = [
        tmp_path / "obj1.png",
        tmp_path / "obj2.png",
        tmp_path / "obj3.png",
    ]
    _make_object(image_paths[0], "blob")
    _make_object(image_paths[1], "diamond")
    _make_object(image_paths[2], "rect")

    result = pack_images(
        image_paths,
        options=PackOptions(
            solver="optimize",
            optimizer_method="lbfgsb",
            optimizer_maxiter=10,
            enable_spread_phase=False,
            target_aspect_ratio=1.3,
            padding=2,
        ),
    )

    assert result.image.mode == "RGBA"
    assert result.image.size == result.canvas_size
    assert len(result.placements) == len(image_paths)
    assert 0 < result.fill_ratio <= 1
