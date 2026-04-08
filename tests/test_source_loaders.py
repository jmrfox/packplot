from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from packplot import PackOptions
from packplot.source_loaders import get_source_loader, infer_source_loader_name, validate_source_inputs


def test_raster_source_loader_loads_objects(tmp_path: Path) -> None:
    path = tmp_path / "alpha.png"
    image = Image.new("RGBA", (32, 32), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((7, 8, 24, 23), fill=(20, 20, 20, 255))
    image.save(path)

    loader = get_source_loader("raster")
    loaded = loader.load([path], PackOptions())
    assert len(loaded) == 1
    assert loaded[0].source_path == path
    assert loaded[0].hull.area > 0


def test_raster_source_loader_loads_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (32, 32), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 9, 23, 24), fill=(20, 20, 20))
    image.save(path, format="JPEG")

    loader = get_source_loader("raster")
    loaded = loader.load([path], PackOptions())
    assert len(loaded) == 1
    assert loaded[0].source_path == path
    assert loaded[0].hull.area > 0


def test_infer_source_loader_name_handles_svg_and_mixed(tmp_path: Path) -> None:
    png = tmp_path / "a.png"
    svg = tmp_path / "b.svg"
    Image.new("RGBA", (4, 4), (255, 255, 255, 255)).save(png)
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
    assert infer_source_loader_name([png]) == "raster"
    assert infer_source_loader_name([svg]) == "svg"
    assert infer_source_loader_name([png, svg]) == "mixed"


def test_svg_source_loader_path_exists(tmp_path: Path) -> None:
    svg_path = tmp_path / "shape.svg"
    svg_path.write_text(
        (
            "<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'>"
            "<rect x='6' y='6' width='20' height='20' fill='black'/>"
            "</svg>"
        ),
        encoding="utf-8",
    )
    loader = get_source_loader("svg")
    try:
        loaded = loader.load([svg_path], PackOptions())
        assert len(loaded) == 1
        assert loaded[0].source_path == svg_path
        assert loaded[0].hull.area > 0
    except RuntimeError as exc:
        assert "cairosvg" in str(exc).lower()


def test_validate_source_inputs_rejects_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "bad.bmp"
    bad.write_text("not-an-image", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported source format"):
        validate_source_inputs([bad])


def test_validate_source_inputs_rejects_missing_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(ValueError, match="not found"):
        validate_source_inputs([missing])
