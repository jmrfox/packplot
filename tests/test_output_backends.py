from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot import list_output_backends, pack_images, render_result_with_backend
from tests.helpers import fast_opt_options


def _make_object(path: Path, shape: str) -> None:
    image = Image.new("RGBA", (30, 30), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if shape == "blob":
        draw.ellipse((6, 6, 24, 24), fill=(20, 90, 220, 255))
    else:
        draw.rectangle((6, 8, 24, 22), fill=(20, 20, 20, 255))
    image.save(path)


def _make_result(tmp_path: Path):
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    _make_object(paths[0], "blob")
    _make_object(paths[1], "rect")
    return pack_images(paths, options=fast_opt_options())[0]


def test_list_output_backends_contains_expected_names() -> None:
    backends = set(list_output_backends())
    assert {"pil", "matplotlib", "seaborn", "plotly", "bokeh", "pyvista", "pygal"} <= backends


def test_render_result_with_pil_backend(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    rendered = render_result_with_backend(result, backend="pil")
    assert rendered.size == result.image.size


def test_render_result_with_unknown_backend_raises(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    try:
        render_result_with_backend(result, backend="nope")
    except ValueError as exc:
        assert "Supported backends" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown backend.")
