"""Tests for packplot.packplot."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from packplot.packplot import packplot


def _white_png(path: Path, size: tuple[int, int] = (40, 30)) -> Path:
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return path


def test_packplot_returns_image(tmp_path: Path) -> None:
    paths = [_white_png(tmp_path / f"img{i}.png") for i in range(2)]

    canvas = packplot(paths)

    assert canvas.width > 0
    assert canvas.height > 0


def test_packplot_writes_output(tmp_path: Path) -> None:
    paths = [_white_png(tmp_path / f"img{i}.png") for i in range(2)]
    output = tmp_path / "out.png"

    canvas = packplot(paths, output_path=output)

    assert output.is_file()
    assert canvas.size == Image.open(output).size


def test_packplot_empty_raises() -> None:
    with pytest.raises(ValueError, match="cannot render an empty arrangement"):
        packplot([])
