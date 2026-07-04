"""Tests for packplot.arrangement."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from packplot.arrangement import Arrangement
from packplot.input import Entry


def _entry(path: Path, size: tuple[int, int]) -> Entry:
    entry = Entry(path)
    entry.cache_image(Image.new("RGB", size, (255, 0, 0)))
    return entry


def test_arrangement_grid_layout(tmp_path: Path) -> None:
    sizes = [(10, 20), (30, 10), (15, 15), (20, 10)]
    entries = [_entry(tmp_path / f"img{i}.png", size) for i, size in enumerate(sizes)]

    arrangement = Arrangement(entries, layout="grid")

    assert arrangement.canvas_width > 0
    assert arrangement.canvas_height > 0
    assert len(arrangement.placements) == len(entries)
    assert all(not placement.rotated for placement in arrangement.placements)
    assert arrangement.canvas_width * arrangement.canvas_height >= sum(
        w * h for w, h in sizes
    )


def test_arrangement_pack_layout(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path / "a.png", (100, 30)),
        _entry(tmp_path / "b.png", (40, 60)),
    ]

    grid = Arrangement(entries, layout="grid")
    packed = Arrangement(entries, layout="pack")

    assert packed.canvas_width * packed.canvas_height <= (
        grid.canvas_width * grid.canvas_height
    )
