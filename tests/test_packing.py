"""Tests for packplot.packing."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from packplot.input import Entry
from packplot.packing import pack_entries


def _entry(path: Path, size: tuple[int, int]) -> Entry:
    entry = Entry(path)
    entry.cache_image(Image.new("RGB", size, (255, 0, 0)))
    return entry


def _grid_canvas_area(entries: list[Entry]) -> int:
    n = len(entries)
    grid_size = math.ceil(math.sqrt(n))
    widths = [entry.width for entry in entries]
    heights = [entry.height for entry in entries]

    col_widths = [0] * grid_size
    row_heights = [0] * grid_size
    for i, (width, height) in enumerate(zip(widths, heights, strict=True)):
        col = i % grid_size
        row = i // grid_size
        col_widths[col] = max(col_widths[col], width)
        row_heights[row] = max(row_heights[row], height)

    return sum(col_widths) * sum(row_heights)


def test_pack_entries_beats_grid_area(tmp_path: Path) -> None:
    sizes = [(100, 30), (40, 60), (30, 30), (70, 70), (100, 50), (30, 30)]
    entries = [_entry(tmp_path / f"img{i}.png", size) for i, size in enumerate(sizes)]

    canvas_width, canvas_height, _ = pack_entries(entries)

    packed_area = canvas_width * canvas_height
    grid_area = _grid_canvas_area(entries)
    assert packed_area < grid_area


def test_pack_entries_rotates_when_beneficial(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path / "wide.png", (100, 30)),
        _entry(tmp_path / "tall.png", (40, 60)),
        _entry(tmp_path / "square.png", (30, 30)),
    ]

    _, _, placements = pack_entries(entries, rotation=True)

    assert any(placement.rotated for placement in placements)


def test_pack_entries_placements_within_canvas(tmp_path: Path) -> None:
    sizes = [(100, 30), (40, 60), (30, 30), (70, 70)]
    entries = [_entry(tmp_path / f"img{i}.png", size) for i, size in enumerate(sizes)]

    canvas_width, canvas_height, placements = pack_entries(entries)

    for placement in placements:
        image = placement.entry.load()
        width = image.height if placement.rotated else image.width
        height = image.width if placement.rotated else image.height
        assert 0 <= placement.x
        assert 0 <= placement.y
        assert placement.x + width <= canvas_width
        assert placement.y + height <= canvas_height


def test_pack_entries_empty() -> None:
    assert pack_entries([]) == (0, 0, [])


def test_pack_entries_respects_target_aspect_ratio(tmp_path: Path) -> None:
    sizes = [(100, 30), (40, 60), (30, 30), (70, 70), (100, 50), (30, 30)]
    entries = [_entry(tmp_path / f"img{i}.png", size) for i, size in enumerate(sizes)]

    width_169, height_169, _ = pack_entries(entries, aspect_ratio="16:9")
    width_34, height_34, _ = pack_entries(entries, aspect_ratio="3:4")

    ratio_169 = width_169 / height_169
    ratio_34 = width_34 / height_34
    assert abs(ratio_169 - 16 / 9) < abs(ratio_34 - 16 / 9)
    assert abs(ratio_34 - 3 / 4) < abs(ratio_169 - 3 / 4)


def test_pack_entries_no_rotation_preserves_orientation(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path / "wide.png", (100, 30)),
        _entry(tmp_path / "tall.png", (40, 60)),
    ]

    _, _, placements = pack_entries(entries, rotation=False)

    assert all(not placement.rotated for placement in placements)
