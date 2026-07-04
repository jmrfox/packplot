"""Tests for packplot.input."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from packplot.input import EntrySet


def _white_png(path: Path, size: tuple[int, int] = (80, 60)) -> Path:
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return path


def _red_square_png(path: Path, size: tuple[int, int] = (40, 40)) -> Path:
    canvas = Image.new("RGB", size, (255, 255, 255))
    canvas.paste(Image.new("RGB", (20, 20), (255, 0, 0)), (10, 10))
    canvas.save(path)
    return path


def test_entry_set_loads_paths(tmp_path: Path) -> None:
    paths = [_white_png(tmp_path / f"img{i}.png") for i in range(3)]

    entry_set = EntrySet(paths)

    assert len(entry_set) == 3
    assert [entry.width for entry in entry_set] == [80, 80, 80]


def test_entry_set_crop_reduces_dimensions(tmp_path: Path) -> None:
    path = _red_square_png(tmp_path / "square.png")

    uncropped = EntrySet([path])
    cropped = EntrySet([path], crop=True)

    assert cropped.entries[0].width < uncropped.entries[0].width
    assert cropped.entries[0].height < uncropped.entries[0].height


def test_entry_set_margin_expands_dimensions(tmp_path: Path) -> None:
    path = _white_png(tmp_path / "img.png", size=(40, 30))
    margin_width = 5

    plain = EntrySet([path])
    margined = EntrySet([path], margin_width=margin_width)

    assert margined.entries[0].width == plain.entries[0].width + 2 * margin_width
    assert margined.entries[0].height == plain.entries[0].height + 2 * margin_width


def test_entry_set_crop_then_margin(tmp_path: Path) -> None:
    path = _red_square_png(tmp_path / "square.png")
    margin_width = 4

    cropped = EntrySet([path], crop=True)
    cropped_and_margined = EntrySet([path], crop=True, margin_width=margin_width)

    assert cropped_and_margined.entries[0].width == cropped.entries[0].width + 2 * margin_width
    assert cropped_and_margined.entries[0].height == cropped.entries[0].height + 2 * margin_width
