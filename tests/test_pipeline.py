from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw

from packplot.arrangement import Arrangement, ArrangementEntry
from packplot import pack_images
from tests.helpers import fast_opt_options


def _make_object(path: Path, shape: str) -> None:
    image = Image.new("RGBA", (34, 34), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if shape == "blob":
        draw.ellipse((6, 6, 28, 28), fill=(0, 120, 240, 255))
    elif shape == "diamond":
        draw.polygon(((17, 3), (31, 17), (17, 31), (3, 17)), fill=(240, 60, 60, 255))
    else:
        draw.rectangle((6, 8, 28, 24), fill=(60, 220, 80, 255))
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

    results = pack_images(
        image_paths,
        options=fast_opt_options(target_aspect_ratio=1.3, padding=2),
    )
    result = results[0]

    assert result.image.mode == "RGBA"
    assert result.image.size == result.canvas_size
    assert len(result.placements) == len(image_paths)
    assert 0 < result.fill_ratio <= 1
    assert result.total_overlap_area >= 0
    assert result.out_of_bounds_count == 0
    assert result.sanity_passed
    assert result.minimum_clearance is not None
    assert result.minimum_clearance >= 0
    assert result.outside_violation_magnitude is not None
    assert result.outside_violation_magnitude == 0
    assert result.solver_method is not None
    assert result.solver_success is not None


def test_pipeline_logs_sanity_warning_when_arrangement_overlaps(tmp_path: Path, caplog) -> None:
    image_paths = [tmp_path / "a.png", tmp_path / "b.png"]
    _make_object(image_paths[0], "blob")
    _make_object(image_paths[1], "rect")
    arrangement = Arrangement(
        version=1,
        canvas_size=(48, 48),
        background_color=(255, 255, 255),
        entries=[
            ArrangementEntry(key="a", center_norm=(0.5, 0.5), angle_degrees=0.0, flipped=False),
            ArrangementEntry(key="b", center_norm=(0.5, 0.5), angle_degrees=0.0, flipped=False),
        ],
    )
    caplog.set_level(logging.WARNING)
    results = pack_images(
        image_paths,
        arrangement=arrangement,
        arrangement_key_mode="stem",
        strict_arrangement=True,
    )
    result = results[0]
    assert not result.sanity_passed
    assert result.total_overlap_area > 0
    assert any("Layout sanity check failed" in rec.message for rec in caplog.records)
