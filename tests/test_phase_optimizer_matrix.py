from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from packplot import (
    DifferentialEvolutionConfig,
    LbfgsbConfig,
    OptimizeConfig,
    OptimizationPhaseConfig,
    pack_images,
)
from tests.helpers import fast_opt_options


def _make_shape(path: Path, shape: str) -> None:
    image = Image.new("RGBA", (30, 30), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    if shape == "rect":
        draw.rectangle((4, 7, 25, 22), fill=(20, 20, 20, 255))
    elif shape == "tri":
        draw.polygon(((15, 4), (26, 25), (4, 25)), fill=(20, 20, 20, 255))
    else:
        draw.ellipse((5, 5, 24, 24), fill=(20, 20, 20, 255))
    image.save(path)


def _matrix_options(*, compact_method: str, clearance_method: str):
    phase_lbfgsb = LbfgsbConfig(max_iterations=10, random_restart_count=1, alternating_refinement_cycles=0)
    phase_de = DifferentialEvolutionConfig(max_generations=3, population_size=7, worker_count=1)
    optimize_cfg = OptimizeConfig(
        compact_layout_backend="optimize",
        compact_layout_best_count=2,
        compact_to_clearance_beam_width=2,
        compact_layout=OptimizationPhaseConfig(
            method=compact_method,
            progress_log_every_evaluations=0,
            lbfgsb=phase_lbfgsb,
            differential_evolution=phase_de,
        ),
        enable_clearance_refinement_phase=True,
        clearance_refinement=OptimizationPhaseConfig(
            method=clearance_method,
            progress_log_every_evaluations=0,
            lbfgsb=phase_lbfgsb,
            differential_evolution=phase_de,
        ),
    )
    return fast_opt_options(
        optimize_config=optimize_cfg,
        target_aspect_ratio=1.25,
        padding=2,
        edge_buffer=1.0,
        random_seed=11,
    )


@pytest.mark.parametrize(
    ("compact_method", "clearance_method"),
    [
        ("lbfgsb", "de"),
        ("de", "nsga2"),
        ("nsga2", "lbfgsb"),
    ],
)
def test_phase_optimizer_matrix_outputs_and_determinism(
    tmp_path: Path,
    compact_method: str,
    clearance_method: str,
) -> None:
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")
    options = _matrix_options(compact_method=compact_method, clearance_method=clearance_method)

    first = pack_images(paths, options=options)
    second = pack_images(paths, options=options)

    assert len(first) >= 1
    assert len(second) == len(first)
    for result in first:
        assert result.sanity_passed is not None
        assert result.minimum_clearance is not None
        assert result.outside_violation_magnitude is not None

    first_signature = [
        (
            result.canvas_size,
            round(result.total_overlap_area, 6),
            round(float(result.minimum_clearance or 0.0), 6),
            round(float(result.outside_violation_magnitude or 0.0), 6),
            round(result.fill_ratio, 6),
        )
        for result in first
    ]
    second_signature = [
        (
            result.canvas_size,
            round(result.total_overlap_area, 6),
            round(float(result.minimum_clearance or 0.0), 6),
            round(float(result.outside_violation_magnitude or 0.0), 6),
            round(result.fill_ratio, 6),
        )
        for result in second
    ]
    assert first_signature == second_signature


def test_pymoo_compact_backend_with_phase_clearance(tmp_path: Path) -> None:
    paths = [tmp_path / "u.png", tmp_path / "v.png", tmp_path / "w.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")

    base = _matrix_options(compact_method="lbfgsb", clearance_method="lbfgsb")
    options = replace(
        base,
        optimize_config=replace(base.optimize_config, compact_layout_backend="pymoo"),
    )
    results = pack_images(paths, options=options)
    assert len(results) >= 1
    assert all(result.minimum_clearance is not None for result in results)
