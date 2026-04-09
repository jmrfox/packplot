from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from packplot import (
    DifferentialEvolutionConfig,
    LbfgsbConfig,
    PipelineConfig,
    SolverConfig,
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


def _matrix_options(*, pack_optimizer: str, refine_optimizer: str):
    phase_lbfgsb = LbfgsbConfig(max_iterations=10, random_restart_count=1, alternating_refinement_cycles=0)
    phase_de = DifferentialEvolutionConfig(max_generations=3, population_size=7, worker_count=1)
    pipeline_cfg = PipelineConfig(
        pack_best_count=2,
        pack_to_refine_beam_width=2,
        pack_phase=SolverConfig(
            optimizer=pack_optimizer,
            progress_log_every_evaluations=0,
            lbfgsb=phase_lbfgsb,
            differential_evolution=phase_de,
        ),
        enable_refine_phase=True,
        refine_phase=SolverConfig(
            optimizer=refine_optimizer,
            progress_log_every_evaluations=0,
            lbfgsb=phase_lbfgsb,
            differential_evolution=phase_de,
        ),
    )
    return fast_opt_options(
        pipeline_config=pipeline_cfg,
        target_aspect_ratio=1.25,
        padding=2,
        edge_buffer=1.0,
        random_seed=11,
    )


@pytest.mark.parametrize(
    ("pack_optimizer", "refine_optimizer"),
    [
        ("scipy-lbfgsb", "scipy-de"),
        ("scipy-de", "pymoo-nsga2"),
        ("scipy-nsga2", "scipy-lbfgsb"),
    ],
)
def test_phase_optimizer_matrix_outputs_and_determinism(
    tmp_path: Path,
    pack_optimizer: str,
    refine_optimizer: str,
) -> None:
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")
    options = _matrix_options(pack_optimizer=pack_optimizer, refine_optimizer=refine_optimizer)

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

    base = _matrix_options(pack_optimizer="scipy-lbfgsb", refine_optimizer="scipy-lbfgsb")
    options = replace(
        base,
        pipeline_config=replace(
            base.pipeline_config,
            pack_phase=replace(base.pipeline_config.pack_phase, optimizer="pymoo-nsga2"),
        ),
    )
    results = pack_images(paths, options=options)
    assert len(results) >= 1
    assert all(result.minimum_clearance is not None for result in results)


def test_pack_multicandidate_flow_through_refine_and_render(tmp_path: Path) -> None:
    paths = [tmp_path / "m1.png", tmp_path / "m2.png", tmp_path / "m3.png"]
    _make_shape(paths[0], "rect")
    _make_shape(paths[1], "tri")
    _make_shape(paths[2], "ellipse")

    pipeline_cfg = PipelineConfig(
        pack_phase=SolverConfig(
            optimizer="scipy-lbfgsb",
            progress_log_every_evaluations=0,
            lbfgsb=LbfgsbConfig(max_iterations=8, random_restart_count=2, alternating_refinement_cycles=0),
            differential_evolution=DifferentialEvolutionConfig(max_generations=2, population_size=6, worker_count=1),
        ),
        pack_best_count=2,
        pack_to_refine_beam_width=2,
        enable_refine_phase=True,
        refine_phase=SolverConfig(
            optimizer="scipy-lbfgsb",
            progress_log_every_evaluations=0,
            lbfgsb=LbfgsbConfig(max_iterations=5, random_restart_count=1, alternating_refinement_cycles=0),
            differential_evolution=DifferentialEvolutionConfig(max_generations=2, population_size=6, worker_count=1),
        ),
    )
    results = pack_images(
        paths,
        options=fast_opt_options(
            pipeline_config=pipeline_cfg,
            random_seed=7,
            target_aspect_ratio=1.1,
        ),
    )
    assert len(results) == 2
    assert all(result.image.size == result.canvas_size for result in results)
