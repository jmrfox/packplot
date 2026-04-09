from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from packplot import PymooConfig, pack_images
from tests.helpers import fast_pymoo_options


def test_pymoo_solver_produces_layout(tmp_path: Path) -> None:
    paths = [tmp_path / "obj1.png", tmp_path / "obj2.png"]
    image = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 6, 18, 17), fill=(20, 20, 20, 255))
    image.save(paths[0])
    image2 = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw2 = ImageDraw.Draw(image2)
    draw2.ellipse((4, 4, 19, 19), fill=(20, 20, 20, 255))
    image2.save(paths[1])

    results = pack_images(paths, options=fast_pymoo_options())
    assert len(results) >= 1
    best = results[0]
    assert len(best.placements) == 2
    assert best.canvas_size[0] > 0
    assert best.canvas_size[1] > 0
    assert best.solver_method is not None
    assert "pymoo" in best.solver_method


def test_pymoo_best_layout_count_option(tmp_path: Path) -> None:
    paths = [tmp_path / "obj1.png", tmp_path / "obj2.png"]
    image = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 6, 18, 17), fill=(20, 20, 20, 255))
    image.save(paths[0])
    image2 = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw2 = ImageDraw.Draw(image2)
    draw2.ellipse((4, 4, 19, 19), fill=(20, 20, 20, 255))
    image2.save(paths[1])

    results = pack_images(
        paths,
        options=fast_pymoo_options(
            pymoo_config=PymooConfig(
                algorithm="nsga2",
                generations=10,
                population_size=16,
                offspring_count=8,
                eliminate_duplicates=True,
                best_layout_count=3,
            )
        ),
    )
    assert len(results) >= 1
    assert len(results) <= 3


def test_pymoo_ranking_is_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    paths = [tmp_path / "obj1.png", tmp_path / "obj2.png", tmp_path / "obj3.png"]
    image = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 6, 18, 17), fill=(20, 20, 20, 255))
    image.save(paths[0])
    image2 = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw2 = ImageDraw.Draw(image2)
    draw2.ellipse((4, 4, 19, 19), fill=(20, 20, 20, 255))
    image2.save(paths[1])
    image3 = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw3 = ImageDraw.Draw(image3)
    draw3.polygon(((12, 3), (20, 20), (4, 20)), fill=(20, 20, 20, 255))
    image3.save(paths[2])

    options = fast_pymoo_options(
        pymoo_config=PymooConfig(
            algorithm="nsga2",
            generations=10,
            population_size=20,
            offspring_count=10,
            eliminate_duplicates=True,
            best_layout_count=3,
        )
    )
    options = replace(
        options,
        optimize_config=replace(options.optimize_config, compact_to_clearance_beam_width=3),
        random_seed=123,
    )

    first = pack_images(paths, options=options)
    second = pack_images(paths, options=options)
    assert len(first) == len(second)
    first_signature = [
        (
            result.canvas_size,
            round(result.total_overlap_area, 6),
            round(float(result.minimum_clearance or 0.0), 6),
            round(float(result.outside_violation_magnitude or 0.0), 6),
        )
        for result in first
    ]
    second_signature = [
        (
            result.canvas_size,
            round(result.total_overlap_area, 6),
            round(float(result.minimum_clearance or 0.0), 6),
            round(float(result.outside_violation_magnitude or 0.0), 6),
        )
        for result in second
    ]
    assert first_signature == second_signature
