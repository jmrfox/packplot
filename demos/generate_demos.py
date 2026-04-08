from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

from PIL import Image
from packplot import (
    CellPackingConfig,
    CircPackerConfig,
    LbfgsbConfig,
    OptimizeConfig,
    OptimizationPhaseConfig,
    PackOptions,
    pack_images,
)

logger = logging.getLogger(__name__)


def _subset_every_other(items: list[Path], start: int) -> list[Path]:
    return [item for idx, item in enumerate(items) if idx % 2 == start]


def _pick_smallest_image(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    best_path: Path | None = None
    best_area = float("inf")
    for path in paths:
        with Image.open(path) as image:
            area = float(image.width * image.height)
        if area < best_area:
            best_area = area
            best_path = path
    return [best_path] if best_path is not None else []


def _options_for_solver(solver: str, aspect_ratio: float) -> PackOptions:
    common = dict(
        solver=solver,
        target_aspect_ratio=aspect_ratio,
        white_threshold=250,
        padding=2,
        edge_buffer=1.0,
    )
    if solver == "optimize":
        return PackOptions(
            **common,
            optimize_config=OptimizeConfig(
                phase1=OptimizationPhaseConfig(
                    method="lbfgsb",
                    progress_log_every_evaluations=2000,
                    lbfgsb=LbfgsbConfig(
                        max_iterations=220,
                        random_restart_count=4,
                        alternating_refinement_cycles=2,
                    ),
                ),
                enable_spread_phase=True,
                spread=OptimizationPhaseConfig(
                    method="lbfgsb",
                    progress_log_every_evaluations=1200,
                    lbfgsb=LbfgsbConfig(
                        max_iterations=120,
                        random_restart_count=2,
                        alternating_refinement_cycles=1,
                    ),
                ),
            ),
        )
    if solver == "heuristic":
        return PackOptions(
            **common,
            rotation_step_degrees=90,
            allow_flip=False,
            fill_ratio=0.25,
            max_grow_steps=16,
            grow_factor=1.2,
        )
    if solver == "circpacker":
        return PackOptions(
            **common,
            circpacker_config=CircPackerConfig(
                initial_depth=3,
                max_depth=6,
                max_canvas_growth_steps=4,
                canvas_growth_factor=1.15,
            ),
        )
    if solver == "cell_packing":
        return PackOptions(
            **common,
            cell_packing_config=CellPackingConfig(
                iterations=120,
                attraction_step=0.55,
                repulsion_step=1.9,
            ),
        )
    raise ValueError(f"Unsupported solver '{solver}'")


def main() -> None:
    level_name = os.getenv("PACKPLOT_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    logger.info("Demo generator started with log level %s", logging.getLevelName(log_level))

    repo_root = Path(__file__).resolve().parent.parent
    inputs_dir = repo_root / "inputs"
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(inputs_dir.glob("*.png"))
    if not all_images:
        logger.error("No PNG files found in %s", inputs_dir)
        raise RuntimeError(f"No PNG files found in {inputs_dir}")

    rng = random.Random(7)
    random_subset = rng.sample(all_images, k=min(3, len(all_images)))

    demo_cases = [
        ("first3_square", all_images[: min(3, len(all_images))], 1.0),
        ("even3_landscape", _subset_every_other(all_images, 0)[: min(3, len(all_images))], 16 / 9),
        ("random3_wide", random_subset, 2.0),
    ]
    solver_order = ["optimize", "heuristic", "circpacker", "cell_packing"]

    logger.info("Found %d source images in %s", len(all_images), inputs_dir)
    for case_name, subset, aspect_ratio in demo_cases:
        if not subset:
            logger.warning("Skipping empty subset for demo case %s", case_name)
            continue
        logger.info("Running case=%s with %d images (aspect=%.3f)", case_name, len(subset), aspect_ratio)
        for solver in solver_order:
            solver_subset = subset
            # Heuristic is intentionally exhaustive and can be very slow on full-res sets.
            # Keep it to the smallest image so each solver demo remains around the 60-second budget.
            if solver == "heuristic":
                solver_subset = _pick_smallest_image(subset)
            options = _options_for_solver(solver, aspect_ratio)
            t0 = time.perf_counter()
            result = pack_images(solver_subset, options=options)
            elapsed = time.perf_counter() - t0
            out_path = outputs_dir / f"{case_name}__{solver}.png"
            result.image.save(out_path)
            status = "OK" if elapsed <= 60.0 else "SLOW"
            logger.info(
                "case=%s solver=%s runtime=%.1fs [%s] n=%d canvas=%s fill=%.3f -> %s",
                case_name,
                solver,
                elapsed,
                status,
                len(solver_subset),
                result.canvas_size,
                result.fill_ratio,
                out_path,
            )


if __name__ == "__main__":
    main()
