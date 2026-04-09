from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

from packplot import (
    LbfgsbConfig,
    PipelineConfig,
    SolverConfig,
    PackOptions,
    PymooConfig,
    pack_images,
)

logger = logging.getLogger(__name__)


def _subset_every_other(items: list[Path], start: int) -> list[Path]:
    return [item for idx, item in enumerate(items) if idx % 2 == start]


def _options_for_solver(solver: str, aspect_ratio: float) -> PackOptions:
    common = dict(
        target_aspect_ratio=aspect_ratio,
        white_threshold=250,
        padding=2,
        edge_buffer=1.0,
    )
    if solver == "scipy-lbfgsb":
        return PackOptions(
            **common,
            pipeline_config=PipelineConfig(
                pack_phase=SolverConfig(
                    optimizer="scipy-lbfgsb",
                    progress_log_every_evaluations=2000,
                    lbfgsb=LbfgsbConfig(
                        max_iterations=600,
                        random_restart_count=8,
                        alternating_refinement_cycles=3,
                    ),
                ),
                enable_refine_phase=True,
                refine_phase=SolverConfig(
                    optimizer="scipy-lbfgsb",
                    progress_log_every_evaluations=1200,
                    lbfgsb=LbfgsbConfig(
                        max_iterations=400,
                        random_restart_count=4,
                        alternating_refinement_cycles=2,
                    ),
                ),
            ),
        )
    if solver == "pymoo-nsga2":
        return PackOptions(
            **common,
            pipeline_config=PipelineConfig(
                pack_phase=SolverConfig(optimizer="pymoo-nsga2"),
                enable_refine_phase=True,
                refine_phase=SolverConfig(
                    optimizer="scipy-lbfgsb",
                    lbfgsb=LbfgsbConfig(
                        max_iterations=400,
                        random_restart_count=4,
                        alternating_refinement_cycles=2,
                    ),
                ),
            ),
            pymoo_config=PymooConfig(
                algorithm="nsga2",
                generations=80,
                population_size=60,
                offspring_count=None,
                eliminate_duplicates=True,
                best_layout_count=3,
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
        # ("even3_landscape", _subset_every_other(all_images, 0)[: min(3, len(all_images))], 16 / 9),
        # ("random3_wide", random_subset, 2.0),
    ]
    solver_order = ["scipy-lbfgsb", "pymoo-nsga2"]

    logger.info("Found %d source images in %s", len(all_images), inputs_dir)
    for case_name, subset, aspect_ratio in demo_cases:
        if not subset:
            logger.warning("Skipping empty subset for demo case %s", case_name)
            continue
        logger.info("Running case=%s with %d images (aspect=%.3f)", case_name, len(subset), aspect_ratio)
        for solver in solver_order:
            solver_subset = subset
            options = _options_for_solver(solver, aspect_ratio)
            t0 = time.perf_counter()
            results = pack_images(solver_subset, options=options)
            elapsed = time.perf_counter() - t0
            status = "OK" if elapsed <= 60.0 else "SLOW"
            for result in results:
                suffix = "" if len(results) == 1 else f"__best{result.rank}"
                out_path = outputs_dir / f"{case_name}__{solver}{suffix}.png"
                result.image.save(out_path)
                logger.info(
                    "case=%s solver=%s rank=%d runtime=%.1fs [%s] n=%d canvas=%s fill=%.3f "
                    "sanity=%s overlap=%.6f oob=%d -> %s",
                    case_name,
                    solver,
                    result.rank,
                    elapsed,
                    status,
                    len(solver_subset),
                    result.canvas_size,
                    result.fill_ratio,
                    result.sanity_passed,
                    result.total_overlap_area,
                    result.out_of_bounds_count,
                    out_path,
                )


if __name__ == "__main__":
    main()
