from __future__ import annotations

import logging
import os
import random
from pathlib import Path

from packplot import PackOptions, pack_images

logger = logging.getLogger(__name__)


def _subset_every_other(items: list[Path], start: int) -> list[Path]:
    return [item for idx, item in enumerate(items) if idx % 2 == start]


def main() -> None:
    level_name = os.getenv("PACKPLOT_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )
    workers = int(os.getenv("PACKPLOT_DE_WORKERS", "1"))
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
    random_subset = rng.sample(all_images, k=min(4, len(all_images)))

    demo_cases = [
        ("demo_first4_square_fullres", all_images[: min(4, len(all_images))], 1.0),
        ("demo_mid4_landscape_fullres", all_images[2 : 2 + min(4, len(all_images) - 2)], 16 / 9),
        ("demo_even4_tall_fullres", _subset_every_other(all_images, 0)[: min(4, len(all_images))], 3 / 4),
        ("demo_random4_wide_fullres", random_subset, 2.2),
    ]

    logger.info("Found %d source images in %s", len(all_images), inputs_dir)
    for demo_name, subset, aspect_ratio in demo_cases:
        if not subset:
            logger.warning("Skipping empty subset for demo case %s", demo_name)
            continue
        logger.info("Running %s with %d images (aspect=%.3f)", demo_name, len(subset), aspect_ratio)
        result = pack_images(
            subset,
            options=PackOptions(
                target_aspect_ratio=aspect_ratio,
                white_threshold=250,
                optimizer_maxiter=1200,
                optimizer_method="lbfgsb",
                optimizer_workers=workers,
                spread_method="lbfgsb",
                spread_workers=workers,
            ),
        )
        out_path = outputs_dir / f"{demo_name}.png"
        result.image.save(out_path)
        logger.info(
            "%s complete: n=%d canvas=%s fill=%.3f -> %s",
            demo_name,
            len(subset),
            result.canvas_size,
            result.fill_ratio,
            out_path,
        )


if __name__ == "__main__":
    main()
