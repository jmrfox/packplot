"""End-to-end packplot pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from packplot.arrangement import Arrangement
from packplot.input import EntrySet
from packplot.packing import AspectRatio


def packplot(
    paths: list[Path | str],
    *,
    output_path: Path | str | None = None,
    crop: bool = False,
    background: tuple[int, ...] = (255, 255, 255),
    threshold: int = 0,
    margin_width: int = 0,
    rotation: bool = True,
    aspect_ratio: AspectRatio = "4:3",
) -> Image.Image:
    """Load *paths*, optionally crop each entry, pack into a canvas, and render."""
    entry_set = EntrySet(
        paths,
        crop=crop,
        background=background,
        threshold=threshold,
        margin_width=margin_width,
    )
    canvas = Arrangement(
        entry_set.entries,
        layout="pack",
        rotation=rotation,
        aspect_ratio=aspect_ratio,
    ).render_image()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)

    return canvas
