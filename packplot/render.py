from __future__ import annotations

import logging
from PIL import Image

from packplot.types import PackedPlacement

logger = logging.getLogger(__name__)


def render_composition(
    canvas_size: tuple[int, int],
    placements: list[PackedPlacement],
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    logger.info(
        "Rendering composition on %s canvas with %d placements (bg=%s).",
        canvas_size,
        len(placements),
        background_color,
    )
    canvas = Image.new("RGBA", canvas_size, (*background_color, 255))
    for placement in placements:
        logger.debug(
            "Compositing %s at %s angle=%.1f flip=%s",
            placement.source_path.name,
            placement.top_left,
            placement.angle_degrees,
            placement.flipped,
        )
        canvas.alpha_composite(placement.image, placement.top_left)
    return canvas
