from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from packplot.arrangement import Arrangement, ArrangementKeyFunc, ArrangementKeyMode, apply_arrangement, load_arrangement
from packplot.external_packers import pack_with_cell_packing, pack_with_circpacker
from packplot.extract import extract_source_objects
from packplot.optimize import optimize_pack
from packplot.packer import pack_polygons
from packplot.render import render_composition
from packplot.types import PackOptions, PackResult

logger = logging.getLogger(__name__)


def pack_images(
    image_paths: Iterable[str | Path],
    options: PackOptions | None = None,
    *,
    target_aspect_ratio: float | None = None,
    arrangement: Arrangement | str | Path | None = None,
    arrangement_key_mode: ArrangementKeyMode = "stem",
    arrangement_key_func: ArrangementKeyFunc | None = None,
    strict_arrangement: bool = True,
) -> PackResult:
    """Pack input images into a single composed figure.

    Args:
        image_paths: Paths to PNG/JPEG files, one primary object per file.
        options: Optional `PackOptions`; defaults are used when omitted.
        target_aspect_ratio: Optional override for `options.target_aspect_ratio`.
        arrangement: Optional arrangement object or JSON path to replay layout.
        arrangement_key_mode: Key matching mode for arrangement replay.
        arrangement_key_func: Optional callable that maps a file path to an ID key.
        strict_arrangement: If True, missing/extra keys raise errors.

    Returns:
        `PackResult` containing the composed image, placement metadata, and
        final canvas statistics.
    """

    paths = [Path(path) for path in image_paths]
    logger.info("pack_images called with %d image paths.", len(paths))
    if options is None:
        options = PackOptions()
    if target_aspect_ratio is not None:
        options = replace(options, target_aspect_ratio=target_aspect_ratio)
    logger.debug("Effective pack options: %s", options)

    source_objects = extract_source_objects(paths, options)
    logger.info("Extracted %d source objects; beginning packing.", len(source_objects))
    background_color = source_objects[0].background_color
    for source in source_objects[1:]:
        if source.background_color != background_color:
            logger.warning(
                "Input backgrounds differ (%s vs %s); using first image background.",
                background_color,
                source.background_color,
            )
            break
    arrangement_obj: Arrangement | None = None
    if arrangement is not None:
        arrangement_obj = load_arrangement(arrangement) if isinstance(arrangement, (str, Path)) else arrangement
        placements, canvas_size, background_color = apply_arrangement(
            source_objects,
            arrangement_obj,
            key_mode=arrangement_key_mode,
            key_func=arrangement_key_func,
            strict=strict_arrangement,
        )
    elif options.solver == "heuristic":
        placements, canvas_size = pack_polygons(source_objects, options)
    elif options.solver == "optimize":
        placements, canvas_size = optimize_pack(source_objects, options)
    elif options.solver == "circpacker":
        placements, canvas_size = pack_with_circpacker(source_objects, options)
    elif options.solver == "cell_packing":
        placements, canvas_size = pack_with_cell_packing(source_objects, options)
    else:
        raise ValueError("Unknown solver. Expected 'heuristic', 'optimize', 'circpacker', or 'cell_packing'.")
    image = render_composition(canvas_size, placements, background_color=background_color)
    fill_ratio = sum(item.polygon.area for item in placements) / float(canvas_size[0] * canvas_size[1])
    logger.info(
        "Packing complete: placements=%d canvas=%s fill_ratio=%.3f",
        len(placements),
        canvas_size,
        fill_ratio,
    )
    return PackResult(
        image=image,
        placements=placements,
        canvas_size=canvas_size,
        target_aspect_ratio=options.target_aspect_ratio,
        fill_ratio=fill_ratio,
        background_color=background_color,
    )
