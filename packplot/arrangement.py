from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from shapely import affinity

from packplot.geometry import Orientation, apply_orientation
from packplot.types import PackResult, PackedPlacement, SourceObject

logger = logging.getLogger(__name__)

ArrangementKeyMode = Literal["stem", "name"]
ArrangementKeyFunc = Callable[[Path], str]


@dataclass(frozen=True)
class ArrangementEntry:
    """Relative placement and orientation for one object identity."""

    key: str
    center_norm: tuple[float, float]
    angle_degrees: float
    flipped: bool


@dataclass(frozen=True)
class Arrangement:
    """Reusable arrangement template for consistent multi-figure layouts."""

    version: int
    canvas_size: tuple[int, int]
    background_color: tuple[int, int, int]
    entries: list[ArrangementEntry]


def _key_for_path(path: Path, key_mode: ArrangementKeyMode, key_func: ArrangementKeyFunc | None = None) -> str:
    if key_func is not None:
        key = key_func(path)
        if not key:
            raise ValueError("Arrangement key function returned an empty key.")
        return key
    if key_mode == "stem":
        return path.stem
    if key_mode == "name":
        return path.name
    raise ValueError("Unknown arrangement key mode. Expected 'stem' or 'name'.")


def _raise_on_duplicate_keys(keys: list[str], *, context: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        dupes = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate arrangement keys detected in {context}: {dupes}")


def create_arrangement(
    result: PackResult | list[PackResult],
    *,
    key_mode: ArrangementKeyMode = "stem",
    key_func: ArrangementKeyFunc | None = None,
) -> Arrangement:
    """Create a reusable arrangement from a prior `pack_images` result."""
    if isinstance(result, list):
        if not result:
            raise ValueError("Cannot create arrangement from an empty results list.")
        result = result[0]
    width, height = result.canvas_size
    entries: list[ArrangementEntry] = []
    keys: list[str] = []
    for placement in result.placements:
        cx, cy = placement.polygon.centroid.x, placement.polygon.centroid.y
        key = _key_for_path(placement.source_path, key_mode, key_func)
        keys.append(key)
        entries.append(
            ArrangementEntry(
                key=key,
                center_norm=(float(cx / width), float(cy / height)),
                angle_degrees=float(placement.angle_degrees),
                flipped=bool(placement.flipped),
            )
        )
    _raise_on_duplicate_keys(keys, context="create_arrangement inputs")
    return Arrangement(
        version=1,
        canvas_size=result.canvas_size,
        background_color=result.background_color,
        entries=entries,
    )


def save_arrangement(arrangement: Arrangement, path: str | Path) -> Path:
    """Serialize an arrangement to a JSON file."""
    out_path = Path(path)
    payload = {
        "version": arrangement.version,
        "canvas_size": list(arrangement.canvas_size),
        "background_color": list(arrangement.background_color),
        "entries": [
            {
                "key": entry.key,
                "center_norm": [entry.center_norm[0], entry.center_norm[1]],
                "angle_degrees": entry.angle_degrees,
                "flipped": entry.flipped,
            }
            for entry in arrangement.entries
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_arrangement(path: str | Path) -> Arrangement:
    """Load an arrangement JSON file produced by `save_arrangement`."""
    in_path = Path(path)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    entries = [
        ArrangementEntry(
            key=str(item["key"]),
            center_norm=(float(item["center_norm"][0]), float(item["center_norm"][1])),
            angle_degrees=float(item["angle_degrees"]),
            flipped=bool(item["flipped"]),
        )
        for item in payload["entries"]
    ]
    return Arrangement(
        version=int(payload["version"]),
        canvas_size=(int(payload["canvas_size"][0]), int(payload["canvas_size"][1])),
        background_color=(
            int(payload["background_color"][0]),
            int(payload["background_color"][1]),
            int(payload["background_color"][2]),
        ),
        entries=entries,
    )


def apply_arrangement(
    source_objects: list[SourceObject],
    arrangement: Arrangement,
    *,
    key_mode: ArrangementKeyMode = "stem",
    key_func: ArrangementKeyFunc | None = None,
    strict: bool = True,
) -> tuple[list[PackedPlacement], tuple[int, int], tuple[int, int, int]]:
    """Apply a saved arrangement to a new source set."""
    source_keys = [_key_for_path(source.source_path, key_mode, key_func) for source in source_objects]
    _raise_on_duplicate_keys(source_keys, context="current input sources")
    arrangement_keys = [entry.key for entry in arrangement.entries]
    _raise_on_duplicate_keys(arrangement_keys, context="arrangement entries")
    source_by_key = {key: source for key, source in zip(source_keys, source_objects)}
    width, height = arrangement.canvas_size

    placements: list[PackedPlacement] = []
    used_keys: set[str] = set()
    for entry in arrangement.entries:
        source = source_by_key.get(entry.key)
        if source is None:
            if strict:
                raise ValueError(f"Arrangement key '{entry.key}' missing in current inputs.")
            logger.warning("Skipping missing arrangement key '%s'.", entry.key)
            continue

        oriented = apply_orientation(
            source.cropped_image,
            source.hull,
            Orientation(angle_degrees=entry.angle_degrees, flipped=entry.flipped),
        )
        target_cx = entry.center_norm[0] * width
        target_cy = entry.center_norm[1] * height
        poly_cx, poly_cy = oriented.polygon.centroid.x, oriented.polygon.centroid.y
        x = int(round(target_cx - poly_cx))
        y = int(round(target_cy - poly_cy))

        if x < 0 or y < 0 or x + oriented.image.width > width or y + oriented.image.height > height:
            if strict:
                raise ValueError(
                    f"Placement for key '{entry.key}' falls outside canvas {arrangement.canvas_size}."
                )
            x = min(max(0, x), max(0, width - oriented.image.width))
            y = min(max(0, y), max(0, height - oriented.image.height))
            logger.warning("Clamped out-of-bounds placement for key '%s'.", entry.key)

        polygon = affinity.translate(oriented.polygon, xoff=x, yoff=y)
        placements.append(
            PackedPlacement(
                source_path=source.source_path,
                polygon=polygon,
                angle_degrees=entry.angle_degrees,
                flipped=entry.flipped,
                top_left=(x, y),
                image=oriented.image,
            )
        )
        used_keys.add(entry.key)

    if strict:
        extra = sorted(set(source_by_key) - used_keys)
        if extra:
            raise ValueError(f"Current inputs contain keys not in arrangement: {extra}")
    return placements, arrangement.canvas_size, arrangement.background_color
