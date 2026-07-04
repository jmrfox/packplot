"""Rectangle packing via rectpack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rectpack import PackingBin, PackingMode, SORT_LSIDE, newPacker
from rectpack.enclose import Enclose
from rectpack.skyline import SkylineBlWm

from packplot.input import Entry

AspectRatio = Literal["4:3", "3:2", "16:9", "1:1", "3:4", "2:3"]

ASPECT_RATIOS: dict[AspectRatio, float] = {
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "16:9": 16 / 9,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "2:3": 2 / 3,
}


@dataclass(frozen=True)
class PackedEntry:
    """One entry positioned on the canvas after packing."""

    entry: Entry
    x: int
    y: int
    rotated: bool


def pack_entries(
    entries: list[Entry],
    *,
    rotation: bool = True,
    aspect_ratio: AspectRatio = "4:3",
) -> tuple[int, int, list[PackedEntry]]:
    """Pack *entries* into a canvas, favoring *aspect_ratio* when possible."""
    if not entries:
        return 0, 0, []

    sizes = [(entry.width, entry.height) for entry in entries]
    canvas_width, canvas_height = _select_canvas_size(
        sizes,
        rotation=rotation,
        aspect_ratio=aspect_ratio,
    )

    repacker = newPacker(
        PackingMode.Offline,
        PackingBin.BFF,
        pack_algo=SkylineBlWm,
        sort_algo=SORT_LSIDE,
        rotation=rotation,
    )
    repacker.add_bin(canvas_width, canvas_height)
    for index, entry in enumerate(entries):
        repacker.add_rect(entry.width, entry.height, rid=index)
    repacker.pack()

    if len(repacker[0]) != len(entries):
        raise ValueError("could not pack entries")

    placements: list[PackedEntry] = []
    for rect in repacker[0]:
        entry = entries[rect.rid]
        placements.append(
            PackedEntry(
                entry=entry,
                x=int(rect.x),
                y=canvas_height - int(rect.top),
                rotated=(rect.width, rect.height) != (entry.width, entry.height),
            )
        )
    return canvas_width, canvas_height, placements


def _select_canvas_size(
    sizes: list[tuple[int, int]],
    *,
    rotation: bool,
    aspect_ratio: AspectRatio,
) -> tuple[int, int]:
    target_ratio = ASPECT_RATIOS[aspect_ratio]
    enclose = Enclose(sizes, rotation=rotation)
    candidates = enclose._container_candidates()
    if not candidates:
        raise ValueError("could not pack entries")

    containers = []
    for candidate in candidates:
        container = enclose._refine_candidate(*candidate)
        if container:
            containers.append(container)
    if not containers:
        raise ValueError("could not pack entries")

    width, height, _ = min(
        containers,
        key=lambda container: (
            _aspect_ratio_error(container[0], container[1], target_ratio),
            container[0] * container[1],
        ),
    )
    return int(width), int(height)


def _aspect_ratio_error(width: int | float, height: int | float, target: float) -> float:
    return abs(width / height - target)
