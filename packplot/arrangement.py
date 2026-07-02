"""Canvas layout for input entries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from packplot.input import Entry


@dataclass(frozen=True)
class Placement:
    """One entry positioned on the canvas."""

    entry: Entry
    x: int
    y: int


class Arrangement:
    """Non-overlapping placement of entries on a canvas."""

    def __init__(self, entries: list[Entry]) -> None:
        self.entries = list(entries)
        self.placements: list[Placement] = []
        self.canvas_width = 0
        self.canvas_height = 0
        self._init_grid()

    def _init_grid(self) -> None:
        """Place entries on a square-ish grid without overlap."""
        n = len(self.entries)
        if n == 0:
            return

        grid_size = math.ceil(math.sqrt(n))
        widths = [entry.width for entry in self.entries]
        heights = [entry.height for entry in self.entries]

        col_widths = [0] * grid_size
        row_heights = [0] * grid_size
        for i, (width, height) in enumerate(zip(widths, heights, strict=True)):
            col = i % grid_size
            row = i // grid_size
            col_widths[col] = max(col_widths[col], width)
            row_heights[row] = max(row_heights[row], height)

        x_offsets = [0]
        for width in col_widths[:-1]:
            x_offsets.append(x_offsets[-1] + width)
        y_offsets = [0]
        for height in row_heights[:-1]:
            y_offsets.append(y_offsets[-1] + height)

        self.placements = [
            Placement(
                entry=entry,
                x=x_offsets[i % grid_size],
                y=y_offsets[i // grid_size],
            )
            for i, entry in enumerate(self.entries)
        ]
        self.canvas_width = sum(col_widths)
        self.canvas_height = sum(row_heights)

    def render(self, path: Path | str) -> Path:
        """Composite placements onto a canvas and write ``path``."""
        output = Path(path)
        if self.canvas_width == 0 or self.canvas_height == 0:
            raise ValueError("cannot render an empty arrangement")

        canvas = Image.new(
            "RGBA",
            (self.canvas_width, self.canvas_height),
            (255, 255, 255, 255),
        )
        for placement in self.placements:
            image = placement.entry.load().convert("RGBA")
            canvas.paste(image, (placement.x, placement.y), image)

        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output)
        return output
