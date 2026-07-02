"""Demo: load sample inputs, arrange on a grid, and render."""

from __future__ import annotations

from pathlib import Path

from packplot.arrangement import Arrangement
from packplot.input import entries_from_files

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "inputs" / "png"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    paths = sorted(INPUT_DIR.glob("TS*_2.png"))[:6]
    if not paths:
        raise SystemExit(f"No PNG inputs found in {INPUT_DIR}")

    entries = entries_from_files(paths)
    arrangement = Arrangement(entries)
    output = OUTPUT_DIR / "grid_demo.png"
    arrangement.render(output)
    print(f"Wrote {output} ({arrangement.canvas_width}x{arrangement.canvas_height})")


if __name__ == "__main__":
    main()
