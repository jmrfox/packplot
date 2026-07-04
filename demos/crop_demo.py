"""Demo: crop a tilted input image to its foreground content."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from packplot.crop import crop_to_content

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "inputs" / "TS1_tilted.png"
OUTPUT = Path(__file__).resolve().parent / "outputs" / "crop_demo.png"


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    original = Image.open(input_path)
    cropped = crop_to_content(
        input_path,
        background=(255, 255, 255),
        threshold=0,
        output_path=OUTPUT,
    )
    print(
        f"Wrote {OUTPUT} "
        f"({original.width}x{original.height} -> {cropped.width}x{cropped.height})"
    )


if __name__ == "__main__":
    main()
