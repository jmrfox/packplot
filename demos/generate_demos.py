"""Demo: load sample inputs and run the packplot pipeline."""

from __future__ import annotations

from pathlib import Path

from packplot.packplot import packplot

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

N_ENTRIES = 9


def main() -> None:
    paths = sorted(INPUT_DIR.glob("TS*_2.png"))[:N_ENTRIES]
    if not paths:
        raise SystemExit(f"No PNG inputs found in {INPUT_DIR}")

    output = OUTPUT_DIR / "packed_demo.png"
    canvas = packplot(
        paths,
        output_path=output,
        crop=True,
        background=(255, 255, 255),
        threshold=0,
        margin_width=10,
        rotation=True,
        aspect_ratio="4:3",
    )
    print(f"Wrote {output} ({canvas.width}x{canvas.height})")

if __name__ == "__main__":
    main()