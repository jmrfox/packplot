from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot.extract import extract_source_object
from packplot.types import PackOptions


def test_extract_uses_alpha_mask(tmp_path: Path) -> None:
    image = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 10, 28, 30), fill=(20, 20, 20, 255))
    path = tmp_path / "alpha.png"
    image.save(path)

    extracted = extract_source_object(path, PackOptions())
    assert extracted.cropped_image.size == (21, 21)
    assert extracted.mask.any()
    assert extracted.hull.area > 0


def test_extract_falls_back_to_white_threshold(tmp_path: Path) -> None:
    image = Image.new("RGB", (36, 36), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((9, 7, 26, 27), fill=(10, 10, 10))
    path = tmp_path / "white_bg.jpg"
    image.save(path)

    extracted = extract_source_object(path, PackOptions(white_threshold=245))
    assert extracted.cropped_image.size[0] < 36
    assert extracted.cropped_image.size[1] < 36
    assert extracted.hull.area > 0
