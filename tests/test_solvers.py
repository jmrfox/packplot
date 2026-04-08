from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot import PackOptions, pack_images


def test_pymoo_solver_is_stubbed_with_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "obj.png"
    image = Image.new("RGBA", (24, 24), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 6, 18, 17), fill=(20, 20, 20, 255))
    image.save(path)

    try:
        pack_images([path], options=PackOptions(solver="pymoo"))
        assert False, "Expected pymoo solver to raise until implemented."
    except NotImplementedError as exc:
        msg = str(exc).lower()
        assert "stub" in msg and "not implemented" in msg
