"""Input image entries and loading."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class Entry:
    """One input image and its associated metadata."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._image: Image.Image | None = None

    @property
    def name(self) -> str:
        """Basename of the source file without extension."""
        return self.path.stem

    @property
    def image(self) -> Image.Image | None:
        """Cached PIL image, or ``None`` if not yet loaded."""
        return self._image

    def load(self) -> Image.Image:
        """Load the image from ``self.path`` and cache it."""
        if self._image is None:
            self._image = Image.open(self.path)
        return self._image

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.load().width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.load().height


def entries_from_files(paths: list[Path | str]) -> list[Entry]:
    """Build one :class:`Entry` per file path."""
    return [Entry(path) for path in paths]
