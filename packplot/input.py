"""Input image entries and loading."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from packplot.crop import crop_to_content


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

    def cache_image(self, image: Image.Image) -> None:
        """Use *image* as the loaded image instead of reading from disk."""
        self._image = image

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


class EntrySet:
    """A collection of entries loaded from paths, with optional cropping."""

    def __init__(
        self,
        paths: list[Path | str],
        *,
        crop: bool = False,
        background: tuple[int, ...] = (255, 255, 255),
        threshold: int = 0,
        margin_width: int = 0,
    ) -> None:
        self.entries = entries_from_files(paths)
        if crop:
            self._crop_entries(background, threshold)
        if margin_width > 0:
            self._margin_entries(background, margin_width)

    def _crop_entries(
        self,
        background: tuple[int, ...],
        threshold: int,
    ) -> None:
        for entry in self.entries:
            cropped = crop_to_content(
                entry.load(),
                background=background,
                threshold=threshold,
                write=False,
            )
            entry.cache_image(cropped)

    def _margin_entries(
        self,
        background: tuple[int, ...],
        margin_width: int,
    ) -> None:
        for entry in self.entries:
            entry.cache_image(
                _add_margin(entry.load(), margin_width, background)
            )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _add_margin(
    image: Image.Image,
    margin_width: int,
    background: tuple[int, ...],
) -> Image.Image:
    padded = Image.new(
        image.mode,
        (image.width + 2 * margin_width, image.height + 2 * margin_width),
        _margin_fillcolor(image.mode, background),
    )
    padded.paste(image, (margin_width, margin_width))
    return padded


def _margin_fillcolor(mode: str, background: tuple[int, ...]) -> tuple[int, ...]:
    if mode == "RGBA":
        if len(background) >= 4:
            return background[:4]
        return (*background[:3], 255)
    return background[:3]
