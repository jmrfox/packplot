from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

from PIL import Image

from packplot.extract import extract_source_object_from_image, extract_source_objects
from packplot.types import PackOptions, SourceObject

RASTER_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}
SVG_SUFFIX = ".svg"


class SourceLoader(ABC):
    """Interface for loading source shapes from input paths."""

    @abstractmethod
    def load(self, paths: list[Path], options: PackOptions) -> list[SourceObject]:
        """Load input paths into extracted source objects."""


class RasterSourceLoader(SourceLoader):
    """Default raster loader using the existing extraction pipeline."""

    def load(self, paths: list[Path], options: PackOptions) -> list[SourceObject]:
        return extract_source_objects(paths, options)


class SvgSourceLoader(SourceLoader):
    """SVG loader that rasterizes vectors to RGBA before extraction."""

    def load(self, paths: list[Path], options: PackOptions) -> list[SourceObject]:
        if not paths:
            raise ValueError("At least one image path is required.")
        try:
            import cairosvg  # pylint: disable=import-error
        except Exception as exc:
            raise RuntimeError(
                "SVG loading requires 'cairosvg'. Install it to enable SVG inputs."
            ) from exc

        objects: list[SourceObject] = []
        for path in paths:
            png_bytes = cairosvg.svg2png(url=str(path))
            image_rgba = Image.open(BytesIO(png_bytes)).convert("RGBA")
            objects.append(extract_source_object_from_image(path, image_rgba, options))
        return objects


class MixedSourceLoader(SourceLoader):
    """Loader for mixed raster/SVG inputs."""

    def load(self, paths: list[Path], options: PackOptions) -> list[SourceObject]:
        if not paths:
            raise ValueError("At least one image path is required.")
        raster_loader = RasterSourceLoader()
        svg_loader = SvgSourceLoader()

        objects: list[SourceObject] = []
        for path in paths:
            if path.suffix.lower() == ".svg":
                objects.extend(svg_loader.load([path], options))
            else:
                objects.extend(raster_loader.load([path], options))
        return objects


def infer_source_loader_name(paths: list[Path]) -> str:
    """Infer loader kind from input suffixes."""
    validate_source_inputs(paths)
    if not paths:
        raise ValueError("At least one image path is required.")
    suffixes = {path.suffix.lower() for path in paths}
    if suffixes == {SVG_SUFFIX}:
        return "svg"
    if SVG_SUFFIX in suffixes:
        return "mixed"
    return "raster"


def validate_source_inputs(paths: list[Path]) -> None:
    """Validate source file paths and extensions with actionable errors."""
    if not paths:
        raise ValueError("No input paths provided. Pass at least one image path.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Input file(s) not found: {', '.join(missing)}")

    bad_types = [str(path) for path in paths if path.is_dir()]
    if bad_types:
        raise ValueError(f"Expected files but got directories: {', '.join(bad_types)}")

    supported = ", ".join(sorted(RASTER_SUFFIXES | {SVG_SUFFIX}))
    unsupported = [
        str(path)
        for path in paths
        if path.suffix.lower() not in RASTER_SUFFIXES and path.suffix.lower() != SVG_SUFFIX
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported source format for: {', '.join(unsupported)}. "
            f"Supported formats are: {supported}."
        )


def get_source_loader(name: str = "raster") -> SourceLoader:
    """Resolve a source loader by name."""
    if name == "raster":
        return RasterSourceLoader()
    if name == "svg":
        return SvgSourceLoader()
    if name == "mixed":
        return MixedSourceLoader()
    raise ValueError("Unknown source loader. Expected 'raster', 'svg', or 'mixed'.")
