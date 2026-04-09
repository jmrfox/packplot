from __future__ import annotations

from typing import Any

import numpy as np

from packplot.types import PackResult

SUPPORTED_OUTPUT_BACKENDS = ("pil", "matplotlib", "seaborn", "plotly", "bokeh", "pyvista", "pygal")


def list_output_backends() -> tuple[str, ...]:
    """Return all output backend names supported by packplot."""
    return SUPPORTED_OUTPUT_BACKENDS


def _optional_import(module_name: str, *, extra_name: str):
    try:
        module = __import__(module_name, fromlist=["*"])
    except Exception as exc:
        raise RuntimeError(
            f"Output backend '{extra_name}' requires optional dependency '{module_name}'. "
            f"Install with `pip install packplot[{extra_name}]`."
        ) from exc
    return module


def _as_rgba_array(result: PackResult) -> np.ndarray:
    return np.asarray(result.image.convert("RGBA"), dtype=np.uint8)


def render_result_with_backend(
    result: PackResult,
    *,
    backend: str = "pil",
    title: str | None = None,
) -> Any:
    """Convert a `PackResult` image into a backend-native figure object."""
    key = backend.strip().lower()
    if key == "pil":
        return result.image.copy()

    if key == "matplotlib":
        plt = _optional_import("matplotlib.pyplot", extra_name="matplotlib")
        fig, ax = plt.subplots()
        ax.imshow(_as_rgba_array(result))
        ax.set_axis_off()
        if title:
            ax.set_title(title)
        return fig

    if key == "seaborn":
        plt = _optional_import("matplotlib.pyplot", extra_name="seaborn")
        sns = _optional_import("seaborn", extra_name="seaborn")
        fig, ax = plt.subplots()
        sns.set_theme()
        ax.imshow(_as_rgba_array(result))
        ax.set_axis_off()
        if title:
            ax.set_title(title)
        return fig

    if key == "plotly":
        go = _optional_import("plotly.graph_objects", extra_name="plotly")
        img = _as_rgba_array(result)
        fig = go.Figure(data=[go.Image(z=img)])
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        if title:
            fig.update_layout(title=title)
        return fig

    if key == "bokeh":
        bokeh_plotting = _optional_import("bokeh.plotting", extra_name="bokeh")
        img = _as_rgba_array(result)
        h, w, _ = img.shape
        # Bokeh expects packed RGBA uint32 with origin at bottom-left.
        rgba = np.flipud(img).view(dtype=np.uint32).reshape((h, w))
        fig = bokeh_plotting.figure(
            x_range=(0, w),
            y_range=(0, h),
            width=max(250, w),
            height=max(250, h),
            toolbar_location=None,
            title=title,
        )
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.grid.visible = False
        fig.image_rgba(image=[rgba], x=0, y=0, dw=w, dh=h)
        return fig

    if key == "pyvista":
        pv = _optional_import("pyvista", extra_name="pyvista")
        rgb = np.asarray(result.image.convert("RGB"), dtype=np.uint8)
        h, w, _ = rgb.shape
        texture = pv.numpy_to_texture(np.flipud(rgb))
        plane = pv.Plane(center=(w / 2.0, h / 2.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=w, j_size=h)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(plane, texture=texture)
        plotter.view_xy()
        if title:
            plotter.add_text(title, position="upper_left", font_size=12)
        return plotter

    if key == "pygal":
        pygal = _optional_import("pygal", extra_name="pygal")
        width, height = result.canvas_size
        chart = pygal.XY(
            stroke=True,
            show_legend=True,
            title=title or "packplot layout",
            x_title="x",
            y_title="y",
            width=max(200, width),
            height=max(200, height),
        )
        for placement in result.placements:
            coords = [(float(x), float(y)) for x, y in placement.polygon.exterior.coords]
            chart.add(placement.source_path.stem, coords)
        return chart

    raise ValueError(
        f"Unknown output backend '{backend}'. Supported backends: {', '.join(SUPPORTED_OUTPUT_BACKENDS)}."
    )

