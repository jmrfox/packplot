packplot
========

`packplot` packs arbitrary object images into a single rectangular figure by:
- extracting one foreground object from each image (alpha-first, white-fallback),
- approximating each object with a convex hull,
- packing hulls with translation/rotation into a target aspect ratio,
- compositing transformed objects into one RGBA output image.

## Pipeline

```
extract -> problem -> initialization -> pack -> refine -> render
```

| Stage            | Module                 | Description |
|------------------|------------------------|-------------|
| **Extract**      | `extract.py`, `source_loaders.py` | Load images, extract foreground mask and convex hull per object. |
| **Problem**      | `problem.py`           | Build the normalized packing problem and Jacobi coordinate mapping. |
| **Initialization** | `initialization.py`  | Generate starting-point layouts (grid, randomized grid) for solvers. |
| **Pack phase**   | `pack_phase.py`    | Minimize bounding-box area subject to overlap, aspect-ratio, and regularization constraints. |
| **Refine phase** | `refine_phase.py`      | Fixed-canvas optimization: maximize shape-to-shape and shape-to-edge clearance. |
| **Render**       | `render.py`            | Compose final image from placements; sanity-check overlap/bounds. |

Supporting modules: `optimizer.py` (algorithm dispatch), `layout_metrics.py` (geometric metrics),
`types.py` (data classes), `arrangement.py` (save/replay layouts), `output_backends.py` (optional figure backends).

`pack_images(...)` returns a best-first list of `PackResult` solutions.

## Optimizer selection

Each phase has a **Solver** slot configured with an **Optimizer** string:

- **Pack phase** (`pipeline_config.pack_phase.optimizer`):
  `"scipy-lbfgsb"`, `"scipy-de"`, `"scipy-hybrid"`, `"scipy-nsga2"`, or `"pymoo-nsga2"`.
- **Refine phase** (`pipeline_config.refine_phase.optimizer`):
  same choices; refine always runs a scalar objective internally.

The `pymoo-nsga2` pack variant uses multi-objective NSGA-II with vector objectives
(`area`, `aspect_error`, `-min_pair_clearance`) and returns the best N layouts
(configured via `PackOptions.pymoo_config.best_layout_count`).

All other pack variants use a single scalarized objective.

Supported source formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp`, `.svg`.
SVG loading requires the optional `svg` extra (`cairosvg`).

## Install

```bash
uv sync
uv pip install -e .
```

Optional extras:

```bash
uv add "packplot[svg]"           # SVG support
uv add "packplot[matplotlib]"    # matplotlib backend
uv add "packplot[plotly]"        # plotly backend
uv add "packplot[all]"           # everything
```

## Quick usage

```python
from pathlib import Path
from packplot import (
    DifferentialEvolutionConfig,
    LbfgsbConfig,
    PipelineConfig,
    SolverConfig,
    PackOptions,
    pack_images,
)

results = pack_images(
    [Path("inputs/a.png"), Path("inputs/b.png"), Path("inputs/c.png")],
    options=PackOptions(
        target_aspect_ratio=16 / 9,
        padding=4,
        edge_buffer=2.0,
        pipeline_config=PipelineConfig(
            pack_phase=SolverConfig(
                optimizer="scipy-lbfgsb",
                lbfgsb=LbfgsbConfig(max_iterations=800, random_restart_count=12),
            ),
            refine_phase=SolverConfig(
                optimizer="scipy-lbfgsb",
            ),
        ),
    ),
)

best = results[0]
best.image.save("packed.png")
print(best.canvas_size, best.fill_ratio, best.solver_method)
```

## Replay a saved arrangement

```python
from packplot import PackOptions, load_arrangement, pack_images

arr = load_arrangement("mesh_arrangement.json")
results = pack_images(
    ["inputs/cellA_skeleton.png", "inputs/cellB_skeleton.png"],
    arrangement=arr,
    arrangement_key_mode="stem",
)
results[0].image.save("skeletons_aligned.png")
```

For modality-specific suffixes (`_mesh`, `_skeleton`), pass a key function:

```python
key_fn = lambda p: p.stem.replace("_mesh", "").replace("_skeleton", "")
results = pack_images(paths, arrangement=arr, arrangement_key_func=key_fn)
```

## Assumptions

- One primary object per input image.
- Foreground masking: alpha-first; fallback to near-white thresholding.
- Convex hull proxies are used for packing (fine concavities ignored).
- `edge_buffer` inflates each hull during optimization to enforce spacing.
- `jacobi_inflation` scales relative Jacobi coordinates (>1 spreads, <1 pulls together).

## Logging

```bash
PACKPLOT_LOG_LEVEL=DEBUG uv run python demos/generate_demos.py
```

## Optional output backends

```python
from packplot import render_result_with_backend, list_output_backends

list_output_backends()  # ['pil', 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'pyvista', 'pygal']
fig = render_result_with_backend(result, backend="plotly", title="Packed layout")
```
