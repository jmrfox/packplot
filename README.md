packplot
========

`packplot` packs arbitrary object images into a single rectangular figure by:
- extracting one foreground object from each image (alpha-first, white-fallback),
- approximating each object with a convex hull,
- packing hulls with translation/rotation into a target aspect ratio (optimizer backends),
- compositing transformed objects into one RGBA output image.

`pack_images(...)` returns a best-first list of `PackResult` solutions. Most solvers return one solution; `pymoo` can return multiple best solutions from one run.

Two compact-layout backends are available:
- `optimize_config.compact_layout_backend="optimize"`: Jacobi-coordinate optimization of relative centers and rotation fractions in `[0, 1]`, minimizing bounding box area plus overlap/aspect penalties.
- `optimize_config.compact_layout_backend="pymoo"`: multi-objective NSGA-II using vector-valued objectives (`area`, `aspect_error`, `-min_pair_clearance`) with overlap feasibility constraints.
  - Returns the best `N` ranked solutions in the overall `pack_images` return list, configured by `PackOptions.pymoo_config.best_layout_count`.

Supported source formats currently include common raster images
(`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tif`, `.tiff`, `.webp`) and `.svg` inputs.
SVG loading uses a dedicated loader path and requires the optional `svg` extra (`cairosvg`).

The optimizer runs in a normalized coordinate system (bounded Jacobi variables and unit-scale geometry), then maps the optimal coordinates back to full image resolution for final rendering.
After that, a second fixed-canvas clearance-refinement phase can improve visual spacing by maximizing a smooth approximation of the minimum shape-shape and shape-edge clearance.

All compact-layout backends (`optimize`, `pymoo`) solve on normalized geometry and map selected arrangements back to full-resolution rendering coordinates.

You can switch optimization engines by phase configuration:
- `optimize_config.compact_layout.method`: `"lbfgsb"`, `"de"` (differential evolution), `"nsga2"`, or `"hybrid"`.
- `optimize_config.clearance_refinement.method`: same choices for the clearance-refinement phase.
- `worker_count` is configured per phase under `differential_evolution`.

Defaults are now tuned for public-quality robustness:
- compact-layout method defaults to `"lbfgsb"` with multi-start and alternating center/rotation refinement
- clearance-refinement method defaults to `"lbfgsb"`
- per-phase differential-evolution `worker_count` defaults to `1` (set higher only when using `"de"`)

`pymoo` defaults (`PackOptions.pymoo_config`) are conservative for quality. For faster exploratory runs, lower `generations` and `population_size`.
Set `best_layout_count` to get multiple renders from one run (for example, the best 3).
Use `initialization_config` to control compact-layout starting points (`grid` or `randomized_grid`).
Use `optimize_config.compact_layout_best_count` to keep multiple compact candidates and
`optimize_config.compact_to_clearance_beam_width` to control how many are refined in phase 2.

### Why NSGA-II first?

`pymoo` has many algorithms. For this project, NSGA-II is the best starting point because:
- it is the most established baseline for 2-3 objective problems,
- it works well with explicit constraints (we enforce overlap feasibility),
- it gives a Pareto set directly, so we avoid collapsing objectives into one weighted scalar too early.

If we later expand to many objectives (for example, >=4 competing layout goals), NSGA-III or MOEA/D are likely next candidates.

## Install

```bash
uv sync
uv pip install -e .
```

Optional extras:

```bash
# SVG support
uv add "packplot[svg]"

# Output backend families
uv add "packplot[matplotlib]"
uv add "packplot[seaborn]"
uv add "packplot[plotly]"
uv add "packplot[bokeh]"
uv add "packplot[pyvista]"
uv add "packplot[pygal]"

# Everything
uv add "packplot[all]"
```

## Quick usage

```python
from pathlib import Path

from packplot import (
    ClearanceRefinementObjectiveConfig,
    CompactLayoutObjectiveConfig,
    DifferentialEvolutionConfig,
    LbfgsbConfig,
    OptimizeConfig,
    OptimizationPhaseConfig,
    PackOptions,
    create_arrangement,
    pack_images,
    render_result_with_backend,
    save_arrangement,
)

paths = [
    Path("inputs/object_a.png"),
    Path("inputs/object_b.jpg"),
    Path("inputs/object_c.png"),
]

results = pack_images(
    paths,
    options=PackOptions(
        target_aspect_ratio=16 / 9,
        padding=4,
        edge_buffer=2.0,
        jacobi_inflation=1.2,
        white_threshold=245,
        optimize_config=OptimizeConfig(
            compact_layout_backend="optimize",
            compact_layout=OptimizationPhaseConfig(
                method="lbfgsb",
                lbfgsb=LbfgsbConfig(
                    max_iterations=800,
                    random_restart_count=12,
                    alternating_refinement_cycles=4,
                ),
            ),
            clearance_refinement=OptimizationPhaseConfig(
                method="de",
                differential_evolution=DifferentialEvolutionConfig(
                    max_generations=35,
                    population_size=12,
                    worker_count=8,
                ),
            ),
            compact_layout_objective=CompactLayoutObjectiveConfig(),
            clearance_refinement_objective=ClearanceRefinementObjectiveConfig(
                softmin_smoothness=20.0,
                lower_quartile_spacing_weight=0.35,
                mean_spacing_weight=0.1,
            ),
        ),
    ),
)

best = results[0]
best.image.save("packed.png")
plotly_fig = render_result_with_backend(best, backend="plotly", title="Packed layout")
print(best.canvas_size, best.fill_ratio)
print(best.minimum_clearance, best.outside_violation_magnitude)
print(best.solver_method, best.solver_iterations, best.solver_success)

# Save this arrangement and replay it on another modality later.
arrangement = create_arrangement(best)
save_arrangement(arrangement, "mesh_arrangement.json")
```

## Replay a saved arrangement

```python
from packplot import PackOptions, load_arrangement, pack_images

arr = load_arrangement("mesh_arrangement.json")
results = pack_images(
    [
        "inputs/cellA_skeleton.png",
        "inputs/cellB_skeleton.png",
        "inputs/cellC_skeleton.png",
    ],
    options=PackOptions(),  # extraction settings still apply
    arrangement=arr,        # reuse placement/orientation
    arrangement_key_mode="stem",
    strict_arrangement=True,
)
results[0].image.save("skeletons_aligned.png")
```

For modality-specific suffixes (`_mesh`, `_skeleton`, `_cable`), pass a key function:

```python
key_fn = lambda p: p.stem.replace("_mesh", "").replace("_skeleton", "").replace("_cable", "")

arr = load_arrangement("mesh_arrangement.json")
results = pack_images(
    ["inputs/cellA_skeleton.png", "inputs/cellB_skeleton.png"],
    arrangement=arr,
    arrangement_key_func=key_fn,
)
```

## Current assumptions

- One primary object per input image.
- Foreground masking is alpha-first; images without useful alpha are segmented by near-white thresholding.
- Convex hull proxies are used for packing, so fine concavities are ignored.
- `edge_buffer` expands each shape during optimization/collision checks to enforce spacing.
- `jacobi_inflation` (>1.0 separates shapes more, <1.0 pulls them closer) scales relative Jacobi coordinates.

## Logging

Library modules emit standard Python logging (`debug`, `info`, `warning`, `error`).
For demo runs, set:

```bash
PACKPLOT_LOG_LEVEL=DEBUG uv run python demos/generate_demos.py
```

## Optional output backends

`pack_images` always returns `PackResult` with a PIL image (`PackResult.image`).
You can convert that result into optional backend-native figures with:
- `render_result_with_backend(result, backend="matplotlib" | "seaborn" | "plotly" | "bokeh" | "pyvista" | "pygal" | "pil")`
- `list_output_backends()` to enumerate supported names.
