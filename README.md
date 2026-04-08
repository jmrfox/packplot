packplot
========

`packplot` packs arbitrary object images into a single rectangular figure by:
- extracting one foreground object from each image (alpha-first, white-fallback),
- approximating each object with a convex hull,
- packing hulls with translation/rotation into a target aspect ratio (heuristic or optimizer),
- compositing transformed objects into one RGBA output image.

Two solver modes are available:
- `solver="heuristic"`: fast, grid-style search with adaptive canvas growth.
- `solver="optimize"`: Jacobi-coordinate optimization of relative centers and rotation fractions in `[0, 1]`, minimizing bounding box area plus overlap/aspect penalties.

The optimizer runs in a normalized coordinate system (bounded Jacobi variables and unit-scale geometry), then maps the optimal coordinates back to full image resolution for final rendering.
After that, a second fixed-canvas spread phase can improve visual spacing by maximizing a smooth approximation of the minimum shape-shape and shape-edge clearance.

You can switch optimization engines:
- `optimizer_method`: `"lbfgsb"`, `"de"` (differential evolution), or `"hybrid"`.
- `spread_method`: same choices for the spread phase.
- `optimizer_workers` / `spread_workers` enable parallel evaluation for `"de"` (set > 1).

Defaults are now tuned for public-quality robustness:
- `optimizer_method="lbfgsb"` with multi-start and alternating center/rotation refinement
- `spread_method="lbfgsb"`
- `optimizer_workers=1` and `spread_workers=1` by default (set higher workers if using `"de"`)

## Install

```bash
uv sync
uv pip install -e .
```

## Quick usage

```python
from pathlib import Path

from packplot import PackOptions, create_arrangement, pack_images, save_arrangement

paths = [
    Path("inputs/object_a.png"),
    Path("inputs/object_b.jpg"),
    Path("inputs/object_c.png"),
]

result = pack_images(
    paths,
    options=PackOptions(
        solver="optimize",
        target_aspect_ratio=16 / 9,
        padding=4,
        edge_buffer=2.0,
        jacobi_inflation=1.2,
        white_threshold=245,
    ),
)

result.image.save("packed.png")
print(result.canvas_size, result.fill_ratio)

# Save this arrangement and replay it on another modality later.
arrangement = create_arrangement(result)
save_arrangement(arrangement, "mesh_arrangement.json")
```

## Replay a saved arrangement

```python
from packplot import PackOptions, load_arrangement, pack_images

arr = load_arrangement("mesh_arrangement.json")
result = pack_images(
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
result.image.save("skeletons_aligned.png")
```

For modality-specific suffixes (`_mesh`, `_skeleton`, `_cable`), pass a key function:

```python
key_fn = lambda p: p.stem.replace("_mesh", "").replace("_skeleton", "").replace("_cable", "")

arr = load_arrangement("mesh_arrangement.json")
result = pack_images(
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
- `jacobi_inflation` (>1.0 spreads shapes apart, <1.0 pulls them closer) scales relative Jacobi coordinates.

## Logging

Library modules emit standard Python logging (`debug`, `info`, `warning`, `error`).
For demo runs, set:

```bash
PACKPLOT_LOG_LEVEL=DEBUG uv run python demos/generate_demos.py
```