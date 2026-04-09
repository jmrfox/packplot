# Packplot TODO

All items completed. Current module structure:

```
packplot/
  __init__.py        Public API exports
  types.py           Data classes and configuration
  extract.py         Image loading, foreground mask, convex hull
  source_loaders.py  SourceLoader interface (raster, SVG, mixed)
  geometry.py        Geometry primitives
  problem.py         PackingProblem (normalized hulls, Jacobi mapping)
  initialization.py  Starting-point generation (grid, randomized grid)
  layout_metrics.py  Shared geometric metrics (overlap, clearance, bbox)
  optimizer.py       Algorithm dispatch (L-BFGS-B, DE, NSGA-II)
  pack_phase.py      Pack phase: scipy + pymoo solvers, Jacobi math, ranking
  refine_phase.py    Refine phase: fixed-canvas spacing, quality gate
  pipeline.py        Main orchestration (extract->pack->refine->render)
  render.py          Final image composition
  arrangement.py     Save/load placement arrangements (JSON)
  output_backends.py Optional figure backends (matplotlib, plotly, etc.)
```

Pipeline: `extract -> problem -> initialization -> pack -> refine -> render`
