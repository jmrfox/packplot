# Packplot Improvement TODO

This list is intentionally ordered. Complete items top-to-bottom.

## 1) P0 correctness and reliability

- [x] Add duplicate-key detection for arrangement creation/replay to avoid silent key collisions.
- [x] Add pipeline-level sanity check after placement (overlap + in-bounds) that warns but still renders.
- [x] Expose sanity metrics on `PackResult` so callers can programmatically detect degraded layouts.
- [x] Make optimize solver flip behavior explicit (`allow_flip` currently ignored in optimize path; warn user).

## 2) P1 architecture for future solver methods

- [x] Introduce a `PackingProblem` model that keeps:
  - full-resolution source objects,
  - normalized geometry,
  - normalization scale metadata.
- [x] Split `optimize.py` into focused modules:
  - [x] objective terms (`optimize_objectives.py`),
  - [x] compact-layout optimization (`optimize_compact_layout.py`),
  - [x] clearance-refinement optimization (`optimize_clearance_refinement.py`),
  - [x] post-processing/cleanup (`optimize_clearance_refinement.py`).
- [x] Add a `LayoutSolver` interface and solver registry to reduce branching in `pipeline.py`.
- [x] Keep `pymoo` dependency in place and add a stub `PymooSolver` integration point (no default activation yet).

## 3) P2 source loading and format extensibility

- [x] Add a source-loader abstraction (`RasterSourceLoader`) and route current extraction through it.
- [x] Add an SVG loader path behind the same loader abstraction (future feature).
- [x] Add clear source format validation/errors with actionable messages.

## 4) P2 quality reporting and observability

- [x] Extend `PackResult` diagnostics with optional fields:
  - minimum clearance,
  - outside-violation magnitude,
  - solver metadata (`method`, `iterations`, `success`).
- [x] Improve warning messages with concise remediation hints.

## 5) P3 test coverage and robustness

- [x] Add tests for arrangement duplicate-key failures (both create and apply paths).
- [x] Add dedicated tests for sanity-check warning behavior.
- [x] Add tests for render edge cases (empty placements, ordering/layering, background handling).
- [x] Add tests for non-PNG source paths and future SVG parsing behavior.

## 6) P0/P1 align to two-phase optimizer pipeline

- [x] Remove legacy search-based solver naming/path; use optimizer-based solver naming consistently.
- [x] Enforce one shared high-level flow for all methods:
  - load sources,
  - build normalized `PackingProblem`,
  - solve `compact_layout`,
  - solve `clearance_refinement`,
  - render.
- [x] Introduce a phase-optimizer interface so each phase can choose optimizer independently (`lbfgsb`, `de`, `nsga2`).
- [x] Refactor compact-layout solving to be backend-agnostic:
  - one objective definition,
  - one variable mapping (Jacobi + rotations),
  - multiple optimizer backends.
- [x] Refactor clearance-refinement solving to be backend-agnostic:
  - one objective definition (fixed-canvas spacing),
  - one variable mapping (normalized centers),
  - multiple optimizer backends.
- [x] Make both phases return ranked solution lists; pass top candidates from phase 1 into phase 2 (beam-style continuation).
- [x] Standardize ranking/selection across backends:
  - feasibility first (overlap/outside),
  - compactness and aspect for compact-layout,
  - spacing quality for clearance-refinement.
- [x] Unify solver outputs as `list[PackResult]` best-first for every method (single-element list when backend is single-solution).
- [x] Expand config to choose optimizer per phase explicitly in `PackOptions` (no solver-specific hidden defaults).
- [x] Add integration tests that run at least one case for each optimizer choice in both phases and verify:
  - non-empty solution list,
  - sanity metrics present,
  - deterministic ranking under fixed seed.
