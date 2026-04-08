"""Public package API for packplot."""

from packplot.arrangement import Arrangement, ArrangementKeyFunc, create_arrangement, load_arrangement, save_arrangement
from packplot.pipeline import pack_images
from packplot.problem import PackingProblem, build_packing_problem
from packplot.solvers import LayoutSolver, get_solver
from packplot.source_loaders import (
    MixedSourceLoader,
    RasterSourceLoader,
    SourceLoader,
    SvgSourceLoader,
    get_source_loader,
    infer_source_loader_name,
)
from packplot.types import (
    ClearanceRefinementObjectiveConfig,
    CompactLayoutObjectiveConfig,
    DifferentialEvolutionConfig,
    LbfgsbConfig,
    OptimizeConfig,
    OptimizationPhaseConfig,
    PackOptions,
    PackResult,
    PackedPlacement,
    SolverMetadata,
    SourceObject,
)

__all__ = [
    "Arrangement",
    "ArrangementKeyFunc",
    "create_arrangement",
    "save_arrangement",
    "load_arrangement",
    "pack_images",
    "PackingProblem",
    "build_packing_problem",
    "LayoutSolver",
    "get_solver",
    "SourceLoader",
    "RasterSourceLoader",
    "SvgSourceLoader",
    "MixedSourceLoader",
    "get_source_loader",
    "infer_source_loader_name",
    "PackOptions",
    "OptimizeConfig",
    "OptimizationPhaseConfig",
    "LbfgsbConfig",
    "DifferentialEvolutionConfig",
    "CompactLayoutObjectiveConfig",
    "ClearanceRefinementObjectiveConfig",
    "PackResult",
    "PackedPlacement",
    "SolverMetadata",
    "SourceObject",
]
