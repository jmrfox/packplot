"""Public package API for packplot."""

from packplot.arrangement import Arrangement, ArrangementKeyFunc, create_arrangement, load_arrangement, save_arrangement
from packplot.output_backends import list_output_backends, render_result_with_backend
from packplot.pipeline import pack_images
from packplot.problem import PackingProblem, build_packing_problem
from packplot.source_loaders import (
    MixedSourceLoader,
    RasterSourceLoader,
    SourceLoader,
    SvgSourceLoader,
    get_source_loader,
    infer_source_loader_name,
)
from packplot.types import (
    RefineObjectiveConfig,
    PackObjectiveConfig,
    DifferentialEvolutionConfig,
    InitializationConfig,
    LbfgsbConfig,
    PipelineConfig,
    SolverConfig,
    PackOptions,
    PackResult,
    PackedPlacement,
    PymooConfig,
    SourceObject,
)

__all__ = [
    "Arrangement",
    "ArrangementKeyFunc",
    "create_arrangement",
    "save_arrangement",
    "load_arrangement",
    "pack_images",
    "list_output_backends",
    "render_result_with_backend",
    "PackingProblem",
    "build_packing_problem",
    "SourceLoader",
    "RasterSourceLoader",
    "SvgSourceLoader",
    "MixedSourceLoader",
    "get_source_loader",
    "infer_source_loader_name",
    "PackOptions",
    "InitializationConfig",
    "PipelineConfig",
    "SolverConfig",
    "PymooConfig",
    "LbfgsbConfig",
    "DifferentialEvolutionConfig",
    "PackObjectiveConfig",
    "RefineObjectiveConfig",
    "PackResult",
    "PackedPlacement",
    "SourceObject",
]
