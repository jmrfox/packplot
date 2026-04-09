from __future__ import annotations

from packplot import (
    LbfgsbConfig,
    PipelineConfig,
    SolverConfig,
    PackOptions,
    PymooConfig,
)


FAST_TEST_PIPELINE_CONFIG = PipelineConfig(
    pack_phase=SolverConfig(
        optimizer="scipy-lbfgsb",
        progress_log_every_evaluations=0,
        lbfgsb=LbfgsbConfig(
            max_iterations=12,
            random_restart_count=1,
            alternating_refinement_cycles=0,
        ),
    ),
    enable_refine_phase=False,
)


def fast_opt_options(**kwargs) -> PackOptions:
    """Small helper for fast, deterministic optimizer-based tests."""
    pipeline_config = kwargs.pop("pipeline_config", FAST_TEST_PIPELINE_CONFIG)
    return PackOptions(
        pipeline_config=pipeline_config,
        **kwargs,
    )


def fast_pymoo_options(**kwargs) -> PackOptions:
    """Small helper for fast pymoo solver tests."""
    pipeline_config = kwargs.pop(
        "pipeline_config",
        PipelineConfig(pack_phase=SolverConfig(optimizer="pymoo-nsga2")),
    )
    pymoo_config = kwargs.pop(
        "pymoo_config",
        PymooConfig(
            algorithm="nsga2",
            generations=10,
            population_size=16,
            offspring_count=8,
            eliminate_duplicates=True,
        ),
    )
    return PackOptions(
        pipeline_config=pipeline_config,
        pymoo_config=pymoo_config,
        **kwargs,
    )
