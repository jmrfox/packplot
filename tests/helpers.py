from __future__ import annotations

from packplot import (
    LbfgsbConfig,
    OptimizeConfig,
    OptimizationPhaseConfig,
    PackOptions,
    PymooConfig,
)


FAST_TEST_OPTIMIZE_CONFIG = OptimizeConfig(
    compact_layout_backend="optimize",
    compact_layout=OptimizationPhaseConfig(
        method="lbfgsb",
        progress_log_every_evaluations=0,
        lbfgsb=LbfgsbConfig(
            max_iterations=12,
            random_restart_count=1,
            alternating_refinement_cycles=0,
        ),
    ),
    enable_clearance_refinement_phase=False,
)


def fast_opt_options(**kwargs) -> PackOptions:
    """Small helper for fast, deterministic optimizer-based tests."""
    optimize_config = kwargs.pop("optimize_config", FAST_TEST_OPTIMIZE_CONFIG)
    return PackOptions(
        optimize_config=optimize_config,
        **kwargs,
    )


def fast_pymoo_options(**kwargs) -> PackOptions:
    """Small helper for fast pymoo solver tests."""
    optimize_config = kwargs.pop(
        "optimize_config",
        OptimizeConfig(compact_layout_backend="pymoo"),
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
        optimize_config=optimize_config,
        pymoo_config=pymoo_config,
        **kwargs,
    )
