from __future__ import annotations

from packplot import (
    LbfgsbConfig,
    OptimizeConfig,
    OptimizationPhaseConfig,
    PackOptions,
)


FAST_TEST_OPTIMIZE_CONFIG = OptimizeConfig(
    phase1=OptimizationPhaseConfig(
        method="lbfgsb",
        progress_log_every_evaluations=0,
        lbfgsb=LbfgsbConfig(
            max_iterations=35,
            random_restart_count=2,
            alternating_refinement_cycles=1,
        ),
    ),
    enable_spread_phase=False,
)


def fast_opt_options(**kwargs) -> PackOptions:
    """Small helper for fast, deterministic optimizer-based tests."""
    return PackOptions(
        solver="optimize",
        optimize_config=FAST_TEST_OPTIMIZE_CONFIG,
        **kwargs,
    )
