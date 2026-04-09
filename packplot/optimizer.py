"""Optimizer dispatch: maps algorithm strings to numerical minimization backends."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import OptimizeResult, differential_evolution, minimize

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    phase: str
    interval: int
    count: int = 0
    best_score: float = float("inf")
    best_overlap: float = float("inf")
    best_aspect_error: float = float("inf")
    heartbeat_seconds: float = 5.0
    _last_log_ts: float = 0.0

    def update(self, *, score: float, overlap: float, aspect_error: float | None = None) -> None:
        self.count += 1
        improved = score < self.best_score
        if improved:
            self.best_score = score
            self.best_overlap = overlap
            if aspect_error is not None:
                self.best_aspect_error = aspect_error

        now = time.monotonic()
        should_log = False
        if self.count == 1:
            should_log = True
        elif self.interval > 0 and self.count % self.interval == 0:
            should_log = True
        elif self.heartbeat_seconds > 0 and (now - self._last_log_ts) >= self.heartbeat_seconds:
            should_log = True

        if should_log:
            self._last_log_ts = now
            if self.best_aspect_error != float("inf"):
                logger.info(
                    "%s progress: eval=%d best_score=%.3f best_overlap=%.6f best_aspect_error=%.4f",
                    self.phase,
                    self.count,
                    self.best_score,
                    self.best_overlap,
                    self.best_aspect_error,
                )
            else:
                logger.info(
                    "%s progress: eval=%d best_score=%.3f best_overlap=%.6f",
                    self.phase,
                    self.count,
                    self.best_score,
                    self.best_overlap,
                )


def run_with_method(
    *,
    phase: str,
    method: str,
    objective,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    maxiter: int,
    de_maxiter: int,
    de_popsize: int,
    workers: int,
    seed: int | None,
) -> OptimizeResult:
    """Dispatch a scalar minimization to the requested algorithm."""
    chosen = method.lower().strip()

    def _run_lbfgsb(start: np.ndarray) -> OptimizeResult:
        return minimize(
            objective,
            x0=start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )

    def _run_de() -> OptimizeResult:
        if workers > 1:
            def _thread_map(func, iterable):
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    return list(executor.map(func, iterable))

            de_workers = _thread_map
        else:
            de_workers = 1
        return differential_evolution(
            objective,
            bounds=bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            seed=seed,
            workers=de_workers,
            updating="deferred" if workers != 1 else "immediate",
            polish=True,
        )

    def _run_nsga2() -> OptimizeResult:
        class _ScalarProblem(ElementwiseProblem):
            def __init__(self):
                xl = np.asarray([bound[0] for bound in bounds], dtype=float)
                xu = np.asarray([bound[1] for bound in bounds], dtype=float)
                super().__init__(n_var=len(bounds), n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = [float(objective(np.asarray(x, dtype=float)))]

        nsga2_pop = max(8, int(de_popsize))
        nsga2_gen = max(1, int(de_maxiter))
        problem = _ScalarProblem()
        algorithm = NSGA2(pop_size=nsga2_pop, n_offsprings=None, eliminate_duplicates=True)
        result = pymoo_minimize(problem, algorithm, ("n_gen", nsga2_gen), seed=seed, verbose=False)
        if result.X is None:
            raise RuntimeError("NSGA-II phase optimization produced no solution.")
        x = np.asarray(result.X, dtype=float)
        f = np.asarray(result.F, dtype=float)
        if x.ndim == 1:
            best_x = x
            best_f = float(f[0]) if f.ndim > 0 else float(f)
        else:
            best_idx = int(np.argmin(f[:, 0]))
            best_x = x[best_idx]
            best_f = float(f[best_idx, 0])
        return OptimizeResult(
            x=best_x,
            fun=best_f,
            success=True,
            nit=int(getattr(result.algorithm, "n_gen", nsga2_gen)),
            message="NSGA-II optimization completed.",
        )

    if chosen == "lbfgsb":
        return _run_lbfgsb(x0)
    if chosen in {"de", "differential_evolution"}:
        return _run_de()
    if chosen == "nsga2":
        return _run_nsga2()
    if chosen == "hybrid":
        first = _run_lbfgsb(x0)
        message = str(first.message).upper()
        if first.success and "ABNORMAL" not in message:
            return first
        logger.warning("%s L-BFGS-B ended with '%s'; falling back to DE.", phase, first.message)
        second = _run_de()
        if float(second.fun) <= float(first.fun):
            return second
        return first
    raise ValueError(f"Unknown optimizer method '{method}' for phase {phase}.")
