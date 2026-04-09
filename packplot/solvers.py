from __future__ import annotations

from abc import ABC, abstractmethod

from packplot.problem import PackingProblem
from packplot.phase_pipeline import solve_two_phase
from packplot.types import PackedPlacement

SolverCandidate = tuple[list[PackedPlacement], tuple[int, int], tuple[float, float, float] | None]


class LayoutSolver(ABC):
    """Interface for packing solvers."""

    @abstractmethod
    def solve(self, problem: PackingProblem) -> tuple[list[SolverCandidate], str, int | None, bool | None]:
        """Return ranked candidate layouts and run metadata."""


class OptimizeLayoutSolver(LayoutSolver):
    """Adapter for normalized optimization-based packing."""

    def solve(self, problem: PackingProblem) -> tuple[list[SolverCandidate], str, int | None, bool | None]:
        return solve_two_phase(problem, compact_backend="optimize")


class PymooLayoutSolver(LayoutSolver):
    """Adapter for multi-objective pymoo-based packing."""

    def solve(self, problem: PackingProblem) -> tuple[list[SolverCandidate], str, int | None, bool | None]:
        return solve_two_phase(problem, compact_backend="pymoo")


def get_solver(name: str) -> LayoutSolver:
    """Return a solver instance by configured solver name."""
    if name == "optimize":
        return OptimizeLayoutSolver()
    if name == "pymoo":
        return PymooLayoutSolver()
    raise ValueError("Unknown solver. Expected 'optimize' or 'pymoo'.")
