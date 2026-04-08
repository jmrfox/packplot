from __future__ import annotations

from abc import ABC, abstractmethod

from packplot.optimize import optimize_pack
from packplot.packer import pack_polygons
from packplot.problem import PackingProblem
from packplot.types import PackedPlacement, SolverMetadata


class LayoutSolver(ABC):
    """Interface for packing solvers."""

    @abstractmethod
    def solve(self, problem: PackingProblem) -> tuple[list[PackedPlacement], tuple[int, int], SolverMetadata]:
        """Return placements, canvas size, and solver execution metadata."""


class HeuristicLayoutSolver(LayoutSolver):
    """Adapter for the heuristic grid-search packer."""

    def solve(self, problem: PackingProblem) -> tuple[list[PackedPlacement], tuple[int, int], SolverMetadata]:
        placements, canvas_size = pack_polygons(problem.source_objects, problem.options)
        return placements, canvas_size, SolverMetadata(method="heuristic", success=True)


class OptimizeLayoutSolver(LayoutSolver):
    """Adapter for normalized optimization-based packing."""

    def solve(self, problem: PackingProblem) -> tuple[list[PackedPlacement], tuple[int, int], SolverMetadata]:
        return optimize_pack(problem.source_objects, problem.options, problem)


class PymooLayoutSolver(LayoutSolver):
    """Placeholder adapter for a future pymoo-based optimizer."""

    def solve(self, problem: PackingProblem) -> tuple[list[PackedPlacement], tuple[int, int], SolverMetadata]:
        # Keep a runtime import so users get a clear dependency error if pymoo
        # was removed from their environment while this solver is selected.
        import pymoo  # noqa: F401

        raise NotImplementedError(
            "solver='pymoo' is a stub integration point and is not implemented yet. "
            "Use solver='optimize' or solver='heuristic' for now."
        )


def get_solver(name: str) -> LayoutSolver:
    """Return a solver instance by configured solver name."""
    if name == "heuristic":
        return HeuristicLayoutSolver()
    if name == "optimize":
        return OptimizeLayoutSolver()
    if name == "pymoo":
        return PymooLayoutSolver()
    raise ValueError("Unknown solver. Expected 'heuristic', 'optimize', or 'pymoo'.")
