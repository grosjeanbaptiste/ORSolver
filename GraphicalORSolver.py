"""
Legacy wrapper for GraphicalORSolver, powered by the robust orsolver engine.
Maintains backward compatibility while providing auto-scaling, axis intersections,
and exact corner-point enumeration.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from orsolver.core.model import Problem, Sense
from orsolver.visualizer.graphical_2d import DynamicGraphicalSolver


class GraphicalORSolver:
    """
    Class to visualize linear programming problems and calculate solutions graphically.
    """

    def __init__(
        self,
        constraints: List[Tuple[List[float], str, float]],
        bounds: Optional[Any] = None,
        objective: Optional[List[float]] = None,
    ):
        self.constraints = constraints
        self.bounds = bounds
        self.objective = objective or [1.0, 1.0]
        self.solver = DynamicGraphicalSolver()
        self._problem = self._build_problem()

    def _build_problem(self) -> Problem:
        p = Problem(name="graphical_problem", sense=Sense.MAXIMIZE)
        p.add_variable("x1", lower_bound=0.0)
        p.add_variable("x2", lower_bound=0.0)

        p.set_objective({"x1": float(self.objective[0]), "x2": float(self.objective[1])})

        for i, (coeffs, op, val) in enumerate(self.constraints):
            p.add_constraint(
                {"x1": float(coeffs[0]), "x2": float(coeffs[1])},
                op,
                float(val),
                name=f"c{i+1}",
            )
        return p

    def calculate_graphical_solution(self) -> Dict[str, Any]:
        res = self.solver.solve(self._problem)
        opt_pt = res["optimal_point"]
        sol_arr = np.array(opt_pt) if opt_pt is not None else None
        return {
            "solution": sol_arr,
            "objective_value": res["optimal_value"],
        }

    def plot(self, solution=None):
        """Plots the solution (compatible with legacy call)."""
        # Save temporary or render
        self.solver.plot_and_save(self._problem, filepath="solution_graphical.png")

    def save_plot(self, filename: str = "solution.png", format: str = "png"):
        """Saves plot to specified file."""
        self.solver.plot_and_save(self._problem, filepath=filename)
