"""
Legacy wrapper for SimplexSolver, powered by the robust orsolver engine.
Maintains full backward compatibility while supporting <=, >=, and == constraints.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from orsolver.core.model import Problem, Sense
from orsolver.lp.simplex import TwoPhaseSimplexSolver


class SimplexSolver:
    """
    Class to solve linear programming problems using the Simplex method.
    """

    def __init__(
        self,
        objective: List[float],
        constraints: List[Tuple[List[float], str, float]],
        bounds: Optional[Any] = None,
    ):
        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds
        self.solver = TwoPhaseSimplexSolver()

    def solve(self) -> Dict[str, Any]:
        p = Problem(name="simplex_problem", sense=Sense.MAXIMIZE)
        num_vars = len(self.objective)
        var_names = [f"x{i+1}" for i in range(num_vars)]

        for v in var_names:
            p.add_variable(v, lower_bound=0.0)

        obj_dict = {f"x{i+1}": float(c) for i, c in enumerate(self.objective)}
        p.set_objective(obj_dict, sense=Sense.MAXIMIZE)

        for i, (coeffs, op, val) in enumerate(self.constraints):
            c_dict = {f"x{j+1}": float(coeffs[j]) for j in range(len(coeffs))}
            p.add_constraint(c_dict, op, float(val), name=f"c{i+1}")

        sol = self.solver.solve(p)

        solution_list = [sol.variable_values.get(v, 0.0) for v in var_names]
        obj_val = sol.objective_value if sol.objective_value is not None else float("nan")

        # Save to resultats_solution.txt for backward compatibility
        with open("resultats_solution.txt", "w") as f:
            f.write("Solution optimale : \n")
            f.write(f"{obj_val}\n")
            f.write(f"Variables: {solution_list}\n")

        print("Solution optimale : ", obj_val)

        return {"solution": solution_list, "objective_value": obj_val}
