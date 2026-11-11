import numpy as np
from typing import Dict, List, Tuple, Union
from orsolver.core.model import Problem, Sense
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def solve_assignment(
    cost_matrix: Union[List[List[float]], np.ndarray],
) -> Tuple[float, Dict[int, int]]:
    """
    Solves the Assignment Problem (Hungarian / matching).
    Given an n x m cost matrix, finds the optimal 1-to-1 matching
    that minimizes the total cost.

    Returns:
      (min_total_cost, {worker_idx: task_idx})
    """
    c = np.array(cost_matrix, dtype=float)
    n, m = c.shape

    # Pad to square matrix if needed
    dim = max(n, m)
    padded_c = np.zeros((dim, dim), dtype=float)
    padded_c[:n, :m] = c
    # Any unfilled cells have large cost or 0 if dummy worker/task
    if dim > n:
        padded_c[n:, :] = 0.0
    if dim > m:
        padded_c[:, m:] = 0.0

    p = Problem(name="assignment", sense=Sense.MINIMIZE)
    obj = {}
    for i in range(dim):
        for j in range(dim):
            v_name = f"x_{i}_{j}"
            p.add_variable(v_name, lower_bound=0.0, upper_bound=1.0)
            obj[v_name] = padded_c[i, j]

    p.set_objective(obj, sense=Sense.MINIMIZE)

    # Worker constraints: each worker assigned exactly once
    for i in range(dim):
        coeffs = {f"x_{i}_{j}": 1.0 for j in range(dim)}
        p.add_constraint(coeffs, "==", 1.0, name=f"worker_{i}")

    # Task constraints: each task assigned exactly once
    for j in range(dim):
        coeffs = {f"x_{i}_{j}": 1.0 for i in range(dim)}
        p.add_constraint(coeffs, "==", 1.0, name=f"task_{j}")

    solver = TwoPhaseSimplexSolver()
    solution = solver.solve(p)

    assignment: Dict[int, int] = {}
    for i in range(n):
        for j in range(m):
            val = solution.variable_values.get(f"x_{i}_{j}", 0.0)
            if val > 0.5:
                assignment[i] = j

    total_cost = sum(c[i, assignment[i]] for i in range(n) if i in assignment)
    return float(total_cost), assignment
