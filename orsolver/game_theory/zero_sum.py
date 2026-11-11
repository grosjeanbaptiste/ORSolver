from typing import Any, Dict, List, Union
import numpy as np

from orsolver.core.model import Problem, Sense
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def solve_zero_sum_game(
    payoff_matrix: Union[List[List[float]], np.ndarray],
) -> Dict[str, Any]:
    """
    Solves a 2-player zero-sum matrix game using Linear Programming.
    Row player maximizes expected payoff; Column player minimizes it.

    Returns:
      {
        "game_value": float,
        "row_strategy": List[float],
        "col_strategy": List[float]
      }
    """
    A = np.array(payoff_matrix, dtype=float)
    m, n = A.shape

    # Ensure all entries are strictly positive by adding a shift constant if needed
    min_val = np.min(A)
    shift = 0.0
    if min_val <= 0:
        shift = abs(min_val) + 1.0
    A_shifted = A + shift

    # LP Formulation:
    # Let x_i = p_i / V'
    # Minimize sum(x_i)
    # s.t. sum_{i=1}^m A_shifted[i, j] * x_i >= 1  for j = 1..n
    #      x_i >= 0
    p = Problem(name="zero_sum_game", sense=Sense.MINIMIZE)

    var_names = [f"x_{i}" for i in range(m)]
    for v in var_names:
        p.add_variable(v, lower_bound=0.0)

    # Objective: Minimize sum(x_i)
    p.set_objective({v: 1.0 for v in var_names}, sense=Sense.MINIMIZE)

    # Constraints: A'^T x >= 1
    for j in range(n):
        coeffs = {f"x_{i}": float(A_shifted[i, j]) for i in range(m)}
        p.add_constraint(coeffs, ">=", 1.0, name=f"col_strat_{j}")

    solver = TwoPhaseSimplexSolver()
    solution = solver.solve(p)

    x_vals = [solution.variable_values.get(v, 0.0) for v in var_names]
    sum_x = sum(x_vals)

    if sum_x <= 1e-9:
        raise ValueError("Failed to solve zero-sum game.")

    v_prime = 1.0 / sum_x
    row_strategy = [x * v_prime for x in x_vals]

    # Column player's strategy from shadow prices (dual variables)
    col_strategy = []
    for j in range(n):
        sp = solution.shadow_prices.get(f"col_strat_{j}", 0.0)
        col_strategy.append(sp * v_prime)

    # Normalize strategies to sum to exactly 1.0
    row_strategy = [float(v / sum(row_strategy)) for v in row_strategy]
    col_sum = sum(col_strategy)
    if col_sum > 1e-9:
        col_strategy = [float(v / col_sum) for v in col_strategy]
    else:
        col_strategy = [1.0 / n] * n

    game_value = float(v_prime - shift)

    return {
        "game_value": game_value,
        "row_strategy": row_strategy,
        "col_strategy": col_strategy,
    }
