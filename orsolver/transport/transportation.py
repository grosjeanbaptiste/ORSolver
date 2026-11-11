import numpy as np
from typing import List, Tuple, Union
from orsolver.core.model import Problem, Sense
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def north_west_corner(
    supply: Union[List[float], np.ndarray],
    demand: Union[List[float], np.ndarray],
) -> np.ndarray:
    """Computes an initial BFS using the North-West Corner method."""
    s = list(map(float, supply))
    d = list(map(float, demand))
    m, n = len(s), len(d)
    alloc = np.zeros((m, n), dtype=float)

    i, j = 0, 0
    while i < m and j < n:
        qty = min(s[i], d[j])
        alloc[i, j] = qty
        s[i] -= qty
        d[j] -= qty
        if s[i] == 0 and i < m - 1:
            i += 1
        elif d[j] == 0:
            j += 1
        else:
            i += 1
            j += 1
    return alloc


def least_cost_method(
    costs: Union[List[List[float]], np.ndarray],
    supply: Union[List[float], np.ndarray],
    demand: Union[List[float], np.ndarray],
) -> np.ndarray:
    """Computes an initial BFS using the Least-Cost (Greedy) method."""
    c = np.array(costs, dtype=float).copy()
    s = list(map(float, supply))
    d = list(map(float, demand))
    m, n = len(s), len(d)
    alloc = np.zeros((m, n), dtype=float)

    visited = np.zeros((m, n), dtype=bool)

    while True:
        min_cost = float("inf")
        min_i, min_j = -1, -1

        for i in range(m):
            if s[i] <= 0:
                continue
            for j in range(n):
                if d[j] <= 0 or visited[i, j]:
                    continue
                if c[i, j] < min_cost:
                    min_cost = c[i, j]
                    min_i, min_j = i, j

        if min_i == -1:
            break

        qty = min(s[min_i], d[min_j])
        alloc[min_i, min_j] = qty
        s[min_i] -= qty
        d[min_j] -= qty
        visited[min_i, min_j] = True

    return alloc


def vogel_approximation_method(
    costs: Union[List[List[float]], np.ndarray],
    supply: Union[List[float], np.ndarray],
    demand: Union[List[float], np.ndarray],
) -> np.ndarray:
    """Computes an initial BFS using Vogel's Approximation Method (VAM)."""
    c = np.array(costs, dtype=float).copy()
    s = list(map(float, supply))
    d = list(map(float, demand))
    m, n = len(s), len(d)
    alloc = np.zeros((m, n), dtype=float)

    rows_active = [True] * m
    cols_active = [True] * n

    while sum(rows_active) > 0 and sum(cols_active) > 0:
        row_penalties = []
        for i in range(m):
            if not rows_active[i]:
                row_penalties.append(-1.0)
                continue
            available_costs = sorted([c[i, j] for j in range(n) if cols_active[j]])
            if len(available_costs) >= 2:
                row_penalties.append(available_costs[1] - available_costs[0])
            elif len(available_costs) == 1:
                row_penalties.append(available_costs[0])
            else:
                row_penalties.append(-1.0)

        col_penalties = []
        for j in range(n):
            if not cols_active[j]:
                col_penalties.append(-1.0)
                continue
            available_costs = sorted([c[i, j] for i in range(m) if rows_active[i]])
            if len(available_costs) >= 2:
                col_penalties.append(available_costs[1] - available_costs[0])
            elif len(available_costs) == 1:
                col_penalties.append(available_costs[0])
            else:
                col_penalties.append(-1.0)

        max_row_pen = max(row_penalties) if row_penalties else -1.0
        max_col_pen = max(col_penalties) if col_penalties else -1.0

        if max_row_pen < 0 and max_col_pen < 0:
            break

        if max_row_pen >= max_col_pen:
            i = row_penalties.index(max_row_pen)
            # Find min cost in active columns for row i
            j_cand = [j for j in range(n) if cols_active[j]]
            j = min(j_cand, key=lambda col: c[i, col])
        else:
            j = col_penalties.index(max_col_pen)
            # Find min cost in active rows for col j
            i_cand = [i for i in range(m) if rows_active[i]]
            i = min(i_cand, key=lambda row: c[row, j])

        qty = min(s[i], d[j])
        alloc[i, j] = qty
        s[i] -= qty
        d[j] -= qty

        if s[i] <= 1e-9:
            rows_active[i] = False
        if d[j] <= 1e-9:
            cols_active[j] = False

    return alloc


def solve_transportation(
    costs: Union[List[List[float]], np.ndarray],
    supply: Union[List[float], np.ndarray],
    demand: Union[List[float], np.ndarray],
    method: str = "vam",
) -> Tuple[float, np.ndarray]:
    """
    Solves the Transportation problem optimally.
    Automatically balances supply and demand if unbalanced.
    Returns:
      (optimal_total_cost, allocation_matrix)
    """
    c = np.array(costs, dtype=float)
    s = list(map(float, supply))
    d = list(map(float, demand))
    orig_m, orig_n = len(s), len(d)

    total_s = sum(s)
    total_d = sum(d)

    # Balance problem with dummy source or destination if needed
    if total_s > total_d:
        diff = total_s - total_d
        d.append(diff)
        dummy_col = np.zeros((orig_m, 1), dtype=float)
        c = np.hstack([c, dummy_col])
    elif total_d > total_s:
        diff = total_d - total_s
        s.append(diff)
        dummy_row = np.zeros((1, orig_n), dtype=float)
        c = np.vstack([c, dummy_row])

    m, n = len(s), len(d)

    # Formulate exact LP model
    p = Problem(name="transportation", sense=Sense.MINIMIZE)
    obj = {}
    for i in range(m):
        for j in range(n):
            var_name = f"x_{i}_{j}"
            p.add_variable(var_name, lower_bound=0.0)
            obj[var_name] = c[i, j]

    p.set_objective(obj, sense=Sense.MINIMIZE)

    # Supply constraints: sum_j x_ij = s_i
    for i in range(m):
        coeffs = {f"x_{i}_{j}": 1.0 for j in range(n)}
        p.add_constraint(coeffs, "==", s[i], name=f"supply_{i}")

    # Demand constraints: sum_i x_ij = d_j
    for j in range(n):
        coeffs = {f"x_{i}_{j}": 1.0 for i in range(m)}
        p.add_constraint(coeffs, "==", d[j], name=f"demand_{j}")

    solver = TwoPhaseSimplexSolver()
    solution = solver.solve(p)

    alloc = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            var_name = f"x_{i}_{j}"
            alloc[i, j] = solution.variable_values.get(var_name, 0.0)

    # Return allocation cropped to original dimensions if dummy was added
    alloc_orig = alloc[:orig_m, :orig_n]
    total_cost = float(np.sum(alloc_orig * np.array(costs, dtype=float)))

    return total_cost, alloc_orig
