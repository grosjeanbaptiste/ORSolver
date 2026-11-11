from typing import Any, Dict, List, Optional
import numpy as np

from orsolver.core.model import Problem, Sense
from orsolver.core.solution import Solution


def analyze_sensitivity(problem: Problem, solution: Solution) -> Dict[str, Any]:
    """
    Performs sensitivity analysis on an optimal linear program:
    - Allowable increase/decrease for objective coefficients (optimality range)
    - Allowable increase/decrease for RHS values (feasibility range)
    - Shadow prices (dual values)
    - Reduced costs
    """
    if not solution.is_optimal or not solution.tableaus:
        raise ValueError("Sensitivity analysis requires an optimal solution with recorded tableaus.")

    final_snapshot = solution.tableaus[-1]
    headers = final_snapshot.headers[:-1]  # exclude 'RHS'
    basis = final_snapshot.basis
    matrix = final_snapshot.matrix
    num_constraints = len(problem.constraints)

    header_to_col = {h: idx for idx, h in enumerate(headers)}
    var_names = list(problem.variables.keys())

    # 1. RHS Ranges (Feasibility ranges)
    rhs_ranges = {}
    for c in problem.constraints:
        c_name = c.name
        s_name = f"s_{c_name}"

        if s_name in header_to_col:
            col_idx = header_to_col[s_name]
            b_inv_col = matrix[:-1, col_idx]
            rhs_vals = matrix[:-1, -1]

            allowable_inc = float("inf")
            allowable_dec = float("inf")

            for i in range(len(b_inv_col)):
                val = b_inv_col[i]
                rhs_i = rhs_vals[i]

                if val > 1e-8:
                    dec = rhs_i / val
                    if dec < allowable_dec:
                        allowable_dec = dec
                elif val < -1e-8:
                    inc = rhs_i / abs(val)
                    if inc < allowable_inc:
                        allowable_inc = inc

            rhs_ranges[c_name] = {
                "current_rhs": c.rhs,
                "shadow_price": solution.shadow_prices.get(c_name, 0.0),
                "allowable_increase": allowable_inc,
                "allowable_decrease": allowable_dec,
            }

    # 2. Objective Coefficient Ranges (Optimality ranges)
    obj_ranges = {}
    row_obj = matrix[-1, :-1]  # Reduced costs row (z_j - c_j)

    for v_name in var_names:
        curr_c = problem.objective.get(v_name, 0.0)
        allowable_inc = float("inf")
        allowable_dec = float("inf")

        if v_name in basis:
            row_idx = basis.index(v_name)
            # Basic variable
            for j, h in enumerate(headers):
                if h not in basis:
                    a_rj = matrix[row_idx, j]
                    rc = row_obj[j]

                    if a_rj > 1e-8:
                        dec = rc / a_rj
                        if dec < allowable_dec:
                            allowable_dec = dec
                    elif a_rj < -1e-8:
                        inc = rc / abs(a_rj)
                        if inc < allowable_inc:
                            allowable_inc = inc
        else:
            # Non-basic variable
            if v_name in header_to_col:
                col_idx = header_to_col[v_name]
                rc = row_obj[col_idx]
                allowable_inc = float(rc)
                allowable_dec = float("inf")

        obj_ranges[v_name] = {
            "current_coeff": curr_c,
            "reduced_cost": solution.reduced_costs.get(v_name, 0.0),
            "allowable_increase": allowable_inc,
            "allowable_decrease": allowable_dec,
        }

    return {
        "objective_ranges": obj_ranges,
        "rhs_ranges": rhs_ranges,
    }
