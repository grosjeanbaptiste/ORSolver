import numpy as np
from typing import Dict, List, Optional, Tuple
from orsolver.core.model import Problem, Sense
from orsolver.core.solution import Solution, SolutionStatus
from orsolver.lp.simplex import TableauSnapshot


class DualSimplexSolver:
    """
    Dual Simplex Solver.
    Solves linear programming problems starting from a dual-feasible basis
    (optimal reduced costs) with negative RHS values.
    Pivots directly without artificial variables.
    """

    def __init__(self, tol: float = 1e-8, max_iterations: int = 1000):
        self.tol = tol
        self.max_iterations = max_iterations

    def solve(self, problem: Problem) -> Solution:
        is_minimize = problem.sense == Sense.MINIMIZE
        var_names = list(problem.variables.keys())
        num_vars = len(var_names)
        num_constraints = len(problem.constraints)

        # Standard form: sum(a_ij * x_j) + s_i = b_i
        # If sense is MINIMIZE, we convert Min c^T x to Max -c^T x
        c_mult = -1.0 if is_minimize else 1.0

        all_headers = list(var_names)
        slack_cols = {}
        col_idx = num_vars

        rows = []
        rhs_list = []

        for i, c in enumerate(problem.constraints):
            row_coeffs = [c.coefficients.get(v, 0.0) for v in var_names]
            rhs = c.rhs
            # If constraint is >=, multiply by -1 to make it <=
            if c.sense == ">=":
                row_coeffs = [-v for v in row_coeffs]
                rhs = -rhs
            elif c.sense == "==":
                # Dual simplex standard form assumes inequalities
                pass

            s_name = f"s_{c.name}"
            slack_cols[c.name] = col_idx
            all_headers.append(s_name)
            col_idx += 1

            rows.append(row_coeffs)
            rhs_list.append(rhs)

        total_cols = col_idx
        T = np.zeros((num_constraints + 1, total_cols + 1), dtype=float)
        basis = []

        for i in range(num_constraints):
            T[i, :num_vars] = rows[i]
            T[i, slack_cols[problem.constraints[i].name]] = 1.0
            T[i, -1] = rhs_list[i]
            basis.append(all_headers[slack_cols[problem.constraints[i].name]])

        # Row 0: -c_mult * c_j for decision variables
        for j, v in enumerate(var_names):
            if v in problem.objective:
                T[-1, j] = -c_mult * problem.objective[v]

        # Check dual feasibility: in max form, Row 0 coefficients must be >= -tol
        for j in range(total_cols):
            if T[-1, j] < -self.tol:
                # If not dual feasible, dual simplex cannot start directly
                # fallback or adjust
                pass

        iters = 0
        tableaus: List[TableauSnapshot] = [
            TableauSnapshot(
                step=0,
                phase=1,
                headers=all_headers + ["RHS"],
                basis=list(basis),
                matrix=T.copy(),
            )
        ]

        while iters < self.max_iterations:
            # Check primal feasibility: are all RHS >= -tol?
            rhs_vals = T[:-1, -1]
            min_rhs_idx = np.argmin(rhs_vals)
            min_rhs = rhs_vals[min_rhs_idx]

            if min_rhs >= -self.tol:
                # Primal feasible + dual feasible = OPTIMAL!
                break

            pivot_row = min_rhs_idx

            # Find entering column: min ratio (Row 0 / |a_ij|) for a_ij < -tol
            pivot_col = -1
            min_ratio = float("inf")

            for j in range(total_cols):
                a_ij = T[pivot_row, j]
                if a_ij < -self.tol:
                    # In max tableau, Row 0 entry is >= 0
                    ratio = abs(T[-1, j] / a_ij)
                    if ratio < min_ratio - self.tol:
                        min_ratio = ratio
                        pivot_col = j

            if pivot_col == -1:
                # No negative element in pivot row: Primal is INFEASIBLE
                return Solution(
                    status=SolutionStatus.INFEASIBLE,
                    iterations=iters,
                    tableaus=tableaus,
                )

            # Perform pivot on (pivot_row, pivot_col)
            entering = all_headers[pivot_col]
            leaving = basis[pivot_row]

            p_val = T[pivot_row, pivot_col]
            T[pivot_row, :] /= p_val

            for r in range(num_constraints + 1):
                if r != pivot_row:
                    factor = T[r, pivot_col]
                    if abs(factor) > self.tol:
                        T[r, :] -= factor * T[pivot_row, :]

            basis[pivot_row] = entering
            iters += 1

            tableaus.append(
                TableauSnapshot(
                    step=iters,
                    phase=1,
                    headers=all_headers + ["RHS"],
                    basis=list(basis),
                    matrix=T.copy(),
                    entering=entering,
                    leaving=leaving,
                    pivot=(pivot_row, pivot_col),
                )
            )

        # Extract solution
        var_values = {v: 0.0 for v in var_names}
        slack_values = {c.name: 0.0 for c in problem.constraints}
        for i, b_var in enumerate(basis):
            val = float(T[i, -1])
            if b_var in var_values:
                var_values[b_var] = val
            elif b_var.startswith("s_"):
                slack_values[b_var[2:]] = val

        obj_val = float(T[-1, -1]) * c_mult

        return Solution(
            status=SolutionStatus.OPTIMAL,
            objective_value=obj_val,
            variable_values=var_values,
            slack_values=slack_values,
            iterations=iters,
            tableaus=tableaus,
        )
