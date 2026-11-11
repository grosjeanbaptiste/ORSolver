import copy
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from orsolver.core.model import Problem, Sense
from orsolver.core.solution import Solution, SolutionStatus


class TableauSnapshot:
    """Stores a human-readable snapshot of the simplex tableau at a step."""

    def __init__(
        self,
        step: int,
        phase: int,
        headers: List[str],
        basis: List[str],
        matrix: np.ndarray,
        entering: Optional[str] = None,
        leaving: Optional[str] = None,
        pivot: Optional[Tuple[int, int]] = None,
    ):
        self.step = step
        self.phase = phase
        self.headers = list(headers)
        self.basis = list(basis)
        self.matrix = matrix.copy()
        self.entering = entering
        self.leaving = leaving
        self.pivot = pivot

    def __repr__(self) -> str:
        s = [f"Step {self.step} (Phase {self.phase}):"]
        header_str = " | ".join(f"{h:>10}" for h in ["Base"] + self.headers)
        s.append(header_str)
        s.append("-" * len(header_str))
        for i, row in enumerate(self.matrix):
            b_name = "Obj" if i == len(self.matrix) - 1 else self.basis[i]
            row_str = f"{b_name:>10} | " + " | ".join(f"{val:>10.4f}" for val in row)
            s.append(row_str)
        if self.entering and self.leaving:
            s.append(f"Pivot: Enter {self.entering}, Leave {self.leaving}")
        return "\n".join(s)


class TwoPhaseSimplexSolver:
    """
    Two-Phase Simplex Solver supporting <=, >=, == constraints,
    Minimization and Maximization, cycling prevention, sensitivity analysis
    (shadow prices & reduced costs), and step-by-step history.
    """

    def __init__(self, tol: float = 1e-9, max_iterations: int = 1000):
        self.tol = tol
        self.max_iterations = max_iterations

    def solve(self, problem: Problem) -> Solution:
        # Standardize problem:
        # Maximize Z = sum(c_j * x_j)
        # If sense is MINIMIZE: Maximize -Z
        sense_multiplier = 1.0 if problem.sense == Sense.MAXIMIZE else -1.0

        var_names = list(problem.variables.keys())
        num_vars = len(var_names)
        num_constraints = len(problem.constraints)

        # Standard form preparation
        # We need to construct rows:
        # Each constraint i:
        # sum(a_ij * x_j) + s_i - e_i + a_i = b_i  (b_i >= 0)
        slack_cols: Dict[str, int] = {}
        surplus_cols: Dict[str, int] = {}
        artificial_cols: Dict[str, int] = {}

        constraint_types = []
        rhs_values = []
        rows_a = []

        for i, c in enumerate(problem.constraints):
            row_coeffs = [c.coefficients.get(v, 0.0) for v in var_names]
            rhs = c.rhs
            sense = c.sense

            # Ensure RHS is non-negative
            if rhs < -self.tol:
                rhs = -rhs
                row_coeffs = [-v for v in row_coeffs]
                if sense == "<=":
                    sense = ">="
                elif sense == ">=":
                    sense = "<="

            rows_a.append(row_coeffs)
            rhs_values.append(rhs)
            constraint_types.append(sense)

        all_headers = list(var_names)
        col_idx = num_vars

        # Assign slack / surplus / artificial variables
        for i, sense in enumerate(constraint_types):
            c_name = problem.constraints[i].name
            if sense == "<=":
                name = f"s_{c_name}"
                slack_cols[c_name] = col_idx
                all_headers.append(name)
                col_idx += 1
            elif sense == ">=":
                e_name = f"e_{c_name}"
                surplus_cols[c_name] = col_idx
                all_headers.append(e_name)
                col_idx += 1

                a_name = f"a_{c_name}"
                artificial_cols[c_name] = col_idx
                all_headers.append(a_name)
                col_idx += 1
            elif sense == "==":
                a_name = f"a_{c_name}"
                artificial_cols[c_name] = col_idx
                all_headers.append(a_name)
                col_idx += 1

        total_cols = col_idx
        tableau_cols = total_cols + 1  # extra 1 for RHS

        # Build constraint rows
        # Matrix size: (num_constraints + 1) x (tableau_cols)
        T = np.zeros((num_constraints + 1, tableau_cols), dtype=float)
        basis = []

        for i, c in enumerate(problem.constraints):
            c_name = c.name
            T[i, :num_vars] = rows_a[i]
            T[i, -1] = rhs_values[i]
            if c_name in slack_cols:
                T[i, slack_cols[c_name]] = 1.0
                basis.append(all_headers[slack_cols[c_name]])
            elif c_name in artificial_cols:
                if c_name in surplus_cols:
                    T[i, surplus_cols[c_name]] = -1.0
                T[i, artificial_cols[c_name]] = 1.0
                basis.append(all_headers[artificial_cols[c_name]])

        tableaus: List[TableauSnapshot] = []
        iteration_count = 0

        # Phase I needed if any artificial variables exist
        needs_phase_1 = len(artificial_cols) > 0

        if needs_phase_1:
            # Phase I objective: Maximize -W = -sum(a_i)
            # In tableau form: (-W) + sum(a_i) = 0
            T[-1, :] = 0.0
            for a_col in artificial_cols.values():
                T[-1, a_col] = 1.0

            # Canonical form: Row 0 -= sum(Row i for rows where artificial is basic)
            for i, b_var in enumerate(basis):
                if b_var.startswith("a_"):
                    T[-1, :] -= T[i, :]

            tableaus.append(
                TableauSnapshot(
                    step=iteration_count,
                    phase=1,
                    headers=all_headers + ["RHS"],
                    basis=basis,
                    matrix=T,
                )
            )

            # Solve Phase I
            status, T, basis, iters = self._pivot_loop(
                T, basis, all_headers, phase=1, snapshots=tableaus, start_step=iteration_count
            )
            iteration_count += iters

            if status == SolutionStatus.UNBOUNDED:
                return Solution(status=SolutionStatus.INFEASIBLE, iterations=iteration_count)

            # Check optimal Phase I objective: min W = -T[-1, -1]
            phase1_val = -T[-1, -1]
            if phase1_val > 1e-6:
                # Artificial variables cannot all be driven to 0 -> Infeasible
                return Solution(
                    status=SolutionStatus.INFEASIBLE,
                    iterations=iteration_count,
                    tableaus=tableaus,
                )

            # If any artificial variable is still in the basis (with value 0),
            # pivot it out with any non-artificial column with non-zero entry.
            rows_to_remove = []
            for i, b_var in enumerate(list(basis)):
                if b_var.startswith("a_"):
                    pivoted = False
                    for j, h in enumerate(all_headers):
                        if not h.startswith("a_") and abs(T[i, j]) > self.tol:
                            # Pivot (i, j)
                            p_val = T[i, j]
                            T[i, :] /= p_val
                            for r in range(T.shape[0]):
                                if r != i:
                                    T[r, :] -= T[r, j] * T[i, :]
                            basis[i] = h
                            pivoted = True
                            break
                    if not pivoted:
                        rows_to_remove.append(i)

            if rows_to_remove:
                keep_rows = [r for r in range(T.shape[0]) if r not in rows_to_remove]
                T = T[keep_rows, :]
                basis = [b for idx, b in enumerate(basis) if idx not in rows_to_remove]

            # Phase I succeeded. Remove artificial columns.
            keep_indices = [
                idx for idx, h in enumerate(all_headers) if not h.startswith("a_")
            ]
            all_headers = [all_headers[idx] for idx in keep_indices]

            # Re-index table
            keep_indices_with_rhs = keep_indices + [tableau_cols - 1]
            T = T[:, keep_indices_with_rhs]
            tableau_cols = len(keep_indices_with_rhs)

        # Prepare Phase II
        # Original objective: Maximize sum(sense_multiplier * c_j * x_j)
        # Tableau Row 0: -c_j for decision variables, 0 for slacks/surplus
        T[-1, :] = 0.0
        for i, v in enumerate(var_names):
            if v in problem.objective:
                T[-1, i] = -sense_multiplier * problem.objective[v]

        # Put Row 0 in canonical form for current basis
        header_to_idx = {h: idx for idx, h in enumerate(all_headers)}
        for i, b_var in enumerate(basis):
            b_col = header_to_idx[b_var]
            multiplier = T[-1, b_col]
            if abs(multiplier) > self.tol:
                T[-1, :] -= multiplier * T[i, :]

        tableaus.append(
            TableauSnapshot(
                step=iteration_count,
                phase=2,
                headers=all_headers + ["RHS"],
                basis=basis,
                matrix=T,
            )
        )

        status, T, basis, iters = self._pivot_loop(
            T, basis, all_headers, phase=2, snapshots=tableaus, start_step=iteration_count
        )
        iteration_count += iters

        if status != SolutionStatus.OPTIMAL:
            return Solution(
                status=status,
                iterations=iteration_count,
                tableaus=tableaus,
            )

        # Extract solution
        header_to_idx = {h: idx for idx, h in enumerate(all_headers)}
        var_values = {v: 0.0 for v in var_names}
        slack_values = {c.name: 0.0 for c in problem.constraints}
        shadow_prices = {c.name: 0.0 for c in problem.constraints}
        reduced_costs = {v: 0.0 for v in var_names}

        for i, b_var in enumerate(basis):
            val = float(T[i, -1])
            if b_var in var_values:
                var_values[b_var] = val
            elif b_var.startswith("s_"):
                c_name = b_var[2:]
                slack_values[c_name] = val
            elif b_var.startswith("e_"):
                c_name = b_var[2:]
                slack_values[c_name] = -val

        # Reduced costs for non-basic decision variables
        # In max tableau: Row 0 entry is -(c_j - z_j) = z_j - c_j
        for i, v in enumerate(var_names):
            reduced_costs[v] = float(T[-1, i]) if sense_multiplier > 0 else float(-T[-1, i])

        # Shadow prices (dual values):
        # The shadow price is the marginal value of relaxing constraint i.
        # For <= constraint: it's the Row 0 coefficient of its slack variable.
        # For >= constraint: negative of Row 0 coefficient of surplus variable.
        for c in problem.constraints:
            c_name = c.name
            s_name = f"s_{c_name}"
            e_name = f"e_{c_name}"
            if s_name in header_to_idx:
                s_idx = header_to_idx[s_name]
                sp = float(T[-1, s_idx])
                shadow_prices[c_name] = sp
            elif e_name in header_to_idx:
                e_idx = header_to_idx[e_name]
                # Surplus column had -1 originally, so reduced cost sign is inverted
                sp = float(-T[-1, e_idx])
                shadow_prices[c_name] = sp

        obj_value = float(T[-1, -1]) * sense_multiplier

        return Solution(
            status=SolutionStatus.OPTIMAL,
            objective_value=obj_value,
            variable_values=var_values,
            slack_values=slack_values,
            shadow_prices=shadow_prices,
            reduced_costs=reduced_costs,
            iterations=iteration_count,
            tableaus=tableaus,
        )

    def _pivot_loop(
        self,
        T: np.ndarray,
        basis: List[str],
        headers: List[str],
        phase: int,
        snapshots: List[TableauSnapshot],
        start_step: int,
    ) -> Tuple[SolutionStatus, np.ndarray, List[str], int]:
        num_rows = T.shape[0] - 1
        num_cols = T.shape[1] - 1
        iters = 0

        while iters < self.max_iterations:
            # Check optimality: all reduced costs in row 0 >= -tol
            row_obj = T[-1, :num_cols]
            min_val = np.min(row_obj)
            if min_val >= -self.tol:
                # Optimal
                return SolutionStatus.OPTIMAL, T, basis, iters

            # Bland's rule: choose first negative column index
            pivot_col = -1
            for j in range(num_cols):
                if row_obj[j] < -self.tol:
                    pivot_col = j
                    break

            # Minimum ratio test
            pivot_row = -1
            min_ratio = float("inf")
            for i in range(num_rows):
                a_ij = T[i, pivot_col]
                if a_ij > self.tol:
                    ratio = T[i, -1] / a_ij
                    if ratio < min_ratio - self.tol:
                        min_ratio = ratio
                        pivot_row = i

            if pivot_row == -1:
                # Unbounded
                return SolutionStatus.UNBOUNDED, T, basis, iters

            # Pivot operation
            entering = headers[pivot_col]
            leaving = basis[pivot_row]

            p_val = T[pivot_row, pivot_col]
            T[pivot_row, :] /= p_val

            for i in range(num_rows + 1):
                if i != pivot_row:
                    factor = T[i, pivot_col]
                    if abs(factor) > self.tol:
                        T[i, :] -= factor * T[pivot_row, :]

            basis[pivot_row] = entering
            iters += 1

            snapshots.append(
                TableauSnapshot(
                    step=start_step + iters,
                    phase=phase,
                    headers=headers + ["RHS"],
                    basis=list(basis),
                    matrix=T.copy(),
                    entering=entering,
                    leaving=leaving,
                    pivot=(pivot_row, pivot_col),
                )
            )

        # Max iterations reached
        return SolutionStatus.FEASIBLE, T, basis, iters
