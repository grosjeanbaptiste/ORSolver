import copy
import math
from typing import Dict, List, Optional
import numpy as np

from orsolver.core.model import Problem, Sense
from orsolver.core.solution import Solution, SolutionStatus
from orsolver.lp.simplex import TwoPhaseSimplexSolver
from orsolver.mip.branch_and_bound import BranchAndBoundSolver


class GomoryCuttingPlaneSolver:
    """
    Gomory's Fractional Cutting Plane algorithm for pure and mixed integer linear programs.
    Generates Gomory cuts from fractional rows in optimal tableau and iteratively tightens
    the LP relaxation.
    """

    def __init__(self, tol: float = 1e-5, max_cuts: int = 50):
        self.tol = tol
        self.max_cuts = max_cuts
        self.simplex_solver = TwoPhaseSimplexSolver(tol=1e-8)
        self.bb_solver = BranchAndBoundSolver(tol=tol)

    def solve(self, problem: Problem) -> Solution:
        curr_prob = copy.deepcopy(problem)
        cut_count = 0

        while cut_count < self.max_cuts:
            sol = self.simplex_solver.solve(curr_prob)
            if not sol.is_optimal:
                return sol

            # Check integrality of variables marked as integer
            fractional_candidates = []
            for v_name, var in problem.variables.items():
                if var.is_integer or var.is_binary:
                    val = sol.variable_values.get(v_name, 0.0)
                    dist = abs(val - round(val))
                    if dist > self.tol:
                        f_0 = val - math.floor(val)
                        fractional_candidates.append((dist, v_name, val, f_0))

            if not fractional_candidates:
                # All required variables are integral!
                cleaned_vars = dict(sol.variable_values)
                for v_name, var in problem.variables.items():
                    if var.is_integer or var.is_binary:
                        cleaned_vars[v_name] = round(cleaned_vars[v_name])
                sol.variable_values = cleaned_vars
                return sol

            # Find row for the most fractional variable
            fractional_candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_v, best_val, f_0 = fractional_candidates[0]

            final_snapshot = sol.tableaus[-1]
            basis = final_snapshot.basis
            matrix = final_snapshot.matrix
            headers = final_snapshot.headers[:-1]

            if best_v not in basis:
                break

            row_idx = basis.index(best_v)
            row_coeffs = matrix[row_idx, :-1]

            # In the tableau: x_i + sum_j a_rj * w_j = b_r
            # Gomory cut in tableau variables: sum_j f_j * w_j >= f_0
            # where f_j = a_rj - floor(a_rj), f_0 = b_r - floor(b_r)
            # We now substitute any slack variables:
            # s_c = c.rhs - sum_k c.coeffs[k] * x_k
            orig_cut_coeffs = {v: 0.0 for v in problem.variables}
            cut_rhs = f_0

            constraint_map = {c.name: c for c in curr_prob.constraints}

            for j, h in enumerate(headers):
                a_rj = row_coeffs[j]
                f_j = a_rj - math.floor(a_rj)
                if abs(f_j) < 1e-8:
                    continue

                if h in problem.variables:
                    orig_cut_coeffs[h] += f_j
                elif h.startswith("s_"):
                    c_name = h[2:]
                    if c_name in constraint_map:
                        c = constraint_map[c_name]
                        # s_c = rhs - sum_k (a_k * x_k)
                        cut_rhs -= f_j * c.rhs
                        for v_name, coeff in c.coefficients.items():
                            if v_name in orig_cut_coeffs:
                                orig_cut_coeffs[v_name] -= f_j * coeff
                elif h.startswith("e_"):
                    c_name = h[2:]
                    if c_name in constraint_map:
                        c = constraint_map[c_name]
                        # e_c = sum_k (a_k * x_k) - rhs
                        cut_rhs += f_j * c.rhs
                        for v_name, coeff in c.coefficients.items():
                            if v_name in orig_cut_coeffs:
                                orig_cut_coeffs[v_name] += f_j * coeff

            # Clean near-zero coefficients
            filtered_coeffs = {
                k: v for k, v in orig_cut_coeffs.items() if abs(v) > 1e-7
            }

            if not filtered_coeffs:
                break

            cut_count += 1
            curr_prob.add_constraint(
                coefficients=filtered_coeffs,
                sense=">=",
                rhs=float(cut_rhs),
                name=f"gomory_cut_{cut_count}",
            )

        # If pure Gomory reached max cuts or numerical limit, finish with Branch & Bound
        return self.bb_solver.solve(curr_prob)
