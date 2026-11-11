import copy
import math
from typing import Dict, List, Optional
from orsolver.core.model import Problem, Sense, Constraint
from orsolver.core.solution import Solution, SolutionStatus
from orsolver.lp.simplex import TwoPhaseSimplexSolver


class BranchAndBoundSolver:
    """
    Branch-and-Bound solver for Mixed-Integer Linear Programming (MILP).
    Uses TwoPhaseSimplexSolver for LP relaxations.
    """

    def __init__(self, tol: float = 1e-5, max_nodes: int = 1000):
        self.tol = tol
        self.max_nodes = max_nodes
        self.simplex_solver = TwoPhaseSimplexSolver(tol=1e-8)

    def solve(self, problem: Problem) -> Solution:
        is_max = problem.sense == Sense.MAXIMIZE
        best_obj = float("-inf") if is_max else float("inf")
        best_solution: Optional[Solution] = None

        # Stack of subproblems: list of Problem instances
        root_problem = copy.deepcopy(problem)
        stack: List[Problem] = [root_problem]
        nodes_explored = 0

        while stack and nodes_explored < self.max_nodes:
            current_problem = stack.pop()
            nodes_explored += 1

            # Solve continuous LP relaxation
            lp_sol = self.simplex_solver.solve(current_problem)

            if lp_sol.status != SolutionStatus.OPTIMAL:
                # Infeasible or unbounded relaxation -> prune
                continue

            obj = lp_sol.objective_value
            # Bound check: can this node beat the best incumbent?
            if is_max:
                if obj <= best_obj + self.tol:
                    continue
            else:
                if obj >= best_obj - self.tol:
                    continue

            # Check integrality of integer-restricted variables
            fractional_var = None
            max_fractionality = -1.0
            is_integral = True

            for v_name, var in problem.variables.items():
                if var.is_integer or var.is_binary:
                    val = lp_sol.variable_values.get(v_name, 0.0)
                    dist_to_int = abs(val - round(val))
                    if dist_to_int > self.tol:
                        is_integral = False
                        # Choose most fractional variable
                        if dist_to_int > max_fractionality:
                            max_fractionality = dist_to_int
                            fractional_var = (v_name, val)

            if is_integral:
                # All integer variables are integers: we found a new feasible integer solution!
                best_obj = obj
                best_solution = copy.deepcopy(lp_sol)
                continue

            # Branching
            v_name, val = fractional_var
            floor_val = math.floor(val)
            ceil_val = math.ceil(val)

            # Left child: x_j <= floor_val
            left_prob = copy.deepcopy(current_problem)
            left_prob.add_constraint(
                coefficients={v_name: 1.0},
                sense="<=",
                rhs=float(floor_val),
                name=f"branch_{v_name}_le_{floor_val}_{nodes_explored}",
            )

            # Right child: x_j >= ceil_val
            right_prob = copy.deepcopy(current_problem)
            right_prob.add_constraint(
                coefficients={v_name: 1.0},
                sense=">=",
                rhs=float(ceil_val),
                name=f"branch_{v_name}_ge_{ceil_val}_{nodes_explored}",
            )

            stack.append(left_prob)
            stack.append(right_prob)

        if best_solution is None:
            return Solution(status=SolutionStatus.INFEASIBLE, iterations=nodes_explored)

        # Round integer variables slightly to clean up numerical noise
        cleaned_vars = dict(best_solution.variable_values)
        for v_name, var in problem.variables.items():
            if var.is_integer or var.is_binary:
                cleaned_vars[v_name] = round(cleaned_vars.get(v_name, 0.0))

        best_solution.variable_values = cleaned_vars
        best_solution.iterations = nodes_explored
        return best_solution
