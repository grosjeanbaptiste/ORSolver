from dataclasses import dataclass
from typing import Dict, List, Optional
import copy

from orsolver.core.model import Problem, Sense, Constraint, Variable
from orsolver.core.solution import Solution, SolutionStatus
from orsolver.lp.simplex import TwoPhaseSimplexSolver


@dataclass
class Goal:
    coefficients: Dict[str, float]
    target: float
    priority: int
    minimize_deviation: str  # "under" (d_minus), "over" (d_plus), or "both"
    name: str


class PreemptiveGoalProgramming:
    """
    Lexicographic / Preemptive Goal Programming Solver.
    Sequentially optimizes goals in order of priority (P1 > P2 > ...).
    """

    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.hard_constraints: List[Constraint] = []
        self.goals: List[Goal] = []
        self.solver = TwoPhaseSimplexSolver()

    def add_variable(
        self,
        name: str,
        lower_bound: Optional[float] = 0.0,
        upper_bound: Optional[float] = None,
    ) -> Variable:
        v = Variable(name=name, lower_bound=lower_bound, upper_bound=upper_bound)
        self.variables[name] = v
        return v

    def add_hard_constraint(
        self,
        coefficients: Dict[str, float],
        sense: str,
        rhs: float,
        name: Optional[str] = None,
    ) -> Constraint:
        for v in coefficients:
            if v not in self.variables:
                self.add_variable(v)
        c = Constraint(
            coefficients=coefficients,
            sense=sense,
            rhs=rhs,
            name=name or f"hard_{len(self.hard_constraints) + 1}",
        )
        self.hard_constraints.append(c)
        return c

    def add_goal(
        self,
        coefficients: Dict[str, float],
        target: float,
        priority: int,
        minimize_deviation: str = "under",
        name: Optional[str] = None,
    ) -> Goal:
        for v in coefficients:
            if v not in self.variables:
                self.add_variable(v)
        g_name = name or f"goal_{len(self.goals) + 1}"
        g = Goal(
            coefficients=coefficients,
            target=target,
            priority=priority,
            minimize_deviation=minimize_deviation,
            name=g_name,
        )
        self.goals.append(g)
        return g

    def solve(self) -> Solution:
        # Build base problem with hard constraints and deviational variables
        prob = Problem(name="goal_programming", sense=Sense.MINIMIZE)

        for v_name, var in self.variables.items():
            prob.add_variable(v_name, lower_bound=var.lower_bound, upper_bound=var.upper_bound)

        for c in self.hard_constraints:
            prob.add_constraint(dict(c.coefficients), c.sense, c.rhs, name=c.name)

        # For each goal: add d_minus and d_plus variables
        # Goal equation: sum(a_j * x_j) + d_minus - d_plus == target
        goal_dev_vars = {}
        for g in self.goals:
            d_minus = f"d_minus_{g.name}"
            d_plus = f"d_plus_{g.name}"
            prob.add_variable(d_minus, lower_bound=0.0)
            prob.add_variable(d_plus, lower_bound=0.0)
            goal_dev_vars[g.name] = (d_minus, d_plus)

            c_coeffs = dict(g.coefficients)
            c_coeffs[d_minus] = 1.0
            c_coeffs[d_plus] = -1.0
            prob.add_constraint(c_coeffs, "==", g.target, name=f"goal_eq_{g.name}")

        # Group goals by priority in ascending order (1 is highest priority)
        priorities = sorted(list(set(g.priority for g in self.goals)))
        last_sol = None

        for p_level in priorities:
            level_goals = [g for g in self.goals if g.priority == p_level]

            # Objective: minimize weighted sum of deviations at this priority level
            obj = {}
            for g in level_goals:
                d_minus, d_plus = goal_dev_vars[g.name]
                if g.minimize_deviation == "under":
                    obj[d_minus] = obj.get(d_minus, 0.0) + 1.0
                elif g.minimize_deviation == "over":
                    obj[d_plus] = obj.get(d_plus, 0.0) + 1.0
                elif g.minimize_deviation == "both":
                    obj[d_minus] = obj.get(d_minus, 0.0) + 1.0
                    obj[d_plus] = obj.get(d_plus, 0.0) + 1.0

            prob.set_objective(obj, sense=Sense.MINIMIZE)
            last_sol = self.solver.solve(prob)

            if not last_sol.is_optimal:
                return last_sol

            opt_dev = last_sol.objective_value
            # Fix this priority level's achievement as a constraint for subsequent levels
            prob.add_constraint(
                coefficients=obj,
                sense="<=",
                rhs=float(opt_dev + 1e-6),
                name=f"p_fixed_level_{p_level}",
            )

        return last_sol
