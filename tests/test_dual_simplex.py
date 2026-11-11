import pytest
from orsolver.core.model import Problem, Sense
from orsolver.core.solution import SolutionStatus
from orsolver.lp.dual_simplex import DualSimplexSolver


def test_dual_simplex_basic():
    """
    Min Z = 2x1 + x2
    s.t.
      3x1 + x2 >= 3  -> -3x1 - x2 <= -3
      4x1 + 3x2 >= 6 -> -4x1 - 3x2 <= -6
      x1 + 2x2 >= 2  -> -x1 - 2x2 <= -2
      x1, x2 >= 0
    Initial tableau has all Row 0 coefficients >= 0 (dual feasible),
    but negative RHS (primal infeasible).
    Dual Simplex directly pivots to optimal solution!
    Optimal: x1 = 3/5 = 0.6, x2 = 6/5 = 1.2, Z = 2.4.
    """
    p = Problem(name="dual_simplex_test", sense=Sense.MINIMIZE)
    p.add_constraint({"x1": 3.0, "x2": 1.0}, ">=", 3.0, name="c1")
    p.add_constraint({"x1": 4.0, "x2": 3.0}, ">=", 6.0, name="c2")
    p.add_constraint({"x1": 1.0, "x2": 2.0}, ">=", 2.0, name="c3")
    p.set_objective({"x1": 2.0, "x2": 1.0})

    solver = DualSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, 1e-4) == 2.4
    assert pytest.approx(sol.variable_values["x1"], 1e-4) == 0.6
    assert pytest.approx(sol.variable_values["x2"], 1e-4) == 1.2


def test_dual_simplex_infeasible():
    """
    Dual simplex detects primal infeasibility when a row has b_i < 0
    but all a_ij >= 0 (no negative pivot candidate).
    """
    p = Problem(name="dual_infeasible", sense=Sense.MINIMIZE)
    # 0 <= -5 -> 0x1 + 0x2 <= -5
    p.add_constraint({"x1": 0.0, "x2": 0.0}, "<=", -5.0, name="c_infeasible")
    p.set_objective({"x1": 1.0, "x2": 1.0})

    solver = DualSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.INFEASIBLE
