import pytest
from orsolver.core.model import Problem, Sense
from orsolver.core.solution import SolutionStatus
from orsolver.mip.cutting_plane import GomoryCuttingPlaneSolver


def test_gomory_cutting_plane():
    """
    Max Z = x1 + x2
    s.t.
      2x1 + x2 <= 5
      x1 + 2x2 <= 5
      x1, x2 >= 0, integers
    Continuous LP:
      Intersection of 2x1 + x2 = 5 and x1 + 2x2 = 5:
      x1 = 5/3 = 1.667, x2 = 5/3 = 1.667 -> Z = 10/3 = 3.333
    Integer optimum:
      (1, 2) -> Z = 3 (2*1 + 2 = 4 <= 5, 1 + 2*2 = 5 <= 5)
      or (2, 1) -> Z = 3
    Gomory cuts eliminate (5/3, 5/3) and converge to integer optimum Z = 3!
    """
    p = Problem(name="gomory_test", sense=Sense.MAXIMIZE)
    p.add_variable("x1", lower_bound=0.0, is_integer=True)
    p.add_variable("x2", lower_bound=0.0, is_integer=True)
    p.add_constraint({"x1": 2.0, "x2": 1.0}, "<=", 5.0, name="c1")
    p.add_constraint({"x1": 1.0, "x2": 2.0}, "<=", 5.0, name="c2")
    p.set_objective({"x1": 1.0, "x2": 1.0})

    solver = GomoryCuttingPlaneSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, 1e-4) == 3.0
    # Values must be integers
    v1 = sol.variable_values["x1"]
    v2 = sol.variable_values["x2"]
    assert pytest.approx(v1, 1e-4) == round(v1)
    assert pytest.approx(v2, 1e-4) == round(v2)
