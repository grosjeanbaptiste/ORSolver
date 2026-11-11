import pytest
from orsolver.core.model import Problem, Sense
from orsolver.lp.simplex import TwoPhaseSimplexSolver
from orsolver.lp.sensitivity import analyze_sensitivity


def test_sensitivity_ranges():
    """
    Max Z = 3x1 + 5x2
    s.t.
      x1 <= 4          (c1)
      2x2 <= 12        (c2)
      3x1 + 2x2 <= 18  (c3)
      x1, x2 >= 0
    Optimal basis: x1=2, x2=6, s1=2.
    Shadow prices: y1=0, y2=1.5, y3=1.0.
    Allowable ranges can be determined from the optimal basis inverse B^-1.
    """
    p = Problem(name="sensitivity_test", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 1.0}, "<=", 4.0, name="c1")
    p.add_constraint({"x2": 2.0}, "<=", 12.0, name="c2")
    p.add_constraint({"x1": 3.0, "x2": 2.0}, "<=", 18.0, name="c3")
    p.set_objective({"x1": 3.0, "x2": 5.0})

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(p)

    report = analyze_sensitivity(p, sol)

    assert "objective_ranges" in report
    assert "rhs_ranges" in report
    assert "x1" in report["objective_ranges"]
    assert "x2" in report["objective_ranges"]
    assert "c1" in report["rhs_ranges"]
    assert "c2" in report["rhs_ranges"]
    assert "c3" in report["rhs_ranges"]

    # For c1 (non-binding constraint, s1=2 > 0):
    # Allowable decrease is 2.0 (since x1=2, decreasing RHS from 4 to 2 doesn't violate)
    # Allowable increase is +inf
    c1_range = report["rhs_ranges"]["c1"]
    assert pytest.approx(c1_range["allowable_decrease"], 1e-3) == 2.0
    assert c1_range["allowable_increase"] == float("inf")
