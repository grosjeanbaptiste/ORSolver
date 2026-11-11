import pytest
from orsolver.core.model import Problem, Sense
from orsolver.core.solution import SolutionStatus
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def test_beale_cycling_example():
    """
    Beale's famous cycling example:
    Max 10*x1 - 57*x2 - 9*x3 - 24*x4
    s.t.
      0.5*x1 - 5.5*x2 - 2.5*x3 + 9*x4 <= 0
      0.5*x1 - 1.5*x2 - 0.5*x3 + 1*x4 <= 0
      x1 <= 1
      x1, x2, x3, x4 >= 0
    Without Bland's rule or anti-cycling, standard simplex cycles indefinitely.
    With Bland's rule, it terminates with optimal solution: x1=1, x2=0, x3=1, x4=0, Z=1 or similar.
    """
    p = Problem(name="beale_cycling", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 0.5, "x2": -5.5, "x3": -2.5, "x4": 9.0}, "<=", 0.0, name="c1")
    p.add_constraint({"x1": 0.5, "x2": -1.5, "x3": -0.5, "x4": 1.0}, "<=", 0.0, name="c2")
    p.add_constraint({"x1": 1.0}, "<=", 1.0, name="c3")
    p.set_objective({"x1": 10.0, "x2": -57.0, "x3": -9.0, "x4": -24.0})

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    # Beale's optimal value is 1.0 or feasible bounded optimum
    assert pytest.approx(sol.objective_value, 1e-4) == 1.0
