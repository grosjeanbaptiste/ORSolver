import pytest
from orsolver.core.model import Problem, Sense
from orsolver.core.solution import SolutionStatus
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def test_standard_max_problem():
    """
    Max Z = 3x1 + 5x2
    s.t.
      x1 <= 4
      2x2 <= 12  -> x2 <= 6
      3x1 + 2x2 <= 18
      x1, x2 >= 0
    Optimal: x1=2, x2=6, Z=36
    """
    p = Problem(name="standard_max", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 1.0}, "<=", 4.0, name="c1")
    p.add_constraint({"x2": 2.0}, "<=", 12.0, name="c2")
    p.add_constraint({"x1": 3.0, "x2": 2.0}, "<=", 18.0, name="c3")
    p.set_objective({"x1": 3.0, "x2": 5.0})

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, 1e-5) == 36.0
    assert pytest.approx(sol.variable_values["x1"], 1e-5) == 2.0
    assert pytest.approx(sol.variable_values["x2"], 1e-5) == 6.0
    # Shadow price of c2: 1.5, c3: 1.0, c1: 0.0
    assert pytest.approx(sol.shadow_prices["c2"], 1e-5) == 1.5
    assert pytest.approx(sol.shadow_prices["c3"], 1e-5) == 1.0
    assert pytest.approx(sol.shadow_prices["c1"], 1e-5) == 0.0


def test_minimization_with_mixed_constraints():
    """
    Min Z = 4x1 + x2
    s.t.
      3x1 + x2 = 3
      4x1 + 3x2 >= 6
      x1 + 2x2 <= 4
      x1, x2 >= 0
    Feasible interval: x1 in [0.4, 0.6], x2 = 3 - 3x1.
    Z = 4x1 + 3 - 3x1 = x1 + 3.
    Min Z at x1 = 0.4, x2 = 1.8, Z = 3.4
    """
    p = Problem(name="min_mixed", sense=Sense.MINIMIZE)
    p.add_constraint({"x1": 3.0, "x2": 1.0}, "==", 3.0, name="c1")
    p.add_constraint({"x1": 4.0, "x2": 3.0}, ">=", 6.0, name="c2")
    p.add_constraint({"x1": 1.0, "x2": 2.0}, "<=", 4.0, name="c3")
    p.set_objective({"x1": 4.0, "x2": 1.0})

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, 1e-5) == 3.4
    assert pytest.approx(sol.variable_values["x1"], 1e-5) == 0.4
    assert pytest.approx(sol.variable_values["x2"], 1e-5) == 1.8


def test_infeasible_problem():
    """
    Max Z = 2x1 + x2
    s.t.
      x1 + x2 <= 2
      x1 + x2 >= 5
    """
    p = Problem(name="infeasible", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 1.0, "x2": 1.0}, "<=", 2.0, name="c1")
    p.add_constraint({"x1": 1.0, "x2": 1.0}, ">=", 5.0, name="c2")
    p.set_objective({"x1": 2.0, "x2": 1.0})

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.INFEASIBLE


def test_unbounded_problem():
    """
    Max Z = 2x1 + x2
    s.t.
      x1 - x2 <= 2
      x1, x2 >= 0
    """
    p = Problem(name="unbounded", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 1.0, "x2": -1.0}, "<=", 2.0, name="c1")
    p.set_objective({"x1": 2.0, "x2": 1.0})

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.UNBOUNDED


def test_step_by_step_tableaus_recorded():
    """Verify that tableaus and pivot steps are recorded for educational display."""
    p = Problem(name="record_steps", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 1.0, "x2": 1.0}, "<=", 4.0, name="c1")
    p.set_objective({"x1": 2.0, "x2": 3.0})

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    assert len(sol.tableaus) > 0
    assert sol.iterations >= 1
