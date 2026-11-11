import pytest
from orsolver.core.model import Problem, Sense
from orsolver.core.solution import SolutionStatus
from orsolver.mip.branch_and_bound import BranchAndBoundSolver
from orsolver.mip.knapsack import solve_knapsack_01


def test_pure_integer_programming():
    """
    Max Z = 8x1 + 5x2
    s.t.
      x1 + x2 <= 6
      9x1 + 5x2 <= 45
      x1, x2 >= 0, integer
    Continuous LP relaxation:
      x1 = 3.75, x2 = 2.25, Z = 41.25
    Integer optimal:
      (x1, x2) = (5, 0) -> Z = 40
      or (x1, x2) = (0, 6) -> Z = 30
      or (x1, x2) = (4, 1) -> Z = 37
      or (x1, x2) = (3, 3) -> Z = 39
      or (x1, x2) = (5, 0): 9*5 + 0 = 45 <= 45, 5+0 = 5 <= 6 -> Z = 40.
    Optimal integer Z = 40 at (5, 0).
    """
    p = Problem(name="pure_ip", sense=Sense.MAXIMIZE)
    p.add_variable("x1", lower_bound=0, is_integer=True)
    p.add_variable("x2", lower_bound=0, is_integer=True)
    p.add_constraint({"x1": 1.0, "x2": 1.0}, "<=", 6.0, name="c1")
    p.add_constraint({"x1": 9.0, "x2": 5.0}, "<=", 45.0, name="c2")
    p.set_objective({"x1": 8.0, "x2": 5.0})

    solver = BranchAndBoundSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, 1e-5) == 40.0
    assert pytest.approx(sol.variable_values["x1"], 1e-5) == 5.0
    assert pytest.approx(sol.variable_values["x2"], 1e-5) == 0.0


def test_mixed_integer_programming():
    """
    Max Z = x1 + 2x2
    s.t.
      x1 + x2 <= 3.5
      x1 >= 0 (continuous)
      x2 >= 0 (integer)
    Optimal:
      If x2 = 3: x1 <= 0.5 -> x1 = 0.5, Z = 0.5 + 6 = 6.5
    """
    p = Problem(name="milp", sense=Sense.MAXIMIZE)
    p.add_variable("x1", lower_bound=0, is_integer=False)
    p.add_variable("x2", lower_bound=0, is_integer=True)
    p.add_constraint({"x1": 1.0, "x2": 1.0}, "<=", 3.5, name="c1")
    p.set_objective({"x1": 1.0, "x2": 2.0})

    solver = BranchAndBoundSolver()
    sol = solver.solve(p)

    assert sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(sol.objective_value, 1e-5) == 6.5
    assert pytest.approx(sol.variable_values["x1"], 1e-5) == 0.5
    assert pytest.approx(sol.variable_values["x2"], 1e-5) == 3.0


def test_binary_knapsack():
    """
    0-1 Knapsack problem:
    Items:
      1: value 60, weight 10
      2: value 100, weight 20
      3: value 120, weight 30
    Capacity: 50
    Optimal choice: items 2 and 3 (weight 50, value 220)
    """
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    total_value, selected_items = solve_knapsack_01(values, weights, capacity)

    assert total_value == 220
    assert sorted(selected_items) == [1, 2]
