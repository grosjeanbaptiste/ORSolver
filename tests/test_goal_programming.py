import pytest
from orsolver.lp.goal_programming import PreemptiveGoalProgramming, Goal


def test_lexicographic_goal_programming():
    """
    Production planning with 2 goals:
    Priority 1: Achieve profit of at least 80 (minimize d1_minus for 40x1 + 30x2 + d1_minus - d1_plus = 80).
    Priority 2: Overtime penalty (labor: 2x1 + x2 <= 3; minimize d2_plus for 2x1 + x2 + d2_minus - d2_plus = 3).
    Decision vars: x1, x2 >= 0.
    """
    gp = PreemptiveGoalProgramming()
    gp.add_variable("x1", lower_bound=0.0)
    gp.add_variable("x2", lower_bound=0.0)

    # Hard constraint: machine capacity x1 + x2 <= 3
    gp.add_hard_constraint({"x1": 1.0, "x2": 1.0}, "<=", 3.0, name="machine")

    # Priority 1: 40x1 + 30x2 >= 80 (under-achievement d_minus penalized)
    gp.add_goal(
        coefficients={"x1": 40.0, "x2": 30.0},
        target=80.0,
        priority=1,
        minimize_deviation="under",  # minimize d_minus
        name="profit_goal"
    )

    # Priority 2: 2x1 + x2 <= 3 (overtime d_plus penalized)
    gp.add_goal(
        coefficients={"x1": 2.0, "x2": 1.0},
        target=3.0,
        priority=2,
        minimize_deviation="over",  # minimize d_plus
        name="labor_goal"
    )

    solution = gp.solve()

    assert solution.is_optimal
    # Check that profit is at least 80 (e.g. x1=0, x2=3 -> 90 >= 80)
    profit = 40.0 * solution.variable_values["x1"] + 30.0 * solution.variable_values["x2"]
    assert profit >= 80.0 - 1e-4
    # Labor used:
    labor = 2.0 * solution.variable_values["x1"] + 1.0 * solution.variable_values["x2"]
    assert labor <= 4.0
