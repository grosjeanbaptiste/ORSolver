import pytest
from orsolver.core.model import Problem, Sense
from orsolver.core.solution import SolutionStatus
from orsolver.lp.duality import build_dual_problem
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def test_strong_duality_max_problem():
    """
    Primal:
    Max Z = 3x1 + 5x2
    s.t.
      x1 <= 4
      2x2 <= 12
      3x1 + 2x2 <= 18
      x1, x2 >= 0
    Dual:
    Min W = 4y1 + 12y2 + 18y3
    s.t.
      y1 + 3y3 >= 3
      2y2 + 2y3 >= 5
      y1, y2, y3 >= 0
    Both must have optimal value 36.0!
    """
    primal = Problem(name="primal_prod", sense=Sense.MAXIMIZE)
    primal.add_constraint({"x1": 1.0}, "<=", 4.0, name="c1")
    primal.add_constraint({"x2": 2.0}, "<=", 12.0, name="c2")
    primal.add_constraint({"x1": 3.0, "x2": 2.0}, "<=", 18.0, name="c3")
    primal.set_objective({"x1": 3.0, "x2": 5.0})

    dual = build_dual_problem(primal)

    assert dual.sense == Sense.MINIMIZE
    assert len(dual.variables) == 3
    assert len(dual.constraints) == 2

    solver = TwoPhaseSimplexSolver()
    p_sol = solver.solve(primal)
    d_sol = solver.solve(dual)

    assert p_sol.status == SolutionStatus.OPTIMAL
    assert d_sol.status == SolutionStatus.OPTIMAL
    assert pytest.approx(p_sol.objective_value, 1e-5) == 36.0
    assert pytest.approx(d_sol.objective_value, 1e-5) == 36.0

    # Primal shadow prices match dual variable values!
    for c_name, sp in p_sol.shadow_prices.items():
        dual_var_name = f"y_{c_name}"
        assert pytest.approx(d_sol.variable_values[dual_var_name], 1e-5) == sp
