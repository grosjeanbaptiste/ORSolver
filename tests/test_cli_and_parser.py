import os
import pytest
from orsolver.cli.parser import parse_lp_string
from orsolver.core.model import Sense
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def test_parse_lp_string():
    lp_text = """
    Maximize
      obj: 3 x1 + 5 x2
    Subject To
      c1: x1 <= 4
      c2: 2 x2 <= 12
      c3: 3 x1 + 2 x2 <= 18
    End
    """
    problem = parse_lp_string(lp_text)

    assert problem.name == "LP"
    assert problem.sense == Sense.MAXIMIZE
    assert len(problem.variables) == 2
    assert len(problem.constraints) == 3
    assert problem.objective["x1"] == 3.0
    assert problem.objective["x2"] == 5.0

    solver = TwoPhaseSimplexSolver()
    sol = solver.solve(problem)
    assert sol.is_optimal
    assert pytest.approx(sol.objective_value, 1e-4) == 36.0
    assert pytest.approx(sol.variable_values["x1"], 1e-4) == 2.0
    assert pytest.approx(sol.variable_values["x2"], 1e-4) == 6.0
