import pytest
from SimplexSolver import SimplexSolver
from GraphicalORSolver import GraphicalORSolver


def test_legacy_simplex_solver():
    constraints = [
        ([4, 3], "<=", 20),
        ([2, -1], "<=", 0),
    ]
    objective = [40, 25]

    solver = SimplexSolver(objective, constraints)
    result = solver.solve()

    assert "solution" in result
    assert "objective_value" in result
    assert pytest.approx(result["objective_value"], 1e-4) == 180.0
    assert pytest.approx(result["solution"][0], 1e-4) == 2.0
    assert pytest.approx(result["solution"][1], 1e-4) == 4.0


def test_legacy_graphical_solver():
    constraints = [
        ([4, 3], "<=", 20),
        ([2, -1], "<=", 0),
    ]
    objective = [40, 25]

    visualizer = GraphicalORSolver(constraints, objective=objective)
    result = visualizer.calculate_graphical_solution()

    assert "solution" in result
    assert "objective_value" in result
    assert pytest.approx(result["objective_value"], 1e-4) == 180.0
    assert pytest.approx(result["solution"][0], 1e-4) == 2.0
    assert pytest.approx(result["solution"][1], 1e-4) == 4.0
