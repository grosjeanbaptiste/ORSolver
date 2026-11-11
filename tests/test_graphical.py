import os
import pytest
from orsolver.core.model import Problem, Sense
from orsolver.visualizer.graphical_2d import DynamicGraphicalSolver


def test_graphical_solver_vertices_and_optimum():
    """
    Max Z = 40x1 + 25x2
    s.t.
      4x1 + 3x2 <= 20
      2x1 - x2 <= 0  ->  x2 >= 2x1
      x1, x2 >= 0
    Vertices of feasible region:
      (0, 0) -> Z = 0
      Intersection of 2x1 - x2 = 0 with 4x1 + 3x2 = 20:
        x2 = 2x1 -> 4x1 + 6x1 = 20 -> 10x1 = 20 -> x1 = 2, x2 = 4 -> Z = 40(2) + 25(4) = 180
      Intersection of 4x1 + 3x2 = 20 with x1 = 0:
        3x2 = 20 -> x2 = 20/3 = 6.6667 -> Z = 25 * 20/3 = 166.6667
    Optimal is at (2, 4) with Z = 180!
    """
    p = Problem(name="graphical_test", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 4.0, "x2": 3.0}, "<=", 20.0, name="c1")
    p.add_constraint({"x1": 2.0, "x2": -1.0}, "<=", 0.0, name="c2")
    p.set_objective({"x1": 40.0, "x2": 25.0})

    solver = DynamicGraphicalSolver()
    result = solver.solve(p)

    assert result["is_feasible"]
    assert pytest.approx(result["optimal_value"], 1e-4) == 180.0
    assert pytest.approx(result["optimal_point"][0], 1e-4) == 2.0
    assert pytest.approx(result["optimal_point"][1], 1e-4) == 4.0

    # Ensure feasible vertices include (0, 0), (2, 4), (0, 20/3)
    pts = result["feasible_vertices"]
    assert len(pts) >= 3


def test_graphical_solver_save_figure(tmp_path):
    p = Problem(name="plot_test", sense=Sense.MAXIMIZE)
    p.add_constraint({"x1": 1.0, "x2": 1.0}, "<=", 5.0, name="c1")
    p.set_objective({"x1": 2.0, "x2": 3.0})

    solver = DynamicGraphicalSolver()
    fig_path = os.path.join(tmp_path, "solution_plot.png")
    solver.plot_and_save(p, filepath=fig_path)

    assert os.path.exists(fig_path)
    assert os.path.getsize(fig_path) > 0
