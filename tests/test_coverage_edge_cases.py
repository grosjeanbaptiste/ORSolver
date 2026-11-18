import pytest
import numpy as np

from orsolver.core.model import Problem, Constraint, Sense
from orsolver.core.solution import Solution, SolutionStatus
from orsolver.transport.transportation import least_cost_method, solve_transportation
from orsolver.visualizer.graphical_2d import DynamicGraphicalSolver
from orsolver.cli.main import main


def test_least_cost_method():
    costs = [
        [2, 3, 1],
        [5, 4, 8],
    ]
    supply = [20, 30]
    demand = [10, 20, 20]

    alloc = least_cost_method(costs, supply, demand)
    assert np.all(np.sum(alloc, axis=1) == supply)
    assert np.all(np.sum(alloc, axis=0) == demand)


def test_transport_demand_exceeds_supply():
    costs = [[2, 4], [3, 1]]
    supply = [10, 20]
    demand = [15, 25]  # demand (40) > supply (30) -> dummy row needed

    cost, alloc = solve_transportation(costs, supply, demand)
    assert alloc.shape == (2, 2)
    assert cost > 0


def test_model_edge_cases():
    # Unsupported constraint sense
    with pytest.raises(ValueError, match="Unsupported constraint sense"):
        Constraint(coefficients={"x": 1.0}, sense="invalid", rhs=5.0, name="c_err")

    p = Problem()
    # Binary variable
    v_bin = p.add_variable("b1", is_binary=True)
    assert v_bin.is_binary
    assert v_bin.is_integer
    assert v_bin.lower_bound == 0.0
    assert v_bin.upper_bound == 1.0

    # Duplicate variable
    with pytest.raises(ValueError, match="already exists"):
        p.add_variable("b1")


def test_graphical_solver_edge_cases(tmp_path):
    solver = DynamicGraphicalSolver()

    # Not 2 variables
    p_3d = Problem()
    p_3d.add_variable("x1")
    p_3d.add_variable("x2")
    p_3d.add_variable("x3")
    p_3d.set_objective({"x1": 1.0, "x2": 1.0, "x3": 1.0})
    with pytest.raises(ValueError, match="only supports 2D"):
        solver.solve(p_3d)

    # Infeasible plot
    p_inf = Problem()
    p_inf.add_variable("x1", lower_bound=0)
    p_inf.add_variable("x2", lower_bound=0)
    p_inf.add_constraint({"x1": 1.0}, "<=", 2.0)
    p_inf.add_constraint({"x1": 1.0}, ">=", 5.0)
    p_inf.set_objective({"x1": 1.0, "x2": 1.0})

    with pytest.raises(ValueError, match="infeasible"):
        solver.plot_and_save(p_inf, filepath=str(tmp_path / "never.png"))


def test_cli_help(capsys):
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "orsolver" in captured.out


def test_shortest_path_edge_cases():
    from orsolver.network.shortest_path import dijkstra, bellman_ford

    # Dijkstra without target
    graph = {"A": {"B": 1.0}, "B": {}}
    dists, preds = dijkstra(graph, "A")
    assert dists["A"] == 0.0
    assert dists["B"] == 1.0

    # Dijkstra unreachable target
    graph_disc = {"A": {}, "B": {}}
    dist, path = dijkstra(graph_disc, "A", "B")
    assert dist == float("inf")
    assert path == []

    # Bellman-Ford negative cycle
    edges_neg = [("A", "B", 1.0), ("B", "C", -2.0), ("C", "A", -1.0)]
    with pytest.raises(ValueError, match="negative weight cycle"):
        bellman_ford(edges_neg, ["A", "B", "C"], "A")

    # Bellman-Ford without target
    edges_ok = [("A", "B", 2.0), ("B", "C", 3.0)]
    dists_bf, preds_bf = bellman_ford(edges_ok, ["A", "B", "C"], "A")
    assert dists_bf["C"] == 5.0

    # Bellman-Ford unreachable target
    edges_disc = [("A", "B", 1.0)]
    dist_bf_unr, path_bf_unr = bellman_ford(edges_disc, ["A", "B", "C"], "A", "C")
    assert dist_bf_unr == float("inf")
    assert path_bf_unr == []


def test_gomory_edge_cases():
    from orsolver.mip.cutting_plane import GomoryCuttingPlaneSolver

    # Infeasible problem
    p_inf = Problem()
    p_inf.add_variable("x", is_integer=True, lower_bound=0)
    p_inf.add_constraint({"x": 1.0}, "<=", 2.0)
    p_inf.add_constraint({"x": 1.0}, ">=", 5.0)
    p_inf.set_objective({"x": 1.0})
    solver = GomoryCuttingPlaneSolver(max_cuts=5)
    sol = solver.solve(p_inf)
    assert not sol.is_optimal

    # Gomory with >= constraint (surplus variable e_)
    p_surplus = Problem(sense=Sense.MINIMIZE)
    p_surplus.add_variable("x", is_integer=True, lower_bound=0)
    p_surplus.add_variable("y", is_integer=True, lower_bound=0)
    p_surplus.add_constraint({"x": 2.0, "y": 3.0}, ">=", 7.0)
    p_surplus.set_objective({"x": 1.0, "y": 1.0})
    sol_surplus = solver.solve(p_surplus)
    assert sol_surplus.is_optimal
    assert sol_surplus.variable_values["x"].is_integer()
    assert sol_surplus.variable_values["y"].is_integer()

