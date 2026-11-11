import pytest
from orsolver.network.shortest_path import dijkstra, bellman_ford, floyd_warshall
from orsolver.network.max_flow import edmonds_karp
from orsolver.network.mst import kruskal, prim
from orsolver.network.pert_cpm import compute_cpm, Task


def test_shortest_path_dijkstra():
    # Graph: (u, v, weight)
    graph = {
        "A": {"B": 1, "C": 4},
        "B": {"C": 2, "D": 5},
        "C": {"D": 1},
        "D": {}
    }
    dist, path = dijkstra(graph, "A", "D")
    assert dist == 4  # A -> B (1) -> C (2) -> D (1) = 4
    assert path == ["A", "B", "C", "D"]


def test_shortest_path_bellman_ford_with_negative_weights():
    # Graph with negative edge weights (no negative cycles)
    edges = [
        ("S", "A", 4),
        ("S", "B", 5),
        ("A", "C", -2),
        ("B", "C", 1),
        ("C", "D", 3),
    ]
    dist, path = bellman_ford(edges, nodes=["S", "A", "B", "C", "D"], source="S", target="D")
    # S -> A (4) -> C (-2) -> D (3) = 5
    assert dist == 5
    assert path == ["S", "A", "C", "D"]


def test_floyd_warshall_all_pairs():
    nodes = ["1", "2", "3", "4"]
    matrix = {
        "1": {"2": 3, "4": 7},
        "2": {"1": 8, "3": 2},
        "3": {"1": 5, "4": 1},
        "4": {"1": 2}
    }
    dist_matrix = floyd_warshall(nodes, matrix)
    assert dist_matrix["1"]["3"] == 5  # 1 -> 2 (3) + 2 -> 3 (2) = 5
    assert dist_matrix["1"]["4"] == 6  # 1 -> 2 (3) + 2 -> 3 (2) + 3 -> 4 (1) = 6


def test_maximum_flow_edmonds_karp():
    """
    Classic flow network:
    Source: s, Sink: t
    Edges with capacities:
      s -> u: 10, s -> v: 10
      u -> v: 2, u -> t: 4
      v -> t: 9
    Max flow is 13:
      path s -> u -> t: 4
      path s -> u -> v -> t: 2
      path s -> v -> t: 7
      Total = 13
    """
    capacities = {
        "s": {"u": 10, "v": 10},
        "u": {"v": 2, "t": 4},
        "v": {"t": 9},
        "t": {}
    }
    max_flow, flow_dict = edmonds_karp(capacities, source="s", sink="t")
    assert max_flow == 13


def test_minimum_spanning_tree():
    edges = [
        ("A", "B", 7),
        ("A", "D", 5),
        ("B", "C", 8),
        ("B", "D", 9),
        ("B", "E", 7),
        ("C", "E", 5),
        ("D", "E", 15),
        ("D", "F", 6),
        ("E", "F", 8),
        ("E", "G", 9),
        ("F", "G", 11),
    ]
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    cost_kruskal, tree_kruskal = kruskal(nodes, edges)
    cost_prim, tree_prim = prim(nodes, edges)

    assert cost_kruskal == 39
    assert cost_prim == 39


def test_pert_cpm():
    """
    Project scheduling:
    Tasks:
      A: duration 3, deps []
      B: duration 4, deps []
      C: duration 2, deps [A]
      D: duration 5, deps [B]
      E: duration 3, deps [C, D]
    Paths:
      A (3) -> C (2) -> E (3) = 8
      B (4) -> D (5) -> E (3) = 12 (Critical Path!)
    """
    tasks = [
        Task(name="A", duration=3, predecessors=[]),
        Task(name="B", duration=4, predecessors=[]),
        Task(name="C", duration=2, predecessors=["A"]),
        Task(name="D", duration=5, predecessors=["B"]),
        Task(name="E", duration=3, predecessors=["C", "D"]),
    ]

    result = compute_cpm(tasks)

    assert result.project_duration == 12
    assert result.critical_path == ["B", "D", "E"]
    assert result.schedule["B"]["slack"] == 0
    assert result.schedule["D"]["slack"] == 0
    assert result.schedule["E"]["slack"] == 0
    assert result.schedule["A"]["slack"] == 4  # ES=0, LS=4 -> slack=4
