import pytest
from orsolver.network.a_star import a_star_search


def test_a_star_grid_with_obstacle():
    """
    2D Grid where each step horizontally/vertically costs 1.
    Start at (0, 0), Target at (3, 3).
    Obstacles at (1, 1), (1, 2), (2, 1).
    Heuristic: Manhattan distance |x1 - x2| + |y1 - y2|.
    """
    def heuristic(node, target):
        return abs(node[0] - target[0]) + abs(node[1] - target[1])

    obstacles = {(1, 1), (1, 2), (2, 1)}

    def neighbors_fn(node):
        x, y = node
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid = []
        for nx, ny in candidates:
            if 0 <= nx <= 3 and 0 <= ny <= 3 and (nx, ny) not in obstacles:
                valid.append(((nx, ny), 1.0))
        return valid

    start = (0, 0)
    target = (3, 3)

    path, cost, nodes_expanded = a_star_search(
        start=start,
        is_goal=lambda n: n == target,
        neighbors_fn=neighbors_fn,
        heuristic=lambda n: heuristic(n, target),
    )

    assert path[0] == start
    assert path[-1] == target
    assert cost == 6.0  # e.g., (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)->(3,3)
    assert len(path) == 7
    assert nodes_expanded > 0


def test_a_star_graph():
    """
    A graph with Euclidean coordinates:
    S (0, 0) -> A (2, 1) cost 2.5
    S (0, 0) -> B (1, 2) cost 3.0
    A (2, 1) -> G (4, 2) cost 2.5
    B (1, 2) -> G (4, 2) cost 4.0
    Optimal path: S -> A -> G with cost 5.0.
    """
    coords = {
        "S": (0.0, 0.0),
        "A": (2.0, 1.0),
        "B": (1.0, 2.0),
        "G": (4.0, 2.0),
    }
    graph = {
        "S": [("A", 2.5), ("B", 3.0)],
        "A": [("G", 2.5)],
        "B": [("G", 4.0)],
        "G": [],
    }

    def heuristic(node):
        x1, y1 = coords[node]
        x2, y2 = coords["G"]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    path, cost, _ = a_star_search(
        start="S",
        is_goal=lambda n: n == "G",
        neighbors_fn=lambda n: graph[n],
        heuristic=heuristic,
    )

    assert path == ["S", "A", "G"]
    assert cost == 5.0
