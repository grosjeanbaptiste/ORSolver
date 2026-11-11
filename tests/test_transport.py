import pytest
import numpy as np
from orsolver.transport.transportation import (
    solve_transportation,
    north_west_corner,
    least_cost_method,
    vogel_approximation_method,
)
from orsolver.transport.assignment import solve_assignment


def test_transportation_vam_and_modi():
    """
    Classic 3 sources, 4 destinations transportation problem:
    Cost matrix:
      [ [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40,  8, 70, 20] ]
    Supply: [7, 9, 18] (Total: 34)
    Demand: [5, 8, 7, 14] (Total: 34)
    Optimal cost: 743 (or standard textbook result)
    """
    costs = [
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40,  8, 70, 20]
    ]
    supply = [7, 9, 18]
    demand = [5, 8, 7, 14]

    # Test initial BFS methods
    alloc_nw = north_west_corner(supply, demand)
    assert np.all(np.sum(alloc_nw, axis=1) == supply)
    assert np.all(np.sum(alloc_nw, axis=0) == demand)

    alloc_vam = vogel_approximation_method(costs, supply, demand)
    assert np.all(np.sum(alloc_vam, axis=1) == supply)
    assert np.all(np.sum(alloc_vam, axis=0) == demand)

    # Test optimal solution via MODI
    optimal_cost, optimal_alloc = solve_transportation(costs, supply, demand, method="vam")
    assert optimal_cost == 743
    assert np.all(np.sum(optimal_alloc, axis=1) == supply)
    assert np.all(np.sum(optimal_alloc, axis=0) == demand)


def test_unbalanced_transportation():
    """When supply > demand, a dummy destination is added with cost 0."""
    costs = [
        [2, 3],
        [5, 4]
    ]
    supply = [30, 40]  # Total 70
    demand = [20, 30]  # Total 50 -> dummy column of 20 needed

    opt_cost, opt_alloc = solve_transportation(costs, supply, demand)
    # Demand satisfied: row 0 sends 20 to col 0, 10 to col 1. row 1 sends 20 to col 1. Total cost: 20*2 + 10*3 + 20*4 = 40+30+80 = 150
    assert opt_cost == 150
    assert opt_alloc[0][0] == 20
    assert opt_alloc[0][1] == 10
    assert opt_alloc[1][1] == 20


def test_hungarian_assignment():
    """
    Assignment problem: 4 agents, 4 tasks
    Cost matrix:
      [ [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4] ]
    Optimal assignment:
      Agent 0 -> Task 1 (cost 2)
      Agent 1 -> Task 0 (cost 6) or Task 2 (cost 3) depending
      Let's check minimum cost:
      Agent 0 -> 1 (2)
      Agent 1 -> 0 (6) [or Agent 2 -> 2 (1)]
      Agent 2 -> 2 (1)
      Agent 3 -> 3 (4)
      Total = 2 + 6 + 1 + 4 = 13.
    """
    cost_matrix = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
    ]

    min_cost, assignment = solve_assignment(cost_matrix)
    assert min_cost == 13
    assert assignment[0] == 1
    assert assignment[1] == 0
    assert assignment[2] == 2
    assert assignment[3] == 3
