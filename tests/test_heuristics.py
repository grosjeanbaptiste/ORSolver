import pytest
import numpy as np
from orsolver.heuristics import nearest_neighbor_tsp, two_opt_tsp, simulated_annealing_tsp


def test_tsp_nearest_neighbor_and_2opt():
    """
    4 cities on a square of side 10:
    (0, 0), (10, 0), (10, 10), (0, 10).
    Optimal round trip: 10 + 10 + 10 + 10 = 40.
    """
    cities = [
        (0.0, 0.0),    # City 0
        (10.0, 0.0),   # City 1
        (10.0, 10.0),  # City 2
        (0.0, 10.0),   # City 3
    ]
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))

    # Test initial tour
    init_tour, init_cost = nearest_neighbor_tsp(dist_matrix, start_city=0)
    assert len(set(init_tour)) == n
    assert init_cost == 40.0

    # Cross tour: [0, 2, 1, 3] which has crossed edges (suboptimal)
    suboptimal_tour = [0, 2, 1, 3]
    opt_tour, opt_cost = two_opt_tsp(dist_matrix, initial_tour=suboptimal_tour)
    assert opt_cost == 40.0
    assert len(set(opt_tour)) == n


def test_tsp_simulated_annealing():
    """
    5 cities in a line: (0, 0), (2, 0), (4, 0), (6, 0), (8, 0).
    Optimal round trip: 2+2+2+2+8 = 16.
    """
    cities = [(0, 0), (2, 0), (4, 0), (6, 0), (8, 0)]
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = abs(cities[i][0] - cities[j][0])

    tour, cost = simulated_annealing_tsp(dist_matrix, initial_temp=100.0, cooling_rate=0.95, max_iter=500)
    assert cost == 16.0
    assert len(set(tour)) == n
