import math
import random
from typing import List, Optional, Tuple, Union
import numpy as np


def compute_tour_cost(tour: List[int], dist_matrix: np.ndarray) -> float:
    """Calculates the round-trip distance of a tour."""
    n = len(tour)
    cost = 0.0
    for i in range(n - 1):
        cost += dist_matrix[tour[i], tour[i + 1]]
    cost += dist_matrix[tour[-1], tour[0]]
    return float(cost)


def nearest_neighbor_tsp(
    dist_matrix: Union[List[List[float]], np.ndarray],
    start_city: int = 0,
) -> Tuple[List[int], float]:
    """
    Constructs a tour using the Nearest Neighbor greedy heuristic.
    """
    D = np.array(dist_matrix, dtype=float)
    n = D.shape[0]

    visited = [False] * n
    tour = [start_city]
    visited[start_city] = True

    curr = start_city
    for _ in range(n - 1):
        next_city = -1
        min_dist = float("inf")
        for j in range(n):
            if not visited[j] and D[curr, j] < min_dist:
                min_dist = D[curr, j]
                next_city = j
        visited[next_city] = True
        tour.append(next_city)
        curr = next_city

    return tour, compute_tour_cost(tour, D)


def two_opt_tsp(
    dist_matrix: Union[List[List[float]], np.ndarray],
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 1000,
) -> Tuple[List[int], float]:
    """
    Improves an initial tour using the 2-Opt local search heuristic.
    Reverses subsegments when an improvement is found until a 2-optimal tour is reached.
    """
    D = np.array(dist_matrix, dtype=float)
    n = D.shape[0]

    if initial_tour is None:
        tour, _ = nearest_neighbor_tsp(D, start_city=0)
    else:
        tour = list(initial_tour)

    best_cost = compute_tour_cost(tour, D)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                # 2-opt swap: reverse tour[i+1 : j+1]
                new_tour = tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]
                new_cost = compute_tour_cost(new_tour, D)
                if new_cost < best_cost - 1e-9:
                    tour = new_tour
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break

    return tour, best_cost


def simulated_annealing_tsp(
    dist_matrix: Union[List[List[float]], np.ndarray],
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    max_iter: int = 1000,
    seed: Optional[int] = 42,
) -> Tuple[List[int], float]:
    """
    Solves TSP using the Simulated Annealing metaheuristic.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    D = np.array(dist_matrix, dtype=float)
    n = D.shape[0]

    current_tour, current_cost = nearest_neighbor_tsp(D, start_city=0)
    best_tour = list(current_tour)
    best_cost = current_cost

    temp = float(initial_temp)

    for _ in range(max_iter):
        if temp < 1e-4:
            break

        # Generate neighbor by 2-opt segment reversal
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)

        candidate_tour = current_tour[:i] + current_tour[i : j + 1][::-1] + current_tour[j + 1 :]
        candidate_cost = compute_tour_cost(candidate_tour, D)

        delta = candidate_cost - current_cost

        # Metropolis acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = candidate_tour
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = list(current_tour)

        temp *= cooling_rate

    # Final local polishing with 2-opt
    best_tour, best_cost = two_opt_tsp(D, initial_tour=best_tour)
    return best_tour, best_cost
