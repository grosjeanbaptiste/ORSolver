import heapq
from typing import Any, Callable, Dict, List, Optional, Tuple


def a_star_search(
    start: Any,
    is_goal: Callable[[Any], bool],
    neighbors_fn: Callable[[Any], List[Tuple[Any, float]]],
    heuristic: Callable[[Any], float],
) -> Tuple[List[Any], float, int]:
    """
    A* Shortest Path Search Algorithm with heuristic.

    Args:
      start: The initial state/node.
      is_goal: Callable returning True if a state is the goal.
      neighbors_fn: Callable returning list of (neighbor_state, edge_cost) for a state.
      heuristic: Admissible heuristic function h(state) returning estimated cost to goal.

    Returns:
      (optimal_path, total_cost, nodes_expanded)
    """
    counter = 0
    # Priority queue: (f_score, counter, state)
    open_set: List[Tuple[float, int, Any]] = []
    heapq.heappush(open_set, (heuristic(start), counter, start))

    came_from: Dict[Any, Any] = {}
    g_score: Dict[Any, float] = {start: 0.0}
    closed_set = set()
    nodes_expanded = 0

    while open_set:
        f_curr, _, current = heapq.heappop(open_set)

        if is_goal(current):
            # Reconstruct path
            path = [current]
            curr = current
            while curr in came_from:
                curr = came_from[curr]
                path.append(curr)
            path.reverse()
            return path, g_score[current], nodes_expanded

        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_expanded += 1

        for neighbor, edge_cost in neighbors_fn(current):
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    return [], float("inf"), nodes_expanded
