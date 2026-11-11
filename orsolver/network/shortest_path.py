import heapq
from typing import Dict, List, Optional, Tuple, Any


def dijkstra(
    graph: Dict[str, Dict[str, float]],
    source: str,
    target: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Dijkstra's shortest path algorithm for graphs with non-negative edge weights.
    Returns:
      If target is given: (distance, path)
      If target is None: (distances_dict, predecessors_dict)
    """
    distances = {node: float("inf") for node in graph}
    distances[source] = 0.0
    predecessors: Dict[str, Optional[str]] = {node: None for node in graph}

    # Priority queue: (dist, node)
    pq = [(0.0, source)]

    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > distances[u]:
            continue

        if target is not None and u == target:
            break

        for v, weight in graph.get(u, {}).items():
            if v not in distances:
                distances[v] = float("inf")
                predecessors[v] = None
            new_dist = curr_dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq, (new_dist, v))

    if target is not None:
        if distances.get(target, float("inf")) == float("inf"):
            return float("inf"), []
        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = predecessors.get(curr)
        path.reverse()
        return distances[target], path

    return distances, predecessors


def bellman_ford(
    edges: List[Tuple[str, str, float]],
    nodes: List[str],
    source: str,
    target: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Bellman-Ford shortest path algorithm. Handles negative weights and detects negative cycles.
    """
    distances = {n: float("inf") for n in nodes}
    distances[source] = 0.0
    predecessors: Dict[str, Optional[str]] = {n: None for n in nodes}

    # Relax edges |V| - 1 times
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u

    # Check for negative cycles
    for u, v, w in edges:
        if distances[u] + w < distances[v]:
            raise ValueError("Graph contains a negative weight cycle!")

    if target is not None:
        if distances[target] == float("inf"):
            return float("inf"), []
        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = predecessors[curr]
        path.reverse()
        return distances[target], path

    return distances, predecessors


def floyd_warshall(
    nodes: List[str],
    graph: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.
    """
    dist: Dict[str, Dict[str, float]] = {
        u: {v: float("inf") for v in nodes} for u in nodes
    }

    for u in nodes:
        dist[u][u] = 0.0
        for v, w in graph.get(u, {}).items():
            if v in dist[u]:
                dist[u][v] = float(w)

    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist
