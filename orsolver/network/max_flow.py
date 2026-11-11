from collections import deque
from typing import Dict, Tuple


def edmonds_karp(
    capacities: Dict[str, Dict[str, float]],
    source: str,
    sink: str,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Edmonds-Karp algorithm (BFS-based Ford-Fulkerson) for Maximum Flow.
    Returns:
      (max_flow_value, flow_dictionary)
    """
    # Collect all unique nodes
    nodes = set(capacities.keys())
    for u in capacities:
        for v in capacities[u]:
            nodes.add(v)

    # Initialize residual capacity graph
    residual: Dict[str, Dict[str, float]] = {u: {} for u in nodes}
    for u in nodes:
        for v in nodes:
            residual[u][v] = 0.0

    for u in capacities:
        for v, cap in capacities[u].items():
            residual[u][v] = float(cap)

    flow: Dict[str, Dict[str, float]] = {u: {v: 0.0 for v in nodes} for u in nodes}
    total_flow = 0.0

    while True:
        # Find augmenting path using BFS
        parent: Dict[str, str] = {}
        queue = deque([source])
        visited = {source}

        while queue and sink not in visited:
            curr = queue.popleft()
            for neighbor in nodes:
                if neighbor not in visited and residual[curr][neighbor] > 1e-9:
                    visited.add(neighbor)
                    parent[neighbor] = curr
                    queue.append(neighbor)

        if sink not in visited:
            break

        # Find bottleneck capacity along path
        path_flow = float("inf")
        curr = sink
        while curr != source:
            prev = parent[curr]
            path_flow = min(path_flow, residual[prev][curr])
            curr = prev

        # Update residual capacities and flow
        curr = sink
        while curr != source:
            prev = parent[curr]
            residual[prev][curr] -= path_flow
            residual[curr][prev] += path_flow
            flow[prev][curr] += path_flow
            flow[curr][prev] -= path_flow
            curr = prev

        total_flow += path_flow

    return total_flow, flow
