import heapq
from typing import Dict, List, Set, Tuple


class DisjointSet:
    def __init__(self, elements: List[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1: str, set2: str) -> bool:
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 == root2:
            return False
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
        return True


def kruskal(
    nodes: List[str],
    edges: List[Tuple[str, str, float]],
) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Kruskal's algorithm for Minimum Spanning Tree.
    Returns:
      (total_cost, mst_edges)
    """
    ds = DisjointSet(nodes)
    sorted_edges = sorted(edges, key=lambda x: x[2])
    mst_edges = []
    total_cost = 0.0

    for u, v, w in sorted_edges:
        if ds.union(u, v):
            mst_edges.append((u, v, w))
            total_cost += w
            if len(mst_edges) == len(nodes) - 1:
                break

    return total_cost, mst_edges


def prim(
    nodes: List[str],
    edges: List[Tuple[str, str, float]],
) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Prim's algorithm for Minimum Spanning Tree.
    Returns:
      (total_cost, mst_edges)
    """
    if not nodes:
        return 0.0, []

    adj: Dict[str, List[Tuple[float, str]]] = {n: [] for n in nodes}
    for u, v, w in edges:
        adj[u].append((w, v))
        adj[v].append((w, u))

    visited: Set[str] = {nodes[0]}
    pq: List[Tuple[float, str, str]] = []
    for w, neighbor in adj[nodes[0]]:
        heapq.heappush(pq, (w, nodes[0], neighbor))

    mst_edges = []
    total_cost = 0.0

    while pq and len(visited) < len(nodes):
        w, u, v = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total_cost += w
        for next_w, neighbor in adj[v]:
            if neighbor not in visited:
                heapq.heappush(pq, (next_w, v, neighbor))

    return total_cost, mst_edges
