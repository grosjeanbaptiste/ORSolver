from typing import List, Tuple


def solve_knapsack_01(
    values: List[float],
    weights: List[int],
    capacity: int,
) -> Tuple[float, List[int]]:
    """
    Solves the 0-1 Knapsack problem using Dynamic Programming.
    Returns:
      (total_value, selected_item_indices_0_based)
    """
    n = len(values)
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        v = values[i - 1]
        w = int(weights[i - 1])
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if c >= w:
                candidate = dp[i - 1][c - w] + v
                if candidate > dp[i][c]:
                    dp[i][c] = candidate

    # Backtrack to find selected items
    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if abs(dp[i][c] - dp[i - 1][c]) > 1e-9:
            selected.append(i - 1)
            c -= int(weights[i - 1])

    return dp[n][capacity], sorted(selected)
