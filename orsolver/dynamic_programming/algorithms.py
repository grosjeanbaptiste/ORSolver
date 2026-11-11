from typing import Dict, List, Tuple


def matrix_chain_multiplication(dimensions: List[int]) -> Tuple[int, str]:
    """
    Solves the Matrix Chain Multiplication problem using dynamic programming.
    Dimensions array p of length n+1 where matrix A_i has dimension p[i-1] x p[i].
    Returns:
      (min_scalar_multiplications, optimal_parenthesization_string)
    """
    n = len(dimensions) - 1
    # m[i][j] = minimum number of scalar multiplications needed to compute A[i..j]
    m = [[0] * (n + 1) for _ in range(n + 1)]
    # s[i][j] = index of the matrix after which the optimal split occurs
    s = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            m[i][j] = float("inf")
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + dimensions[i - 1] * dimensions[k] * dimensions[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    def _build_parens(i: int, j: int) -> str:
        if i == j:
            return f"A{i}"
        k = s[i][j]
        left = _build_parens(i, k)
        right = _build_parens(k + 1, j)
        return f"({left} {right})"

    return int(m[1][n]), _build_parens(1, n)


def coin_change(coins: List[int], amount: int) -> Tuple[int, Dict[int, int]]:
    """
    Solves the Coin Change Problem using dynamic programming.
    Finds the minimum number of coins needed to make up the given amount.
    Returns:
      (min_coins, {coin_value: count_used})
    """
    dp = [float("inf")] * (amount + 1)
    coin_used = [-1] * (amount + 1)
    dp[0] = 0

    for a in range(1, amount + 1):
        for c in coins:
            if a >= c and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
                coin_used[a] = c

    if dp[amount] == float("inf"):
        raise ValueError(f"Amount {amount} cannot be formed with the given coins.")

    # Reconstruct coin frequencies
    used_dict: Dict[int, int] = {}
    curr = amount
    while curr > 0:
        c = coin_used[curr]
        used_dict[c] = used_dict.get(c, 0) + 1
        curr -= c

    return int(dp[amount]), used_dict
