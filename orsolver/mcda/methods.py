from dataclasses import dataclass
from typing import List, Union
import numpy as np


@dataclass
class AHPResult:
    weights: List[float]
    lambda_max: float
    consistency_index: float
    consistency_ratio: float
    is_consistent: bool


@dataclass
class TOPSISResult:
    scores: List[float]
    ranking: List[int]  # Indices of alternatives sorted best to worst
    ideal_best: List[float]
    ideal_worst: List[float]


# Saaty Random Consistency Index (RI) table for matrix sizes 1 to 10
RANDOM_INDEX = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def solve_ahp(pairwise_matrix: Union[List[List[float]], np.ndarray]) -> AHPResult:
    """
    Analytic Hierarchy Process (AHP) for calculating priority weights from pairwise comparisons.
    Computes lambda_max, Consistency Index (CI), and Consistency Ratio (CR).
    """
    A = np.array(pairwise_matrix, dtype=float)
    n = A.shape[0]

    # Geometric mean method for weights
    geom_mean = np.prod(A, axis=1) ** (1.0 / n)
    weights = geom_mean / np.sum(geom_mean)

    # Principal eigenvalue lambda_max estimation
    Aw = np.dot(A, weights)
    lambda_max = float(np.mean(Aw / weights))

    if n <= 2:
        ci = 0.0
        cr = 0.0
    else:
        ci = (lambda_max - n) / (n - 1)
        ri = RANDOM_INDEX.get(n, 1.49)
        cr = ci / ri if ri > 0 else 0.0

    return AHPResult(
        weights=[float(w) for w in weights],
        lambda_max=lambda_max,
        consistency_index=ci,
        consistency_ratio=cr,
        is_consistent=cr < 0.10,
    )


def solve_topsis(
    decision_matrix: Union[List[List[float]], np.ndarray],
    weights: List[float],
    criteria_types: List[str],  # 'benefit' (max) or 'cost' (min)
) -> TOPSISResult:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).
    """
    X = np.array(decision_matrix, dtype=float)
    m, n = X.shape
    w = np.array(weights, dtype=float) / sum(weights)

    # 1. Vector normalization: r_ij = x_ij / sqrt(sum_k x_kj^2)
    col_norms = np.sqrt(np.sum(X ** 2, axis=0))
    col_norms[col_norms == 0] = 1.0
    R = X / col_norms

    # 2. Weighted normalized matrix
    V = R * w

    # 3. Determine ideal best (A+) and ideal worst (A-)
    ideal_best = np.zeros(n)
    ideal_worst = np.zeros(n)

    for j in range(n):
        c_type = criteria_types[j].lower()
        if c_type in ("benefit", "max", "+"):
            ideal_best[j] = np.max(V[:, j])
            ideal_worst[j] = np.min(V[:, j])
        else:  # cost / min
            ideal_best[j] = np.min(V[:, j])
            ideal_worst[j] = np.max(V[:, j])

    # 4. Separation measures (Euclidean distances)
    d_pos = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))

    # 5. Relative closeness: C_i = d_neg / (d_pos + d_neg)
    total_dist = d_pos + d_neg
    total_dist[total_dist == 0] = 1.0
    scores = d_neg / total_dist

    # Ranking: highest score first
    ranking = list(np.argsort(-scores))

    return TOPSISResult(
        scores=[float(s) for s in scores],
        ranking=[int(idx) for idx in ranking],
        ideal_best=[float(b) for b in ideal_best],
        ideal_worst=[float(w) for w in ideal_worst],
    )
