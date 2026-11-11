import math
from typing import List, Tuple
import numpy as np


def series_reliability(reliabilities: List[float]) -> float:
    """Calculates reliability of a series system: R_sys = prod(R_i)."""
    return float(np.prod(reliabilities))


def parallel_reliability(reliabilities: List[float]) -> float:
    """Calculates reliability of a parallel system: R_sys = 1 - prod(1 - R_i)."""
    unreliabilities = [1.0 - r for r in reliabilities]
    return float(1.0 - np.prod(unreliabilities))


def k_out_of_n_reliability(k: int, n: int, component_reliability: float) -> float:
    """
    Calculates reliability of a k-out-of-n system with identical components.
    R_sys = sum_{i=k}^n binom(n, i) * R^i * (1 - R)^(n - i).
    """
    R = float(component_reliability)
    total_r = 0.0
    for i in range(k, n + 1):
        comb = math.comb(n, i)
        prob = comb * (R ** i) * ((1.0 - R) ** (n - i))
        total_r += prob
    return float(total_r)


def exponential_reliability(rate_lambda: float, t: float) -> Tuple[float, float]:
    """
    Reliability and MTTF for exponentially distributed failure time:
    R(t) = exp(-lambda * t), MTTF = 1 / lambda.
    """
    lam = float(rate_lambda)
    time = float(t)
    r = math.exp(-lam * time)
    mttf = 1.0 / lam
    return float(r), float(mttf)


def weibull_reliability(scale_eta: float, shape_beta: float, t: float) -> float:
    """
    Reliability for Weibull distributed failure time:
    R(t) = exp(-(t / eta)^beta).
    """
    eta = float(scale_eta)
    beta = float(shape_beta)
    time = float(t)
    return float(math.exp(-((time / eta) ** beta)))
