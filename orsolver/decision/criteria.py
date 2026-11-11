from dataclasses import dataclass
from typing import List, Union
import numpy as np


@dataclass
class UncertaintyResult:
    maximin_action: int
    maximin_value: float
    maximax_action: int
    maximax_value: float
    hurwicz_action: int
    hurwicz_value: float
    laplace_action: int
    laplace_value: float
    savage_action: int
    savage_min_max_regret: float
    regret_matrix: List[List[float]]


@dataclass
class RiskResult:
    best_action: int
    best_emv: float
    emv_values: List[float]
    expected_payoff_certainty: float
    evpi: float


def decision_under_uncertainty(
    payoff_matrix: Union[List[List[float]], np.ndarray],
    hurwicz_alpha: float = 0.5,
) -> UncertaintyResult:
    """
    Evaluates actions under complete uncertainty (no probabilities):
      - Maximin (Wald)
      - Maximax
      - Hurwicz (alpha)
      - Laplace (equal likelihood)
      - Savage (Minimax Regret)
    """
    P = np.array(payoff_matrix, dtype=float)
    m, n = P.shape

    # 1. Maximin (Wald)
    row_mins = np.min(P, axis=1)
    maximin_idx = int(np.argmax(row_mins))
    maximin_val = float(row_mins[maximin_idx])

    # 2. Maximax
    row_maxs = np.max(P, axis=1)
    maximax_idx = int(np.argmax(row_maxs))
    maximax_val = float(row_maxs[maximax_idx])

    # 3. Hurwicz
    hurwicz_vals = hurwicz_alpha * row_maxs + (1.0 - hurwicz_alpha) * row_mins
    hurwicz_idx = int(np.argmax(hurwicz_vals))
    hurwicz_val = float(hurwicz_vals[hurwicz_idx])

    # 4. Laplace
    laplace_vals = np.mean(P, axis=1)
    laplace_idx = int(np.argmax(laplace_vals))
    laplace_val = float(laplace_vals[laplace_idx])

    # 5. Savage Minimax Regret
    col_maxs = np.max(P, axis=0)
    R = col_maxs - P  # Regret matrix >= 0
    max_regrets = np.max(R, axis=1)
    savage_idx = int(np.argmin(max_regrets))
    savage_val = float(max_regrets[savage_idx])

    return UncertaintyResult(
        maximin_action=maximin_idx,
        maximin_value=maximin_val,
        maximax_action=maximax_idx,
        maximax_value=maximax_val,
        hurwicz_action=hurwicz_idx,
        hurwicz_value=hurwicz_val,
        laplace_action=laplace_idx,
        laplace_value=laplace_val,
        savage_action=savage_idx,
        savage_min_max_regret=savage_val,
        regret_matrix=R.tolist(),
    )


def decision_under_risk(
    payoff_matrix: Union[List[List[float]], np.ndarray],
    probabilities: List[float],
) -> RiskResult:
    """
    Evaluates actions under risk (given probabilities for states of nature):
      - Expected Monetary Value (EMV)
      - Expected Payoff with Perfect Information (EPC)
      - Expected Value of Perfect Information (EVPI = EPC - max EMV)
    """
    P = np.array(payoff_matrix, dtype=float)
    probs = np.array(probabilities, dtype=float)
    probs = probs / np.sum(probs)

    emv = np.dot(P, probs)
    best_action = int(np.argmax(emv))
    best_emv = float(emv[best_action])

    # Expected payoff under certainty
    best_per_state = np.max(P, axis=0)
    epc = float(np.dot(best_per_state, probs))
    evpi = float(epc - best_emv)

    return RiskResult(
        best_action=best_action,
        best_emv=best_emv,
        emv_values=[float(v) for v in emv],
        expected_payoff_certainty=epc,
        evpi=evpi,
    )
