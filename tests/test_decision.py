import pytest
from orsolver.decision import decision_under_uncertainty, decision_under_risk


def test_decision_under_uncertainty():
    """
    Payoff table: 3 actions (A1, A2, A3), 3 states of nature (S1, S2, S3):
          S1   S2   S3
      A1  10   20   30   -> Min: 10, Max: 30, Mean: 20
      A2   5   40   15   -> Min:  5, Max: 40, Mean: 20
      A3  15   15   15   -> Min: 15, Max: 15, Mean: 15

    - Maximin (Wald): A3 has highest minimum (15)
    - Maximax: A2 has highest maximum (40)
    - Hurwicz (alpha=0.6):
        A1: 0.6*30 + 0.4*10 = 18 + 4 = 22
        A2: 0.6*40 + 0.4*5  = 24 + 2 = 26 -> Best!
        A3: 0.6*15 + 0.4*15 = 15
    - Laplace: A1 and A2 tie at 20.0
    - Savage Regret:
        Max per state: S1=15, S2=40, S3=30
        Regret table:
          A1: [15-10, 40-20, 30-30] = [ 5, 20,  0] -> Max regret = 20
          A2: [15-5,  40-40, 30-15] = [10,  0, 15] -> Max regret = 15 -> Best!
          A3: [15-15, 40-15, 30-15] = [ 0, 25, 15] -> Max regret = 25
    """
    payoffs = [
        [10.0, 20.0, 30.0],
        [5.0, 40.0, 15.0],
        [15.0, 15.0, 15.0],
    ]
    res = decision_under_uncertainty(payoffs, hurwicz_alpha=0.6)

    assert res.maximin_action == 2  # A3 (0-based)
    assert res.maximax_action == 1  # A2
    assert res.hurwicz_action == 1  # A2
    assert res.savage_action == 1   # A2
    assert res.savage_min_max_regret == 15.0


def test_decision_under_risk():
    """
    Payoffs:
      A1: [10, 20, 30]
      A2: [ 5, 40, 15]
    Probabilities: [0.3, 0.5, 0.2]
    EMV(A1) = 0.3*10 + 0.5*20 + 0.2*30 = 3 + 10 + 6 = 19
    EMV(A2) = 0.3*5  + 0.5*40 + 0.2*15 = 1.5 + 20 + 3 = 24.5 -> Best EMV = 24.5 (A2)
    EPC (certainty):
      S1: max(10, 5) = 10 -> 0.3 * 10 = 3
      S2: max(20, 40) = 40 -> 0.5 * 40 = 20
      S3: max(30, 15) = 30 -> 0.2 * 30 = 6
      EPC = 3 + 20 + 6 = 29
    EVPI = EPC - EMV* = 29 - 24.5 = 4.5
    """
    payoffs = [
        [10.0, 20.0, 30.0],
        [5.0, 40.0, 15.0],
    ]
    probs = [0.3, 0.5, 0.2]

    res = decision_under_risk(payoffs, probs)

    assert res.best_action == 1
    assert pytest.approx(res.emv_values[0], 1e-4) == 19.0
    assert pytest.approx(res.emv_values[1], 1e-4) == 24.5
    assert pytest.approx(res.evpi, 1e-4) == 4.5
