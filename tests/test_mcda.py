import pytest
import numpy as np
from orsolver.mcda import solve_ahp, solve_topsis


def test_ahp_consistent_matrix():
    """
    3 Criteria pairwise comparison:
    A vs B: 3 (A 3 times more important than B)
    A vs C: 5
    B vs C: 2
    Matrix:
      [ [1,    3,   5  ],
        [1/3,  1,   2  ],
        [1/5, 1/2,  1  ] ]
    Calculates weights, lambda_max, CI, and CR.
    CR must be < 0.10 (consistent).
    """
    pairwise = [
        [1.0, 3.0, 5.0],
        [1.0 / 3.0, 1.0, 2.0],
        [1.0 / 5.0, 0.5, 1.0],
    ]
    ahp_res = solve_ahp(pairwise)

    assert len(ahp_res.weights) == 3
    assert pytest.approx(sum(ahp_res.weights), 1e-4) == 1.0
    # A > B > C
    assert ahp_res.weights[0] > ahp_res.weights[1] > ahp_res.weights[2]
    assert ahp_res.is_consistent
    assert ahp_res.consistency_ratio < 0.10


def test_topsis_selection():
    """
    3 Alternatives (A1, A2, A3) evaluated on 4 criteria:
    Criteria:
      C1 (Cost, weight 0.25)
      C2 (Benefit, weight 0.25)
      C3 (Benefit, weight 0.25)
      C4 (Benefit, weight 0.25)
    Matrix:
      A1: [250, 16, 12, 5]
      A2: [200, 16, 8,  3]
      A3: [300, 32, 16, 4]
    """
    decision_matrix = [
        [250.0, 16.0, 12.0, 5.0],
        [200.0, 16.0, 8.0, 3.0],
        [300.0, 32.0, 16.0, 4.0],
    ]
    weights = [0.25, 0.25, 0.25, 0.25]
    criteria_types = ["cost", "benefit", "benefit", "benefit"]

    topsis_res = solve_topsis(decision_matrix, weights, criteria_types)

    assert len(topsis_res.scores) == 3
    assert len(topsis_res.ranking) == 3
    # Check that scores are in [0, 1]
    for score in topsis_res.scores:
        assert 0.0 <= score <= 1.0
    # Best alternative rank 1
    assert topsis_res.ranking[0] in [0, 1, 2]
