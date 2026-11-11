import pytest
import numpy as np
from orsolver.game_theory import solve_zero_sum_game


def test_matching_pennies():
    """
    Matching pennies (zero-sum):
    Payoff matrix for Row player:
          H     T
      H  [ 1,  -1]
      T  [-1,   1]
    Optimal mixed strategy: p = [0.5, 0.5], q = [0.5, 0.5], game value V = 0.0.
    """
    payoff_matrix = [
        [1.0, -1.0],
        [-1.0, 1.0]
    ]

    result = solve_zero_sum_game(payoff_matrix)

    assert pytest.approx(result["game_value"], 1e-4) == 0.0
    assert pytest.approx(result["row_strategy"][0], 1e-4) == 0.5
    assert pytest.approx(result["row_strategy"][1], 1e-4) == 0.5
    assert pytest.approx(result["col_strategy"][0], 1e-4) == 0.5
    assert pytest.approx(result["col_strategy"][1], 1e-4) == 0.5


def test_rock_paper_scissors():
    """
    Rock-Paper-Scissors:
           R   P   S
      R [  0, -1,  1]
      P [  1,  0, -1]
      S [ -1,  1,  0]
    Optimal strategy: 1/3 each, game value = 0.
    """
    payoff_matrix = [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0]
    ]

    result = solve_zero_sum_game(payoff_matrix)

    assert pytest.approx(result["game_value"], 1e-4) == 0.0
    for p in result["row_strategy"]:
        assert pytest.approx(p, 1e-4) == 1.0 / 3.0
    for q in result["col_strategy"]:
        assert pytest.approx(q, 1e-4) == 1.0 / 3.0
