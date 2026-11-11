import pytest
from orsolver.dynamic_programming import matrix_chain_multiplication, coin_change


def test_matrix_chain_multiplication():
    """
    4 Matrices:
      A1: 10 x 30
      A2: 30 x 5
      A3: 5 x 60
    Dimensions array: [10, 30, 5, 60]
    Multiplication options:
      (A1 * A2) * A3: 10*30*5 + 10*5*60 = 1500 + 3000 = 4500 multiplications!
      A1 * (A2 * A3): 30*5*60 + 10*30*60 = 9000 + 18000 = 27000 multiplications.
    Optimal: 4500 with ((A1 A2) A3).
    """
    dims = [10, 30, 5, 60]
    min_ops, parens = matrix_chain_multiplication(dims)

    assert min_ops == 4500
    assert parens == "((A1 A2) A3)"


def test_coin_change():
    """
    Coins: [1, 2, 5, 10, 20, 50]
    Amount: 67
    Optimal coins: 50 + 10 + 5 + 2 = 4 coins!
    """
    coins = [1, 2, 5, 10, 20, 50]
    amount = 67

    min_coins, used_coins = coin_change(coins, amount)

    assert min_coins == 4
    assert used_coins[50] == 1
    assert used_coins[10] == 1
    assert used_coins[5] == 1
    assert used_coins[2] == 1
    assert sum(coin * count for coin, count in used_coins.items()) == 67
