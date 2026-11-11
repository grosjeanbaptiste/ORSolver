import pytest
from orsolver.game_theory.adversarial import minimax_alpha_beta, GameNode


def test_minimax_alpha_beta_pruning():
    """
    Classic game tree (Depth 3):
    Root (MAX)
      ├── A (MIN)
      │     ├── D (MAX, leaf values: [3, 5])
      │     └── E (MAX, leaf values: [6, 9])
      └── B (MIN)
            ├── F (MAX, leaf values: [1, 2])
            └── G (MAX, leaf values: [0, -1])

    Evaluation:
    D = min(3, 5) = 3 (if D is MIN or if D is leaf values)
    Let's define standard tree:
    Root is MAX:
      Child A (MIN):
        Leaf L1 = 3
        Leaf L2 = 5
        -> A = min(3, 5) = 3
      Child B (MIN):
        Leaf L3 = 2
        Leaf L4 = 9 (will be pruned because L3=2 <= alpha=3!)
        -> B = 2
    Root = max(3, 2) = 3, best move = 'A'.
    """
    tree = GameNode(
        name="Root",
        children=[
            GameNode(name="A", children=[
                GameNode(name="L1", value=3.0),
                GameNode(name="L2", value=5.0),
            ]),
            GameNode(name="B", children=[
                GameNode(name="L3", value=2.0),
                GameNode(name="L4", value=9.0),
            ]),
        ]
    )

    val, best_move, pruned_count = minimax_alpha_beta(
        node=tree,
        depth=2,
        is_maximizing=True,
    )

    assert val == 3.0
    assert best_move == "A"
    assert pruned_count >= 1  # L4 was pruned because 2 <= alpha(3)


def test_tic_tac_toe_terminal_state():
    """
    Test with a state evaluation function and custom successor function.
    """
    # Simple nim-like game: starting with pile of N stones.
    # Player can take 1 or 2 stones.
    # Player who takes the last stone wins (+1 for MAX, -1 for MIN).
    def get_actions(state):
        stones, player = state
        actions = []
        if stones >= 1:
            actions.append((1, (stones - 1, "MIN" if player == "MAX" else "MAX")))
        if stones >= 2:
            actions.append((2, (stones - 2, "MIN" if player == "MAX" else "MAX")))
        return actions

    def is_terminal(state):
        stones, _ = state
        return stones == 0

    def evaluate(state):
        stones, player = state
        # The player whose turn it is has just lost because previous player took the last stone
        return -1.0 if player == "MAX" else 1.0

    # For 3 stones with MAX to play:
    # Take 1 -> 2 left for MIN. MIN can take 2 and win.
    # Take 2 -> 1 left for MIN. MIN takes 1 and wins.
    # So 3 stones is a losing position for MAX: val = -1.0.
    val, move, _ = minimax_alpha_beta(
        state=(3, "MAX"),
        depth=5,
        is_maximizing=True,
        get_actions_fn=get_actions,
        is_terminal_fn=is_terminal,
        eval_fn=evaluate,
    )
    assert val == -1.0
    assert move in (1, 2)

    # For 4 stones with MAX to play:
    # MAX takes 1 stone -> leaves 3 stones for MIN (losing for MIN -> winning for MAX)!
    val_4, move_4, _ = minimax_alpha_beta(
        state=(4, "MAX"),
        depth=5,
        is_maximizing=True,
        get_actions_fn=get_actions,
        is_terminal_fn=is_terminal,
        eval_fn=evaluate,
    )
    assert val_4 == 1.0
    assert move_4 == 1
