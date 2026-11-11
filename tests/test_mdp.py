import pytest
from orsolver.mdp import MDP, value_iteration, policy_iteration


def test_mdp_gridworld_corridor():
    """
    Simple 3-state corridor: 0 <-> 1 <-> 2 (Goal with reward 100).
    Actions: 'L', 'R'.
    Deterministic transitions:
      At state 0: 'R' -> 1, 'L' -> 0
      At state 1: 'R' -> 2, 'L' -> 0
      At state 2: terminal (no reward from leaving)
    Discount factor gamma = 0.9.
    Optimal policy:
      At 0: 'R' (leads to 1)
      At 1: 'R' (leads to 2, earning 100)
    Optimal values:
      V*(2) = 0 (or terminal)
      V*(1) = 100 + 0.9 * 0 = 100
      V*(0) = 0 + 0.9 * V*(1) = 90
    """
    mdp = MDP(
        states=[0, 1, 2],
        actions=["L", "R"],
        terminal_states=[2],
        gamma=0.9
    )
    # Transitions: (s, a) -> [(prob, next_state, reward)]
    mdp.add_transition(0, "R", [(1.0, 1, 0.0)])
    mdp.add_transition(0, "L", [(1.0, 0, 0.0)])
    mdp.add_transition(1, "R", [(1.0, 2, 100.0)])
    mdp.add_transition(1, "L", [(1.0, 0, 0.0)])

    v_vi, pi_vi = value_iteration(mdp, tol=1e-5)
    v_pi, pi_pi = policy_iteration(mdp)

    # Check Value Iteration
    assert pytest.approx(v_vi[0], 1e-3) == 90.0
    assert pytest.approx(v_vi[1], 1e-3) == 100.0
    assert pi_vi[0] == "R"
    assert pi_vi[1] == "R"

    # Check Policy Iteration
    assert pytest.approx(v_pi[0], 1e-3) == 90.0
    assert pytest.approx(v_pi[1], 1e-3) == 100.0
    assert pi_pi[0] == "R"
    assert pi_pi[1] == "R"
