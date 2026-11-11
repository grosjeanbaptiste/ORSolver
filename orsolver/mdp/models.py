from typing import Any, Dict, List, Optional, Set, Tuple


class MDP:
    """
    Markov Decision Process model (S, A, P, R, gamma).
    """

    def __init__(
        self,
        states: List[Any],
        actions: List[Any],
        terminal_states: Optional[List[Any]] = None,
        gamma: float = 0.95,
    ):
        self.states = list(states)
        self.actions = list(actions)
        self.terminal_states: Set[Any] = set(terminal_states or [])
        self.gamma = float(gamma)
        # transitions[(s, a)] = [(prob, s_prime, reward), ...]
        self.transitions: Dict[Tuple[Any, Any], List[Tuple[float, Any, float]]] = {}

    def add_transition(
        self,
        state: Any,
        action: Any,
        outcomes: List[Tuple[float, Any, float]],
    ) -> None:
        self.transitions[(state, action)] = list(outcomes)


def value_iteration(
    mdp: MDP,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """
    Computes optimal values V*(s) and policy pi*(s) using Value Iteration.
    """
    V: Dict[Any, float] = {s: 0.0 for s in mdp.states}

    for _ in range(max_iter):
        delta = 0.0
        V_new = dict(V)

        for s in mdp.states:
            if s in mdp.terminal_states:
                V_new[s] = 0.0
                continue

            q_values = []
            for a in mdp.actions:
                outcomes = mdp.transitions.get((s, a), [])
                if not outcomes:
                    continue
                q = sum(prob * (reward + mdp.gamma * V[s_next]) for prob, s_next, reward in outcomes)
                q_values.append(q)

            if q_values:
                best_q = max(q_values)
                delta = max(delta, abs(best_q - V[s]))
                V_new[s] = best_q

        V = V_new
        if delta < tol:
            break

    # Extract optimal policy
    policy: Dict[Any, Any] = {}
    for s in mdp.states:
        if s in mdp.terminal_states:
            policy[s] = None
            continue

        best_a = None
        best_q = float("-inf")
        for a in mdp.actions:
            outcomes = mdp.transitions.get((s, a), [])
            if not outcomes:
                continue
            q = sum(prob * (reward + mdp.gamma * V[s_next]) for prob, s_next, reward in outcomes)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a

    return V, policy


def policy_iteration(
    mdp: MDP,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """
    Computes optimal values V*(s) and policy pi*(s) using Policy Iteration.
    """
    # Initialize arbitrary policy
    policy: Dict[Any, Any] = {}
    for s in mdp.states:
        if s in mdp.terminal_states:
            policy[s] = None
        else:
            policy[s] = mdp.actions[0]

    V: Dict[Any, float] = {s: 0.0 for s in mdp.states}

    for _ in range(max_iter):
        # 1. Policy Evaluation
        while True:
            delta = 0.0
            V_new = dict(V)
            for s in mdp.states:
                if s in mdp.terminal_states:
                    V_new[s] = 0.0
                    continue
                a = policy[s]
                outcomes = mdp.transitions.get((s, a), [])
                if outcomes:
                    val = sum(prob * (reward + mdp.gamma * V[s_next]) for prob, s_next, reward in outcomes)
                    delta = max(delta, abs(val - V[s]))
                    V_new[s] = val
            V = V_new
            if delta < tol:
                break

        # 2. Policy Improvement
        policy_stable = True
        for s in mdp.states:
            if s in mdp.terminal_states:
                continue

            old_action = policy[s]
            best_action = None
            best_q = float("-inf")

            for a in mdp.actions:
                outcomes = mdp.transitions.get((s, a), [])
                if not outcomes:
                    continue
                q = sum(prob * (reward + mdp.gamma * V[s_next]) for prob, s_next, reward in outcomes)
                if q > best_q:
                    best_q = q
                    best_action = a

            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return V, policy
