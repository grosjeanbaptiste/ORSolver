from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


@dataclass
class GameNode:
    name: str
    value: Optional[float] = None
    children: List["GameNode"] = field(default_factory=list)


def minimax_alpha_beta(
    node: Optional[GameNode] = None,
    state: Any = None,
    depth: int = 10,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    is_maximizing: bool = True,
    get_actions_fn: Optional[Callable[[Any], List[Tuple[Any, Any]]]] = None,
    is_terminal_fn: Optional[Callable[[Any], bool]] = None,
    eval_fn: Optional[Callable[[Any], float]] = None,
) -> Tuple[float, Any, int]:
    """
    Minimax Search Algorithm with Alpha-Beta Pruning.

    Supports either:
      1) An explicit GameNode tree (pass node)
      2) Dynamic game transitions (pass state, get_actions_fn, is_terminal_fn, eval_fn)

    Returns:
      (optimal_value, best_action_or_node_name, total_pruned_branches)
    """
    pruned_counter = [0]

    # Mode 1: Explicit GameNode
    if node is not None:
        def _search_node(curr: GameNode, d: int, a: float, b: float, maximizing: bool):
            if d == 0 or not curr.children:
                val = curr.value if curr.value is not None else 0.0
                return val, None

            if maximizing:
                max_eval = float("-inf")
                best_child = None
                for child in curr.children:
                    eval_val, _ = _search_node(child, d - 1, a, b, False)
                    if eval_val > max_eval:
                        max_eval = eval_val
                        best_child = child.name
                    a = max(a, eval_val)
                    if b <= a:
                        pruned_counter[0] += 1
                        break
                return max_eval, best_child
            else:
                min_eval = float("inf")
                best_child = None
                for child in curr.children:
                    eval_val, _ = _search_node(child, d - 1, a, b, True)
                    if eval_val < min_eval:
                        min_eval = eval_val
                        best_child = child.name
                    b = min(b, eval_val)
                    if b <= a:
                        pruned_counter[0] += 1
                        break
                return min_eval, best_child

        opt_val, move = _search_node(node, depth, alpha, beta, is_maximizing)
        return opt_val, move, pruned_counter[0]

    # Mode 2: Dynamic State Space
    def _search_state(curr_state: Any, d: int, a: float, b: float, maximizing: bool):
        if d == 0 or (is_terminal_fn and is_terminal_fn(curr_state)):
            val = eval_fn(curr_state) if eval_fn else 0.0
            return val, None

        actions = get_actions_fn(curr_state) if get_actions_fn else []
        if not actions:
            val = eval_fn(curr_state) if eval_fn else 0.0
            return val, None

        if maximizing:
            max_eval = float("-inf")
            best_action = None
            for action, next_state in actions:
                eval_val, _ = _search_state(next_state, d - 1, a, b, False)
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_action = action
                a = max(a, eval_val)
                if b <= a:
                    pruned_counter[0] += 1
                    break
            return max_eval, best_action
        else:
            min_eval = float("inf")
            best_action = None
            for action, next_state in actions:
                eval_val, _ = _search_state(next_state, d - 1, a, b, True)
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_action = action
                b = min(b, eval_val)
                if b <= a:
                    pruned_counter[0] += 1
                    break
            return min_eval, best_action

    opt_val, move = _search_state(state, depth, alpha, beta, is_maximizing)
    return opt_val, move, pruned_counter[0]
