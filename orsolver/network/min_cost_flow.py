from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from orsolver.core.model import Problem, Sense
from orsolver.lp.simplex import TwoPhaseSimplexSolver


@dataclass
class FlowArc:
    u: str
    v: str
    capacity: float
    cost: float
    name: Optional[str] = None


def solve_min_cost_flow(
    supplies: Dict[str, float],
    arcs: List[FlowArc],
) -> Tuple[float, Dict[Tuple[str, str], float]]:
    """
    Solves the Minimum Cost Network Flow (MCNF) / Transshipment problem.

    Args:
      supplies: {node_name: supply_value} (positive = supply, negative = demand, 0 = transshipment)
      arcs: list of FlowArc instances

    Returns:
      (min_cost, {(u, v): flow_value})
    """
    p = Problem(name="min_cost_flow", sense=Sense.MINIMIZE)

    obj = {}
    arc_names = {}
    for idx, arc in enumerate(arcs):
        var_name = arc.name or f"flow_{arc.u}_{arc.v}_{idx}"
        arc_names[var_name] = (arc.u, arc.v)
        p.add_variable(var_name, lower_bound=0.0)
        obj[var_name] = arc.cost

        # Capacity constraint: x_uv <= capacity
        if arc.capacity < float("inf"):
            p.add_constraint({var_name: 1.0}, "<=", arc.capacity, name=f"cap_{var_name}")

    p.set_objective(obj, sense=Sense.MINIMIZE)

    # Node flow conservation constraints:
    # sum(outflow) - sum(inflow) == supply(node)
    all_nodes = set(supplies.keys())
    for arc in arcs:
        all_nodes.add(arc.u)
        all_nodes.add(arc.v)

    for node in all_nodes:
        coeffs = {}
        for idx, arc in enumerate(arcs):
            var_name = arc.name or f"flow_{arc.u}_{arc.v}_{idx}"
            if arc.u == node:
                coeffs[var_name] = coeffs.get(var_name, 0.0) + 1.0
            if arc.v == node:
                coeffs[var_name] = coeffs.get(var_name, 0.0) - 1.0

        b = supplies.get(node, 0.0)
        p.add_constraint(coeffs, "==", b, name=f"node_{node}")

    solver = TwoPhaseSimplexSolver()
    solution = solver.solve(p)

    flow_dict: Dict[Tuple[str, str], float] = {}
    for var_name, (u, v) in arc_names.items():
        val = solution.variable_values.get(var_name, 0.0)
        flow_dict[(u, v)] = flow_dict.get((u, v), 0.0) + val

    total_cost = float(solution.objective_value) if solution.objective_value is not None else float("inf")
    return total_cost, flow_dict
