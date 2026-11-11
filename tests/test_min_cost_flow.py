import pytest
from orsolver.network.min_cost_flow import solve_min_cost_flow, FlowArc


def test_min_cost_flow_transshipment():
    """
    Transshipment network:
    Nodes:
      1 (Source, supply = 20)
      2 (Transshipment, supply = 0)
      3 (Sink, demand = 20, supply = -20)
    Arcs:
      1 -> 2: cap=15, cost=2
      1 -> 3: cap=10, cost=5
      2 -> 3: cap=20, cost=1
    Flow should send:
      15 units via 1 -> 2 -> 3 (cost: 15*2 + 15*1 = 45)
      5 units via 1 -> 3 (cost: 5*5 = 25)
      Total cost = 45 + 25 = 70.
    """
    supplies = {"1": 20.0, "2": 0.0, "3": -20.0}
    arcs = [
        FlowArc(u="1", v="2", capacity=15.0, cost=2.0),
        FlowArc(u="1", v="3", capacity=10.0, cost=5.0),
        FlowArc(u="2", v="3", capacity=20.0, cost=1.0),
    ]

    cost, flow_dict = solve_min_cost_flow(supplies, arcs)

    assert pytest.approx(cost, 1e-4) == 70.0
    assert pytest.approx(flow_dict[("1", "2")], 1e-4) == 15.0
    assert pytest.approx(flow_dict[("2", "3")], 1e-4) == 15.0
    assert pytest.approx(flow_dict[("1", "3")], 1e-4) == 5.0
