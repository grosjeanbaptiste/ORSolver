import pytest
from orsolver.simulation import monte_carlo_pert, StochasticTask


def test_monte_carlo_pert():
    """
    Stochastic Project:
    Task A: min=2, mode=3, max=4, deps=[]
    Task B: min=3, mode=4, max=5, deps=[]
    Task C: min=1, mode=2, max=3, deps=[A]
    Task D: min=4, mode=5, max=6, deps=[B]
    Path 1: A -> C (mean approx 3 + 2 = 5)
    Path 2: B -> D (mean approx 4 + 5 = 9) -> dominates critical path!
    """
    tasks = [
        StochasticTask(name="A", optimistic=2.0, most_likely=3.0, pessimistic=4.0, predecessors=[]),
        StochasticTask(name="B", optimistic=3.0, most_likely=4.0, pessimistic=5.0, predecessors=[]),
        StochasticTask(name="C", optimistic=1.0, most_likely=2.0, pessimistic=3.0, predecessors=["A"]),
        StochasticTask(name="D", optimistic=4.0, most_likely=5.0, pessimistic=6.0, predecessors=["B"]),
    ]

    res = monte_carlo_pert(tasks, num_simulations=1000, seed=42)

    # Mean duration should be close to 4 + 5 = 9
    assert pytest.approx(res.mean_duration, abs=0.5) == 9.0
    assert res.std_dev > 0.0
    # Tasks B and D should have high criticality index (~100%)
    assert res.criticality_index["B"] > 0.95
    assert res.criticality_index["D"] > 0.95
    # Tasks A and C should have very low criticality index
    assert res.criticality_index["A"] < 0.05

    # Probability of completion within 12 days should be 1.0
    p_comp = res.probability_of_completion(deadline=12.0)
    assert p_comp > 0.99
