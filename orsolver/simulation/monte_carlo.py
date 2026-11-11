from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from orsolver.network.pert_cpm import compute_cpm, Task


@dataclass
class StochasticTask:
    name: str
    optimistic: float
    most_likely: float
    pessimistic: float
    predecessors: List[str] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    mean_duration: float
    std_dev: float
    simulated_durations: List[float]
    criticality_index: Dict[str, float]

    def probability_of_completion(self, deadline: float) -> float:
        """Returns the empirical probability that the project finishes on or before the deadline."""
        arr = np.array(self.simulated_durations)
        return float(np.mean(arr <= deadline))


def monte_carlo_pert(
    tasks: List[StochasticTask],
    num_simulations: int = 1000,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Simulates project completion times using Monte Carlo sampling on triangular task durations.
    Calculates mean, variance, completion probabilities, and task criticality indices.
    """
    if seed is not None:
        np.random.seed(seed)

    durations = []
    crit_counts = {t.name: 0 for t in tasks}

    for _ in range(num_simulations):
        # Sample durations for each task
        cpm_tasks = []
        for t in tasks:
            sampled_dur = np.random.triangular(t.optimistic, t.most_likely, t.pessimistic)
            cpm_tasks.append(Task(name=t.name, duration=float(sampled_dur), predecessors=t.predecessors))

        res = compute_cpm(cpm_tasks)
        durations.append(res.project_duration)

        for c_task in res.critical_path:
            crit_counts[c_task] += 1

    mean_d = float(np.mean(durations))
    std_d = float(np.std(durations))
    crit_indices = {k: v / num_simulations for k, v in crit_counts.items()}

    return MonteCarloResult(
        mean_duration=mean_d,
        std_dev=std_d,
        simulated_durations=durations,
        criticality_index=crit_indices,
    )
