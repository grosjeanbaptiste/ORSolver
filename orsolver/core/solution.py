from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class SolutionStatus(Enum):
    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    FEASIBLE = "FEASIBLE"
    NOT_SOLVED = "NOT_SOLVED"


@dataclass
class Solution:
    status: SolutionStatus
    objective_value: Optional[float] = None
    variable_values: Dict[str, float] = field(default_factory=dict)
    slack_values: Dict[str, float] = field(default_factory=dict)
    shadow_prices: Dict[str, float] = field(default_factory=dict)
    reduced_costs: Dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    tableaus: List[Any] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_optimal(self) -> bool:
        return self.status == SolutionStatus.OPTIMAL

    def __repr__(self) -> str:
        return (
            f"Solution(status={self.status.value}, "
            f"objective={self.objective_value}, "
            f"variables={self.variable_values})"
        )
