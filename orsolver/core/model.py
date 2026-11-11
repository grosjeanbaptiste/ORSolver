from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Sense(Enum):
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


@dataclass
class Variable:
    name: str
    lower_bound: Optional[float] = 0.0
    upper_bound: Optional[float] = None
    is_integer: bool = False
    is_binary: bool = False


@dataclass
class Constraint:
    coefficients: Dict[str, float]
    sense: str  # '<=', '>=', '==' or '='
    rhs: float
    name: str

    def __post_init__(self):
        if self.sense == "=":
            self.sense = "=="
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(f"Unsupported constraint sense: {self.sense}")


@dataclass
class Problem:
    name: str = "LinearProgram"
    sense: Sense = Sense.MAXIMIZE
    variables: Dict[str, Variable] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    objective: Dict[str, float] = field(default_factory=dict)

    def add_variable(
        self,
        name: str,
        lower_bound: Optional[float] = 0.0,
        upper_bound: Optional[float] = None,
        is_integer: bool = False,
        is_binary: bool = False,
    ) -> Variable:
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists in problem.")
        if is_binary:
            lower_bound = 0.0
            upper_bound = 1.0
            is_integer = True
        var = Variable(
            name=name,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_integer=is_integer,
            is_binary=is_binary,
        )
        self.variables[name] = var
        return var

    def set_objective(self, coefficients: Dict[str, float], sense: Optional[Sense] = None) -> None:
        if sense is not None:
            self.sense = sense
        for v in coefficients:
            if v not in self.variables:
                self.add_variable(v)
        self.objective = {k: float(v) for k, v in coefficients.items()}

    def add_constraint(
        self,
        coefficients: Dict[str, float],
        sense: str,
        rhs: float,
        name: Optional[str] = None,
    ) -> Constraint:
        for v in coefficients:
            if v not in self.variables:
                self.add_variable(v)
        c_name = name or f"c_{len(self.constraints) + 1}"
        c = Constraint(
            coefficients={k: float(v) for k, v in coefficients.items()},
            sense=sense,
            rhs=float(rhs),
            name=c_name,
        )
        self.constraints.append(c)
        return c
