import pytest
from orsolver.core.model import Problem, Sense, Constraint, Variable
from orsolver.core.solution import Solution, SolutionStatus


def test_variable_creation():
    x1 = Variable(name="x1", lower_bound=0, upper_bound=None, is_integer=False)
    assert x1.name == "x1"
    assert x1.lower_bound == 0
    assert x1.upper_bound is None
    assert not x1.is_integer


def test_constraint_creation():
    c = Constraint(coefficients={"x1": 2.0, "x2": 3.0}, sense="<=", rhs=10.0, name="c1")
    assert c.coefficients == {"x1": 2.0, "x2": 3.0}
    assert c.sense == "<="
    assert c.rhs == 10.0
    assert c.name == "c1"


def test_problem_definition():
    prob = Problem(name="production_planning", sense=Sense.MAXIMIZE)
    prob.add_variable("x1", lower_bound=0)
    prob.add_variable("x2", lower_bound=0)
    
    prob.set_objective({"x1": 40.0, "x2": 30.0})
    prob.add_constraint({"x1": 1.0, "x2": 1.0}, "<=", 10.0, name="labor")
    prob.add_constraint({"x1": 2.0, "x2": 1.0}, "<=", 16.0, name="material")
    
    assert prob.name == "production_planning"
    assert prob.sense == Sense.MAXIMIZE
    assert len(prob.variables) == 2
    assert len(prob.constraints) == 2
    assert prob.objective["x1"] == 40.0
    assert prob.objective["x2"] == 30.0


def test_solution_object():
    sol = Solution(
        status=SolutionStatus.OPTIMAL,
        objective_value=340.0,
        variable_values={"x1": 6.0, "x2": 4.0},
        slack_values={"labor": 0.0, "material": 0.0},
        shadow_prices={"labor": 20.0, "material": 10.0},
        reduced_costs={"x1": 0.0, "x2": 0.0},
        iterations=2
    )
    assert sol.is_optimal
    assert sol.objective_value == 340.0
    assert sol.variable_values["x1"] == 6.0
    assert sol.shadow_prices["labor"] == 20.0
