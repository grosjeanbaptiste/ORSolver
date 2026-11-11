from orsolver.core.model import Problem, Sense


def build_dual_problem(primal: Problem) -> Problem:
    """
    Constructs the mathematical dual of a given linear program.

    Primal (MAX):
      Max c^T x
      s.t. A x <= b, x >= 0
    Dual (MIN):
      Min b^T y
      s.t. A^T y >= c, y >= 0

    Primal (MIN):
      Min c^T x
      s.t. A x >= b, x >= 0
    Dual (MAX):
      Max b^T y
      s.t. A^T y <= c, y >= 0
    """
    is_max = primal.sense == Sense.MAXIMIZE
    dual_sense = Sense.MINIMIZE if is_max else Sense.MAXIMIZE
    dual = Problem(name=f"dual_of_{primal.name}", sense=dual_sense)

    # 1. Dual variables: one per primal constraint
    dual_obj = {}
    for c in primal.constraints:
        y_name = f"y_{c.name}"
        # For standard primal MAX with <= constraints, y >= 0
        dual.add_variable(y_name, lower_bound=0.0)
        dual_obj[y_name] = c.rhs

    dual.set_objective(dual_obj, sense=dual_sense)

    # 2. Dual constraints: one per primal variable
    constraint_sense = ">=" if is_max else "<="

    for var_name, var in primal.variables.items():
        row_coeffs = {}
        for c in primal.constraints:
            y_name = f"y_{c.name}"
            coeff = c.coefficients.get(var_name, 0.0)
            if abs(coeff) > 1e-9:
                row_coeffs[y_name] = coeff

        rhs = primal.objective.get(var_name, 0.0)
        dual.add_constraint(
            coefficients=row_coeffs,
            sense=constraint_sense,
            rhs=rhs,
            name=f"dual_c_{var_name}",
        )

    return dual
