import re
from typing import Dict, Tuple
from orsolver.core.model import Problem, Sense


def _parse_linear_expression(expr_str: str) -> Dict[str, float]:
    """Parses expressions like '3 x1 + 5 x2 - 2.5 x3' into {'x1': 3.0, 'x2': 5.0, 'x3': -2.5}."""
    pattern = re.compile(r"([+-]?\s*\d*(?:\.\d+)?)\s*([a-zA-Z_]\w*)")
    coeffs = {}
    tokens = pattern.findall(expr_str)
    for sign_num, var in tokens:
        clean_num = sign_num.replace(" ", "")
        if clean_num in ("", "+"):
            val = 1.0
        elif clean_num == "-":
            val = -1.0
        else:
            val = float(clean_num)
        coeffs[var] = coeffs.get(var, 0.0) + val
    return coeffs


def parse_lp_string(lp_str: str) -> Problem:
    """Parses a standard LP (CPLEX format) string into a Problem instance."""
    lines = [line.strip() for line in lp_str.strip().split("\n") if line.strip() and not line.strip().startswith("\\")]
    problem = Problem(name="LP")

    mode = None
    constraint_idx = 0

    for line in lines:
        upper = line.upper()
        if upper in ("MAXIMIZE", "MAX"):
            mode = "OBJ"
            problem.sense = Sense.MAXIMIZE
            continue
        elif upper in ("MINIMIZE", "MIN"):
            mode = "OBJ"
            problem.sense = Sense.MINIMIZE
            continue
        elif upper in ("SUBJECT TO", "SUCH THAT", "ST", "S.T."):
            mode = "CONSTRAINTS"
            continue
        elif upper in ("BOUNDS", "GENERALS", "INTEGERS", "BINARY", "END"):
            if upper == "END":
                break
            continue

        if mode == "OBJ":
            # Strip objective name if present (e.g. 'obj: 3 x1 + 5 x2')
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            obj_coeffs = _parse_linear_expression(line)
            problem.set_objective(obj_coeffs)
        elif mode == "CONSTRAINTS":
            c_name = None
            if ":" in line:
                parts = line.split(":", 1)
                c_name = parts[0].strip()
                line = parts[1].strip()

            # Find operator
            match = re.search(r"(<=|>=|==|=|<|>)", line)
            if not match:
                continue

            op = match.group(1)
            lhs_str, rhs_str = line.split(op, 1)
            op = "==" if op == "=" else op
            rhs = float(rhs_str.strip())
            coeffs = _parse_linear_expression(lhs_str)

            constraint_idx += 1
            name = c_name or f"c_{constraint_idx}"
            problem.add_constraint(coeffs, op, rhs, name=name)

    return problem
