import re
from typing import Dict, Optional, Tuple
from orsolver.core.model import Problem, Sense, Variable


def _parse_linear_expression(expr_str: str) -> Dict[str, float]:
    """
    Parses a linear expression string like "3 x1 - 5.5 x2 + x3" into a dict:
    {'x1': 3.0, 'x2': -5.5, 'x3': 1.0}
    """
    coeffs: Dict[str, float] = {}
    tokens = re.findall(r"([+-]?\s*[\d.]*\s*[a-zA-Z_][a-zA-Z0-9_]*)", expr_str)

    for tok in tokens:
        tok = tok.replace(" ", "")
        if not tok:
            continue
        match = re.match(r"^([+-]?[\d.]*)([a-zA-Z_][a-zA-Z0-9_]*)$", tok)
        if match:
            c_str, var_name = match.groups()
            if c_str in ("", "+"):
                coeff = 1.0
            elif c_str == "-":
                coeff = -1.0
            else:
                coeff = float(c_str)
            coeffs[var_name] = coeffs.get(var_name, 0.0) + coeff
    return coeffs


def _parse_bound_line(line: str, problem: Problem) -> None:
    """
    Parse a bound specification line such as:
    - 0 <= x1 <= 40
    - x2 >= 0
    - 0 <= x1
    - x1 <= 100
    - x1 free
    """
    line = line.strip()
    if not line:
        return

    # Case: "x1 free"
    if line.lower().endswith("free"):
        var_name = line[:-4].strip()
        v = problem.get_variable(var_name) or problem.add_variable(var_name)
        v.lower_bound = float("-inf")
        v.upper_bound = float("inf")
        return

    # Double bound: "0 <= x1 <= 40"
    m_double = re.match(r"^([+-]?[\d.]+)\s*<=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*<=\s*([+-]?[\d.]+)$", line)
    if m_double:
        lb_val = float(m_double.group(1))
        var_name = m_double.group(2)
        ub_val = float(m_double.group(3))
        v = problem.get_variable(var_name) or problem.add_variable(var_name)
        v.lower_bound = lb_val
        v.upper_bound = ub_val
        return

    # Single bound: "0 <= x1"
    m_lb1 = re.match(r"^([+-]?[\d.]+)\s*<=\s*([a-zA-Z_][a-zA-Z0-9_]*)$", line)
    if m_lb1:
        lb_val = float(m_lb1.group(1))
        var_name = m_lb1.group(2)
        v = problem.get_variable(var_name) or problem.add_variable(var_name)
        v.lower_bound = lb_val
        return

    # Single bound: "x1 >= 0"
    m_lb2 = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*>=\s*([+-]?[\d.]+)$", line)
    if m_lb2:
        var_name = m_lb2.group(1)
        lb_val = float(m_lb2.group(2))
        v = problem.get_variable(var_name) or problem.add_variable(var_name)
        v.lower_bound = lb_val
        return

    # Single bound: "x1 <= 40"
    m_ub = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*<=\s*([+-]?[\d.]+)$", line)
    if m_ub:
        var_name = m_ub.group(1)
        ub_val = float(m_ub.group(2))
        v = problem.get_variable(var_name) or problem.add_variable(var_name)
        v.upper_bound = ub_val
        return


def parse_lp_string(lp_content: str, name: str = "LP") -> Problem:
    """
    Parses a string formatted in standard CPLEX LP format into an ORSolver Problem object.
    Supports objective (Maximize / Minimize), constraints (Subject To), Bounds,
    Generals/Integers, and Binary sections.
    """
    problem = Problem(name=name)
    lines = lp_content.strip().splitlines()

    mode: Optional[str] = None
    constraint_idx = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("\\") or line.startswith("#"):
            continue

        upper = line.upper()
        if upper in ("MAXIMIZE", "MAX"):
            problem.sense = Sense.MAXIMIZE
            mode = "OBJ"
            continue
        elif upper in ("MINIMIZE", "MIN"):
            problem.sense = Sense.MINIMIZE
            mode = "OBJ"
            continue
        elif upper in ("SUBJECT TO", "SUCH THAT", "ST", "S.T."):
            mode = "CONSTRAINTS"
            continue
        elif upper in ("BOUNDS", "BOUND"):
            mode = "BOUNDS"
            continue
        elif upper in ("GENERALS", "GENERAL", "INTEGERS", "INTEGER"):
            mode = "GENERALS"
            continue
        elif upper in ("BINARY", "BINARIES", "BIN"):
            mode = "BINARY"
            continue
        elif upper == "END":
            break

        if mode == "OBJ":
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

            match = re.search(r"(<=|>=|==|=|<|>)", line)
            if not match:
                continue

            op = match.group(1)
            lhs_str, rhs_str = line.split(op, 1)
            op = "==" if op == "=" else op
            rhs = float(rhs_str.strip())
            coeffs = _parse_linear_expression(lhs_str)

            constraint_idx += 1
            name_c = c_name or f"c_{constraint_idx}"
            problem.add_constraint(coeffs, op, rhs, name=name_c)
        elif mode == "BOUNDS":
            _parse_bound_line(line, problem)
        elif mode == "GENERALS":
            var_names = line.split()
            for v_name in var_names:
                v = problem.get_variable(v_name) or problem.add_variable(v_name)
                v.is_integer = True
        elif mode == "BINARY":
            var_names = line.split()
            for v_name in var_names:
                v = problem.get_variable(v_name) or problem.add_variable(v_name)
                v.is_binary = True
                v.is_integer = True
                v.lower_bound = 0.0
                v.upper_bound = 1.0

    return problem
