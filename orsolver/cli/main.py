import argparse
import os
import sys
from typing import List, Optional

from orsolver import __version__
from orsolver.cli.parser import parse_lp_string
from orsolver.lp.simplex import TwoPhaseSimplexSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orsolver",
        description="ORSolver - Comprehensive Operations Research Solver CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"orsolver {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve an optimization model from file")
    solve_parser.add_argument("file", help="Path to the model file (.lp)")
    solve_parser.add_argument(
        "-o", "--output", help="Optional output file path to write results"
    )

    return parser


def format_solution(solution) -> str:
    lines = [
        "=" * 40,
        f"Status: {solution.status.value}",
        f"Objective Value: {solution.objective_value}",
        f"Iterations: {solution.iterations}",
        "-" * 40,
        "Variable Values:",
    ]
    for var, val in sorted(solution.variable_values.items()):
        lines.append(f"  {var} = {val:.6g}")

    if solution.shadow_prices:
        lines.append("-" * 40)
        lines.append("Shadow Prices (Duals):")
        for c_name, sp in sorted(solution.shadow_prices.items()):
            lines.append(f"  {c_name} = {sp:.6g}")

    lines.append("=" * 40)
    return "\n".join(lines)


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "solve":
        file_path = parsed_args.file
        if not os.path.exists(file_path):
            sys.stderr.write(f"Error: File '{file_path}' not found.\n")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        problem = parse_lp_string(content)
        solver = TwoPhaseSimplexSolver()
        solution = solver.solve(problem)

        formatted = format_solution(solution)
        print(formatted)

        if parsed_args.output:
            if parsed_args.output.endswith(".json"):
                import json
                data = {
                    "status": solution.status.value,
                    "objective_value": solution.objective_value,
                    "iterations": solution.iterations,
                    "variables": solution.variable_values,
                    "shadow_prices": solution.shadow_prices,
                }
                with open(parsed_args.output, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(parsed_args.output, "w", encoding="utf-8") as f:
                    f.write(formatted + "\n")

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
