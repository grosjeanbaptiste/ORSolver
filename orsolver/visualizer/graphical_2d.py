import math
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from orsolver.core.model import Problem, Sense


class DynamicGraphicalSolver:
    """
    2D Graphical Linear Programming Solver.
    Features:
      - Automatic intersection calculation with constraints and axes.
      - Convex polygon feasible region sorting and shading.
      - Dynamic auto-scaling of plot boundaries.
      - Objective function contour line plotting.
    """

    def __init__(self, tol: float = 1e-7):
        self.tol = tol

    def solve(self, problem: Problem) -> Dict[str, Any]:
        vars_list = list(problem.variables.keys())
        if len(vars_list) != 2:
            raise ValueError(
                f"Graphical solver only supports 2D problems. Got {len(vars_list)} variables."
            )

        x_name, y_name = vars_list[0], vars_list[1]
        c_x = problem.objective.get(x_name, 0.0)
        c_y = problem.objective.get(y_name, 0.0)

        # Collect all boundary lines: (a1, a2, rhs, sense, name)
        lines = []
        for c in problem.constraints:
            a1 = c.coefficients.get(x_name, 0.0)
            a2 = c.coefficients.get(y_name, 0.0)
            lines.append((a1, a2, c.rhs, c.sense, c.name))

        # Add non-negativity axes: x1 >= 0, x2 >= 0
        lines.append((1.0, 0.0, 0.0, ">=", f"{x_name} >= 0"))
        lines.append((0.0, 1.0, 0.0, ">=", f"{y_name} >= 0"))

        # Find all pairwise intersections
        candidate_points = []
        n_lines = len(lines)
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                a1, b1, c1, _, _ = lines[i]
                a2, b2, c2, _, _ = lines[j]

                det = a1 * b2 - a2 * b1
                if abs(det) < self.tol:
                    continue  # Parallel lines

                px = (c1 * b2 - c2 * b1) / det
                py = (a1 * c2 - a2 * c1) / det
                candidate_points.append((px, py))

        # Filter points that satisfy all constraints
        feasible_points = []
        for px, py in candidate_points:
            # Check non-negativity
            if px < -self.tol or py < -self.tol:
                continue

            point_is_feasible = True
            for c in problem.constraints:
                val = c.coefficients.get(x_name, 0.0) * px + c.coefficients.get(y_name, 0.0) * py
                if c.sense == "<=" and val > c.rhs + self.tol:
                    point_is_feasible = False
                    break
                elif c.sense == ">=" and val < c.rhs - self.tol:
                    point_is_feasible = False
                    break
                elif c.sense == "==" and abs(val - c.rhs) > self.tol:
                    point_is_feasible = False
                    break

            if point_is_feasible:
                # Add if not already present
                if not any(
                    abs(px - ex) < 1e-5 and abs(py - ey) < 1e-5
                    for ex, ey in feasible_points
                ):
                    feasible_points.append((round(px, 6), round(py, 6)))

        if not feasible_points:
            return {
                "is_feasible": False,
                "feasible_vertices": [],
                "optimal_point": None,
                "optimal_value": None,
            }

        # Order vertices cyclically around centroid
        cx = sum(p[0] for p in feasible_points) / len(feasible_points)
        cy = sum(p[1] for p in feasible_points) / len(feasible_points)
        feasible_points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        # Evaluate objective function
        is_max = problem.sense == Sense.MAXIMIZE
        best_val = float("-inf") if is_max else float("inf")
        best_point = None

        for pt in feasible_points:
            val = c_x * pt[0] + c_y * pt[1]
            if is_max:
                if val > best_val:
                    best_val = val
                    best_point = pt
            else:
                if val < best_val:
                    best_val = val
                    best_point = pt

        return {
            "is_feasible": True,
            "feasible_vertices": feasible_points,
            "optimal_point": best_point,
            "optimal_value": best_val,
            "x_name": x_name,
            "y_name": y_name,
            "c_x": c_x,
            "c_y": c_y,
        }

    def plot_and_save(
        self,
        problem: Problem,
        filepath: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculates solution, plots high-quality visual, and saves to file."""
        res = self.solve(problem)
        if not res["is_feasible"]:
            raise ValueError("Cannot plot: problem is infeasible.")

        x_name, y_name = res["x_name"], res["y_name"]
        vertices = res["feasible_vertices"]
        opt_pt = res["optimal_point"]
        opt_val = res["optimal_value"]
        c_x, c_y = res["c_x"], res["c_y"]

        # Compute dynamic bounds with margin
        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        max_x = max(max(xs) * 1.3, 5.0)
        max_y = max(max(ys) * 1.3, 5.0)

        fig, ax = plt.subplots(figsize=(10, 7))
        x_vals = np.linspace(0, max_x, 400)

        # Plot constraint lines
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for idx, c in enumerate(problem.constraints):
            a1 = c.coefficients.get(x_name, 0.0)
            a2 = c.coefficients.get(y_name, 0.0)
            color = colors[idx % len(colors)]
            label = f"{a1:g}{x_name} + {a2:g}{y_name} {c.sense} {c.rhs:g}"

            if abs(a2) > 1e-9:
                y_line = (c.rhs - a1 * x_vals) / a2
                ax.plot(x_vals, y_line, label=label, color=color, linewidth=1.5)
            else:
                x_val = c.rhs / a1
                ax.axvline(x=x_val, label=label, color=color, linewidth=1.5)

        # Shade feasible polygon
        poly = plt.Polygon(
            vertices,
            closed=True,
            facecolor="#66c2a5",
            edgecolor="#2ca25f",
            alpha=0.4,
            label="Feasible Region",
        )
        ax.add_patch(poly)

        # Plot vertices
        for pt in vertices:
            ax.plot(pt[0], pt[1], "o", color="#2b8cbe", markersize=6)
            ax.text(
                pt[0] + 0.05 * max_x,
                pt[1] + 0.02 * max_y,
                f"({pt[0]:.2f}, {pt[1]:.2f})",
                fontsize=8,
            )

        # Plot optimal level curve: c_x * x + c_y * y = opt_val
        if abs(c_y) > 1e-9:
            opt_line = (opt_val - c_x * x_vals) / c_y
            ax.plot(
                x_vals,
                opt_line,
                "--",
                color="#e41a1c",
                linewidth=2.0,
                label=f"Optimal Obj: {c_x:g}{x_name} + {c_y:g}{y_name} = {opt_val:.2f}",
            )

        # Highlight optimal point
        ax.plot(
            opt_pt[0],
            opt_pt[1],
            "*",
            color="#e41a1c",
            markersize=14,
            label=f"Optimum ({opt_pt[0]:.2f}, {opt_pt[1]:.2f})",
        )

        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.set_xlabel(x_name, fontsize=12)
        ax.set_ylabel(y_name, fontsize=12)
        ax.set_title(title or f"2D Linear Program: {problem.name}", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)

        return res
