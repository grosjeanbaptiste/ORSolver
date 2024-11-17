import matplotlib.pyplot as plt
import numpy as np


class LinearProgramVisualizer:
    """
    Class to visualize linear programming problems and calculate solutions graphically.
    """

    def __init__(self, constraints, bounds=None, objective=None):
        self.constraints = constraints
        self.bounds = bounds
        self.objective = objective

    def calculate_graphical_solution(self):
        """
        Finds the optimal solution graphically by checking intersections of constraints.

        Returns:
        - solution: The optimal solution [x1, x2].
        - objective_value: Value of the objective function at the optimal solution.
        """
        feasible_points = []

        for i in range(len(self.constraints)):
            for j in range(i + 1, len(self.constraints)):
                coeff1, op1, val1 = self.constraints[i]
                coeff2, op2, val2 = self.constraints[j]

                A = np.array([coeff1, coeff2])
                B = np.array([val1, val2])

                try:
                    intersection = np.linalg.solve(A, B)
                except np.linalg.LinAlgError:
                    continue

                if all(
                    self.is_feasible(intersection, c, op, v)
                    for c, op, v in self.constraints
                ):
                    feasible_points.append(intersection)

        optimal_point = None
        optimal_value = float("-inf")

        for point in feasible_points:
            value = sum(c * p for c, p in zip(self.objective, point))
            if value > optimal_value:
                optimal_value = value
                optimal_point = point

        return {"solution": optimal_point, "objective_value": optimal_value}

    @staticmethod
    def is_feasible(point, coefficients, operator, value):
        """
        Checks if a point satisfies a given constraint.

        Args:
        - point: [x1, x2] coordinates.
        - coefficients: Coefficients of the constraint.
        - operator: Constraint operator ('<=', '>=', or '=').
        - value: RHS value of the constraint.

        Returns:
        - bool: True if the point satisfies the constraint.
        """
        lhs = sum(c * p for c, p in zip(coefficients, point))
        if operator == "<=":
            return lhs <= value
        elif operator == ">=":
            return lhs >= value
        elif operator == "=":
            return np.isclose(lhs, value)
        return False

    def plot(self, solution):
        """
        Visualize the constraints, feasible region, and solution.
        """
        x = np.linspace(0, 10, 400)
        y = np.linspace(0, 10, 400)
        X, Y = np.meshgrid(x, y)
        feasible_region = np.ones_like(X, dtype=bool)

        plt.figure(figsize=(12, 8))

        for coefficients, operator, value in self.constraints:
            if operator == "<=":
                constraint = coefficients[0] * X + coefficients[1] * Y <= value
            elif operator == ">=":
                constraint = coefficients[0] * X + coefficients[1] * Y >= value
            elif operator == "=":
                constraint = np.isclose(
                    coefficients[0] * X + coefficients[1] * Y, value
                )
            feasible_region &= constraint

        plt.contourf(
            X, Y, feasible_region, levels=[0, 1], colors=["#CDE7F0"], alpha=0.4
        )
        plt.contour(
            X, Y, feasible_region, levels=[0.5], colors=["#1f77b4"], linewidths=1.5
        )

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.constraints)))
        for idx, (coefficients, operator, value) in enumerate(self.constraints):
            y_line = (value - coefficients[0] * x) / coefficients[1]
            label = f"{coefficients[0]}x₁ + {coefficients[1]}x₂ {operator} {value}"
            plt.plot(x, y_line, label=label, color=colors[idx], linestyle="--")

            mid_x = x[len(x) // 2]
            mid_y = (value - coefficients[0] * mid_x) / coefficients[1]
            if 0 <= mid_y <= 10:
                plt.text(
                    mid_x,
                    mid_y,
                    f"C{idx + 1}",
                    color=colors[idx],
                    fontsize=10,
                    ha="center",
                )

        if solution is not None:
            plt.scatter(
                solution[0],
                solution[1],
                color="red",
                s=120,
                edgecolor="black",
                label="Optimal Solution",
            )
            plt.annotate(
                f"({solution[0]:.2f}, {solution[1]:.2f})\nObj: {sum(c * s for c, s in zip(self.objective, solution)):.2f}",
                (solution[0], solution[1]),
                textcoords="offset points",
                xytext=(10, -15),
                fontsize=12,
                color="black",
                ha="center",
            )

        if self.objective and solution is not None:
            z_optimal = sum(c * s for c, s in zip(self.objective, solution))
            obj_y = (z_optimal - self.objective[0] * x) / self.objective[1]
            plt.plot(
                x,
                obj_y,
                "-",
                color="orange",
                linewidth=2,
                label=f"Objective Function (z = {z_optimal:.2f})",
            )

        plt.title(
            "Linear Program Constraints and Feasible Region", fontsize=16, weight="bold"
        )
        plt.xlabel("x₁", fontsize=14)
        plt.ylabel("x₂", fontsize=14)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", fontsize=12)
        plt.show()