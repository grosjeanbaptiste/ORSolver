import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class SimplexSolver:
    """
    Class to solve linear programming problems using the Simplex method.
    """

    def __init__(self, objective, constraints, bounds=None):
        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds
        self.tableau = None
        self.basic_variables = []
        self.non_basic_variables = []

    def initialize_tableau(self):
        num_vars = len(self.objective)
        num_constraints = len(self.constraints)

        self.tableau = []
        for coefficients, operator, value in self.constraints:
            if operator != "<=":
                raise NotImplementedError(
                    "Only '<=' constraints are currently supported."
                )
            row = coefficients + [0] * num_constraints + [value]
            self.tableau.append(row)

        for i in range(num_constraints):
            self.tableau[i][num_vars + i] = 1

        self.tableau.append([-c for c in self.objective] + [0] * (num_constraints + 1))

        self.basic_variables = list(range(num_vars, num_vars + num_constraints))
        self.non_basic_variables = list(range(num_vars))

    def is_optimal(self):
        return all(c >= 0 for c in self.tableau[-1][:-1])

    def get_pivot_column(self):
        return np.argmin(self.tableau[-1][:-1])

    def get_pivot_row(self, pivot_column):
        ratios = []
        for i, row in enumerate(self.tableau[:-1]):
            if row[pivot_column] > 0:
                ratios.append((row[-1] / row[pivot_column], i))
        if not ratios:
            raise ValueError("The linear program is unbounded.")
        return min(ratios, key=lambda x: x[0])[1]

    def perform_iteration(self):
        pivot_column = self.get_pivot_column()
        pivot_row = self.get_pivot_row(pivot_column)

        pivot_value = self.tableau[pivot_row][pivot_column]
        self.tableau[pivot_row] = [x / pivot_value for x in self.tableau[pivot_row]]

        for i, row in enumerate(self.tableau):
            if i != pivot_row:
                multiplier = row[pivot_column]
                self.tableau[i] = [
                    a - multiplier * b for a, b in zip(row, self.tableau[pivot_row])
                ]

        self.basic_variables[pivot_row] = pivot_column

    def solve(self):
        self.initialize_tableau()

        while not self.is_optimal():
            self.perform_iteration()

        num_vars = len(self.objective)
        solution = [0] * num_vars
        for i, basic_var in enumerate(self.basic_variables):
            if basic_var < num_vars:
                solution[basic_var] = self.tableau[i][-1]
        objective_value = -self.tableau[-1][-1]

        return {"solution": solution, "objective_value": objective_value}


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


if __name__ == "__main__":
    constraints_example = [
        ([6, 4], "<=", 24),
        ([1, 2], "<=", 6),
        ([0, 1], "<=", 2),
        ([-1, 1], "<=", 1),
    ]
    objective_example = [5, 4]

    simplex_solver = SimplexSolver(objective_example, constraints_example)
    simplex_result = simplex_solver.solve()

    print("Simplex Method:")
    print("Optimal Solution:", simplex_result)

    visualizer = LinearProgramVisualizer(
        constraints_example, objective=objective_example
    )
    graphical_result = visualizer.calculate_graphical_solution()

    print("\nGraphical Method:")
    print("Optimal Solution:", graphical_result)

    visualizer.plot(graphical_result["solution"])
