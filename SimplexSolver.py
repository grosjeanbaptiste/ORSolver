import numpy as np


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