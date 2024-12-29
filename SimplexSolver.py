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

        print("État initial :")
        column_names = [f"X{i+1}" for i in range(len(self.objective))] + [f"S{i+1}" for i in range(len(self.objective))]
        print(f"""\n{" | ".join(column_names)}\n{"-" * (len(column_names) * 10)}""")
        for row in self.tableau:
            print(" | ".join([f"{val:10.2f}" for val in row]))
        print("=" * 40)

        with open('resultats_solution.txt', 'a') as f:
            f.write("État initial :\n")
            f.write(f"""\n{" | ".join(column_names)}\n{"-" * (len(column_names) * 10)}\n""")
            for row in self.tableau:
                f.write(" | ".join([f"{val:10.2f}" for val in row]) + '\n')
            f.write("=" * 40 + '\n')

        print(f"\n=== Itération : ===")
        print(f"Variable entrant : X{pivot_column+1}, Variable sortante : S{self.basic_variables[pivot_row]-len(self.objective)+1}")

        with open('resultats_solution.txt', 'a') as f:
            f.write(f"\n=== Itération : ===\n")
            f.write(f"Variable entrant : X{pivot_column+1}, Variable sortante : S{self.basic_variables[pivot_row]-len(self.objective)+1}\n")
        
        slack_variables = [f'S{i+1}' for i in range(len(self.objective))]

        print("Détails de l'itération :")
        print(f"Variables de base : {[f'S{i+1}' if i >= len(self.objective) else f'X{i+1}' for i in self.basic_variables]}")
        print(f"Variables non de base : {[f'S{i+1}' if i >= len(self.objective) else f'X{i+1}' for i in range(len(self.tableau[0]) - 1) if i not in self.basic_variables]}\n")
        ratios = []
        for i, row in enumerate(self.tableau[:-1]):
            if row[pivot_column] > 0:
                ratios.append((row[-1] / row[pivot_column], i))
        print(f"Ratios calculés : {ratios}")
        print(f"Colonne pivot : {pivot_column}, Ligne pivot : {pivot_row}")

        with open('resultats_solution.txt', 'a') as f:
            f.write("Détails de l'itération :\n")
            f.write(f"Variables de base : {[f'S{i+1}' if i >= len(self.objective) else f'X{i+1}' for i in self.basic_variables]}\n")
            f.write(f"Variables non de base : {[f'S{i+1}' if i >= len(self.objective) else f'X{i+1}' for i in range(len(self.tableau[0]) - 1) if i not in self.basic_variables]}\n")
            f.write(f"Ratios calculés : {ratios}\n")
            f.write(f"Colonne pivot : {pivot_column}, Ligne pivot : {pivot_row}\n")

        pivot_value = self.tableau[pivot_row][pivot_column]
        self.tableau[pivot_row] = [x / pivot_value for x in self.tableau[pivot_row]]

        for i, row in enumerate(self.tableau):
            if i != pivot_row:
                multiplier = row[pivot_column]
                self.tableau[i] = [
                    a - multiplier * b for a, b in zip(row, self.tableau[pivot_row])
                ]
        print("État du tableau après l'itération :")
        column_names = [f"X{i+1}" for i in range(len(self.objective))] + [f"S{i+1}" for i in range(len(self.objective))]
        print(f"""\n{" | ".join(column_names)}\n{"-" * (len(column_names) * 10)}""")
        for row in self.tableau:
            print(" | ".join([f"{val:10.2f}" for val in row]))
        print("=" * 40)

        with open('resultats_solution.txt', 'a') as f:
            f.write("État du tableau après l'itération :\n")
            f.write(f"""\n{" | ".join(column_names)}\n{"-" * (len(column_names) * 10)}\n""")
            for row in self.tableau:
                f.write(" | ".join([f"{val:10.2f}" for val in row]) + '\n')
            f.write("=" * 40 + '\n')

        self.basic_variables[pivot_row] = pivot_column

    def solve(self):
        self.initialize_tableau()

        with open('resultats_solution.txt', 'w') as f:
            f.write("État initial :\n")
            column_names = [f"X{i+1}" for i in range(len(self.objective))] + [f"S{i+1}" for i in range(len(self.objective))]
            f.write(f"""\n{" | ".join(column_names)}\n{"-" * (len(column_names) * 10)}\n""")
            for row in self.tableau:
                f.write(" | ".join([f"{val:10.2f}" for val in row]) + '\n')
            f.write("=" * 40 + '\n')

        while not self.is_optimal():
            self.perform_iteration()

        with open('resultats_solution.txt', 'a') as f:
            f.write("Solution optimale : \n")
            f.write(str(-self.tableau[-1][-1]) + '\n')

        print("Solution optimale : ", -self.tableau[-1][-1])

        num_vars = len(self.objective)
        solution = [0] * num_vars
        for i, basic_var in enumerate(self.basic_variables):
            if basic_var < num_vars:
                solution[basic_var] = self.tableau[i][-1]
        objective_value = -self.tableau[-1][-1]

        return {"solution": solution, "objective_value": objective_value}