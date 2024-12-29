import logging
import os
import sys

from GraphicalORSolver import GraphicalORSolver
from SimplexSolver import SimplexSolver

logging.basicConfig(level=logging.INFO)


def save_results_to_file(results, filename):
    """Save results to a specified file."""
    with open(filename, 'w') as file:
        file.write(results)


if __name__ == "__main__":
    constraints_example = [
        ([4, 3], "<=", 20),
        ([2, -1], "<=", 0),
    ]
    objective_example = [40, 25]

    simplex_solver = SimplexSolver(objective_example, constraints_example)
    simplex_result = simplex_solver.solve()

    print("Simplex Method:")
    print("Optimal Solution:", simplex_result)

    visualizer = GraphicalORSolver(
        constraints_example, objective=objective_example
    )
    graphical_result = visualizer.calculate_graphical_solution()

    print("\nGraphical Method:")
    print("Optimal Solution:", graphical_result)

    visualizer.plot(graphical_result["solution"])

    formula = ' + '.join([f'{coef[0]}x + {coef[1]}y {op} {val}' for coef, op, val in constraints_example])
    formula += f' (Maximize: {objective_example[0]}x + {objective_example[1]}y)'
    
    calculation_folder = f'solutions/calculation_{formula.replace(" ", "_").replace("<=", "leq").replace("+", "plus").replace("(", "_").replace(")", "_").replace(":", "_").replace(" ", "_")}'
    if not os.path.exists('solutions'):
        os.makedirs('solutions')
    if not os.path.exists(calculation_folder):
        os.makedirs(calculation_folder)

    image_filename_graphical = f'{calculation_folder}/graphique_solution.png'
    visualizer.save_plot(filename=image_filename_graphical, format='png')


    filename = f'{calculation_folder}/resultats_solution.txt'

    results = 'Voici les résultats de la solution : \n Simplex Method: \n Optimal Solution: {}\n\n Graphical Method: \n Optimal Solution: {}'.format(simplex_result, graphical_result)
    save_results_to_file(results, filename)

    sys.exit()
