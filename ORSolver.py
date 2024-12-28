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

    visualizer = GraphicalORSolver(
        constraints_example, objective=objective_example
    )
    graphical_result = visualizer.calculate_graphical_solution()

    print("\nGraphical Method:")
    print("Optimal Solution:", graphical_result)

    visualizer.plot(graphical_result["solution"])

    formula = ' + '.join([f'{coef[0]}x + {coef[1]}y {op} {val}' for coef, op, val in constraints_example])
    formula += f' (Maximize: {objective_example[0]}x + {objective_example[1]}y)'
    image_filename = f'solutions/graphique_solution_{formula.replace(" ", "_").replace("<=", "leq").replace("+", "plus").replace("(", "_").replace(")", "_").replace(":", "_").replace(" ", "_")}.png'
    visualizer.save_plot(image_filename)

    if not os.path.exists('solutions'):
        os.makedirs('solutions')

    filename = f'solutions/resultats_solution_{formula.replace(" ", "_").replace("<=", "leq").replace("+", "plus").replace("(", "_").replace(")", "_")}.txt'

    results = 'Voici les résultats de la solution : \n Simplex Method: \n Optimal Solution: {}\n\n Graphical Method: \n Optimal Solution: {}'.format(simplex_result, graphical_result)
    save_results_to_file(results, filename)

    sys.exit()  
    sys.exit()  
