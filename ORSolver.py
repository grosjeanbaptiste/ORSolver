import logging

from GraphicalORSolver import GraphicalORSolver
from SimplexSolver import SimplexSolver

logging.basicConfig(level=logging.INFO)


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
