# Linear Programming Solver

This project is a Python-based tool designed to solve linear programming problems using two approaches:

1. **Simplex Method**: A numerical algorithm to find the optimal solution to linear programming problems.
2. **Graphical Method**: A visual representation of the problem, highlighting the feasible region, constraints, and optimal solution.

---

## Features

- **Dual Calculation**:
  - Solves problems using the Simplex method for exact numerical results.
  - Provides graphical representation to visualize the feasible region and optimal solution.
- **Dynamic Visualization**:
  - Highlights the feasible region clearly.
  - Displays constraints with clear labels.
  - Marks the optimal solution explicitly with annotations.
  - Plots the optimal objective function line dynamically.
- **Comparison**:
  - Outputs results from both the Simplex and Graphical methods for verification and understanding.

---

## Input Description

### Constraints

The constraints represent the problem's limitations and are defined as inequalities or equalities involving variables \(x_1, x_2, \ldots\). Each constraint specifies:

- **Coefficients**: The weights of the variables in the equation.
- **Operator**: The type of relationship (\(\leq, \geq, =\)).
- **Value**: The constant on the right-hand side of the equation.

**Example**:  
\(6x_1 + 4x_2 \leq 24\)

### Objective Function

The objective function is the formula to be optimized (maximized or minimized), typically represented as:

\[
z = c_1x_1 + c_2x_2 + \ldots
\]

- **Coefficients**: Represent the weights or contributions of each variable to the function.
- **Objective**: The target to either maximize or minimize.

**Example**:  
Maximize \(z = 5x_1 + 4x_2\)

---

## Visualization

The graphical method includes:

- **Constraints**:
  - Plotted as lines with distinct colors for easy differentiation.
  - Labeled directly on the graph for clarity.
- **Feasible Region**:
  - Highlighted with a transparent fill to clearly show the valid solution space.
- **Optimal Solution**:
  - Marked with a bold red dot.
  - Annotated with the coordinates and objective value.
- **Objective Function**:
  - Displayed as a dashed line passing through the optimal solution.
  - Annotated with its value.

---

## Output

### Simplex Method
The Simplex method outputs:
- **Optimal Solution**: Values of the variables (\(x_1, x_2, \ldots\)) that optimize the objective function.
- **Objective Value**: The optimized value of \(z\).

### Graphical Method
The graphical method displays:
- A clear visualization of constraints.
- The feasible region and its boundaries.
- The optimal solution and the corresponding objective function.

---

## Application

This tool is ideal for:
- Teaching and learning linear programming concepts.
- Visualizing and solving small-scale optimization problems.
- Comparing numerical and graphical approaches for understanding and verification.

---

## License

This project is licensed under the MIT License. Use it freely for educational and professional purposes.

---

## Acknowledgments

Special thanks to the creators and maintainers of `matplotlib` and `numpy` for providing the tools used to develop this solver.