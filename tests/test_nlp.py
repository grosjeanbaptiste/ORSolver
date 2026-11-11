import pytest
import numpy as np
from orsolver.nlp import golden_section_search, gradient_descent


def test_golden_section_search():
    """
    Min f(x) = (x - 3)^2 + 5 on [0, 10].
    Minimum at x = 3, f(3) = 5.
    """
    f = lambda x: (x - 3.0) ** 2 + 5.0
    x_opt, f_opt = golden_section_search(f, a=0.0, b=10.0, tol=1e-5)

    assert pytest.approx(x_opt, 1e-4) == 3.0
    assert pytest.approx(f_opt, 1e-4) == 5.0


def test_gradient_descent_quadratic():
    """
    Min f(x1, x2) = 2*(x1 - 1)^2 + 3*(x2 - 2)^2 + 4.
    Grad: [4*(x1 - 1), 6*(x2 - 2)].
    Minimum at (1, 2) with value 4.
    """
    f = lambda x: 2.0 * (x[0] - 1.0) ** 2 + 3.0 * (x[1] - 2.0) ** 2 + 4.0
    grad = lambda x: np.array([4.0 * (x[0] - 1.0), 6.0 * (x[1] - 2.0)])

    x_init = np.array([5.0, 5.0])
    x_opt, f_opt, iters = gradient_descent(f, grad, x0=x_init, tol=1e-5)

    assert pytest.approx(x_opt[0], 1e-4) == 1.0
    assert pytest.approx(x_opt[1], 1e-4) == 2.0
    assert pytest.approx(f_opt, 1e-4) == 4.0
