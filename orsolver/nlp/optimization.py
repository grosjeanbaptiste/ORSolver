import math
from typing import Callable, Tuple, Union
import numpy as np


def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5,
    max_iter: int = 500,
) -> Tuple[float, float]:
    """
    Finds the minimum of a unimodal 1D function f on [a, b] using Golden Section Search.
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.6180339887

    a = float(a)
    b = float(b)
    c = b - phi * (b - a)
    d = a + phi * (b - a)

    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)

    x_min = (a + b) / 2.0
    return x_min, float(f(x_min))


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: Union[list, np.ndarray],
    tol: float = 1e-5,
    max_iter: int = 1000,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
) -> Tuple[np.ndarray, float, int]:
    """
    Unconstrained nonlinear optimization using Gradient Descent with
    Armijo Backtracking Line Search.
    """
    x = np.array(x0, dtype=float).copy()

    for it in range(max_iter):
        g = np.array(grad(x), dtype=float)
        norm_g = np.linalg.norm(g)

        if norm_g < tol:
            return x, float(f(x)), it

        d = -g  # Descent direction
        alpha = alpha_init

        # Armijo condition: f(x + alpha * d) <= f(x) + c * alpha * (g^T d)
        f_x = f(x)
        dir_deriv = np.dot(g, d)

        while f(x + alpha * d) > f_x + c * alpha * dir_deriv:
            alpha *= beta
            if alpha < 1e-12:
                break

        x += alpha * d

    return x, float(f(x)), max_iter
