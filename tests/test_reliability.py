import pytest
from orsolver.reliability import (
    series_reliability,
    parallel_reliability,
    k_out_of_n_reliability,
    exponential_reliability,
    weibull_reliability,
)


def test_series_and_parallel_reliability():
    # 3 components with reliabilities 0.9, 0.8, 0.7
    r = [0.9, 0.8, 0.7]

    # Series: 0.9 * 0.8 * 0.7 = 0.504
    r_ser = series_reliability(r)
    assert pytest.approx(r_ser, 1e-4) == 0.504

    # Parallel: 1 - (1-0.9)*(1-0.8)*(1-0.7) = 1 - 0.1 * 0.2 * 0.3 = 1 - 0.006 = 0.994
    r_par = parallel_reliability(r)
    assert pytest.approx(r_par, 1e-4) == 0.994


def test_k_out_of_n_reliability():
    # 2 out of 3 engines needed, each has reliability R = 0.9
    # R_2/3 = binom(3,2) * 0.9^2 * 0.1 + binom(3,3) * 0.9^3 * 1
    #       = 3 * 0.81 * 0.1 + 1 * 0.729 = 0.243 + 0.729 = 0.972
    r_k_n = k_out_of_n_reliability(k=2, n=3, component_reliability=0.9)
    assert pytest.approx(r_k_n, 1e-4) == 0.972


def test_time_dependent_reliability():
    # Exponential with rate lambda = 0.01 per hour at t = 100 hours
    # R(100) = exp(-0.01 * 100) = exp(-1) = 0.3678794
    # MTTF = 1 / lambda = 100
    r_exp, mttf = exponential_reliability(rate_lambda=0.01, t=100.0)
    assert pytest.approx(r_exp, 1e-4) == 0.3678794
    assert pytest.approx(mttf, 1e-4) == 100.0

    # Weibull with scale eta=100, shape beta=2.0 at t=100
    # R(100) = exp(-(100/100)^2) = exp(-1) = 0.3678794
    r_weib = weibull_reliability(scale_eta=100.0, shape_beta=2.0, t=100.0)
    assert pytest.approx(r_weib, 1e-4) == 0.3678794
