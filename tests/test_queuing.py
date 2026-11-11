import pytest
from orsolver.queuing import mm1_queue, mms_queue


def test_mm1_queue():
    """
    M/M/1 Queue:
    Arrival rate lambda = 3 customers/hour
    Service rate mu = 5 customers/hour
    rho = 3/5 = 0.6
    P0 = 1 - 0.6 = 0.4
    L = 3 / (5 - 3) = 1.5 customers
    Lq = 1.5 - 0.6 = 0.9 customers
    W = 1 / (5 - 3) = 0.5 hours (30 min)
    Wq = 0.9 / 3 = 0.3 hours (18 min)
    """
    res = mm1_queue(arrival_rate=3.0, service_rate=5.0)

    assert pytest.approx(res.utilization, 1e-4) == 0.6
    assert pytest.approx(res.p0, 1e-4) == 0.4
    assert pytest.approx(res.L, 1e-4) == 1.5
    assert pytest.approx(res.Lq, 1e-4) == 0.9
    assert pytest.approx(res.W, 1e-4) == 0.5
    assert pytest.approx(res.Wq, 1e-4) == 0.3


def test_mms_queue():
    """
    M/M/s Queue:
    Arrival rate lambda = 8 customers/hour
    Service rate mu = 5 customers/hour per server
    Servers s = 2
    r = lambda / mu = 8 / 5 = 1.6
    rho = 1.6 / 2 = 0.8
    P0 = [1 + 1.6 + (1.6^2 / 2) / (1 - 0.8)]^-1
       = [2.6 + 1.28 / 0.2]^-1 = [2.6 + 6.4]^-1 = 1 / 9 = 0.11111
    P_wait (Erlang-C) = P0 * (r^s / (s! * (1 - rho))) = (1/9) * (1.28 / 0.2) = 6.4 / 9 = 0.7111
    Lq = rho * P_wait / (1 - rho) = 0.8 * (6.4 / 9) / 0.2 = 4 * (6.4 / 9) = 25.6 / 9 = 2.8444
    Wq = Lq / lambda = (25.6 / 9) / 8 = 3.2 / 9 = 0.35555
    """
    res = mms_queue(arrival_rate=8.0, service_rate=5.0, servers=2)

    assert pytest.approx(res.utilization, 1e-4) == 0.8
    assert pytest.approx(res.p0, 1e-4) == 1.0 / 9.0
    assert pytest.approx(res.Lq, 1e-4) == 25.6 / 9.0
    assert pytest.approx(res.Wq, 1e-4) == 3.2 / 9.0
    assert pytest.approx(res.W, 1e-4) == res.Wq + 1.0 / 5.0
    assert pytest.approx(res.L, 1e-4) == 8.0 * res.W


def test_unstable_queue_raises_error():
    with pytest.raises(ValueError, match="unstable"):
        mm1_queue(arrival_rate=5.0, service_rate=5.0)

    with pytest.raises(ValueError, match="unstable"):
        mms_queue(arrival_rate=12.0, service_rate=5.0, servers=2)
