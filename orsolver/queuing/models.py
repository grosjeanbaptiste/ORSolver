import math
from dataclasses import dataclass


@dataclass
class QueuePerformance:
    arrival_rate: float
    service_rate: float
    servers: int
    utilization: float
    p0: float
    L: float
    Lq: float
    W: float
    Wq: float


def mm1_queue(arrival_rate: float, service_rate: float) -> QueuePerformance:
    """
    Analyzes an M/M/1 queue (Poisson arrivals, exponential service, 1 server).
    """
    lam = float(arrival_rate)
    mu = float(service_rate)

    if lam >= mu:
        raise ValueError(f"Queue is unstable: arrival rate ({lam}) >= service rate ({mu}).")

    rho = lam / mu
    p0 = 1.0 - rho
    L = lam / (mu - lam)
    Lq = (lam ** 2) / (mu * (mu - lam))
    W = 1.0 / (mu - lam)
    Wq = lam / (mu * (mu - lam))

    return QueuePerformance(
        arrival_rate=lam,
        service_rate=mu,
        servers=1,
        utilization=rho,
        p0=p0,
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
    )


def mms_queue(arrival_rate: float, service_rate: float, servers: int) -> QueuePerformance:
    """
    Analyzes an M/M/s queue (Poisson arrivals, exponential service, s servers).
    Uses Erlang-C formula for waiting probability.
    """
    lam = float(arrival_rate)
    mu = float(service_rate)
    s = int(servers)

    if lam >= s * mu:
        raise ValueError(f"Queue is unstable: arrival rate ({lam}) >= total service rate ({s * mu}).")

    r = lam / mu
    rho = r / s

    # Calculate P0
    sum_terms = sum((r ** n) / math.factorial(n) for n in range(s))
    last_term = (r ** s) / (math.factorial(s) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)

    # Erlang-C formula: P(Wait)
    p_wait = p0 * last_term

    # Queue length and waiting times
    Lq = (rho * p_wait) / (1.0 - rho)
    Wq = Lq / lam
    W = Wq + 1.0 / mu
    L = lam * W

    return QueuePerformance(
        arrival_rate=lam,
        service_rate=mu,
        servers=s,
        utilization=rho,
        p0=p0,
        L=L,
        Lq=Lq,
        W=W,
        Wq=Wq,
    )
