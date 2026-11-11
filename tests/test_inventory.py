import pytest
from orsolver.inventory import eoq_basic, eoq_with_shortages, reorder_point


def test_eoq_basic():
    """
    Annual demand D = 10,000 units
    Ordering cost S = $50 per order
    Holding cost H = $2 per unit per year
    Unit cost C = $10
    Q* = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) = 707.10678
    N* = 10000 / 707.10678 = 14.14 orders/year
    Total inventory holding + order cost = sqrt(2 * 10000 * 50 * 2) = 1414.21356
    """
    res = eoq_basic(demand=10000, order_cost=50, holding_cost=2, unit_cost=10)

    assert pytest.approx(res.optimal_quantity, 1e-4) == 707.10678
    assert pytest.approx(res.orders_per_year, 1e-4) == 14.1421356
    assert pytest.approx(res.total_variable_cost, 1e-3) == 1414.214
    assert pytest.approx(res.total_cost, 1e-3) == 101414.214


def test_eoq_with_shortages():
    """
    D = 10,000, S = $50, H = $2, Shortage penalty B = $8
    Q* = sqrt(2 * 10000 * 50 / 2) * sqrt((2 + 8) / 8) = 707.10678 * sqrt(1.25) = 790.5694
    Max inventory S_max = Q* * (8 / 10) = 632.4555
    Max backorders = Q* - S_max = 158.1139
    """
    res = eoq_with_shortages(demand=10000, order_cost=50, holding_cost=2, shortage_cost=8)

    assert pytest.approx(res.optimal_quantity, 1e-3) == 790.569
    assert pytest.approx(res.max_inventory, 1e-3) == 632.456
    assert pytest.approx(res.max_backorders, 1e-3) == 158.114


def test_reorder_point():
    """
    Daily demand = 50 units
    Lead time = 4 days
    Std dev of daily demand = 5
    Service level Z = 1.65 (approx 95% cycle service level)
    Safety stock SS = 1.65 * 5 * sqrt(4) = 1.65 * 10 = 16.5
    ROP = 50 * 4 + 16.5 = 216.5
    """
    res = reorder_point(daily_demand=50, lead_time_days=4, std_dev_demand=5, service_factor_z=1.65)

    assert pytest.approx(res.safety_stock, 1e-2) == 16.5
    assert pytest.approx(res.reorder_point, 1e-2) == 216.5
