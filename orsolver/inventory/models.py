import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class EOQResult:
    optimal_quantity: float
    orders_per_year: float
    cycle_time_years: float
    annual_order_cost: float
    annual_holding_cost: float
    total_variable_cost: float
    total_cost: float
    max_inventory: Optional[float] = None
    max_backorders: Optional[float] = None


@dataclass
class ROPResult:
    expected_lead_time_demand: float
    safety_stock: float
    reorder_point: float


def eoq_basic(
    demand: float,
    order_cost: float,
    holding_cost: float,
    unit_cost: float = 0.0,
) -> EOQResult:
    """
    Classic Economic Order Quantity (Wilson formula).
    """
    D = float(demand)
    S = float(order_cost)
    H = float(holding_cost)
    C = float(unit_cost)

    Q = math.sqrt((2.0 * D * S) / H)
    N = D / Q
    T = Q / D

    annual_order = (D / Q) * S
    annual_holding = (Q / 2.0) * H
    var_cost = annual_order + annual_holding
    tot_cost = D * C + var_cost

    return EOQResult(
        optimal_quantity=Q,
        orders_per_year=N,
        cycle_time_years=T,
        annual_order_cost=annual_order,
        annual_holding_cost=annual_holding,
        total_variable_cost=var_cost,
        total_cost=tot_cost,
        max_inventory=Q,
        max_backorders=0.0,
    )


def eoq_with_shortages(
    demand: float,
    order_cost: float,
    holding_cost: float,
    shortage_cost: float,
    unit_cost: float = 0.0,
) -> EOQResult:
    """
    EOQ with planned shortages / backorders.
    """
    D = float(demand)
    S = float(order_cost)
    H = float(holding_cost)
    B = float(shortage_cost)
    C = float(unit_cost)

    Q = math.sqrt((2.0 * D * S) / H) * math.sqrt((H + B) / B)
    S_max = Q * (B / (H + B))
    b_max = Q - S_max

    N = D / Q
    T = Q / D

    annual_order = (D / Q) * S
    annual_holding = ((S_max ** 2) / (2.0 * Q)) * H
    annual_shortage = ((b_max ** 2) / (2.0 * Q)) * B
    var_cost = annual_order + annual_holding + annual_shortage
    tot_cost = D * C + var_cost

    return EOQResult(
        optimal_quantity=Q,
        orders_per_year=N,
        cycle_time_years=T,
        annual_order_cost=annual_order,
        annual_holding_cost=annual_holding,
        total_variable_cost=var_cost,
        total_cost=tot_cost,
        max_inventory=S_max,
        max_backorders=b_max,
    )


def reorder_point(
    daily_demand: float,
    lead_time_days: float,
    std_dev_demand: float = 0.0,
    service_factor_z: float = 0.0,
) -> ROPResult:
    """
    Calculates Reorder Point (ROP) with safety stock.
    ROP = (daily_demand * lead_time) + (Z * std_dev * sqrt(lead_time))
    """
    d = float(daily_demand)
    L = float(lead_time_days)
    sigma_d = float(std_dev_demand)
    z = float(service_factor_z)

    expected_demand = d * L
    safety_stock = z * sigma_d * math.sqrt(L)
    rop = expected_demand + safety_stock

    return ROPResult(
        expected_lead_time_demand=expected_demand,
        safety_stock=safety_stock,
        reorder_point=rop,
    )
