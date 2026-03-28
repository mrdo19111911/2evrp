"""Shared fixtures for ALNS operator tests. ALL i64 per INTERFACE.md."""
import numpy as np
import pytest

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS, SOL_TRUCK_LOADS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS, SOL_BIKE_LOADS,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_META, META_N_TRUCKS, META_N_BIKES, META_N_CUSTOMERS,
    COL_DEMAND,
)
from src.solution.structure import create_solution


def _assign_customer_simple(sol, customer_id, vtype, vid, pos, demand):
    """Manually assign a customer to a route position (no dist_matrix needed)."""
    if vtype == VEH_TRUCK:
        stops = sol[SOL_TRUCK_STOPS]
        actions = sol[SOL_TRUCK_ACTIONS]
        lengths = sol[SOL_TRUCK_LENGTHS]
        loads = sol[SOL_TRUCK_LOADS]
    else:
        stops = sol[SOL_BIKE_STOPS]
        actions = sol[SOL_BIKE_ACTIONS]
        lengths = sol[SOL_BIKE_LENGTHS]
        loads = sol[SOL_BIKE_LOADS]

    stops[vid, pos] = customer_id
    actions[vid, pos] = ACT_DELIVER
    lengths[vid] = max(lengths[vid], pos + 1)
    loads[vid] += demand

    n_trucks = sol[SOL_META][META_N_TRUCKS]
    sol[SOL_CUST_VEHICLE][customer_id] = vid if vtype == VEH_TRUCK else n_trucks + vid
    sol[SOL_CUST_VTYPE][customer_id] = vtype
    sol[SOL_CUST_ROUTE_POS][customer_id] = pos


@pytest.fixture
def sol_5_assigned(tiny_instance, tiny_dist_matrix):
    """5 customers all assigned: C0 to truck, C1-C4 to bikes."""
    data = tiny_instance
    n_trucks = data["n_trucks"]
    n_bikes = data["n_bikes"]
    n_cust = data["n_customers"]
    customers = data["customers"]

    sol = create_solution(n_trucks, n_bikes, n_cust)

    # C0 (100kg) -> truck 0
    _assign_customer_simple(sol, 0, VEH_TRUCK, 0, 0, customers[0, COL_DEMAND])
    # C1 (20kg) -> bike 0
    _assign_customer_simple(sol, 1, VEH_BIKE, 0, 0, customers[1, COL_DEMAND])
    # C2 (10kg) -> bike 0
    _assign_customer_simple(sol, 2, VEH_BIKE, 0, 1, customers[2, COL_DEMAND])
    # C3 (8kg) -> bike 1
    _assign_customer_simple(sol, 3, VEH_BIKE, 1, 0, customers[3, COL_DEMAND])
    # C4 (5kg) -> bike 1
    _assign_customer_simple(sol, 4, VEH_BIKE, 1, 1, customers[4, COL_DEMAND])

    return {
        "sol": sol,
        "customers": customers,
        "vehicles": data["vehicles"],
        "dist_matrix": tiny_dist_matrix,
        "n_trucks": n_trucks,
        "n_bikes": n_bikes,
        "N": n_cust,
    }
