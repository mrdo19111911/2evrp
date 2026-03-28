"""Quick O(1) constraint checks for ALNS -- no simulation needed."""
import numpy as np

from src.data.constants import VEH_TRUCK, VEH_BIKE, COL_DEMAND
from src.data.cost import TRUCK_CAPACITY, BIKE_CAPACITY
from src.solution.query import is_customer_assigned, get_customer_info


def _get_loads(sol, vtype):
    return sol["truck_loads"] if vtype == VEH_TRUCK else sol["bike_loads"]


def _get_lengths(sol, vtype):
    return sol["truck_lengths"] if vtype == VEH_TRUCK else sol["bike_lengths"]


def can_insert_customer(sol, vtype, vid, customer, customers, restricted, vehicle_capacity):
    """O(1) feasibility: capacity + restriction + route space."""
    if vtype == VEH_TRUCK and restricted[customer] == 1:
        return False
    demand = customers[customer, COL_DEMAND]
    if _get_loads(sol, vtype)[vid] + demand > vehicle_capacity:
        return False
    if _get_lengths(sol, vtype)[vid] >= sol["max_route_len"] - 1:
        return False
    return True


def can_move_customer(sol, customer, to_vtype, to_vid, customers, restricted, vehicle_capacity):
    """O(1) feasibility: can customer move to target route?"""
    if not is_customer_assigned(sol, customer):
        return False
    return can_insert_customer(sol, to_vtype, to_vid, customer, customers, restricted,
                               vehicle_capacity)


def can_swap_customers(sol, cust_a, cust_b, customers, restricted):
    """O(1) feasibility: can swap 2 customers between their routes?"""
    vtype_a, vid_a, _ = get_customer_info(sol, cust_a)
    vtype_b, vid_b, _ = get_customer_info(sol, cust_b)

    if vtype_a == -1 or vtype_b == -1:
        return False

    demand_a = customers[cust_a, COL_DEMAND]
    demand_b = customers[cust_b, COL_DEMAND]

    # Route A loses demand_a, gains demand_b
    load_a = _get_loads(sol, vtype_a)[vid_a] - demand_a + demand_b
    cap_a = TRUCK_CAPACITY if vtype_a == VEH_TRUCK else BIKE_CAPACITY
    if load_a > cap_a:
        return False

    # Route B loses demand_b, gains demand_a
    load_b = _get_loads(sol, vtype_b)[vid_b] - demand_b + demand_a
    cap_b = TRUCK_CAPACITY if vtype_b == VEH_TRUCK else BIKE_CAPACITY
    if load_b > cap_b:
        return False

    # Restriction checks
    if vtype_b == VEH_TRUCK and restricted[cust_a] == 1:
        return False
    if vtype_a == VEH_TRUCK and restricted[cust_b] == 1:
        return False

    return True


def check_route_capacity_quick(sol, vtype, vid, vehicle_capacity):
    """O(1) total demand <= capacity?"""
    return bool(_get_loads(sol, vtype)[vid] <= vehicle_capacity)


def check_all_assigned(sol, n_customers):
    """O(N) all customers delivered?"""
    return bool(np.all(sol["cust_vehicle"][:n_customers] >= 0))
