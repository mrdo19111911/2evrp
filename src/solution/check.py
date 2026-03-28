"""Quick O(1) constraint checks for ALNS. @njit."""
import numpy as np
from numba import njit

from src.data.constants import (
    VEH_TRUCK, VEH_BIKE, COL_DEMAND, COL_RESTRICTED,
    SOL_CUST_VEHICLE, SOL_META, META_MAX_ROUTE_LEN,
)
from src.solution._helpers import get_route_arrays, get_loads
from src.solution.query import is_customer_assigned, get_customer_info


@njit(cache=True)
def can_insert_customer(sol, vtype, vid, customer, customers, vehicle_capacity):
    """O(1) feasibility: capacity + restriction + route space."""
    if vtype == VEH_TRUCK and customers[customer, COL_RESTRICTED] == 1:
        return False
    demand = customers[customer, COL_DEMAND]
    if get_loads(sol, vtype)[vid] + demand > vehicle_capacity:
        return False
    _, _, lengths = get_route_arrays(sol, vtype)
    if lengths[vid] >= sol[SOL_META][META_MAX_ROUTE_LEN]:
        return False
    return True


@njit(cache=True)
def can_swap_customers(sol, cust_a, cust_b, customers, vehicles):
    """O(1) feasibility: can swap 2 customers between their routes?"""
    vtype_a, vid_a, _ = get_customer_info(sol, cust_a)
    vtype_b, vid_b, _ = get_customer_info(sol, cust_b)
    if vtype_a == -1 or vtype_b == -1:
        return False

    demand_a = customers[cust_a, COL_DEMAND]
    demand_b = customers[cust_b, COL_DEMAND]

    n_trucks = sol[SOL_META][0]
    load_a = get_loads(sol, vtype_a)[vid_a] - demand_a + demand_b
    gvid_a = vid_a if vtype_a == VEH_TRUCK else n_trucks + vid_a
    if load_a > vehicles[gvid_a, 1]:
        return False

    load_b = get_loads(sol, vtype_b)[vid_b] - demand_b + demand_a
    gvid_b = vid_b if vtype_b == VEH_TRUCK else n_trucks + vid_b
    if load_b > vehicles[gvid_b, 1]:
        return False

    if vtype_b == VEH_TRUCK and customers[cust_a, COL_RESTRICTED] == 1:
        return False
    if vtype_a == VEH_TRUCK and customers[cust_b, COL_RESTRICTED] == 1:
        return False
    return True


@njit(cache=True)
def check_route_capacity_quick(sol, vtype, vid, vehicle_capacity):
    return get_loads(sol, vtype)[vid] <= vehicle_capacity


@njit(cache=True)
def check_all_assigned(sol, n_customers):
    for i in range(n_customers):
        if sol[SOL_CUST_VEHICLE][i] < 0:
            return False
    return True
