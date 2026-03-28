"""O(1) lookups and solution queries. @njit where possible."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_META, META_N_TRUCKS,
)
from src.solution._helpers import get_route_arrays as _get_route_arrays


@njit(cache=True)
def get_unassigned_customers(sol, n_customers):
    """Customers not yet delivered."""
    return np.where(sol[SOL_CUST_VEHICLE][:n_customers] == -1)[0]


@njit(cache=True)
def get_assigned_customers(sol, n_customers):
    """Customers already delivered."""
    return np.where(sol[SOL_CUST_VEHICLE][:n_customers] >= 0)[0]


@njit(cache=True)
def get_route_customers_only(sol, vtype, vid):
    """Only DELIVER customers in route. Returns int32 array."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    if L == 0:
        return np.empty(0, dtype=np.int32)
    mask = actions[vid, :L] == ACT_DELIVER
    result = stops[vid, :L][mask]
    out = np.empty(len(result), dtype=np.int32)
    for i in range(len(result)):
        out[i] = result[i]
    return out


@njit(cache=True)
def is_customer_assigned(sol, customer):
    """O(1) check."""
    return sol[SOL_CUST_VEHICLE][customer] >= 0


@njit(cache=True)
def get_customer_info(sol, customer):
    """O(1) lookup. Returns (vtype, vid, pos) or (-1,-1,-1)."""
    veh = sol[SOL_CUST_VEHICLE][customer]
    if veh == -1:
        return -1, -1, -1
    vtype = sol[SOL_CUST_VTYPE][customer]
    pos = sol[SOL_CUST_ROUTE_POS][customer]
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    if vtype == VEH_TRUCK:
        vid = veh
    else:
        vid = veh - n_trucks
    return int(vtype), int(vid), int(pos)
