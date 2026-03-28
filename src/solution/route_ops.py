"""Single-route operations: insert, remove, swap, reverse. All @njit."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_META, META_N_TRUCKS, META_MAX_ROUTE_LEN,
)
from src.solution._helpers import (
    get_route_arrays as _get_route_arrays,
    get_loads as _get_loads,
    global_vid as _global_vid,
    update_route_distance as _update_route_distance,
)


@njit(cache=True)
def insert_stop(sol, vtype, vid, pos, customer, action, dist_matrix, customers):
    """Insert stop at pos in route. Update index + distance + load cache."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    max_route_len = sol[SOL_META][META_MAX_ROUTE_LEN]

    if L >= max_route_len:
        return

    for j in range(L, pos, -1):
        stops[vid, j] = stops[vid, j - 1]
        actions[vid, j] = actions[vid, j - 1]

    stops[vid, pos] = customer
    actions[vid, pos] = action
    lengths[vid] += 1

    cust_vehicle = sol[SOL_CUST_VEHICLE]
    cust_vtype = sol[SOL_CUST_VTYPE]
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]

    if action == ACT_DELIVER:
        cust_vehicle[customer] = _global_vid(vtype, vid, n_trucks)
        cust_vtype[customer] = vtype
        cust_route_pos[customer] = pos

    for j in range(pos + 1, lengths[vid]):
        if actions[vid, j] == ACT_DELIVER:
            cust_route_pos[stops[vid, j]] = j

    _update_route_distance(sol, vtype, vid, dist_matrix)

    if action == ACT_DELIVER:
        _get_loads(sol, vtype)[vid] += customers[customer, COL_DEMAND]


@njit(cache=True)
def remove_stop(sol, vtype, vid, pos, dist_matrix, customers):
    """Remove stop at pos. Returns (removed_customer, removed_action)."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    if pos >= L or L == 0:
        return -1, -1

    removed_customer = stops[vid, pos]
    removed_action = actions[vid, pos]

    cust_vehicle = sol[SOL_CUST_VEHICLE]
    cust_vtype = sol[SOL_CUST_VTYPE]
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]

    if removed_action == ACT_DELIVER:
        cust_vehicle[removed_customer] = -1
        cust_vtype[removed_customer] = -1
        cust_route_pos[removed_customer] = -1

    for j in range(pos, L - 1):
        stops[vid, j] = stops[vid, j + 1]
        actions[vid, j] = actions[vid, j + 1]
    stops[vid, L - 1] = -1
    actions[vid, L - 1] = ACT_PAD
    lengths[vid] -= 1

    for j in range(pos, lengths[vid]):
        if actions[vid, j] == ACT_DELIVER:
            cust_route_pos[stops[vid, j]] = j

    _update_route_distance(sol, vtype, vid, dist_matrix)

    if removed_action == ACT_DELIVER:
        _get_loads(sol, vtype)[vid] -= customers[removed_customer, COL_DEMAND]

    return int(removed_customer), int(removed_action)


@njit(cache=True)
def swap_stops_within(sol, vtype, vid, pos_a, pos_b, dist_matrix):
    """Swap 2 stops in same route."""
    stops, actions, _ = _get_route_arrays(sol, vtype)
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]

    tmp_s = stops[vid, pos_a]
    tmp_a = actions[vid, pos_a]
    stops[vid, pos_a] = stops[vid, pos_b]
    actions[vid, pos_a] = actions[vid, pos_b]
    stops[vid, pos_b] = tmp_s
    actions[vid, pos_b] = tmp_a

    if actions[vid, pos_a] == ACT_DELIVER:
        cust_route_pos[stops[vid, pos_a]] = pos_a
    if actions[vid, pos_b] == ACT_DELIVER:
        cust_route_pos[stops[vid, pos_b]] = pos_b

    _update_route_distance(sol, vtype, vid, dist_matrix)


@njit(cache=True)
def reverse_segment(sol, vtype, vid, start, end, dist_matrix):
    """Reverse segment [start, end] inclusive. For 2-opt."""
    stops, actions, _ = _get_route_arrays(sol, vtype)
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]

    lo, hi = start, end
    while lo < hi:
        tmp_s = stops[vid, lo]
        tmp_a = actions[vid, lo]
        stops[vid, lo] = stops[vid, hi]
        actions[vid, lo] = actions[vid, hi]
        stops[vid, hi] = tmp_s
        actions[vid, hi] = tmp_a
        lo += 1
        hi -= 1

    for j in range(start, end + 1):
        if actions[vid, j] == ACT_DELIVER:
            cust_route_pos[stops[vid, j]] = j

    _update_route_distance(sol, vtype, vid, dist_matrix)
