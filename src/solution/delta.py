"""Delta cost evaluation. O(1) per operation -- speed-critical. All @njit.

All distances/costs are i64 (meters). No float math on hot path.
Speed comes from vehicles array (VCOL_SPEED = microseconds per meter).
Travel time: time_s = dist_m * speed_us_per_m // 1_000_000
"""
import numpy as np
from numba import njit

from src.data.constants import (
    VEH_TRUCK, VEH_BIKE, COL_DEMAND, COL_TW_CLOSE, COL_TW_OPEN,
    COL_SERVICE, ACT_DELIVER, VCOL_SPEED, VCOL_CAPACITY,
    SOL_META, META_N_TRUCKS, META_N_BIKES,
    travel_time_s,
)
from src.solution._helpers import (
    get_route_arrays as _get_route_arrays,
    get_loads as _get_loads,
)


@njit(cache=True)
def insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix):
    """O(1) delta distance (i64 meters) if inserting customer at pos."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    cn = customer + 1

    if L == 0:
        return dist_matrix[0, cn] + dist_matrix[cn, 0]
    if pos == 0:
        old_first = stops[vid, 0] + 1
        return dist_matrix[0, cn] + dist_matrix[cn, old_first] - dist_matrix[0, old_first]
    if pos == L:
        old_last = stops[vid, L - 1] + 1
        return dist_matrix[old_last, cn] + dist_matrix[cn, 0] - dist_matrix[old_last, 0]

    a = stops[vid, pos - 1] + 1
    b = stops[vid, pos] + 1
    return dist_matrix[a, cn] + dist_matrix[cn, b] - dist_matrix[a, b]


@njit(cache=True)
def removal_cost_delta(sol, vtype, vid, pos, dist_matrix):
    """O(1) delta distance (i64 meters) if removing stop at pos."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    cn = stops[vid, pos] + 1

    if L == 1:
        return -(dist_matrix[0, cn] + dist_matrix[cn, 0])
    if pos == 0:
        nxt = stops[vid, 1] + 1
        return dist_matrix[0, nxt] - dist_matrix[0, cn] - dist_matrix[cn, nxt]
    if pos == L - 1:
        prv = stops[vid, L - 2] + 1
        return dist_matrix[prv, 0] - dist_matrix[prv, cn] - dist_matrix[cn, 0]

    prv = stops[vid, pos - 1] + 1
    nxt = stops[vid, pos + 1] + 1
    return dist_matrix[prv, nxt] - dist_matrix[prv, cn] - dist_matrix[cn, nxt]


@njit(cache=True)
def best_insertion_pos(sol, vtype, vid, customer, dist_matrix):
    """Best position in 1 route. O(route_len). Returns (pos: i32, delta: i64)."""
    _, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    best_pos = np.int32(0)
    best_delta = np.int64(2_000_000_000_000)  # large sentinel

    for pos in range(L + 1):
        delta = insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix)
        if delta < best_delta:
            best_delta = delta
            best_pos = np.int32(pos)
    return best_pos, best_delta


@njit(cache=True)
def _estimate_arrival_at_pos(sol, vtype, vid, pos, dist_matrix, customers, vehicles):
    """Estimate arrival time (seconds, i64) at position pos in route. O(pos) scan."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    gvid = vid if vtype == VEH_TRUCK else n_trucks + vid
    speed_us = vehicles[gvid, VCOL_SPEED]

    clock = np.int64(0)
    prev_dm = np.int32(0)
    target = min(pos, L)
    for i in range(target):
        c = stops[vid, i]
        dist_m = dist_matrix[prev_dm, c + 1]
        clock += travel_time_s(dist_m, speed_us)
        tw_open = customers[c, COL_TW_OPEN]
        if clock < tw_open:
            clock = tw_open
        clock += customers[c, COL_SERVICE]
        prev_dm = c + 1
    return clock, prev_dm


@njit(cache=True)
def _estimate_route_return_time(sol, vtype, vid, dist_matrix, customers, vehicles):
    """Estimate total route time (seconds, i64) including return to depot."""
    _, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    if L == 0:
        return np.int64(0)
    clock, prev_dm = _estimate_arrival_at_pos(sol, vtype, vid, L, dist_matrix, customers, vehicles)
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    gvid = vid if vtype == VEH_TRUCK else n_trucks + vid
    speed_us = vehicles[gvid, VCOL_SPEED]
    clock += travel_time_s(dist_matrix[prev_dm, 0], speed_us)
    return clock


@njit(cache=True)
def _check_tw_at_insertion(sol, vtype, vid, pos, customer, dist_matrix, customers, vehicles):
    """Quick check: would inserting customer at pos violate its TW?"""
    clock, prev_dm = _estimate_arrival_at_pos(sol, vtype, vid, pos, dist_matrix, customers, vehicles)
    cn = customer + 1
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    gvid = vid if vtype == VEH_TRUCK else n_trucks + vid
    speed_us = vehicles[gvid, VCOL_SPEED]
    arrival = clock + travel_time_s(dist_matrix[prev_dm, cn], speed_us)
    tw_close = customers[customer, COL_TW_CLOSE]
    return arrival <= tw_close


@njit(cache=True)
def cascade_removal_value(sol, vtype, vid, pos, dist_matrix, customers, vehicles):
    """Value of removing customer at pos (i64 meters). Higher = more valuable. O(L)."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    gvid = vid if vtype == VEH_TRUCK else n_trucks + vid
    speed_us = vehicles[gvid, VCOL_SPEED]

    if L <= 1:
        return -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)

    dist_savings = -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)
    cn = stops[vid, pos] + 1
    service_s = customers[stops[vid, pos], COL_SERVICE]

    prev_dm = np.int32(0) if pos == 0 else stops[vid, pos - 1] + 1
    next_dm = np.int32(0) if pos == L - 1 else stops[vid, pos + 1] + 1

    old_travel_s = travel_time_s(dist_matrix[prev_dm, cn] + dist_matrix[cn, next_dm], speed_us)
    new_travel_s = travel_time_s(dist_matrix[prev_dm, next_dm], speed_us)
    time_saved_s = old_travel_s - new_travel_s + service_s

    n_downstream = np.int32(0)
    for i in range(pos + 1, L):
        if actions[vid, i] == ACT_DELIVER:
            n_downstream += 1

    # Value in meters equivalent: dist_savings + time bonus scaled to distance
    # Use dist_savings as primary, add time_saved proportional to downstream impact
    cascade_bonus = n_downstream * time_saved_s
    return dist_savings + cascade_bonus


@njit(cache=True)
def find_best_insertion_all_routes(sol, vtype, customer, dist_matrix, customers, vehicles):
    """Best (vid, pos, delta) across all routes of vtype. delta is i64 meters."""
    demand = customers[customer, COL_DEMAND]
    meta = sol[SOL_META]
    n_vehicles = meta[META_N_TRUCKS] if vtype == VEH_TRUCK else meta[META_N_BIKES]
    loads = _get_loads(sol, vtype)

    best_vid = np.int32(-1)
    best_pos = np.int32(-1)
    best_delta = np.int64(2_000_000_000_000)  # large sentinel

    n_trucks = meta[META_N_TRUCKS]
    for vid in range(n_vehicles):
        gvid = vid if vtype == VEH_TRUCK else n_trucks + vid
        cap = vehicles[gvid, VCOL_CAPACITY]
        if loads[vid] + demand > cap:
            continue
        pos, delta = best_insertion_pos(sol, vtype, vid, customer, dist_matrix)
        if not _check_tw_at_insertion(sol, vtype, vid, pos, customer, dist_matrix, customers, vehicles):
            continue
        if delta < best_delta:
            best_vid, best_pos, best_delta = np.int32(vid), np.int32(pos), delta

    return best_vid, best_pos, best_delta
