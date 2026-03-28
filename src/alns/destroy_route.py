"""Route-level destroy operators: route removal, satellite, split, cluster.
Signature: (sol, customers_i64, dist_matrix_i64, seed_i32, config_f64) -> i32[:]
No restricted param. All @njit.
"""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, SAT_CUST, SAT_BIKE,
    SOL_SATELLITES, SOL_META, META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
    CFG_Q_MIN, CFG_Q_MAX, CFG_ZONE_PCT, VCOL_SPEED,
)
from src.solution.query import (
    get_assigned_customers, get_customer_info, get_route_customers_only,
)
from src.solution.route_ops import remove_stop
from src.solution.solution_ops import clear_route
from src.solution.satellite_ops import remove_satellites_for_customer, remove_satellites_for_bike
from src.solution._helpers import get_route_arrays
from src.alns.destroy_helpers import remove_targets, sort_by_pos_desc, nonempty_routes


@njit(cache=True)
def route_removal(sol, customers, dist_matrix, seed, config):
    """Remove entire route."""
    np.random.seed(seed)
    ne = nonempty_routes(sol)
    if len(ne) == 0:
        return np.empty(0, dtype=np.int32)
    idx = np.random.randint(len(ne))
    vtype, vid = int(ne[idx, 0]), int(ne[idx, 1])
    removed_pairs = clear_route(sol, vtype, vid)
    removed = np.empty(len(removed_pairs), dtype=np.int32)
    n = 0
    for i in range(len(removed_pairs)):
        c, action = removed_pairs[i, 0], removed_pairs[i, 1]
        remove_satellites_for_customer(sol, c)
        if action == ACT_DELIVER:
            removed[n] = c
            n += 1
    return removed[:n]


@njit(cache=True)
def satellite_removal(sol, customers, dist_matrix, seed, config):
    """Remove satellite + all dependent bike customers."""
    np.random.seed(seed)
    sats = sol[SOL_SATELLITES]
    n_sats = int(sol[SOL_META][META_N_SATELLITES])
    if n_sats == 0:
        return np.empty(0, dtype=np.int32)
    # Manual unique of sat_cust values
    seen = np.zeros(len(customers) + 100, dtype=np.int8)
    unique_buf = np.empty(n_sats, dtype=np.int32)
    n_unique = 0
    for i in range(n_sats):
        sc = int(sats[i, SAT_CUST])
        if seen[sc] == 0:
            seen[sc] = 1
            unique_buf[n_unique] = sc
            n_unique += 1
    if n_unique == 0:
        return np.empty(0, dtype=np.int32)
    sat_cust = int(unique_buf[np.random.randint(0, n_unique)])
    # Collect bike_ids for this sat_cust — manual unique
    bike_seen = np.zeros(int(sol[SOL_META][META_N_BIKES]) + 1, dtype=np.int8)
    bike_buf = np.empty(n_sats, dtype=np.int32)
    n_bikes_found = 0
    for i in range(n_sats):
        if int(sats[i, SAT_CUST]) == sat_cust:
            bid = int(sats[i, SAT_BIKE])
            if bid < len(bike_seen) and bike_seen[bid] == 0:
                bike_seen[bid] = 1
                bike_buf[n_bikes_found] = bid
                n_bikes_found += 1
    n_bikes = int(sol[SOL_META][META_N_BIKES])
    removed = np.empty(200, dtype=np.int32)
    n = 0
    for bi in range(n_bikes_found):
        bid = int(bike_buf[bi])
        if bid >= n_bikes:
            continue
        deliver_custs = get_route_customers_only(sol, VEH_BIKE, bid)
        sorted_c = sort_by_pos_desc(deliver_custs, sol)
        for ci in range(len(sorted_c)):
            c = int(sorted_c[ci])
            vtype, vid, pos = get_customer_info(sol, c)
            if vtype == -1:
                continue
            remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
            if n < 200:
                removed[n] = c
                n += 1
        clear_route(sol, VEH_BIKE, bid)
        remove_satellites_for_bike(sol, bid)
    return removed[:n]


@njit(cache=True)
def route_split_removal(sol, customers, dist_matrix, seed, config):
    """Split longest route: keep first half, remove second half."""
    np.random.seed(seed)
    meta = sol[SOL_META]
    worst_vtype, worst_vid, worst_L = -1, -1, 0
    for vtype in (VEH_TRUCK, VEH_BIKE):
        _, _, lengths = get_route_arrays(sol, vtype)
        n_veh = int(meta[0]) if vtype == VEH_TRUCK else int(meta[1])
        for v in range(n_veh):
            L = int(lengths[v])
            if L > worst_L:
                worst_L = L
                worst_vtype = vtype
                worst_vid = v
    if worst_L < 6:
        return np.empty(0, dtype=np.int32)

    split_pos = worst_L // 2
    if split_pos < 3:
        split_pos = 3
    if split_pos > worst_L - 3:
        split_pos = worst_L - 3

    stops, actions, _ = get_route_arrays(sol, worst_vtype)
    removed = np.empty(worst_L, dtype=np.int32)
    n = 0
    for i in range(worst_L - 1, split_pos - 1, -1):
        if int(actions[worst_vid, i]) == ACT_DELIVER:
            c = int(stops[worst_vid, i])
            vtype2, vid2, pos2 = get_customer_info(sol, c)
            if vtype2 != -1:
                remove_stop(sol, vtype2, vid2, pos2, dist_matrix, customers)
                remove_satellites_for_customer(sol, c)
                removed[n] = c
                n += 1
    return removed[:n]


@njit(cache=True)
def cluster_removal(sol, customers, dist_matrix, seed, config):
    """Remove ALL routes serving a geographic zone. Coordinates i64 meters."""
    np.random.seed(seed)
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)
    center = int(assigned[np.random.randint(0, len(assigned))])
    cx = customers[center, COL_X]
    cy = customers[center, COL_Y]
    # Squared distances
    dists_sq = np.empty(len(assigned), dtype=np.int64)
    for i in range(len(assigned)):
        dx = customers[assigned[i], COL_X] - cx
        dy = customers[assigned[i], COL_Y] - cy
        dists_sq[i] = dx * dx + dy * dy
    pct = config[CFG_ZONE_PCT]
    sorted_d = np.sort(dists_sq)
    idx = int(len(sorted_d) * pct / 100.0)
    if idx > len(sorted_d) - 1:
        idx = len(sorted_d) - 1
    radius_sq = sorted_d[idx]
    # Collect zone customers
    zone_buf = np.empty(len(assigned), dtype=np.int32)
    n_zone = 0
    for i in range(len(assigned)):
        if dists_sq[i] <= radius_sq:
            zone_buf[n_zone] = assigned[i]
            n_zone += 1
    # Collect unique routes via boolean 2D seen array
    max_trucks = int(sol[SOL_META][META_N_TRUCKS])
    max_bikes = int(sol[SOL_META][META_N_BIKES])
    max_vid = max_trucks
    if max_bikes > max_vid:
        max_vid = max_bikes
    route_seen = np.zeros((3, max_vid + 1), dtype=np.int8)
    route_vt = np.empty(n_zone, dtype=np.int32)
    route_vi = np.empty(n_zone, dtype=np.int32)
    n_routes = 0
    for i in range(n_zone):
        c = int(zone_buf[i])
        vtype, vid, _ = get_customer_info(sol, c)
        if vtype != -1 and route_seen[vtype, vid] == 0:
            route_seen[vtype, vid] = 1
            route_vt[n_routes] = vtype
            route_vi[n_routes] = vid
            n_routes += 1
    removed = np.empty(N, dtype=np.int32)
    n = 0
    for ri in range(n_routes):
        vtype = int(route_vt[ri])
        vid = int(route_vi[ri])
        removed_pairs = clear_route(sol, vtype, vid)
        for i in range(len(removed_pairs)):
            c = removed_pairs[i, 0]
            action = removed_pairs[i, 1]
            remove_satellites_for_customer(sol, int(c))
            if action == ACT_DELIVER:
                removed[n] = c
                n += 1
    return removed[:n]
