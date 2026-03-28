"""Echelon rebalance — move customers between satellite zones. All @njit.
Signature: (sol, customers_i64, dist_matrix_i64, vehicles_i64, seed, config_f64) -> bool
No restricted param.
"""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_BIKE, COL_DEMAND,
    VCOL_CAPACITY, SAT_CUST, SAT_BIKE,
    SOL_SATELLITES, SOL_META, META_N_BIKES, META_N_SATELLITES,
)
from src.solution.query import get_assigned_customers, get_customer_info, get_route_customers_only
from src.solution.route_ops import remove_stop, insert_stop
from src.solution.delta import best_insertion_pos
from src.solution._helpers import get_loads, get_route_arrays


@njit(cache=True)
def echelon_rebalance(sol, customers, dist_matrix, vehicles, seed, config):
    """Move customers from overloaded to underloaded satellites. i64 loads."""
    np.random.seed(seed)
    sats = sol[SOL_SATELLITES]
    n_sats = sol[SOL_META][META_N_SATELLITES]
    if n_sats == 0:
        return False
    N = len(customers)
    n_bikes = sol[SOL_META][META_N_BIKES]

    # Get bike capacity from vehicles
    bike_cap = np.int64(0)
    for v in range(len(vehicles)):
        if vehicles[v, 0] == VEH_BIKE and vehicles[v, VCOL_CAPACITY] > bike_cap:
            bike_cap = vehicles[v, VCOL_CAPACITY]

    bike_loads = get_loads(sol, VEH_BIKE)

    # Manual unique satellites
    seen_sat = np.zeros(N, dtype=np.bool_)
    unique_sats = np.empty(n_sats, dtype=np.int32)
    n_unique = 0
    for i in range(n_sats):
        v = np.int32(sats[i, SAT_CUST])
        if not seen_sat[v]:
            seen_sat[v] = True
            unique_sats[n_unique] = v
            n_unique += 1
    if n_unique < 2:
        return False

    # Build satellite -> bikes mapping using 2D array [max_sats x max_bikes_per_sat]
    # sat_bike_count[i] = number of bikes for unique_sats[i]
    # sat_bike_ids[i, j] = j-th bike for unique_sats[i]
    max_bps = n_bikes  # max bikes per satellite
    sat_bike_ids = np.zeros((n_unique, max_bps), dtype=np.int32)
    sat_bike_count = np.zeros(n_unique, dtype=np.int32)
    # seen_pair: bool array to avoid duplicate bike assignments per sat
    seen_pair = np.zeros((n_unique, n_bikes), dtype=np.bool_)

    # Map satellite customer id -> index in unique_sats
    sat_idx_map = np.full(N, -1, dtype=np.int32)
    for i in range(n_unique):
        sat_idx_map[unique_sats[i]] = i

    for i in range(n_sats):
        sn = np.int32(sats[i, SAT_CUST])
        bid = np.int32(sats[i, SAT_BIKE])
        si = sat_idx_map[sn]
        if si >= 0 and bid < n_bikes and not seen_pair[si, bid]:
            seen_pair[si, bid] = True
            sat_bike_ids[si, sat_bike_count[si]] = bid
            sat_bike_count[si] += 1

    # Compute demand per satellite and collect customers
    # Max customers per satellite: generous upper bound
    max_custs_per_sat = N
    sat_demand = np.zeros(n_unique, dtype=np.int64)
    # Flat arrays for customer info per satellite
    # sat_cust_ids[sat_idx][k], sat_cust_demand[sat_idx][k], sat_cust_dist[sat_idx][k]
    sat_cust_ids = np.full((n_unique, max_custs_per_sat), -1, dtype=np.int32)
    sat_cust_demand = np.zeros((n_unique, max_custs_per_sat), dtype=np.int64)
    sat_cust_dist = np.zeros((n_unique, max_custs_per_sat), dtype=np.int64)
    sat_cust_count = np.zeros(n_unique, dtype=np.int32)

    for si in range(n_unique):
        sn = unique_sats[si]
        for bj in range(sat_bike_count[si]):
            bid = sat_bike_ids[si, bj]
            if bid >= n_bikes:
                continue
            rc = get_route_customers_only(sol, VEH_BIKE, bid)
            for ci in range(len(rc)):
                c = int(rc[ci])
                d = customers[c, COL_DEMAND]
                sat_demand[si] += d
                k = sat_cust_count[si]
                sat_cust_ids[si, k] = c
                sat_cust_demand[si, k] = d
                sat_cust_dist[si, k] = dist_matrix[c + 1, sn + 1]
                sat_cust_count[si] += 1

    # Find overloaded and underloaded satellites
    over_buf = np.empty(n_unique, dtype=np.int32)
    n_over = 0
    under_buf = np.empty(n_unique, dtype=np.int32)
    under_slack = np.empty(n_unique, dtype=np.int64)
    n_under = 0

    for si in range(n_unique):
        total_cap = np.int64(sat_bike_count[si]) * bike_cap
        if sat_demand[si] > total_cap * 9 // 10:
            over_buf[n_over] = si
            n_over += 1
        if sat_demand[si] < total_cap * 7 // 10:
            under_buf[n_under] = si
            under_slack[n_under] = total_cap - sat_demand[si]
            n_under += 1

    if n_over == 0 or n_under == 0:
        return False

    # Shuffle overloaded (Fisher-Yates)
    for i in range(n_over - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        tmp = over_buf[i]
        over_buf[i] = over_buf[j]
        over_buf[j] = tmp

    for oi in range(n_over):
        si = over_buf[oi]
        nc = sat_cust_count[si]
        if nc == 0:
            continue
        # Sort candidates by distance descending (farthest first)
        # Build index array and sort by -dist
        idx = np.arange(nc)
        # Simple selection sort (nc is small)
        for a in range(nc):
            best_k = a
            for b in range(a + 1, nc):
                if sat_cust_dist[si, idx[b]] > sat_cust_dist[si, idx[best_k]]:
                    best_k = b
            if best_k != a:
                tmp = idx[a]
                idx[a] = idx[best_k]
                idx[best_k] = tmp

        for ci in range(nc):
            k = idx[ci]
            c = sat_cust_ids[si, k]
            demand = sat_cust_demand[si, k]
            # Find best underloaded satellite
            best_under_idx = -1
            best_dist = np.int64(9999999999)
            for ui in range(n_under):
                if under_slack[ui] < demand:
                    continue
                d = dist_matrix[c + 1, unique_sats[under_buf[ui]] + 1]
                if d < best_dist:
                    best_dist = d
                    best_under_idx = ui
            if best_under_idx == -1:
                continue
            vtype, vid, pos = get_customer_info(sol, c)
            if vtype != VEH_BIKE or vid < 0:
                continue
            remove_stop(sol, VEH_BIKE, vid, pos, dist_matrix, customers)
            # Try to insert into a bike of the underloaded satellite
            usi = under_buf[best_under_idx]
            best_bid = np.int32(-1)
            best_pos = np.int32(-1)
            best_delta = np.int64(9999999999)
            for bj in range(sat_bike_count[usi]):
                bid = sat_bike_ids[usi, bj]
                if bid >= n_bikes or bike_loads[bid] + demand > bike_cap:
                    continue
                p, delta = best_insertion_pos(sol, VEH_BIKE, bid, c, dist_matrix)
                if delta < best_delta:
                    best_bid = bid
                    best_pos = p
                    best_delta = delta
            if best_bid >= 0:
                insert_stop(sol, VEH_BIKE, best_bid, best_pos, c, ACT_DELIVER,
                            dist_matrix, customers)
                return True
            else:
                _, _, lengths = get_route_arrays(sol, VEH_BIKE)
                reins = min(pos, int(lengths[vid]))
                insert_stop(sol, VEH_BIKE, vid, reins, c, ACT_DELIVER,
                            dist_matrix, customers)
    return False
