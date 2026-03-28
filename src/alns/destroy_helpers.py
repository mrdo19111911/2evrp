"""Destroy operator helpers — sort, remove targets, relatedness. @njit where possible."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
    SOL_META, META_N_TRUCKS, META_N_BIKES,
)
from src.solution.query import get_assigned_customers, get_customer_info, get_route_customers_only
from src.solution.route_ops import remove_stop
from src.solution.satellite_ops import remove_satellites_for_customer
from src.solution._helpers import get_route_arrays


@njit(cache=True)
def sort_by_pos_desc(targets, sol):
    """Sort targets by (vtype, vid, -pos) so highest pos removed first per route."""
    n = len(targets)
    if n == 0:
        return targets
    sort_keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        vtype, vid, pos = get_customer_info(sol, targets[i])
        sort_keys[i] = vtype * 100000000 + vid * 10000 + (9999 - pos)
    order = np.argsort(sort_keys)
    return targets[order]


@njit(cache=True)
def remove_targets(sol, targets, dist_matrix, customers):
    """Remove targets safely. Re-query position each time. Full @njit."""
    sorted_t = sort_by_pos_desc(targets, sol)
    removed = np.empty(len(sorted_t), dtype=np.int32)
    n = 0
    for idx in range(len(sorted_t)):
        c = sorted_t[idx]
        vtype, vid, pos = get_customer_info(sol, c)
        if vtype == -1:
            continue
        remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
        remove_satellites_for_customer(sol, c)
        removed[n] = c
        n += 1
    return removed[:n]


@njit(cache=True)
def shaw_relatedness(c, targets, customers, dist_matrix, sol, max_dist, max_demand):
    """Multi-dimensional relatedness: distance + demand + TW overlap + same-route.
    All inputs i64. Returns i64 scores (scaled x1000 for precision)."""
    w1, w2, w3, w4 = 9, 3, 2, 5
    c_open = customers[c, COL_TW_OPEN]
    c_close = customers[c, COL_TW_CLOSE]
    c_demand = customers[c, COL_DEMAND]
    c_vtype, c_vid, _ = get_customer_info(sol, c)

    scores = np.empty(len(targets), dtype=np.int64)
    safe_dist = max_dist if max_dist > 0 else 1
    safe_demand = max_demand if max_demand > 0 else 1
    for i in range(len(targets)):
        t = targets[i]
        d = customers[c, COL_DEMAND]  # placeholder — distance term
        d_term = dist_matrix[c + 1, t + 1] * 1000 // safe_dist
        dd_term = abs(c_demand - customers[t, COL_DEMAND]) * 1000 // safe_demand
        t_open = customers[t, COL_TW_OPEN]
        t_close = customers[t, COL_TW_CLOSE]
        overlap = max(np.int64(0), min(c_close, t_close) - max(c_open, t_open))
        span = max(c_close - c_open, t_close - t_open)
        if span < 1:
            span = 1
        tw_term = 1000 - overlap * 1000 // span
        t_vtype, t_vid, _ = get_customer_info(sol, t)
        same_route = 0 if (c_vtype == t_vtype and c_vid == t_vid and c_vtype != -1) else 1000
        scores[i] = w1 * d_term + w2 * dd_term + w3 * tw_term + w4 * same_route
    return scores


@njit(cache=True)
def nonempty_routes(sol):
    """Return ndarray of (vtype, vid) for non-empty routes."""
    meta = sol[SOL_META]
    n_trucks = meta[META_N_TRUCKS]
    n_bikes = meta[META_N_BIKES]
    _, _, lengths_t = get_route_arrays(sol, VEH_TRUCK)
    _, _, lengths_b = get_route_arrays(sol, VEH_BIKE)
    result = np.empty((n_trucks + n_bikes, 2), dtype=np.int32)
    k = 0
    for t in range(n_trucks):
        if lengths_t[t] > 0:
            result[k, 0] = VEH_TRUCK
            result[k, 1] = t
            k += 1
    for b in range(n_bikes):
        if lengths_b[b] > 0:
            result[k, 0] = VEH_BIKE
            result[k, 1] = b
            k += 1
    return result[:k]
