"""Sync cost between truck and bike at satellite nodes. All i64."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_RELOAD,
    ST_ACTION, ST_CUST, ST_DEPART, ST_ARRIVE,
    SAT_CUST, SAT_BIKE, SAT_TRUCK,
)
from src.data.cost import PENALTY_MISSING_RELOAD, PENALTY_SYNC_GAP_SEC


@njit(cache=True)
def _find_depart_3d(sim, lengths, vid, cust):
    """Find RELOAD depart time (i64) in 3D sim. Returns -1 if not found."""
    if vid >= len(lengths):
        return np.int64(-1)
    L = np.int32(lengths[vid])
    for i in range(L):
        if np.int32(sim[vid, i, ST_CUST]) == cust and np.int32(sim[vid, i, ST_ACTION]) == ACT_RELOAD:
            return sim[vid, i, ST_DEPART]
    return np.int64(-1)


@njit(cache=True)
def _find_arrive_3d(sim, lengths, vid, cust):
    """Find RELOAD arrive time (i64) in 3D sim. Returns -1 if not found."""
    if vid >= len(lengths):
        return np.int64(-1)
    L = np.int32(lengths[vid])
    for i in range(L):
        if np.int32(sim[vid, i, ST_CUST]) == cust and np.int32(sim[vid, i, ST_ACTION]) == ACT_RELOAD:
            return sim[vid, i, ST_ARRIVE]
    return np.int64(-1)


@njit(cache=True)
def compute_sync_cost(truck_sim, truck_lengths, bike_sim, bike_lengths,
                      satellites, n_satellites, delta_t_s):
    """Sync cost in VND (i64). Returns total_sync_cost."""
    total = np.int64(0)
    missing = np.int64(PENALTY_MISSING_RELOAD)
    gap_per_s = np.int64(PENALTY_SYNC_GAP_SEC)

    for s in range(n_satellites):
        cust = np.int32(satellites[s, SAT_CUST])
        bike_id = np.int32(satellites[s, SAT_BIKE])
        truck_id = np.int32(satellites[s, SAT_TRUCK])

        truck_depart = _find_depart_3d(truck_sim, truck_lengths, truck_id, cust)
        if truck_depart < 0:
            total += missing
            continue

        bike_arrive = _find_arrive_3d(bike_sim, bike_lengths, bike_id, cust)
        if bike_arrive < 0:
            total += missing
            continue

        gap = abs(truck_depart - bike_arrive)
        if gap > delta_t_s:
            total += gap_per_s * (gap - delta_t_s)

    return total
