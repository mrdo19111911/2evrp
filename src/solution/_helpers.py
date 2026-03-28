"""Shared helpers for solution module. All @njit, all int64 on hot path."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_META, META_N_TRUCKS,
)


@njit(cache=True)
def get_route_arrays(sol, vtype):
    """Returns (stops i32[:,:], actions i8[:,:], lengths i32[:])."""
    if vtype == VEH_TRUCK:
        return sol[SOL_TRUCK_STOPS], sol[SOL_TRUCK_ACTIONS], sol[SOL_TRUCK_LENGTHS]
    return sol[SOL_BIKE_STOPS], sol[SOL_BIKE_ACTIONS], sol[SOL_BIKE_LENGTHS]


@njit(cache=True)
def get_loads(sol, vtype):
    """Returns i64[:] loads (grams)."""
    if vtype == VEH_TRUCK:
        return sol[SOL_TRUCK_LOADS]
    return sol[SOL_BIKE_LOADS]


@njit(cache=True)
def get_distances(sol, vtype):
    """Returns i64[:] distances (meters)."""
    if vtype == VEH_TRUCK:
        return sol[SOL_TRUCK_DISTANCES]
    return sol[SOL_BIKE_DISTANCES]


@njit(cache=True)
def global_vid(vtype, vid, n_trucks):
    if vtype == VEH_TRUCK:
        return vid
    return n_trucks + vid


@njit(cache=True)
def update_route_distance(sol, vtype, vid, dist_matrix):
    """Recompute route distance (meters). dist_matrix is i64."""
    stops, _, lengths = get_route_arrays(sol, vtype)
    L = lengths[vid]
    dists = get_distances(sol, vtype)
    if L == 0:
        dists[vid] = 0
        return
    total = dist_matrix[0, stops[vid, 0] + 1]
    for i in range(L - 1):
        total += dist_matrix[stops[vid, i] + 1, stops[vid, i + 1] + 1]
    total += dist_matrix[stops[vid, L - 1] + 1, 0]
    dists[vid] = total


@njit(cache=True)
def update_route_load(sol, vtype, vid, customers):
    """Recompute route load (grams). customers is i64."""
    stops, actions, lengths = get_route_arrays(sol, vtype)
    L = lengths[vid]
    total = np.int64(0)
    for i in range(L):
        if actions[vid, i] == ACT_DELIVER:
            total += customers[stops[vid, i], COL_DEMAND]
    get_loads(sol, vtype)[vid] = total
