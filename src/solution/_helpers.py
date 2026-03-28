"""Shared helpers for solution module — single source of truth."""
import numpy as np

from src.data.constants import ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_DEMAND


def get_route_arrays(sol, vtype):
    """Returns (stops, actions, lengths) for vtype."""
    if vtype == VEH_TRUCK:
        return sol["truck_stops"], sol["truck_actions"], sol["truck_lengths"]
    return sol["bike_stops"], sol["bike_actions"], sol["bike_lengths"]


def get_loads(sol, vtype):
    return sol["truck_loads"] if vtype == VEH_TRUCK else sol["bike_loads"]


def get_distances(sol, vtype):
    return sol["truck_distances"] if vtype == VEH_TRUCK else sol["bike_distances"]


def global_vid(vtype, vid, n_trucks):
    return vid if vtype == VEH_TRUCK else n_trucks + vid


def update_route_distance(sol, vtype, vid, dist_matrix):
    """Recompute route distance from scratch."""
    stops, _, lengths = get_route_arrays(sol, vtype)
    L = lengths[vid]
    if L == 0:
        get_distances(sol, vtype)[vid] = 0.0
        return
    total = dist_matrix[0, stops[vid, 0] + 1]
    for i in range(L - 1):
        total += dist_matrix[stops[vid, i] + 1, stops[vid, i + 1] + 1]
    total += dist_matrix[stops[vid, L - 1] + 1, 0]
    get_distances(sol, vtype)[vid] = total


def update_route_load(sol, vtype, vid, customers):
    """Recompute route load from scratch."""
    stops, actions, lengths = get_route_arrays(sol, vtype)
    L = lengths[vid]
    total = 0.0
    for i in range(L):
        if actions[vid, i] == ACT_DELIVER:
            total += customers[stops[vid, i], COL_DEMAND]
    get_loads(sol, vtype)[vid] = total
