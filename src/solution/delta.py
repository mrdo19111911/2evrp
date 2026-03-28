"""Delta cost evaluation. O(1) per operation -- speed-critical for ALNS."""
import numpy as np

from src.data.constants import VEH_TRUCK, VEH_BIKE, COL_DEMAND


def _get_route_arrays(sol, vtype):
    if vtype == VEH_TRUCK:
        return sol["truck_stops"], sol["truck_actions"], sol["truck_lengths"]
    return sol["bike_stops"], sol["bike_actions"], sol["bike_lengths"]


def _get_loads(sol, vtype):
    return sol["truck_loads"] if vtype == VEH_TRUCK else sol["bike_loads"]


def insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix):
    """O(1) delta distance if inserting customer at pos."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    cn = customer + 1  # dist_matrix index

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


def removal_cost_delta(sol, vtype, vid, pos, dist_matrix):
    """O(1) delta distance if removing stop at pos."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    cn = stops[vid, pos] + 1

    if L == 1:
        return -(dist_matrix[0, cn] + dist_matrix[cn, 0])

    if pos == 0:
        next_node = stops[vid, 1] + 1
        return dist_matrix[0, next_node] - dist_matrix[0, cn] - dist_matrix[cn, next_node]

    if pos == L - 1:
        prev_node = stops[vid, L - 2] + 1
        return dist_matrix[prev_node, 0] - dist_matrix[prev_node, cn] - dist_matrix[cn, 0]

    prev_node = stops[vid, pos - 1] + 1
    next_node = stops[vid, pos + 1] + 1
    return (dist_matrix[prev_node, next_node]
            - dist_matrix[prev_node, cn] - dist_matrix[cn, next_node])


def insertion_cost_delta_full(sol, vtype, vid, pos, customer, dist_matrix, customers):
    """O(1) delta cost in VND (distance + time + service)."""
    return insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix)


def best_insertion_pos(sol, vtype, vid, customer, dist_matrix):
    """Best position in 1 route. O(route_len). Returns (pos, delta)."""
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    best_pos = 0
    best_delta = np.inf

    for pos in range(L + 1):
        delta = insertion_cost_delta(sol, vtype, vid, pos, customer, dist_matrix)
        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    return best_pos, best_delta


def find_best_insertion_all_routes(sol, vtype, customer, dist_matrix, customers,
                                   vehicle_capacity):
    """Best (vid, pos, delta) across all routes of vtype. Returns (-1,-1,inf) if none."""
    demand = customers[customer, COL_DEMAND]
    n_vehicles = sol["n_trucks"] if vtype == VEH_TRUCK else sol["n_bikes"]
    loads = _get_loads(sol, vtype)

    best_vid, best_pos, best_delta = -1, -1, np.inf

    for vid in range(n_vehicles):
        if loads[vid] + demand > vehicle_capacity:
            continue
        pos, delta = best_insertion_pos(sol, vtype, vid, customer, dist_matrix)
        if delta < best_delta:
            best_vid, best_pos, best_delta = vid, pos, delta

    return best_vid, best_pos, best_delta
