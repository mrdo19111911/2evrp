"""Cross-route operations: move, swap between routes, clear, create."""
import numpy as np

from src.data.constants import ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND
from src.solution.route_ops import insert_stop, remove_stop


def _get_route_arrays(sol, vtype):
    if vtype == VEH_TRUCK:
        return sol["truck_stops"], sol["truck_actions"], sol["truck_lengths"]
    return sol["bike_stops"], sol["bike_actions"], sol["bike_lengths"]


def _get_loads(sol, vtype):
    return sol["truck_loads"] if vtype == VEH_TRUCK else sol["bike_loads"]


def _get_distances(sol, vtype):
    return sol["truck_distances"] if vtype == VEH_TRUCK else sol["bike_distances"]


def _global_vid(vtype, vid, n_trucks):
    return vid if vtype == VEH_TRUCK else n_trucks + vid


def _update_route_distance(sol, vtype, vid, dist_matrix):
    stops, _, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    if L == 0:
        _get_distances(sol, vtype)[vid] = 0.0
        return
    total = dist_matrix[0, stops[vid, 0] + 1]
    for i in range(L - 1):
        total += dist_matrix[stops[vid, i] + 1, stops[vid, i + 1] + 1]
    total += dist_matrix[stops[vid, L - 1] + 1, 0]
    _get_distances(sol, vtype)[vid] = total


def _update_route_load(sol, vtype, vid, customers):
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    total = 0.0
    for i in range(L):
        if actions[vid, i] == ACT_DELIVER:
            total += customers[stops[vid, i], COL_DEMAND]
    _get_loads(sol, vtype)[vid] = total


def move_stop(sol, from_vtype, from_vid, from_pos, to_vtype, to_vid, to_pos,
              dist_matrix, customers):
    """Move stop from one route to another. Atomic remove+insert."""
    cust, action = remove_stop(sol, from_vtype, from_vid, from_pos, dist_matrix, customers)
    insert_stop(sol, to_vtype, to_vid, to_pos, cust, action, dist_matrix, customers)


def swap_stops_between(sol, vtype_a, vid_a, pos_a, vtype_b, vid_b, pos_b,
                       dist_matrix, customers):
    """Swap stops between 2 different routes."""
    cust_a, act_a = remove_stop(sol, vtype_a, vid_a, pos_a, dist_matrix, customers)
    cust_b, act_b = remove_stop(sol, vtype_b, vid_b, pos_b, dist_matrix, customers)
    insert_stop(sol, vtype_a, vid_a, pos_a, cust_b, act_b, dist_matrix, customers)
    insert_stop(sol, vtype_b, vid_b, pos_b, cust_a, act_a, dist_matrix, customers)


def clear_route(sol, vtype, vid):
    """Remove all stops. Returns list of (customer, action) pairs."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    removed = []

    for i in range(L):
        c = int(stops[vid, i])
        a = int(actions[vid, i])
        removed.append((c, a))
        if a == ACT_DELIVER:
            sol["cust_vehicle"][c] = -1
            sol["cust_vtype"][c] = -1
            sol["cust_route_pos"][c] = -1

    stops[vid, :] = -1
    actions[vid, :] = ACT_PAD
    lengths[vid] = 0
    _get_loads(sol, vtype)[vid] = 0.0
    _get_distances(sol, vtype)[vid] = 0.0

    return removed


def create_route_from_list(sol, vtype, vid, stop_list, dist_matrix, customers):
    """Build route from [(customer, action), ...] list. Overwrites existing."""
    clear_route(sol, vtype, vid)

    stops, actions, lengths = _get_route_arrays(sol, vtype)
    for i, (c, a) in enumerate(stop_list):
        stops[vid, i] = c
        actions[vid, i] = a
        if a == ACT_DELIVER:
            sol["cust_vehicle"][c] = _global_vid(vtype, vid, sol["n_trucks"])
            sol["cust_vtype"][c] = vtype
            sol["cust_route_pos"][c] = i

    lengths[vid] = len(stop_list)
    _update_route_distance(sol, vtype, vid, dist_matrix)
    _update_route_load(sol, vtype, vid, customers)
