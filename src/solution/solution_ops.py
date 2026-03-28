"""Cross-route operations: move, swap between routes, clear, create."""
import numpy as np

from src.data.constants import ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND
from src.solution.route_ops import insert_stop, remove_stop
from src.solution._helpers import (
    get_route_arrays as _get_route_arrays,
    get_loads as _get_loads,
    get_distances as _get_distances,
    global_vid as _global_vid,
    update_route_distance as _update_route_distance,
    update_route_load as _update_route_load,
)


def move_stop(sol, from_vtype, from_vid, from_pos, to_vtype, to_vid, to_pos,
              dist_matrix, customers):
    """Move stop from one route to another. Atomic remove+insert."""
    cust, action = remove_stop(sol, from_vtype, from_vid, from_pos, dist_matrix, customers)
    # Adjust to_pos for same-route moves: after removal, positions shift left
    if from_vtype == to_vtype and from_vid == to_vid and to_pos > from_pos:
        to_pos -= 1
    insert_stop(sol, to_vtype, to_vid, to_pos, cust, action, dist_matrix, customers)


def swap_stops_between(sol, vtype_a, vid_a, pos_a, vtype_b, vid_b, pos_b,
                       dist_matrix, customers):
    """Swap stops between 2 routes. Must be different routes."""
    same_route = (vtype_a == vtype_b and vid_a == vid_b)
    cust_a, act_a = remove_stop(sol, vtype_a, vid_a, pos_a, dist_matrix, customers)
    # Adjust pos_b if same route and pos_b was after pos_a
    adj_pos_b = pos_b
    if same_route and pos_b > pos_a:
        adj_pos_b -= 1
    cust_b, act_b = remove_stop(sol, vtype_b, vid_b, adj_pos_b, dist_matrix, customers)
    # After both removals, insert back. Positions shift again for same route.
    insert_stop(sol, vtype_a, vid_a, min(pos_a, _get_route_arrays(sol, vtype_a)[2][vid_a]),
                cust_b, act_b, dist_matrix, customers)
    adj_b = min(pos_b, _get_route_arrays(sol, vtype_b)[2][vid_b])
    insert_stop(sol, vtype_b, vid_b, adj_b,
                cust_a, act_a, dist_matrix, customers)


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
