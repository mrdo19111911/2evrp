"""Single-route operations: insert, remove, swap, reverse."""
import numpy as np

from src.data.constants import ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND
from src.solution._helpers import (
    get_route_arrays as _get_route_arrays,
    get_loads as _get_loads,
    get_distances as _get_distances,
    global_vid as _global_vid,
    update_route_distance as _update_route_distance,
)


def insert_stop(sol, vtype, vid, pos, customer, action, dist_matrix, customers):
    """Insert stop at pos in route. Update index + distance + load cache."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]

    if L >= sol["max_route_len"]:
        return  # route full, cannot insert

    # Shift right
    stops[vid, pos + 1:L + 1] = stops[vid, pos:L]
    actions[vid, pos + 1:L + 1] = actions[vid, pos:L]

    # Insert
    stops[vid, pos] = customer
    actions[vid, pos] = action
    lengths[vid] += 1

    # Update customer index if DELIVER
    if action == ACT_DELIVER:
        sol["cust_vehicle"][customer] = _global_vid(vtype, vid, sol["n_trucks"])
        sol["cust_vtype"][customer] = vtype
        sol["cust_route_pos"][customer] = pos
        for j in range(pos + 1, lengths[vid]):
            c = stops[vid, j]
            if actions[vid, j] == ACT_DELIVER:
                sol["cust_route_pos"][c] = j

    _update_route_distance(sol, vtype, vid, dist_matrix)

    if action == ACT_DELIVER:
        _get_loads(sol, vtype)[vid] += customers[customer, COL_DEMAND]


def remove_stop(sol, vtype, vid, pos, dist_matrix, customers):
    """Remove stop at pos. Returns (removed_customer, removed_action)."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = int(lengths[vid])
    if pos >= L or L == 0:
        return -1, -1  # invalid position, nothing to remove
    removed_customer = int(stops[vid, pos])
    removed_action = int(actions[vid, pos])

    # Clear index if DELIVER
    if removed_action == ACT_DELIVER:
        sol["cust_vehicle"][removed_customer] = -1
        sol["cust_vtype"][removed_customer] = -1
        sol["cust_route_pos"][removed_customer] = -1

    # Shift left
    stops[vid, pos:L - 1] = stops[vid, pos + 1:L]
    actions[vid, pos:L - 1] = actions[vid, pos + 1:L]
    stops[vid, L - 1] = -1
    actions[vid, L - 1] = ACT_PAD
    lengths[vid] -= 1

    # Update route_pos for customers after pos
    for j in range(pos, lengths[vid]):
        c = stops[vid, j]
        if actions[vid, j] == ACT_DELIVER:
            sol["cust_route_pos"][c] = j

    _update_route_distance(sol, vtype, vid, dist_matrix)

    if removed_action == ACT_DELIVER:
        _get_loads(sol, vtype)[vid] -= customers[removed_customer, COL_DEMAND]

    return removed_customer, removed_action


def swap_stops_within(sol, vtype, vid, pos_a, pos_b, dist_matrix):
    """Swap 2 stops in same route."""
    stops, actions, _ = _get_route_arrays(sol, vtype)

    # Swap stops and actions
    stops[vid, pos_a], stops[vid, pos_b] = stops[vid, pos_b], stops[vid, pos_a]
    actions[vid, pos_a], actions[vid, pos_b] = actions[vid, pos_b], actions[vid, pos_a]

    # Update route_pos index
    ca, cb = stops[vid, pos_a], stops[vid, pos_b]
    if actions[vid, pos_a] == ACT_DELIVER:
        sol["cust_route_pos"][ca] = pos_a
    if actions[vid, pos_b] == ACT_DELIVER:
        sol["cust_route_pos"][cb] = pos_b

    _update_route_distance(sol, vtype, vid, dist_matrix)


def reverse_segment(sol, vtype, vid, start, end, dist_matrix):
    """Reverse segment [start, end] inclusive. For 2-opt."""
    stops, actions, _ = _get_route_arrays(sol, vtype)

    stops[vid, start:end + 1] = stops[vid, start:end + 1][::-1]
    actions[vid, start:end + 1] = actions[vid, start:end + 1][::-1]

    for j in range(start, end + 1):
        c = stops[vid, j]
        if actions[vid, j] == ACT_DELIVER:
            sol["cust_route_pos"][c] = j

    _update_route_distance(sol, vtype, vid, dist_matrix)
