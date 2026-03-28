"""Helper utilities for local search operators."""
import numpy as np

from src.data.constants import ACT_DELIVER, DM_DIST
from src.solution.route_ops import _update_route_distance


def _two_opt_delta(sol, vid, i, j, dist_matrix):
    """O(1) delta distance if reversing segment [i,j]."""
    stops = sol["stops"]
    lengths = sol["lengths"]
    L = lengths[vid]
    dm = dist_matrix[DM_DIST]

    a_prev = stops[vid, i - 1] if i > 0 else stops[vid, 0]
    a = stops[vid, i]
    b = stops[vid, j]
    b_next = stops[vid, j + 1] if j < L - 1 else stops[vid, L - 1]

    if i == 0:
        a_prev = a
    if j == L - 1:
        b_next = b

    old_cost = dm[a_prev, a] + dm[b, b_next]
    new_cost = dm[a_prev, b] + dm[a, b_next]
    return new_cost - old_cost


def _or_opt_delta(sol, vid, seg_start, seg_len, insert_pos, dist_matrix):
    """O(1) delta for moving segment [seg_start..seg_start+seg_len-1]."""
    stops = sol["stops"]
    lengths = sol["lengths"]
    L = lengths[vid]
    dm = dist_matrix[DM_DIST]
    seg_end = seg_start + seg_len - 1

    prev_seg = stops[vid, seg_start - 1] if seg_start > 0 else stops[vid, 0]
    first_seg = stops[vid, seg_start]
    last_seg = stops[vid, seg_end]
    next_seg = stops[vid, seg_end + 1] if seg_end < L - 1 else stops[vid, seg_end]

    if seg_start == 0:
        prev_seg = first_seg
    if seg_end == L - 1:
        next_seg = last_seg

    remove_cost = (dm[prev_seg, next_seg]
                   - dm[prev_seg, first_seg]
                   - dm[last_seg, next_seg])

    if insert_pos > seg_end:
        adj_pos = insert_pos - seg_len
    else:
        adj_pos = insert_pos

    route = []
    for idx in range(L):
        if idx < seg_start or idx > seg_end:
            route.append(stops[vid, idx])

    if len(route) == 0:
        ins_prev, ins_next = first_seg, first_seg
    elif adj_pos == 0:
        ins_prev, ins_next = route[0], route[0]
    elif adj_pos >= len(route):
        ins_prev, ins_next = route[-1], route[-1]
    else:
        ins_prev, ins_next = route[adj_pos - 1], route[adj_pos]

    insert_cost = (dm[ins_prev, first_seg]
                   + dm[last_seg, ins_next]
                   - dm[ins_prev, ins_next])
    return remove_cost + insert_cost


def _do_or_opt_move(sol, vid, seg_start, seg_len, insert_pos, dist_matrix,
                    vehicles=None):
    """Execute or-opt move by rebuilding route."""
    stops = sol["stops"]
    actions = sol["actions"]
    L = sol["lengths"][vid]

    # Build loc->customer map BEFORE rearranging
    custs_on_v = np.where(sol["cust_vehicle"] == vid)[0]
    loc_to_cust_map = {}
    for c in custs_on_v:
        old_pos = sol["cust_route_pos"][c]
        if 0 <= old_pos < L:
            loc_to_cust_map[int(stops[vid, old_pos])] = int(c)

    seg_stops = stops[vid, seg_start:seg_start + seg_len].copy()
    seg_actions = actions[vid, seg_start:seg_start + seg_len].copy()

    new_stops, new_actions = [], []
    for idx in range(L):
        if idx < seg_start or idx >= seg_start + seg_len:
            new_stops.append(stops[vid, idx])
            new_actions.append(actions[vid, idx])

    adj_pos = insert_pos if insert_pos <= seg_start else insert_pos - seg_len
    for k in range(seg_len):
        new_stops.insert(adj_pos + k, seg_stops[k])
        new_actions.insert(adj_pos + k, seg_actions[k])

    for idx in range(L):
        stops[vid, idx] = new_stops[idx]
        actions[vid, idx] = new_actions[idx]

    # Rebuild cust_route_pos from loc->customer map
    for idx in range(L):
        if actions[vid, idx] == ACT_DELIVER:
            loc = int(stops[vid, idx])
            c = loc_to_cust_map.get(loc, -1)
            if c >= 0:
                sol["cust_route_pos"][c] = idx

    _update_route_distance(sol, vid, dist_matrix, vehicles)
