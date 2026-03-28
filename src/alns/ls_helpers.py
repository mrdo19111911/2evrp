"""Helper utilities for local search operators."""
import numpy as np

from src.data.constants import ACT_DELIVER, VEH_TRUCK, VEH_BIKE


def _get_route_arrays(sol, vtype):
    if vtype == VEH_TRUCK:
        return sol["truck_stops"], sol["truck_actions"]
    return sol["bike_stops"], sol["bike_actions"]


def _get_lengths(sol, vtype):
    return sol["truck_lengths"] if vtype == VEH_TRUCK else sol["bike_lengths"]


def _get_stops(sol, vtype):
    return sol["truck_stops"] if vtype == VEH_TRUCK else sol["bike_stops"]


def _get_actions(sol, vtype):
    return sol["truck_actions"] if vtype == VEH_TRUCK else sol["bike_actions"]


def _get_loads(sol, vtype):
    return sol["truck_loads"] if vtype == VEH_TRUCK else sol["bike_loads"]


def _two_opt_delta(sol, vtype, vid, i, j, dist_matrix):
    """O(1) delta distance if reversing segment [i,j]."""
    stops, _ = _get_route_arrays(sol, vtype)
    L = int(_get_lengths(sol, vtype)[vid])

    # Depot is index 0 in dist_matrix, customers are +1
    a = int(stops[vid, i]) + 1
    b = int(stops[vid, j]) + 1
    a_prev = int(stops[vid, i - 1]) + 1 if i > 0 else 0  # depot
    b_next = int(stops[vid, j + 1]) + 1 if j < L - 1 else 0  # depot

    old_cost = dist_matrix[a_prev, a] + dist_matrix[b, b_next]
    new_cost = dist_matrix[a_prev, b] + dist_matrix[a, b_next]
    return new_cost - old_cost


def _or_opt_delta(sol, vtype, vid, seg_start, seg_len, insert_pos, dist_matrix):
    """O(1) delta for moving segment [seg_start..seg_start+seg_len-1]."""
    stops, _ = _get_route_arrays(sol, vtype)
    L = int(_get_lengths(sol, vtype)[vid])
    seg_end = seg_start + seg_len - 1

    def dm_node(pos):
        """Map position to dist_matrix index. -1 means depot (index 0)."""
        if pos < 0 or pos >= L:
            return 0
        return int(stops[vid, pos]) + 1

    prev_seg = dm_node(seg_start - 1)
    first_seg = dm_node(seg_start)
    last_seg = dm_node(seg_end)
    next_seg = dm_node(seg_end + 1)

    # Cost of removing segment
    remove_cost = (dist_matrix[prev_seg, next_seg]
                   - dist_matrix[prev_seg, first_seg]
                   - dist_matrix[last_seg, next_seg])

    # Adjusted insert position after removal
    if insert_pos > seg_end:
        adj_pos = insert_pos - seg_len
    else:
        adj_pos = insert_pos

    # Build route without segment to find insert neighbors
    route_nodes = []
    for idx in range(L):
        if idx < seg_start or idx > seg_end:
            route_nodes.append(dm_node(idx))

    if len(route_nodes) == 0:
        ins_prev, ins_next = 0, 0  # depot
    elif adj_pos == 0:
        ins_prev, ins_next = 0, route_nodes[0]  # depot -> first
    elif adj_pos >= len(route_nodes):
        ins_prev, ins_next = route_nodes[-1], 0  # last -> depot
    else:
        ins_prev, ins_next = route_nodes[adj_pos - 1], route_nodes[adj_pos]

    insert_cost = (dist_matrix[ins_prev, first_seg]
                   + dist_matrix[last_seg, ins_next]
                   - dist_matrix[ins_prev, ins_next])
    return remove_cost + insert_cost


def _do_or_opt_move(sol, vtype, vid, seg_start, seg_len, insert_pos,
                    dist_matrix):
    """Execute or-opt move by rebuilding route."""
    stops, actions = _get_route_arrays(sol, vtype)
    L = int(_get_lengths(sol, vtype)[vid])

    # Build loc->customer map BEFORE rearranging
    global_vid = vid if vtype == VEH_TRUCK else sol["n_trucks"] + vid
    custs_on_v = np.where(sol["cust_vehicle"] == global_vid)[0]
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

    # Update distance cache
    _update_route_distance(sol, vtype, vid, dist_matrix)


def _update_route_distance(sol, vtype, vid, dist_matrix):
    """Recompute route distance from scratch."""
    stops, _ = _get_route_arrays(sol, vtype)
    L = int(_get_lengths(sol, vtype)[vid])
    distances = sol["truck_distances"] if vtype == VEH_TRUCK else sol["bike_distances"]

    if L == 0:
        distances[vid] = 0.0
        return

    total = dist_matrix[0, int(stops[vid, 0]) + 1]
    for i in range(1, L):
        total += dist_matrix[int(stops[vid, i - 1]) + 1, int(stops[vid, i]) + 1]
    total += dist_matrix[int(stops[vid, L - 1]) + 1, 0]
    distances[vid] = total
