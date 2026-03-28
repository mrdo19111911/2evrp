"""LS helpers. @njit, i64 distances/loads, delta returns i64."""
import numpy as np
from numba import njit
from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_BIKE_STOPS, SOL_BIKE_ACTIONS,
    SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS,
    SOL_TRUCK_LOADS, SOL_BIKE_LOADS,
    SOL_TRUCK_DISTANCES, SOL_BIKE_DISTANCES,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_META, META_N_TRUCKS, META_MAX_ROUTE_LEN,
)

@njit(cache=True)
def get_stops(sol, vtype):
    return sol[SOL_TRUCK_STOPS] if vtype == VEH_TRUCK else sol[SOL_BIKE_STOPS]

@njit(cache=True)
def get_actions(sol, vtype):
    return sol[SOL_TRUCK_ACTIONS] if vtype == VEH_TRUCK else sol[SOL_BIKE_ACTIONS]

@njit(cache=True)
def get_lengths(sol, vtype):
    return sol[SOL_TRUCK_LENGTHS] if vtype == VEH_TRUCK else sol[SOL_BIKE_LENGTHS]

@njit(cache=True)
def get_loads(sol, vtype):
    return sol[SOL_TRUCK_LOADS] if vtype == VEH_TRUCK else sol[SOL_BIKE_LOADS]

@njit(cache=True)
def get_distances(sol, vtype):
    return sol[SOL_TRUCK_DISTANCES] if vtype == VEH_TRUCK else sol[SOL_BIKE_DISTANCES]

@njit(cache=True)
def n_vehicles(sol, vtype):
    return sol[SOL_META][META_N_TRUCKS] if vtype == VEH_TRUCK else sol[SOL_META][1]


@njit(cache=True)
def two_opt_delta(sol, vtype, vid, i, j, dist_matrix):
    """O(1) delta distance (i64 meters) if reversing segment [i,j]."""
    stops = get_stops(sol, vtype)
    L = get_lengths(sol, vtype)[vid]
    a = stops[vid, i] + 1
    b = stops[vid, j] + 1
    a_prev = stops[vid, i - 1] + 1 if i > 0 else 0
    b_next = stops[vid, j + 1] + 1 if j < L - 1 else 0
    return (dist_matrix[a_prev, b] + dist_matrix[a, b_next]
            - dist_matrix[a_prev, a] - dist_matrix[b, b_next])


@njit(cache=True)
def or_opt_delta(sol, vtype, vid, seg_start, seg_len, insert_pos, dist_matrix):
    """O(1) delta (i64 meters) for moving segment."""
    stops = get_stops(sol, vtype)
    L = get_lengths(sol, vtype)[vid]
    seg_end = seg_start + seg_len - 1

    prev_seg = 0 if seg_start == 0 else stops[vid, seg_start - 1] + 1
    first_seg = stops[vid, seg_start] + 1
    last_seg = stops[vid, seg_end] + 1
    next_seg = 0 if seg_end >= L - 1 else stops[vid, seg_end + 1] + 1

    remove_cost = (dist_matrix[prev_seg, next_seg]
                   - dist_matrix[prev_seg, first_seg]
                   - dist_matrix[last_seg, next_seg])

    adj_pos = insert_pos - seg_len if insert_pos > seg_end else insert_pos
    n_remaining = L - seg_len

    if n_remaining == 0:
        ins_prev = ins_next = 0
    elif adj_pos == 0:
        ins_prev = 0
        for idx in range(L):
            if idx < seg_start or idx > seg_end:
                ins_next = stops[vid, idx] + 1
                break
        else:
            ins_next = 0
    elif adj_pos >= n_remaining:
        ins_next = 0
        for idx in range(L - 1, -1, -1):
            if idx < seg_start or idx > seg_end:
                ins_prev = stops[vid, idx] + 1
                break
    else:
        k = 0
        ins_prev = 0
        ins_next = 0
        for idx in range(L):
            if idx < seg_start or idx > seg_end:
                if k == adj_pos - 1:
                    ins_prev = stops[vid, idx] + 1
                if k == adj_pos:
                    ins_next = stops[vid, idx] + 1
                    break
                k += 1

    insert_cost = (dist_matrix[ins_prev, first_seg]
                   + dist_matrix[last_seg, ins_next]
                   - dist_matrix[ins_prev, ins_next])
    return remove_cost + insert_cost


@njit(cache=True)
def do_or_opt_move(sol, vtype, vid, seg_start, seg_len, insert_pos, dist_matrix):
    """Execute or-opt move by rebuilding route."""
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)
    L = get_lengths(sol, vtype)[vid]
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]

    tmp_s = np.empty(seg_len, dtype=np.int32)
    tmp_a = np.empty(seg_len, dtype=np.int8)
    for k in range(seg_len):
        tmp_s[k] = stops[vid, seg_start + k]
        tmp_a[k] = actions[vid, seg_start + k]

    buf_s = np.empty(L, dtype=np.int32)
    buf_a = np.empty(L, dtype=np.int8)
    n = 0
    for idx in range(L):
        if idx < seg_start or idx >= seg_start + seg_len:
            buf_s[n] = stops[vid, idx]
            buf_a[n] = actions[vid, idx]
            n += 1

    adj = insert_pos if insert_pos <= seg_start else insert_pos - seg_len
    for idx in range(n - 1, adj - 1, -1):
        buf_s[idx + seg_len] = buf_s[idx]
        buf_a[idx + seg_len] = buf_a[idx]
    for k in range(seg_len):
        buf_s[adj + k] = tmp_s[k]
        buf_a[adj + k] = tmp_a[k]

    for idx in range(L):
        stops[vid, idx] = buf_s[idx]
        actions[vid, idx] = buf_a[idx]

    for idx in range(L):
        if actions[vid, idx] == ACT_DELIVER:
            cust_route_pos[stops[vid, idx]] = idx

    update_route_distance(sol, vtype, vid, dist_matrix)


@njit(cache=True)
def update_route_distance(sol, vtype, vid, dist_matrix):
    """Recompute route distance (i64 meters) from scratch."""
    stops = get_stops(sol, vtype)
    L = get_lengths(sol, vtype)[vid]
    dists = get_distances(sol, vtype)
    if L == 0:
        dists[vid] = 0
        return
    total = dist_matrix[0, stops[vid, 0] + 1]
    for i in range(1, L):
        total += dist_matrix[stops[vid, i - 1] + 1, stops[vid, i] + 1]
    total += dist_matrix[stops[vid, L - 1] + 1, 0]
    dists[vid] = total


@njit(cache=True)
def clear_tail_index(sol, vtype, vid, start, end):
    """Clear cust index for DELIVER stops in [start..end)."""
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)
    cv = sol[SOL_CUST_VEHICLE]
    ct = sol[SOL_CUST_VTYPE]
    cp = sol[SOL_CUST_ROUTE_POS]
    for idx in range(start, end):
        if actions[vid, idx] == ACT_DELIVER:
            c = stops[vid, idx]
            if c >= 0:
                cv[c] = -1
                ct[c] = -1
                cp[c] = -1


@njit(cache=True)
def rebuild_route_index(sol, vtype, vid, L):
    """Rebuild cust index for route."""
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    gv = vid if vtype == VEH_TRUCK else n_trucks + vid
    cv = sol[SOL_CUST_VEHICLE]
    ct = sol[SOL_CUST_VTYPE]
    cp = sol[SOL_CUST_ROUTE_POS]
    for idx in range(L):
        if actions[vid, idx] == ACT_DELIVER:
            c = stops[vid, idx]
            cv[c] = gv
            ct[c] = vtype
            cp[c] = idx
