"""Cross-route operations: move, swap between routes, clear, create. All @njit."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    SOL_CUST_VEHICLE, SOL_CUST_VTYPE, SOL_CUST_ROUTE_POS,
    SOL_META, META_N_TRUCKS,
)
from src.solution.route_ops import insert_stop, remove_stop
from src.solution.satellite_ops import remove_satellites_for_customer
from src.solution._helpers import (
    get_route_arrays as _get_route_arrays,
    get_loads as _get_loads,
    get_distances as _get_distances,
    global_vid as _global_vid,
    update_route_distance as _update_route_distance,
    update_route_load as _update_route_load,
)


@njit(cache=True)
def move_stop(sol, from_vtype, from_vid, from_pos, to_vtype, to_vid, to_pos,
              dist_matrix, customers):
    """Move stop from one route to another. Atomic remove+insert."""
    cust, action = remove_stop(sol, from_vtype, from_vid, from_pos, dist_matrix, customers)
    if from_vtype == to_vtype and from_vid == to_vid:
        if to_pos > from_pos:
            to_pos -= 1
        lengths = _get_route_arrays(sol, to_vtype)[2]
        L = lengths[to_vid]
        if to_pos > L:
            to_pos = L
    insert_stop(sol, to_vtype, to_vid, to_pos, cust, action, dist_matrix, customers)


@njit(cache=True)
def swap_stops_between(sol, vtype_a, vid_a, pos_a, vtype_b, vid_b, pos_b,
                       dist_matrix, customers):
    """Swap stops between 2 routes. Must be different routes."""
    same_route = (vtype_a == vtype_b and vid_a == vid_b)
    cust_a, act_a = remove_stop(sol, vtype_a, vid_a, pos_a, dist_matrix, customers)
    adj_pos_b = pos_b
    if same_route and pos_b > pos_a:
        adj_pos_b -= 1
    cust_b, act_b = remove_stop(sol, vtype_b, vid_b, adj_pos_b, dist_matrix, customers)
    len_a = _get_route_arrays(sol, vtype_a)[2][vid_a]
    ins_pos_a = pos_a if pos_a <= len_a else len_a
    insert_stop(sol, vtype_a, vid_a, ins_pos_a,
                cust_b, act_b, dist_matrix, customers)
    len_b = _get_route_arrays(sol, vtype_b)[2][vid_b]
    ins_pos_b = pos_b if pos_b <= len_b else len_b
    insert_stop(sol, vtype_b, vid_b, ins_pos_b,
                cust_a, act_a, dist_matrix, customers)


@njit(cache=True)
def clear_route(sol, vtype, vid):
    """Remove all stops. Returns ndarray of (customer, action) pairs shape (L, 2)."""
    stops, actions, lengths = _get_route_arrays(sol, vtype)
    L = lengths[vid]
    cust_vehicle = sol[SOL_CUST_VEHICLE]
    cust_vtype = sol[SOL_CUST_VTYPE]
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]

    removed = np.empty((L, 2), dtype=np.int32)
    for i in range(L):
        c = int(stops[vid, i])
        a = int(actions[vid, i])
        removed[i, 0] = c
        removed[i, 1] = a
        if a == ACT_DELIVER:
            cust_vehicle[c] = -1
            cust_vtype[c] = -1
            cust_route_pos[c] = -1

    for i in range(L):
        if removed[i, 1] == ACT_DELIVER:
            remove_satellites_for_customer(sol, removed[i, 0])

    stops[vid, :] = -1
    actions[vid, :] = ACT_PAD
    lengths[vid] = 0
    _get_loads(sol, vtype)[vid] = np.int64(0)
    _get_distances(sol, vtype)[vid] = np.int64(0)

    return removed


@njit(cache=True)
def create_route_from_sequence(sol, vtype, vid, cust_arr, act_arr, n_stops,
                               dist_matrix, customers):
    """Build route from customer and action arrays. Overwrites existing."""
    clear_route(sol, vtype, vid)
    n_trucks = sol[SOL_META][META_N_TRUCKS]

    stops, actions, lengths = _get_route_arrays(sol, vtype)
    cust_vehicle = sol[SOL_CUST_VEHICLE]
    cust_vtype_arr = sol[SOL_CUST_VTYPE]
    cust_route_pos = sol[SOL_CUST_ROUTE_POS]

    for i in range(n_stops):
        c = cust_arr[i]
        a = act_arr[i]
        stops[vid, i] = c
        actions[vid, i] = a
        if a == ACT_DELIVER:
            cust_vehicle[c] = _global_vid(vtype, vid, n_trucks)
            cust_vtype_arr[c] = vtype
            cust_route_pos[c] = i

    lengths[vid] = n_stops
    _update_route_distance(sol, vtype, vid, dist_matrix)
    _update_route_load(sol, vtype, vid, customers)
