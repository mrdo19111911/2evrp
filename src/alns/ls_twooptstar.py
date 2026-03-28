"""2-opt* inter-route operator. @njit, i64 types."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND,
    VCOL_CAPACITY, SOL_META, META_N_TRUCKS, META_MAX_ROUTE_LEN,
)
from src.solution._helpers import update_route_load
from src.alns.ls_helpers import (
    get_stops, get_actions, get_lengths, get_loads,
    n_vehicles, update_route_distance,
    clear_tail_index, rebuild_route_index,
)


@njit(cache=True)
def _dm_node(stops, vid, pos, L):
    if pos < 0 or pos >= L:
        return 0
    return stops[vid, pos] + 1


@njit(cache=True)
def _partial_load(stops, actions, vid, start, end, customers):
    """Partial load (i64 grams) for stops[vid, start..end]."""
    if start > end:
        return np.int64(0)
    total = np.int64(0)
    for k in range(start, end + 1):
        if actions[vid, k] == ACT_DELIVER:
            total += customers[stops[vid, k], COL_DEMAND]
    return total


@njit(cache=True)
def _delta(stops, va, vb, i, j, La, Lb, dm):
    """Delta distance (i64 meters) for tail swap."""
    ai = _dm_node(stops, va, i, La)
    ai1 = _dm_node(stops, va, i + 1, La)
    aL = _dm_node(stops, va, La - 1, La)
    bj = _dm_node(stops, vb, j, Lb)
    bj1 = _dm_node(stops, vb, j + 1, Lb)
    bL = _dm_node(stops, vb, Lb - 1, Lb)

    old = dm[ai, ai1] + dm[aL, 0] + dm[bj, bj1] + dm[bL, 0]
    if j + 1 >= Lb:
        new_a = dm[ai, 0]
    else:
        new_a = dm[ai, bj1] + dm[bL, 0]
    if i + 1 >= La:
        new_b = dm[bj, 0]
    else:
        new_b = dm[bj, ai1] + dm[aL, 0]
    return new_a + new_b - old


@njit(cache=True)
def two_opt_star(sol, vtype, dist_matrix, customers, vehicles):
    """2-opt*: swap tails between two routes. i64 loads/distances."""
    nv = n_vehicles(sol, vtype)
    if nv < 2:
        return False
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    if vtype == VEH_TRUCK:
        capacity = vehicles[0, VCOL_CAPACITY]
    else:
        capacity = vehicles[n_trucks, VCOL_CAPACITY]
    max_len = sol[SOL_META][META_MAX_ROUTE_LEN]
    lengths = get_lengths(sol, vtype)
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)

    for va in range(nv):
        La = lengths[va]
        if La == 0:
            continue
        for vb in range(va + 1, nv):
            Lb = lengths[vb]
            if Lb == 0:
                continue
            for i in range(La):
                for j in range(Lb):
                    la = (_partial_load(stops, actions, va, 0, i, customers)
                          + _partial_load(stops, actions, vb, j + 1, Lb - 1,
                                          customers))
                    lb = (_partial_load(stops, actions, vb, 0, j, customers)
                          + _partial_load(stops, actions, va, i + 1, La - 1,
                                          customers))
                    if la > capacity or lb > capacity:
                        continue
                    nLa = (i + 1) + (Lb - j - 1)
                    nLb = (j + 1) + (La - i - 1)
                    if nLa > max_len or nLb > max_len:
                        continue
                    delta = _delta(stops, va, vb, i, j, La, Lb, dist_matrix)
                    if delta < -1:
                        _execute(sol, vtype, va, vb, i, j, La, Lb,
                                 dist_matrix, customers)
                        return True
    return False


@njit(cache=True)
def _execute(sol, vtype, va, vb, i, j, La, Lb, dist_matrix, customers):
    """Execute tail swap between routes va and vb."""
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)
    lengths = get_lengths(sol, vtype)
    clear_tail_index(sol, vtype, va, i + 1, La)
    clear_tail_index(sol, vtype, vb, j + 1, Lb)

    len_at = La - (i + 1)
    len_bt = Lb - (j + 1)
    at_s = np.empty(len_at, dtype=np.int32)
    at_a = np.empty(len_at, dtype=np.int8)
    bt_s = np.empty(len_bt, dtype=np.int32)
    bt_a = np.empty(len_bt, dtype=np.int8)
    for k in range(len_at):
        at_s[k] = stops[va, i + 1 + k]
        at_a[k] = actions[va, i + 1 + k]
    for k in range(len_bt):
        bt_s[k] = stops[vb, j + 1 + k]
        bt_a[k] = actions[vb, j + 1 + k]

    nLa = (i + 1) + len_bt
    nLb = (j + 1) + len_at

    for k in range(len_bt):
        stops[va, i + 1 + k] = bt_s[k]
        actions[va, i + 1 + k] = bt_a[k]
    for k in range(nLa, La):
        stops[va, k] = -1
        actions[va, k] = ACT_PAD

    for k in range(len_at):
        stops[vb, j + 1 + k] = at_s[k]
        actions[vb, j + 1 + k] = at_a[k]
    for k in range(nLb, Lb):
        stops[vb, k] = -1
        actions[vb, k] = ACT_PAD

    lengths[va] = nLa
    lengths[vb] = nLb
    rebuild_route_index(sol, vtype, va, nLa)
    rebuild_route_index(sol, vtype, vb, nLb)
    update_route_load(sol, vtype, va, customers)
    update_route_load(sol, vtype, vb, customers)
    update_route_distance(sol, vtype, va, dist_matrix)
    update_route_distance(sol, vtype, vb, dist_matrix)
