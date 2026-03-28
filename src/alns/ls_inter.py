"""Inter-route local search: relocate, exchange. @njit, i64 types."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_DEMAND, COL_RESTRICTED,
    VCOL_CAPACITY, SOL_META, META_N_TRUCKS,
)
from src.solution.route_ops import remove_stop, insert_stop
from src.solution.delta import removal_cost_delta, best_insertion_pos
from src.solution.check import can_swap_customers
from src.alns.ls_helpers import (
    get_stops, get_actions, get_lengths, get_loads, n_vehicles,
)


@njit(cache=True)
def _get_capacity(vtype, vehicles, n_trucks):
    """Get capacity for vtype from vehicles array."""
    if vtype == VEH_TRUCK:
        return vehicles[0, VCOL_CAPACITY]
    return vehicles[n_trucks, VCOL_CAPACITY]


@njit(cache=True)
def relocate_inter_route(sol, vtype, dist_matrix, customers, vehicles):
    """Move customer between routes of same vtype. All i64."""
    nv = n_vehicles(sol, vtype)
    if nv < 2:
        return False
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    capacity = _get_capacity(vtype, vehicles, n_trucks)
    lengths = get_lengths(sol, vtype)
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)
    loads = get_loads(sol, vtype)

    for va in range(nv):
        La = lengths[va]
        for pos_a in range(La):
            if actions[va, pos_a] != ACT_DELIVER:
                continue
            c = stops[va, pos_a]
            demand = customers[c, COL_DEMAND]
            rem_delta = removal_cost_delta(sol, vtype, va, pos_a, dist_matrix)
            for vb in range(nv):
                if vb == va or loads[vb] + demand > capacity:
                    continue
                ins_pos, ins_delta = best_insertion_pos(
                    sol, vtype, vb, c, dist_matrix)
                if rem_delta + ins_delta < -1:
                    rc, ra = remove_stop(
                        sol, vtype, va, pos_a, dist_matrix, customers)
                    insert_stop(
                        sol, vtype, vb, ins_pos, rc, ra, dist_matrix, customers)
                    return True
    return False


@njit(cache=True)
def _swap_ins_delta(stops, lengths, vid, pos, new_cust, dist_matrix):
    """Cost delta (i64 meters) of new_cust at pos instead of current."""
    L = lengths[vid]
    cn = new_cust + 1
    prev_dm = 0 if pos == 0 else stops[vid, pos - 1] + 1
    next_dm = 0 if pos >= L - 1 else stops[vid, pos + 1] + 1
    return (dist_matrix[prev_dm, cn] + dist_matrix[cn, next_dm]
            - dist_matrix[prev_dm, next_dm])


@njit(cache=True)
def exchange_inter_route(sol, vtype, dist_matrix, customers):
    """Swap 1 customer between 2 routes of same vtype. No restricted param."""
    nv = n_vehicles(sol, vtype)
    if nv < 2:
        return False
    lengths = get_lengths(sol, vtype)
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)

    for va in range(nv):
        La = lengths[va]
        for pa in range(La):
            if actions[va, pa] != ACT_DELIVER:
                continue
            ca = stops[va, pa]
            for vb in range(va + 1, nv):
                Lb = lengths[vb]
                for pb in range(Lb):
                    if actions[vb, pb] != ACT_DELIVER:
                        continue
                    cb = stops[vb, pb]
                    # Restriction check: truck cannot serve bike-only
                    if vtype == VEH_TRUCK:
                        if customers[ca, COL_RESTRICTED] == 1:
                            continue
                        if customers[cb, COL_RESTRICTED] == 1:
                            continue
                    rem_a = removal_cost_delta(
                        sol, vtype, va, pa, dist_matrix)
                    rem_b = removal_cost_delta(
                        sol, vtype, vb, pb, dist_matrix)
                    ins_b_a = _swap_ins_delta(
                        stops, lengths, va, pa, cb, dist_matrix)
                    ins_a_b = _swap_ins_delta(
                        stops, lengths, vb, pb, ca, dist_matrix)
                    if rem_a + rem_b + ins_b_a + ins_a_b < -1:
                        c_a, a_a = remove_stop(
                            sol, vtype, va, pa, dist_matrix, customers)
                        c_b, a_b = remove_stop(
                            sol, vtype, vb, pb, dist_matrix, customers)
                        p1, _ = best_insertion_pos(
                            sol, vtype, va, c_b, dist_matrix)
                        insert_stop(
                            sol, vtype, va, p1, c_b, a_b,
                            dist_matrix, customers)
                        p2, _ = best_insertion_pos(
                            sol, vtype, vb, c_a, dist_matrix)
                        insert_stop(
                            sol, vtype, vb, p2, c_a, a_a,
                            dist_matrix, customers)
                        return True
    return False
