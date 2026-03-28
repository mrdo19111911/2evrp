"""Advanced inter-route LS: swap*, ejection chain. @njit, i64 types."""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE, COL_DEMAND, COL_RESTRICTED,
    VCOL_CAPACITY, SOL_CUST_ROUTE_POS, SOL_META, META_N_TRUCKS,
)
from src.solution.route_ops import remove_stop, insert_stop
from src.solution.delta import removal_cost_delta, best_insertion_pos
from src.alns.ls_helpers import (
    get_stops, get_actions, get_lengths, get_loads, n_vehicles,
)


@njit(cache=True)
def swap_star(sol, vtype, dist_matrix, customers):
    """SWAP*: remove both, reinsert at best pos in other route.

    Reads restricted from customers[:, COL_RESTRICTED]. No separate param.
    """
    nv = n_vehicles(sol, vtype)
    if nv < 2:
        return False
    lengths = get_lengths(sol, vtype)
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)
    for va in range(nv):
        for pa in range(lengths[va]):
            if actions[va, pa] != ACT_DELIVER:
                continue
            ca = stops[va, pa]
            for vb in range(va + 1, nv):
                for pb in range(lengths[vb]):
                    if actions[vb, pb] != ACT_DELIVER:
                        continue
                    cb = stops[vb, pb]
                    # Restriction: truck cannot serve bike-only
                    if vtype == VEH_TRUCK:
                        if customers[ca, COL_RESTRICTED] == 1:
                            continue
                        if customers[cb, COL_RESTRICTED] == 1:
                            continue
                    ra = removal_cost_delta(
                        sol, vtype, va, pa, dist_matrix)
                    rb = removal_cost_delta(
                        sol, vtype, vb, pb, dist_matrix)
                    _, iab = best_insertion_pos(
                        sol, vtype, vb, ca, dist_matrix)
                    _, iba = best_insertion_pos(
                        sol, vtype, va, cb, dist_matrix)
                    if ra + rb + iab + iba < -1:
                        ca2, aa = remove_stop(
                            sol, vtype, va, pa, dist_matrix, customers)
                        cb2, ab = remove_stop(
                            sol, vtype, vb, pb, dist_matrix, customers)
                        p1, _ = best_insertion_pos(
                            sol, vtype, vb, ca2, dist_matrix)
                        insert_stop(sol, vtype, vb, p1, ca2, aa,
                                    dist_matrix, customers)
                        p2, _ = best_insertion_pos(
                            sol, vtype, va, cb2, dist_matrix)
                        insert_stop(sol, vtype, va, p2, cb2, ab,
                                    dist_matrix, customers)
                        return True
    return False


@njit(cache=True)
def ejection_chain_3(sol, vtype, dist_matrix, customers, vehicles):
    """3-way cyclic transfer: c1:A->B, c2:B->C, c3:C->A. i64 types."""
    nv = n_vehicles(sol, vtype)
    if nv < 3:
        return False
    n_trucks = sol[SOL_META][META_N_TRUCKS]
    if vtype == VEH_TRUCK:
        capacity = vehicles[0, VCOL_CAPACITY]
    else:
        capacity = vehicles[n_trucks, VCOL_CAPACITY]
    lengths = get_lengths(sol, vtype)
    stops = get_stops(sol, vtype)
    actions = get_actions(sol, vtype)
    loads = get_loads(sol, vtype)
    crp = sol[SOL_CUST_ROUTE_POS]

    max_per_route = 100
    rc_custs = np.empty((nv, max_per_route), dtype=np.int32)
    rc_rems = np.empty((nv, max_per_route), dtype=np.int64)
    rc_counts = np.zeros(nv, dtype=np.int32)

    for vid in range(nv):
        L = lengths[vid]
        k = 0
        for p in range(L):
            if actions[vid, p] == ACT_DELIVER and k < max_per_route:
                rc_custs[vid, k] = stops[vid, p]
                rc_rems[vid, k] = removal_cost_delta(
                    sol, vtype, vid, p, dist_matrix)
                k += 1
        rc_counts[vid] = k

    for va in range(nv):
        na = rc_counts[va]
        for vb in range(nv):
            if vb == va:
                continue
            nb = rc_counts[vb]
            for vc in range(nv):
                if vc == va or vc == vb:
                    continue
                nc = rc_counts[vc]
                for ia in range(na):
                    c1 = rc_custs[va, ia]
                    r1 = rc_rems[va, ia]
                    _, i1 = best_insertion_pos(
                        sol, vtype, vb, c1, dist_matrix)
                    for ib in range(nb):
                        c2 = rc_custs[vb, ib]
                        r2 = rc_rems[vb, ib]
                        _, i2 = best_insertion_pos(
                            sol, vtype, vc, c2, dist_matrix)
                        for ic in range(nc):
                            c3 = rc_custs[vc, ic]
                            r3 = rc_rems[vc, ic]
                            _, i3 = best_insertion_pos(
                                sol, vtype, va, c3, dist_matrix)
                            if r1 + i1 + r2 + i2 + r3 + i3 >= -1:
                                continue
                            d1 = customers[c1, COL_DEMAND]
                            d2 = customers[c2, COL_DEMAND]
                            d3 = customers[c3, COL_DEMAND]
                            if loads[va] - d1 + d3 > capacity:
                                continue
                            if loads[vb] - d2 + d1 > capacity:
                                continue
                            if loads[vc] - d3 + d2 > capacity:
                                continue
                            remove_stop(sol, vtype, va, crp[c1],
                                        dist_matrix, customers)
                            remove_stop(sol, vtype, vb, crp[c2],
                                        dist_matrix, customers)
                            remove_stop(sol, vtype, vc, crp[c3],
                                        dist_matrix, customers)
                            p1, _ = best_insertion_pos(
                                sol, vtype, vb, c1, dist_matrix)
                            insert_stop(sol, vtype, vb, p1, c1,
                                        ACT_DELIVER, dist_matrix, customers)
                            p2, _ = best_insertion_pos(
                                sol, vtype, vc, c2, dist_matrix)
                            insert_stop(sol, vtype, vc, p2, c2,
                                        ACT_DELIVER, dist_matrix, customers)
                            p3, _ = best_insertion_pos(
                                sol, vtype, va, c3, dist_matrix)
                            insert_stop(sol, vtype, va, p3, c3,
                                        ACT_DELIVER, dist_matrix, customers)
                            return True
    return False
