"""Repair operators. All @njit(cache=True).
Signature: (sol, removed, customers_i64, dist_matrix_i64, vehicles_i64, seed_i32, config_f64)"""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE,
    COL_DEMAND, COL_RESTRICTED, VCOL_CAPACITY, SAT_CUST,
    SOL_SATELLITES, SOL_CUST_VEHICLE,
    SOL_META, META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
    CFG_REGRET_K, CFG_REGRET_NOISE, CFG_SAT_THRESHOLD_KM, CFG_BLINK_PROB,
)
from src.solution.route_ops import insert_stop
from src.solution.delta import find_best_insertion_all_routes, best_insertion_pos
from src.solution._helpers import get_loads, get_route_arrays
from src.alns.repair_helpers import force_insert, greedy_insert_single, find_top_k_insertions


@njit(cache=True)
def greedy_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config):
    """Insert each customer at cheapest position. Farthest first."""
    n = len(removed)
    dists = np.empty(n, dtype=np.int64)
    for i in range(n):
        dists[i] = -dist_matrix[0, removed[i] + 1]
    order = np.argsort(dists)
    for idx in order:
        greedy_insert_single(sol, int(removed[idx]), customers, dist_matrix, vehicles)


@njit(cache=True)
def _regret_k_core(sol, remaining, n_rem, k, noise_factor,
                    customers, dist_matrix, vehicles, seed):
    """Core regret-k loop. Full @njit."""
    np.random.seed(seed)
    while n_rem > 0:
        best_regret, best_idx = np.int64(-9999999999), -1
        best_vtype, best_vid, best_pos = np.int32(-1), np.int32(-1), np.int32(-1)
        has_best = False
        max_cost = np.int64(0)
        for ri in range(n_rem):
            c = remaining[ri]
            deltas, vtypes, vids, poss, nf = find_top_k_insertions(
                sol, c, k, customers, dist_matrix, vehicles)
            if nf <= 1:
                regret = np.int64(9999999999)
                if regret > best_regret:
                    best_regret, best_idx = regret, ri
                    if nf == 1:
                        best_vtype, best_vid, best_pos = vtypes[0], vids[0], poss[0]
                        has_best = True
                    else:
                        has_best = False
            else:
                regret = np.int64(0)
                for j in range(1, nf):
                    regret += deltas[j] - deltas[0]
                if deltas[nf - 1] > max_cost:
                    max_cost = deltas[nf - 1]
                noisy = regret
                if max_cost > 0:
                    noisy += np.int64(np.random.random() * noise_factor * max_cost)
                if noisy > best_regret:
                    best_regret, best_idx = noisy, ri
                    best_vtype, best_vid, best_pos = vtypes[0], vids[0], poss[0]
                    has_best = True
        if best_idx < 0:
            break
        if has_best:
            insert_stop(sol, best_vtype, best_vid, best_pos,
                         remaining[best_idx], ACT_DELIVER, dist_matrix, customers)
        else:
            force_insert(sol, remaining[best_idx], dist_matrix, customers, vehicles)
        remaining[best_idx] = remaining[n_rem - 1]
        n_rem -= 1


@njit(cache=True)
def regret_k_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config):
    """Regret-k insertion. Full @njit."""
    remaining = removed.copy()
    _regret_k_core(sol, remaining, len(remaining), int(config[CFG_REGRET_K]),
                    config[CFG_REGRET_NOISE], customers, dist_matrix, vehicles, seed)


@njit(cache=True)
def _try_satellite_bike(sol, c, customers, dist_matrix, active, n_active,
                        bike_cap, n_bikes, threshold):
    """Try inserting c into bike route near a satellite. Returns True if inserted."""
    if n_active == 0 or customers[c, COL_DEMAND] > bike_cap:
        return False
    # Find nearest active satellite
    best_dist = np.int64(9999999999)
    nearest = np.int32(-1)
    for i in range(n_active):
        d = dist_matrix[c + 1, active[i] + 1]
        if d < best_dist:
            best_dist = d
            nearest = active[i]
    if best_dist >= threshold:
        return False
    stops_b, actions_b, lengths_b = get_route_arrays(sol, VEH_BIKE)
    loads_b = get_loads(sol, VEH_BIKE)
    for b in range(n_bikes):
        if loads_b[b] + customers[c, COL_DEMAND] > bike_cap:
            continue
        L = int(lengths_b[b])
        found_reload = False
        for j in range(L):
            if stops_b[b, j] == nearest and actions_b[b, j] == ACT_RELOAD:
                found_reload = True
                break
        if not found_reload:
            continue
        pos, _ = best_insertion_pos(sol, VEH_BIKE, b, c, dist_matrix)
        insert_stop(sol, VEH_BIKE, b, pos, c, ACT_DELIVER, dist_matrix, customers)
        return True
    return False


@njit(cache=True)
def satellite_aware_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config):
    """Prefer inserting near active satellites."""
    np.random.seed(seed)
    sats = sol[SOL_SATELLITES]
    n_sats = sol[SOL_META][META_N_SATELLITES]
    # Collect unique active satellite customer indices
    active = np.empty(n_sats, dtype=np.int32)
    n_active = 0
    if n_sats > 0:
        for i in range(n_sats):
            val = np.int32(sats[i, SAT_CUST])
            found = False
            for j in range(n_active):
                if active[j] == val:
                    found = True
                    break
            if not found:
                active[n_active] = val
                n_active += 1
    # Compute order: sort by min distance to active satellites
    n = len(removed)
    keys = np.empty(n, dtype=np.int64)
    if n_active > 0:
        for i in range(n):
            c = removed[i]
            min_d = np.int64(9999999999)
            for j in range(n_active):
                d = dist_matrix[c + 1, active[j] + 1]
                if d < min_d:
                    min_d = d
            keys[i] = min_d
        order = np.argsort(keys)
    else:
        order = np.random.permutation(n)
    # Find bike capacity
    bike_cap = np.int64(0)
    for v in range(len(vehicles)):
        if vehicles[v, 0] == VEH_BIKE and vehicles[v, VCOL_CAPACITY] > bike_cap:
            bike_cap = vehicles[v, VCOL_CAPACITY]
    n_bikes = sol[SOL_META][META_N_BIKES]
    threshold = np.int64(config[CFG_SAT_THRESHOLD_KM])
    for idx in order:
        c = int(removed[idx])
        if _try_satellite_bike(sol, c, customers, dist_matrix, active, n_active,
                                bike_cap, n_bikes, threshold):
            continue
        greedy_insert_single(sol, c, customers, dist_matrix, vehicles)


@njit(cache=True)
def selective_drop_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config):
    """Insert far customers first. Drop near-depot if no TW-feasible position."""
    n = len(removed)
    dists = np.empty(n, dtype=np.int64)
    for i in range(n):
        dists[i] = -dist_matrix[0, removed[i] + 1]
    order = np.argsort(dists)
    # max depot distance
    max_d = np.int64(1)
    n_nodes = dist_matrix.shape[1]
    for i in range(1, n_nodes):
        if dist_matrix[0, i] > max_d:
            max_d = dist_matrix[0, i]
    for idx in order:
        c = int(removed[idx])
        importance = dist_matrix[0, c + 1] * 1000 // max_d
        best_vtype, best_vid, best_pos = np.int32(-1), np.int32(-1), np.int32(-1)
        best_cost = np.int64(9999999999)
        if customers[c, COL_RESTRICTED] != 1:
            vid, pos, delta = find_best_insertion_all_routes(
                sol, VEH_TRUCK, c, dist_matrix, customers, vehicles)
            if delta < best_cost:
                best_cost, best_vtype, best_vid, best_pos = delta, VEH_TRUCK, vid, pos
        vid, pos, delta = find_best_insertion_all_routes(
            sol, VEH_BIKE, c, dist_matrix, customers, vehicles)
        if delta < best_cost:
            best_vtype, best_vid, best_pos = VEH_BIKE, vid, pos
        if best_vid >= 0:
            insert_stop(sol, best_vtype, best_vid, best_pos, c, ACT_DELIVER,
                        dist_matrix, customers)
        elif importance > 500:
            force_insert(sol, c, dist_matrix, customers, vehicles)


@njit(cache=True)
def _blinks_core(sol, removed, customers, dist_matrix, vehicles, p_blink, seed):
    """Blinks insertion core. @njit."""
    np.random.seed(seed)
    n = len(removed)
    if n == 0:
        return
    meta = sol[SOL_META]
    dists = np.empty(n, dtype=np.int64)
    for i in range(n):
        dists[i] = dist_matrix[0, removed[i] + 1]
    order = np.argsort(-dists)
    truck_cap, bike_cap = np.int64(0), np.int64(0)
    for v in range(len(vehicles)):
        if vehicles[v, 0] == VEH_TRUCK and vehicles[v, VCOL_CAPACITY] > truck_cap:
            truck_cap = vehicles[v, VCOL_CAPACITY]
        if vehicles[v, 0] == VEH_BIKE and vehicles[v, VCOL_CAPACITY] > bike_cap:
            bike_cap = vehicles[v, VCOL_CAPACITY]
    cust_vehicle = sol[SOL_CUST_VEHICLE]
    for idx in range(n):
        c = removed[order[idx]]
        if cust_vehicle[c] != -1:
            continue
        demand = customers[c, COL_DEMAND]
        is_restr = customers[c, COL_RESTRICTED]
        best_vid, best_vtype, best_pos = np.int32(-1), np.int32(0), np.int32(0)
        best_cost = np.int64(9999999999)
        for vtype in (VEH_TRUCK, VEH_BIKE):
            if is_restr == 1 and vtype == VEH_TRUCK:
                continue
            cap = truck_cap if vtype == VEH_TRUCK else bike_cap
            n_veh = meta[META_N_TRUCKS] if vtype == VEH_TRUCK else meta[META_N_BIKES]
            loads = sol[6] if vtype == VEH_TRUCK else sol[7]
            for vid in range(n_veh):
                if loads[vid] + demand > cap:
                    continue
                pos, delta = best_insertion_pos(sol, vtype, vid, c, dist_matrix)
                if np.random.random() < p_blink:
                    continue
                if delta < best_cost:
                    best_cost, best_vid, best_vtype, best_pos = delta, vid, vtype, pos
        if best_vid >= 0:
            insert_stop(sol, best_vtype, best_vid, best_pos, c, ACT_DELIVER,
                        dist_matrix, customers)
        else:
            force_insert(sol, c, dist_matrix, customers, vehicles)


@njit(cache=True)
def blinks_insertion(sol, removed, customers, dist_matrix, vehicles, seed, config):
    """Wrapper. @njit."""
    _blinks_core(sol, removed, customers, dist_matrix, vehicles,
                  config[CFG_BLINK_PROB], seed)
