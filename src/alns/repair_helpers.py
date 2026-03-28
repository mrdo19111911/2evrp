"""Repair operator helpers — force insert, greedy insert single, top-k. @njit.
vehicles param added. No restricted param — read customers[:, COL_RESTRICTED].
All loads i64 grams, distances i64 meters, deltas i64.
"""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE,
    COL_DEMAND, COL_RESTRICTED, VCOL_CAPACITY,
    SOL_META, META_N_TRUCKS, META_N_BIKES,
)
from src.data.cost import DAY_LENGTH
from src.solution.route_ops import insert_stop
from src.solution.delta import (
    find_best_insertion_all_routes, best_insertion_pos,
    _check_tw_at_insertion, _estimate_route_return_time,
)
from src.solution._helpers import get_loads


@njit(cache=True)
def force_insert(sol, customer, dist_matrix, customers, vehicles):
    """Insert at cheapest TW+time-feasible position. vehicles i64 (K,4)."""
    meta = sol[SOL_META]
    demand = customers[customer, COL_DEMAND]
    best_vid, best_vtype, best_pos = -1, np.int32(0), np.int32(0)
    best_cost = np.int64(9999999999)
    has_feasible = False

    # Find truck/bike capacities from vehicles array
    truck_cap = np.int64(0)
    bike_cap = np.int64(0)
    for v in range(len(vehicles)):
        if vehicles[v, 0] == VEH_TRUCK and vehicles[v, VCOL_CAPACITY] > truck_cap:
            truck_cap = vehicles[v, VCOL_CAPACITY]
        if vehicles[v, 0] == VEH_BIKE and vehicles[v, VCOL_CAPACITY] > bike_cap:
            bike_cap = vehicles[v, VCOL_CAPACITY]

    for t in range(meta[META_N_TRUCKS]):
        rt = _estimate_route_return_time(sol, VEH_TRUCK, t, dist_matrix, customers, vehicles)
        if rt > DAY_LENGTH:
            continue
        pos, delta = best_insertion_pos(sol, VEH_TRUCK, t, customer, dist_matrix)
        tw_ok = _check_tw_at_insertion(sol, VEH_TRUCK, t, pos, customer,
                                        dist_matrix, customers, vehicles)
        cost = delta if tw_ok else delta + np.int64(100000000)
        if tw_ok:
            has_feasible = True
        if cost < best_cost:
            best_cost, best_vid, best_vtype, best_pos = cost, t, VEH_TRUCK, pos

    if demand <= bike_cap:
        for b in range(meta[META_N_BIKES]):
            rt = _estimate_route_return_time(sol, VEH_BIKE, b, dist_matrix, customers, vehicles)
            if rt > DAY_LENGTH:
                continue
            pos, delta = best_insertion_pos(sol, VEH_BIKE, b, customer, dist_matrix)
            tw_ok = _check_tw_at_insertion(sol, VEH_BIKE, b, pos, customer,
                                            dist_matrix, customers, vehicles)
            cost = delta if tw_ok else delta + np.int64(100000000)
            if tw_ok:
                has_feasible = True
            if cost < best_cost:
                best_cost, best_vid, best_vtype, best_pos = cost, b, VEH_BIKE, pos

    if best_vid >= 0 and has_feasible:
        insert_stop(sol, best_vtype, best_vid, best_pos, customer, ACT_DELIVER,
                     dist_matrix, customers)


@njit(cache=True)
def greedy_insert_single(sol, customer, customers, dist_matrix, vehicles):
    """Greedy insert 1 customer: best across truck + bike. No restricted param."""
    demand = customers[customer, COL_DEMAND]
    is_restricted = customers[customer, COL_RESTRICTED]
    best_cost = np.int64(9999999999)
    best_vtype, best_vid, best_pos = np.int32(-1), np.int32(-1), np.int32(-1)

    if is_restricted != 1:
        vid, pos, delta = find_best_insertion_all_routes(
            sol, VEH_TRUCK, customer, dist_matrix, customers, vehicles)
        if delta < best_cost:
            best_cost = delta
            best_vtype, best_vid, best_pos = VEH_TRUCK, vid, pos

    # Find bike capacity
    bike_cap = np.int64(0)
    for v in range(len(vehicles)):
        if vehicles[v, 0] == VEH_BIKE and vehicles[v, VCOL_CAPACITY] > bike_cap:
            bike_cap = vehicles[v, VCOL_CAPACITY]

    if demand <= bike_cap:
        vid, pos, delta = find_best_insertion_all_routes(
            sol, VEH_BIKE, customer, dist_matrix, customers, vehicles)
        if delta < best_cost:
            best_vtype, best_vid, best_pos = VEH_BIKE, vid, pos

    if best_vid >= 0:
        insert_stop(sol, best_vtype, best_vid, best_pos, customer, ACT_DELIVER,
                     dist_matrix, customers)
    else:
        force_insert(sol, customer, dist_matrix, customers, vehicles)


@njit(cache=True)
def find_top_k_insertions(sol, customer, k, customers, dist_matrix, vehicles):
    """Top-k insertions. Returns (deltas_i64, vtypes, vids, positions, n_found).
    No restricted param — read customers[customer, COL_RESTRICTED].
    """
    meta = sol[SOL_META]
    n_trucks = meta[META_N_TRUCKS]
    n_bikes = meta[META_N_BIKES]
    max_cands = n_trucks + n_bikes
    demand = customers[customer, COL_DEMAND]
    is_restricted = customers[customer, COL_RESTRICTED]

    deltas = np.empty(max_cands, dtype=np.int64)
    vtypes = np.empty(max_cands, dtype=np.int32)
    vids = np.empty(max_cands, dtype=np.int32)
    positions = np.empty(max_cands, dtype=np.int32)
    n = 0

    # Get capacities from vehicles
    truck_cap = np.int64(0)
    bike_cap = np.int64(0)
    for v in range(len(vehicles)):
        if vehicles[v, 0] == VEH_TRUCK and vehicles[v, VCOL_CAPACITY] > truck_cap:
            truck_cap = vehicles[v, VCOL_CAPACITY]
        if vehicles[v, 0] == VEH_BIKE and vehicles[v, VCOL_CAPACITY] > bike_cap:
            bike_cap = vehicles[v, VCOL_CAPACITY]

    if is_restricted != 1:
        loads = get_loads(sol, VEH_TRUCK)
        for t in range(n_trucks):
            if loads[t] + demand > truck_cap:
                continue
            pos, delta = best_insertion_pos(sol, VEH_TRUCK, t, customer, dist_matrix)
            deltas[n] = delta
            vtypes[n] = VEH_TRUCK
            vids[n] = t
            positions[n] = pos
            n += 1

    if demand <= bike_cap:
        loads = get_loads(sol, VEH_BIKE)
        for b in range(n_bikes):
            if loads[b] + demand > bike_cap:
                continue
            pos, delta = best_insertion_pos(sol, VEH_BIKE, b, customer, dist_matrix)
            deltas[n] = delta
            vtypes[n] = VEH_BIKE
            vids[n] = b
            positions[n] = pos
            n += 1

    # Insertion sort by delta ascending
    for i in range(1, n):
        j = i
        while j > 0 and deltas[j] < deltas[j - 1]:
            deltas[j], deltas[j-1] = deltas[j-1], deltas[j]
            vtypes[j], vtypes[j-1] = vtypes[j-1], vtypes[j]
            vids[j], vids[j-1] = vids[j-1], vids[j]
            positions[j], positions[j-1] = positions[j-1], positions[j]
            j -= 1

    out_n = min(k, n)
    return deltas[:out_n], vtypes[:out_n], vids[:out_n], positions[:out_n], out_n
