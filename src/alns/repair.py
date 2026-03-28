"""Repair operators -- reinsert removed customers."""
import numpy as np

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE, COL_DEMAND, SAT_CUST,
)
from src.data.cost import TRUCK_CAPACITY, BIKE_CAPACITY
from src.solution.route_ops import insert_stop
from src.solution.delta import (
    find_best_insertion_all_routes, best_insertion_pos,
)
from src.solution.query import get_route_as_list
from src.solution.check import can_insert_customer


def _force_insert(sol, customer, dist_matrix, customers):
    """Insert into shortest feasible route. Heavy → truck only, never bike."""
    demand = customers[customer, COL_DEMAND]
    best_vid, best_vtype, best_len = -1, VEH_TRUCK, 999999

    # Always try trucks first
    for t in range(sol["n_trucks"]):
        if sol["truck_lengths"][t] < best_len:
            best_len = sol["truck_lengths"][t]
            best_vid, best_vtype = t, VEH_TRUCK

    # Only try bikes if demand fits
    if demand <= BIKE_CAPACITY:
        for b in range(sol["n_bikes"]):
            if sol["bike_lengths"][b] < best_len:
                best_len = sol["bike_lengths"][b]
                best_vid, best_vtype = b, VEH_BIKE

    if best_vid >= 0:
        pos, _ = best_insertion_pos(sol, best_vtype, best_vid, customer, dist_matrix)
        insert_stop(sol, best_vtype, best_vid, pos, customer, ACT_DELIVER,
                     dist_matrix, customers)


def _greedy_insert_single(sol, customer, customers, restricted, dist_matrix):
    """Greedy insert 1 customer: best across truck + bike."""
    demand = customers[customer, COL_DEMAND]
    best_cost = np.inf
    best_move = None

    if restricted[customer] != 1:
        vid, pos, delta = find_best_insertion_all_routes(
            sol, VEH_TRUCK, customer, dist_matrix, customers, TRUCK_CAPACITY)
        if delta < best_cost:
            best_cost = delta
            best_move = (VEH_TRUCK, vid, pos)

    if demand <= BIKE_CAPACITY:
        vid, pos, delta = find_best_insertion_all_routes(
            sol, VEH_BIKE, customer, dist_matrix, customers, BIKE_CAPACITY)
        if delta < best_cost:
            best_cost = delta
            best_move = (VEH_BIKE, vid, pos)

    if best_move is not None:
        vtype, vid, pos = best_move
        insert_stop(sol, vtype, vid, pos, customer, ACT_DELIVER,
                     dist_matrix, customers)
    else:
        _force_insert(sol, customer, dist_matrix, customers)


def greedy_insertion(sol, removed, customers, restricted, dist_matrix,
                     vehicles, rng, params):
    """Insert each customer at cheapest position."""
    order = rng.permutation(removed)
    for c in order:
        _greedy_insert_single(sol, int(c), customers, restricted, dist_matrix)


def _find_top_k_insertions(sol, customer, k, customers, restricted, dist_matrix):
    """Find top-k insertion positions across all routes. Returns sorted list."""
    demand = customers[customer, COL_DEMAND]
    candidates = []

    if restricted[customer] != 1:
        for t in range(sol["n_trucks"]):
            if sol["truck_loads"][t] + demand > TRUCK_CAPACITY:
                continue
            pos, delta = best_insertion_pos(sol, VEH_TRUCK, t, customer, dist_matrix)
            candidates.append((delta, VEH_TRUCK, t, pos))

    if demand <= BIKE_CAPACITY:
        for b in range(sol["n_bikes"]):
            if sol["bike_loads"][b] + demand > BIKE_CAPACITY:
                continue
            pos, delta = best_insertion_pos(sol, VEH_BIKE, b, customer, dist_matrix)
            candidates.append((delta, VEH_BIKE, b, pos))

    candidates.sort(key=lambda x: x[0])
    return candidates[:k]


def regret_k_insertion(sol, removed, customers, restricted, dist_matrix,
                       vehicles, rng, params):
    """Insert hardest-to-place customer first (highest regret)."""
    k = params.get("regret_k", 3)
    remaining = list(removed)

    while remaining:
        best_regret = -np.inf
        best_cust = None
        best_move = None

        for c in remaining:
            top_k = _find_top_k_insertions(sol, int(c), k, customers,
                                            restricted, dist_matrix)
            if len(top_k) == 0:
                regret = np.inf
                move = None
            elif len(top_k) == 1:
                regret = np.inf
                move = top_k[0]
            else:
                regret = sum(top_k[i][0] - top_k[0][0]
                             for i in range(1, len(top_k)))
                move = top_k[0]

            if regret > best_regret:
                best_regret = regret
                best_cust = c
                best_move = move

        if best_move is not None:
            _, vtype, vid, pos = best_move
            insert_stop(sol, vtype, vid, pos, int(best_cust), ACT_DELIVER,
                         dist_matrix, customers)
        else:
            _force_insert(sol, int(best_cust), dist_matrix, customers)

        remaining.remove(best_cust)


def satellite_aware_insertion(sol, removed, customers, restricted, dist_matrix,
                              vehicles, rng, params):
    """Prefer inserting near active satellites."""
    sat_threshold = params.get("sat_threshold_km", 5.0)
    sats = sol["satellites"]
    active_sats = np.unique(sats[:, SAT_CUST].astype(int)) if len(sats) > 0 else np.array([], dtype=int)

    if len(active_sats) > 0:
        min_dist_to_sat = np.array([
            dist_matrix[c + 1, active_sats + 1].min() for c in removed
        ])
        order = np.argsort(min_dist_to_sat)
    else:
        order = rng.permutation(len(removed))

    for idx in order:
        c = int(removed[idx])
        demand = customers[c, COL_DEMAND]
        inserted = False

        if len(active_sats) > 0 and demand <= BIKE_CAPACITY:
            nearest_sat = active_sats[np.argmin(dist_matrix[c + 1, active_sats + 1])]
            dist_to_sat = dist_matrix[c + 1, int(nearest_sat) + 1]

            if dist_to_sat < sat_threshold:
                for b in range(sol["n_bikes"]):
                    if sol["bike_loads"][b] + demand > BIKE_CAPACITY:
                        continue
                    route = get_route_as_list(sol, VEH_BIKE, b)
                    has_sat = any(node == nearest_sat and act == ACT_RELOAD
                                  for node, act in route)
                    if has_sat:
                        pos, _ = best_insertion_pos(sol, VEH_BIKE, b, c,
                                                     dist_matrix)
                        insert_stop(sol, VEH_BIKE, b, pos, c, ACT_DELIVER,
                                     dist_matrix, customers)
                        inserted = True
                        break

        if not inserted:
            _greedy_insert_single(sol, c, customers, restricted, dist_matrix)


REPAIR_OPS = [
    greedy_insertion,
    regret_k_insertion,
    satellite_aware_insertion,
]
