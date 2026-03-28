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
    """Try to insert at cheapest TW+time-feasible position.
    Skip routes already over DAY_LENGTH. Leave unserved if no feasible route."""
    from src.solution.delta import (
        _check_tw_at_insertion, _estimate_route_return_time,
    )
    from src.data.cost import DAY_LENGTH

    demand = customers[customer, COL_DEMAND]
    best_vid, best_vtype, best_pos, best_cost = -1, VEH_TRUCK, 0, np.inf
    has_feasible = False

    for t in range(sol["n_trucks"]):
        rt = _estimate_route_return_time(sol, VEH_TRUCK, t, dist_matrix, customers)
        if rt > DAY_LENGTH:
            continue
        pos, delta = best_insertion_pos(sol, VEH_TRUCK, t, customer, dist_matrix)
        tw_ok = _check_tw_at_insertion(sol, VEH_TRUCK, t, pos, customer,
                                        dist_matrix, customers)
        cost = delta if tw_ok else delta + 1e8
        if tw_ok:
            has_feasible = True
        if cost < best_cost:
            best_cost = cost
            best_vid, best_vtype, best_pos = t, VEH_TRUCK, pos

    if demand <= BIKE_CAPACITY:
        for b in range(sol["n_bikes"]):
            rt = _estimate_route_return_time(sol, VEH_BIKE, b, dist_matrix, customers)
            if rt > DAY_LENGTH:
                continue
            pos, delta = best_insertion_pos(sol, VEH_BIKE, b, customer, dist_matrix)
            tw_ok = _check_tw_at_insertion(sol, VEH_BIKE, b, pos, customer,
                                            dist_matrix, customers)
            cost = delta if tw_ok else delta + 1e8
            if tw_ok:
                has_feasible = True
            if cost < best_cost:
                best_cost = cost
                best_vid, best_vtype, best_pos = b, VEH_BIKE, pos

    if best_vid >= 0 and has_feasible:
        insert_stop(sol, best_vtype, best_vid, best_pos, customer, ACT_DELIVER,
                     dist_matrix, customers)
    # else: all routes full or over DAY_LENGTH → leave unserved


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
    """Insert each customer at cheapest position. Farthest from depot first."""
    # Sort farthest-first: hard-to-place customers get priority
    dists = np.array([dist_matrix[0, int(c) + 1] for c in removed])
    order = removed[np.argsort(-dists)]  # descending distance from depot
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


def selective_drop_insertion(sol, removed, customers, restricted, dist_matrix,
                             vehicles, rng, params):
    """Insert far customers first. Drop near-depot customers if no TW-feasible position.
    Real-world: prioritize hard-to-reach customers, sacrifice easy ones."""
    from src.solution.delta import _check_tw_at_insertion

    # Sort farthest-first
    dists = np.array([dist_matrix[0, int(c) + 1] for c in removed])
    order = removed[np.argsort(-dists)]

    max_d = float(dist_matrix[0, 1:].max()) if dist_matrix.shape[1] > 1 else 1.0

    for c in order:
        c = int(c)
        demand = customers[c, COL_DEMAND]
        d_depot = dist_matrix[0, c + 1]
        importance = d_depot / max(max_d, 1.0)  # 0 = at depot, 1 = farthest

        # Try TW-feasible insertion first
        best_move = None
        best_cost = np.inf

        if restricted[c] != 1:
            vid, pos, delta = find_best_insertion_all_routes(
                sol, VEH_TRUCK, c, dist_matrix, customers, TRUCK_CAPACITY)
            if delta < best_cost:
                best_cost = delta
                best_move = (VEH_TRUCK, vid, pos)

        if demand <= BIKE_CAPACITY:
            vid, pos, delta = find_best_insertion_all_routes(
                sol, VEH_BIKE, c, dist_matrix, customers, BIKE_CAPACITY)
            if delta < best_cost:
                best_cost = delta
                best_move = (VEH_BIKE, vid, pos)

        if best_move is not None:
            vtype, vid, pos = best_move
            insert_stop(sol, vtype, vid, pos, c, ACT_DELIVER,
                         dist_matrix, customers)
        elif importance > 0.5:
            # Far customer, no TW-feasible position: force insert anyway
            _force_insert(sol, c, dist_matrix, customers)
        # else: near/mid-depot customer with no feasible position → DROP


REPAIR_OPS = [
    greedy_insertion,
    regret_k_insertion,
    satellite_aware_insertion,
    selective_drop_insertion,
]
