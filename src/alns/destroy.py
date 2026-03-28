"""Destroy operators -- remove customers from solution."""
import numpy as np
from itertools import groupby

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN,
    SAT_CUST, SAT_BIKE,
)
from src.data.cost import DAY_LENGTH
from src.solution.query import (
    get_assigned_customers, get_customer_info, get_route_customers_only,
)
from src.solution.route_ops import remove_stop
from src.solution.solution_ops import clear_route
from src.solution.satellite_ops import (
    remove_satellites_for_customer, remove_satellites_for_bike,
)
from src.solution.delta import removal_cost_delta


def _sort_by_pos_desc(customers_list, sol):
    """Sort customers by (vtype, vid, -pos) so highest pos removed first per route."""
    infos = [(c, *get_customer_info(sol, c)) for c in customers_list]
    infos.sort(key=lambda x: (x[1], x[2], -x[3]))
    result = []
    for _, grp in groupby(infos, key=lambda x: (x[1], x[2])):
        group_list = sorted(grp, key=lambda x: -x[3])
        result.extend([g[0] for g in group_list])
    return result


def random_removal(sol, customers, dist_matrix, rng, params):
    """Remove q random customers. Returns removed ndarray."""
    N = len(customers)
    q = rng.integers(params["q_min"], params["q_max"] + 1)
    assigned = get_assigned_customers(sol, N)
    q = min(q, len(assigned))
    if q == 0:
        return np.empty(0, dtype=np.int32)
    targets = rng.choice(assigned, size=q, replace=False)
    return _remove_targets(sol, targets, dist_matrix, customers)


def worst_cost_removal(sol, customers, dist_matrix, rng, params):
    """Remove q most expensive customers. Returns removed ndarray."""
    N = len(customers)
    noise = params.get("worst_noise", 0.1)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    costs = np.zeros(len(assigned), dtype=np.float64)
    for i, c in enumerate(assigned):
        vtype, vid, pos = get_customer_info(sol, c)
        costs[i] = -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)
        if costs[i] > 0:
            costs[i] += rng.uniform(0, noise * costs[i])

    q = rng.integers(params["q_min"], params["q_max"] + 1)
    q = min(q, len(assigned))
    targets = assigned[np.argsort(costs)[-q:]]
    return _remove_targets(sol, targets, dist_matrix, customers)


def shaw_removal(sol, customers, dist_matrix, rng, params):
    """Remove q similar customers. Returns removed ndarray."""
    N = len(customers)
    randomness = params.get("shaw_randomness", 6.0)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    seed = int(rng.choice(assigned))
    q = rng.integers(params["q_min"], params["q_max"] + 1)
    q = min(q, len(assigned))

    targets = [seed]
    remaining = [int(c) for c in assigned if c != seed]

    while len(targets) < q and remaining:
        scores = np.array([
            min(dist_matrix[c + 1, t + 1] for t in targets) for c in remaining
        ])
        pos = int(rng.random() ** randomness * len(remaining))
        pos = min(pos, len(remaining) - 1)
        sorted_idx = np.argsort(scores)
        chosen = remaining[sorted_idx[pos]]
        targets.append(chosen)
        remaining.remove(chosen)

    return _remove_targets(sol, np.array(targets, dtype=np.int32),
                           dist_matrix, customers)


def route_removal(sol, customers, dist_matrix, rng, params):
    """Remove entire route. Returns removed ndarray."""
    nonempty = []
    for t in range(sol["n_trucks"]):
        if sol["truck_lengths"][t] > 0:
            nonempty.append((VEH_TRUCK, t))
    for b in range(sol["n_bikes"]):
        if sol["bike_lengths"][b] > 0:
            nonempty.append((VEH_BIKE, b))

    if not nonempty:
        return np.empty(0, dtype=np.int32)

    vtype, vid = nonempty[rng.integers(len(nonempty))]
    removed_pairs = clear_route(sol, vtype, vid)

    removed = []
    for c, action in removed_pairs:
        if action == ACT_DELIVER:
            removed.append(c)
        remove_satellites_for_customer(sol, c)

    return np.array(removed, dtype=np.int32)


def satellite_removal(sol, customers, dist_matrix, rng, params):
    """Remove satellite + all dependent bike customers. Returns removed ndarray."""
    sats = sol["satellites"]
    if len(sats) == 0:
        return np.empty(0, dtype=np.int32)

    unique_sat_custs = np.unique(sats[:, SAT_CUST].astype(int))
    sat_cust = int(rng.choice(unique_sat_custs))

    mask = sats[:, SAT_CUST] == sat_cust
    bike_ids = np.unique(sats[mask, SAT_BIKE].astype(int))

    removed = []
    n_bikes = sol["n_bikes"]
    for bid in bike_ids:
        if bid >= n_bikes:
            continue
        deliver_custs = get_route_customers_only(sol, VEH_BIKE, bid)
        for c in _sort_by_pos_desc(deliver_custs.tolist(), sol):
            vtype, vid, pos = get_customer_info(sol, c)
            if vtype == -1:
                continue
            remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
            removed.append(c)
        clear_route(sol, VEH_BIKE, bid)
        remove_satellites_for_bike(sol, bid)

    return np.array(removed, dtype=np.int32)


def zone_removal(sol, customers, dist_matrix, rng, params):
    """Remove customers in geographic zone. Returns removed ndarray."""
    N = len(customers)
    zone_pct = params.get("zone_pct", 15)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    center = int(rng.choice(assigned))
    cx, cy = customers[center, COL_X], customers[center, COL_Y]
    dists_to_center = np.sqrt(
        (customers[assigned, COL_X] - cx) ** 2 +
        (customers[assigned, COL_Y] - cy) ** 2
    )
    radius = np.percentile(dists_to_center, zone_pct)
    targets = assigned[dists_to_center <= radius]
    return _remove_targets(sol, targets, dist_matrix, customers)


def _remove_targets(sol, targets, dist_matrix, customers):
    """Remove targets safely. Re-query position each time (positions shift after remove)."""
    removed = []
    for c in _sort_by_pos_desc(targets.tolist(), sol):
        vtype, vid, pos = get_customer_info(sol, c)
        if vtype == -1:
            continue  # already removed
        remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
        remove_satellites_for_customer(sol, c)
        removed.append(c)
    return np.array(removed, dtype=np.int32)


def time_pressure_removal(sol, customers, dist_matrix, rng, params):
    """Remove near-depot customers from longest routes to reduce route time.
    Real-world logic: drop easy (near) customers to make routes feasible."""
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    # Find customers on routes that are too long (high stop count)
    overloaded = []
    for c_raw in assigned:
        c = int(c_raw)
        vtype, vid, pos = get_customer_info(sol, c)
        if vtype == -1:
            continue
        L = int(sol["truck_lengths"][vid] if vtype == VEH_TRUCK else sol["bike_lengths"][vid])
        if L > 15:  # route has too many stops for DAY_LENGTH
            d_depot = dist_matrix[0, c + 1]
            overloaded.append((d_depot, c))

    if not overloaded:
        return np.empty(0, dtype=np.int32)

    # Remove nearest-to-depot customers from overloaded routes
    overloaded.sort()  # ascending distance → nearest first
    q = rng.integers(params["q_min"], min(params["q_max"] + 1, len(overloaded) + 1))
    targets = np.array([c for _, c in overloaded[:q]], dtype=np.int32)
    return _remove_targets(sol, targets, dist_matrix, customers)


def route_split_removal(sol, customers, dist_matrix, rng, params):
    """Split longest route in half: keep first half, remove second half.
    Fundamentally different from customer-level destroy: restructures routes."""
    # Find route with most stops across both vehicle types
    worst_vtype, worst_vid, worst_L = -1, -1, 0
    for t in range(sol["n_trucks"]):
        L = int(sol["truck_lengths"][t])
        if L > worst_L:
            worst_L, worst_vtype, worst_vid = L, VEH_TRUCK, t
    for b in range(sol["n_bikes"]):
        L = int(sol["bike_lengths"][b])
        if L > worst_L:
            worst_L, worst_vtype, worst_vid = L, VEH_BIKE, b

    if worst_L < 6:
        return np.empty(0, dtype=np.int32)

    # Split at time-based position: keep first ~DAY_LENGTH worth of stops
    from src.solution.delta import _estimate_arrival_at_pos
    from src.data.cost import DAY_LENGTH
    split_pos = worst_L // 2  # default: halfway
    for p in range(3, worst_L - 2):
        clock, _ = _estimate_arrival_at_pos(sol, worst_vtype, worst_vid, p,
                                             dist_matrix, customers)
        if clock > DAY_LENGTH * 0.8:
            split_pos = p
            break
    split_pos = max(3, min(split_pos, worst_L - 3))

    # Remove second half (positions split_pos..worst_L-1, backward)
    stops = sol["truck_stops"] if worst_vtype == VEH_TRUCK else sol["bike_stops"]
    actions = sol["truck_actions"] if worst_vtype == VEH_TRUCK else sol["bike_actions"]

    removed = []
    for i in range(worst_L - 1, split_pos - 1, -1):
        if int(actions[worst_vid, i]) == ACT_DELIVER:
            c = int(stops[worst_vid, i])
            vtype, vid, pos = get_customer_info(sol, c)
            if vtype != -1:
                remove_stop(sol, vtype, vid, pos, dist_matrix, customers)
                remove_satellites_for_customer(sol, c)
                removed.append(c)

    return np.array(removed, dtype=np.int32)


def cascade_worst_removal(sol, customers, dist_matrix, rng, params):
    """Remove customers with highest cascade removal value (time + TW savings)."""
    from src.solution.delta import cascade_removal_value

    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    values = np.zeros(len(assigned), dtype=np.float64)
    for i, c in enumerate(assigned):
        vtype, vid, pos = get_customer_info(sol, c)
        values[i] = cascade_removal_value(sol, vtype, vid, pos, dist_matrix, customers)
        noise = params.get("worst_noise", 0.1)
        if values[i] > 0:
            values[i] += rng.uniform(0, noise * values[i])

    q = rng.integers(params["q_min"], params["q_max"] + 1)
    q = min(q, len(assigned))
    targets = assigned[np.argsort(values)[-q:]]
    return _remove_targets(sol, targets, dist_matrix, customers)


def inject_unserved(sol, customers, dist_matrix, rng, params):
    """'Destroy' that removes nothing but returns unserved customers for repair to insert.
    This bridges init gaps: customers that init failed to place get a chance every iteration."""
    N = len(customers)
    unserved = np.where(sol["cust_vehicle"][:N] == -1)[0].astype(np.int32)
    if len(unserved) == 0:
        return np.empty(0, dtype=np.int32)
    q = rng.integers(params["q_min"], min(params["q_max"] + 1, len(unserved) + 1))
    # Pick farthest-from-depot first (highest priority)
    dists = dist_matrix[0, unserved + 1]
    order = np.argsort(-dists)
    return unserved[order[:q]]


DESTROY_OPS = [
    random_removal,
    worst_cost_removal,
    shaw_removal,
    route_removal,
    satellite_removal,
    zone_removal,
    time_pressure_removal,
    route_split_removal,
    cascade_worst_removal,
    inject_unserved,
]
