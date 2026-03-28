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


DESTROY_OPS = [
    random_removal,
    worst_cost_removal,
    shaw_removal,
    route_removal,
    satellite_removal,
    zone_removal,
]
