"""Basic destroy operators: random, worst-cost, shaw, zone. @njit.
Signature: (sol, customers_i64, dist_matrix_i64, seed_i32, config_f64) -> i32[:]
No restricted param — read customers[:, COL_RESTRICTED] if needed.
"""
import numpy as np
from numba import njit

from src.data.constants import (
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
    CFG_Q_MIN, CFG_Q_MAX, CFG_WORST_NOISE, CFG_SHAW_RANDOMNESS, CFG_ZONE_PCT,
)
from src.solution.query import get_assigned_customers, get_customer_info
from src.solution.delta import removal_cost_delta
from src.alns.destroy_helpers import remove_targets, shaw_relatedness


@njit(cache=True)
def random_removal(sol, customers, dist_matrix, seed, config):
    """Remove q random customers."""
    np.random.seed(seed)
    N = len(customers)
    q = np.random.randint(int(config[CFG_Q_MIN]), int(config[CFG_Q_MAX]) + 1)
    assigned = get_assigned_customers(sol, N)
    q = min(q, len(assigned))
    if q == 0:
        return np.empty(0, dtype=np.int32)
    perm = np.random.permutation(len(assigned))
    targets = assigned[perm[:q]]
    return remove_targets(sol, targets, dist_matrix, customers)


@njit(cache=True)
def worst_cost_removal(sol, customers, dist_matrix, seed, config):
    """Remove q most expensive customers. Delta is i64; noise scaled."""
    np.random.seed(seed)
    N = len(customers)
    noise = config[CFG_WORST_NOISE]
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    costs = np.zeros(len(assigned), dtype=np.int64)
    for i in range(len(assigned)):
        c = assigned[i]
        vtype, vid, pos = get_customer_info(sol, c)
        costs[i] = -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)
        if costs[i] > 0:
            costs[i] += np.int64(np.random.random() * noise * costs[i])

    q = np.random.randint(int(config[CFG_Q_MIN]), int(config[CFG_Q_MAX]) + 1)
    q = min(q, len(assigned))
    targets = assigned[np.argsort(costs)[-q:]]
    return remove_targets(sol, targets, dist_matrix, customers)


@njit(cache=True)
def shaw_removal(sol, customers, dist_matrix, seed, config):
    """Remove q related customers."""
    np.random.seed(seed)
    N = len(customers)
    randomness = config[CFG_SHAW_RANDOMNESS]
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    seed_c = assigned[np.random.randint(0, len(assigned))]
    q = np.random.randint(int(config[CFG_Q_MIN]), int(config[CFG_Q_MAX]) + 1)
    q = min(q, len(assigned))

    max_dist = np.int64(1)
    for i in range(1, dist_matrix.shape[0]):
        for j in range(1, dist_matrix.shape[1]):
            if dist_matrix[i, j] > max_dist:
                max_dist = dist_matrix[i, j]
    max_demand = np.int64(1)
    for i in range(N):
        if customers[i, COL_DEMAND] > max_demand:
            max_demand = customers[i, COL_DEMAND]

    targets = np.empty(q, dtype=np.int32)
    targets[0] = seed_c
    n_targets = 1
    remaining = np.empty(len(assigned) - 1, dtype=np.int32)
    k = 0
    for i in range(len(assigned)):
        if assigned[i] != seed_c:
            remaining[k] = assigned[i]
            k += 1
    n_remaining = k

    while n_targets < q and n_remaining > 0:
        ref = targets[np.random.randint(0, n_targets)]
        scores = shaw_relatedness(ref, remaining[:n_remaining], customers,
                                  dist_matrix, sol, max_dist, max_demand)
        pos = int(np.random.random() ** randomness * n_remaining)
        pos = min(pos, n_remaining - 1)
        sorted_idx = np.argsort(scores)
        chosen = remaining[sorted_idx[pos]]
        targets[n_targets] = chosen
        n_targets += 1
        for i in range(n_remaining):
            if remaining[i] == chosen:
                remaining[i] = remaining[n_remaining - 1]
                n_remaining -= 1
                break

    return remove_targets(sol, targets[:n_targets], dist_matrix, customers)


@njit(cache=True)
def zone_removal(sol, customers, dist_matrix, seed, config):
    """Remove customers in geographic zone. Coordinates are i64 meters."""
    np.random.seed(seed)
    N = len(customers)
    zone_pct = config[CFG_ZONE_PCT]
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)

    center = assigned[np.random.randint(0, len(assigned))]
    cx = customers[center, COL_X]
    cy = customers[center, COL_Y]
    # Squared distances to avoid sqrt (i64 meters^2)
    dists_sq = np.empty(len(assigned), dtype=np.int64)
    for i in range(len(assigned)):
        dx = customers[assigned[i], COL_X] - cx
        dy = customers[assigned[i], COL_Y] - cy
        dists_sq[i] = dx * dx + dy * dy

    sorted_d = np.sort(dists_sq)
    idx = min(int(len(sorted_d) * zone_pct / 100.0), len(sorted_d) - 1)
    radius_sq = sorted_d[idx]

    targets = assigned[dists_sq <= radius_sq]
    return remove_targets(sol, targets, dist_matrix, customers)
