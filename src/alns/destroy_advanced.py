"""Advanced destroy operators: time pressure, cascade, inject, SISR, history.
Signature: (sol, customers_i64, dist_matrix_i64, seed_i32, config_f64) -> i32[:]
No restricted param. All @njit.
"""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, VEH_TRUCK, VEH_BIKE,
    SOL_CUST_VEHICLE, SOL_META, META_N_TRUCKS, META_N_BIKES,
    CFG_Q_MIN, CFG_Q_MAX, CFG_WORST_NOISE, CFG_HISTORY_NOISE,
    CFG_SISR_MAX_STRING_LEN, CFG_SISR_MAX_ROUTES,
)
from src.solution.query import get_assigned_customers, get_customer_info, get_route_customers_only
from src.solution._helpers import get_route_arrays
from src.solution.delta import removal_cost_delta
from src.alns.destroy_helpers import remove_targets


@njit(cache=True)
def time_pressure_removal(sol, customers, dist_matrix, seed, config):
    """Remove near-depot customers from longest routes. dist_matrix i64 meters."""
    np.random.seed(seed)
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)
    overloaded_d = np.empty(len(assigned), dtype=np.int64)
    overloaded_c = np.empty(len(assigned), dtype=np.int32)
    n_over = 0
    for c_raw in assigned:
        c = int(c_raw)
        vtype, vid, pos = get_customer_info(sol, c)
        if vtype == -1:
            continue
        _, _, lengths = get_route_arrays(sol, vtype)
        L = int(lengths[vid])
        if L > 15:
            overloaded_d[n_over] = dist_matrix[0, c + 1]
            overloaded_c[n_over] = c
            n_over += 1
    if n_over == 0:
        return np.empty(0, dtype=np.int32)
    order = np.argsort(overloaded_d[:n_over])
    q_hi = int(config[CFG_Q_MAX]) + 1
    if q_hi > n_over + 1:
        q_hi = n_over + 1
    q = np.random.randint(int(config[CFG_Q_MIN]), q_hi)
    targets = overloaded_c[order[:q]]
    return remove_targets(sol, targets, dist_matrix, customers)


@njit(cache=True)
def cascade_worst_removal(sol, customers, dist_matrix, seed, config):
    """Remove customers with highest removal cost savings. Returns i64."""
    np.random.seed(seed)
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)
    values = np.zeros(len(assigned), dtype=np.int64)
    noise = config[CFG_WORST_NOISE]
    for i in range(len(assigned)):
        c = assigned[i]
        vtype, vid, pos = get_customer_info(sol, c)
        values[i] = -removal_cost_delta(sol, vtype, vid, pos, dist_matrix)
        if values[i] > 0:
            values[i] += np.int64(np.random.random() * noise * values[i])
    q = np.random.randint(int(config[CFG_Q_MIN]), int(config[CFG_Q_MAX]) + 1)
    if q > len(assigned):
        q = len(assigned)
    targets = assigned[np.argsort(values)[-q:]]
    return remove_targets(sol, targets, dist_matrix, customers)


@njit(cache=True)
def inject_unserved(sol, customers, dist_matrix, seed, config):
    """Return unserved customers for repair to insert."""
    np.random.seed(seed)
    N = len(customers)
    # Manual np.where replacement
    cust_veh = sol[SOL_CUST_VEHICLE][:N]
    n_unserved = 0
    for i in range(N):
        if cust_veh[i] == -1:
            n_unserved += 1
    if n_unserved == 0:
        return np.empty(0, dtype=np.int32)
    unserved = np.empty(n_unserved, dtype=np.int32)
    k = 0
    for i in range(N):
        if cust_veh[i] == -1:
            unserved[k] = i
            k += 1
    q_lo = int(config[CFG_Q_MIN])
    q_hi = int(config[CFG_Q_MAX]) + 1
    if q_hi > n_unserved + 1:
        q_hi = n_unserved + 1
    if q_lo >= q_hi:
        q = q_lo
        if q > n_unserved:
            q = n_unserved
    else:
        q = np.random.randint(q_lo, q_hi)
    dists = np.empty(n_unserved, dtype=np.int64)
    for i in range(n_unserved):
        dists[i] = -dist_matrix[0, unserved[i] + 1]
    order = np.argsort(dists)
    return unserved[order[:q]]


@njit(cache=True)
def sisr_removal(sol, customers, dist_matrix, seed, config):
    """SISR: remove strings of consecutive delivers from nearby routes."""
    np.random.seed(seed)
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)
    q = np.random.randint(int(config[CFG_Q_MIN]), int(config[CFG_Q_MAX]) + 1)
    if q > len(assigned):
        q = len(assigned)
    if q == 0:
        return np.empty(0, dtype=np.int32)
    max_str_len = int(config[CFG_SISR_MAX_STRING_LEN])
    max_routes = int(config[CFG_SISR_MAX_ROUTES])
    seed_c = int(assigned[np.random.randint(0, len(assigned))])
    seed_vtype, seed_vid, seed_pos = get_customer_info(sol, seed_c)
    if seed_vtype == -1:
        return np.empty(0, dtype=np.int32)
    buf_size = max_str_len * max_routes + q
    if q * 3 > buf_size:
        buf_size = q * 3
    targets = np.empty(buf_size, dtype=np.int32)
    n_targets = 0
    n_targets = _collect_deliver_string(sol, seed_vtype, seed_vid, seed_pos,
                                         max_str_len, seed, targets, n_targets)
    nearby_vt, nearby_vi, nearby_po, n_nearby = _find_nearby_routes(
        sol, seed_c, customers, dist_matrix)
    limit = max_routes - 1
    if n_nearby < limit:
        limit = n_nearby
    for ri in range(limit):
        if n_targets >= q:
            break
        str_len = np.random.randint(1, max_str_len + 1)
        n_targets = _collect_deliver_string(sol, nearby_vt[ri], nearby_vi[ri],
                                             nearby_po[ri], str_len, seed,
                                             targets, n_targets)
    if n_targets == 0:
        return np.empty(0, dtype=np.int32)
    # Manual unique via boolean array
    seen = np.zeros(N, dtype=np.int8)
    unique = np.empty(n_targets, dtype=np.int32)
    n_unique = 0
    for i in range(n_targets):
        c = targets[i]
        if c >= 0 and c < N and seen[c] == 0:
            seen[c] = 1
            unique[n_unique] = c
            n_unique += 1
            if n_unique >= q:
                break
    return remove_targets(sol, unique[:n_unique], dist_matrix, customers)


@njit(cache=True)
def history_removal(sol, customers, dist_matrix, seed, config, history_freq):
    """Remove q customers with lowest history frequency."""
    np.random.seed(seed)
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    if len(assigned) == 0:
        return np.empty(0, dtype=np.int32)
    q = np.random.randint(int(config[CFG_Q_MIN]), int(config[CFG_Q_MAX]) + 1)
    if q > len(assigned):
        q = len(assigned)
    freq_sum = np.int64(0)
    for i in range(len(history_freq)):
        freq_sum += history_freq[i]
    if freq_sum == 0:
        perm = np.random.permutation(len(assigned))
        targets = np.empty(q, dtype=np.int32)
        for i in range(q):
            targets[i] = assigned[perm[i]]
        return remove_targets(sol, targets, dist_matrix, customers)
    freqs = np.empty(len(assigned), dtype=np.float64)
    for i in range(len(assigned)):
        freqs[i] = np.float64(history_freq[assigned[i]])
    noise = config[CFG_HISTORY_NOISE]
    max_freq = freqs[0]
    for i in range(1, len(freqs)):
        if freqs[i] > max_freq:
            max_freq = freqs[i]
    if max_freq > 0.0:
        for i in range(len(freqs)):
            freqs[i] += np.random.random() * noise * max_freq
    targets = assigned[np.argsort(freqs)[:q]]
    return remove_targets(sol, targets, dist_matrix, customers)


@njit(cache=True)
def _collect_deliver_string(sol, vtype, vid, center_pos, max_len, seed, out, n_out):
    """Collect consecutive DELIVER customers into out array. Returns new n_out."""
    stops, actions, lengths = get_route_arrays(sol, vtype)
    L = int(lengths[vid])
    if L == 0:
        return n_out
    # Manual collect deliver positions
    n_deliver = 0
    for i in range(L):
        if actions[vid, i] == ACT_DELIVER:
            n_deliver += 1
    if n_deliver == 0:
        return n_out
    deliver_pos = np.empty(n_deliver, dtype=np.int32)
    k = 0
    for i in range(L):
        if actions[vid, i] == ACT_DELIVER:
            deliver_pos[k] = i
            k += 1
    # searchsorted manually
    center_idx = n_deliver
    for i in range(n_deliver):
        if deliver_pos[i] >= center_pos:
            center_idx = i
            break
    if center_idx > n_deliver - 1:
        center_idx = n_deliver - 1
    str_max = max_len
    if n_deliver < str_max:
        str_max = n_deliver
    str_len = np.random.randint(1, str_max + 1)
    half = str_len // 2
    start = center_idx - half
    if start < 0:
        start = 0
    end = start + str_len
    if end > n_deliver:
        end = n_deliver
    start = end - str_len
    if start < 0:
        start = 0
    for idx in range(start, end):
        if n_out >= len(out):
            break
        out[n_out] = int(stops[vid, deliver_pos[idx]])
        n_out += 1
    return n_out


@njit(cache=True)
def _find_nearby_routes(sol, seed_cust, customers, dist_matrix):
    """Find routes with customers near seed. Returns (vtypes, vids, pos, count)."""
    seed_vtype, seed_vid, _ = get_customer_info(sol, seed_cust)
    meta = sol[SOL_META]
    n_trucks = int(meta[META_N_TRUCKS])
    n_bikes = int(meta[META_N_BIKES])
    max_cands = n_trucks + n_bikes
    cand_dist = np.empty(max_cands, dtype=np.int64)
    cand_vtype = np.empty(max_cands, dtype=np.int32)
    cand_vid = np.empty(max_cands, dtype=np.int32)
    cand_pos = np.empty(max_cands, dtype=np.int32)
    n_cand = 0
    for vtype in (VEH_TRUCK, VEH_BIKE):
        n_veh = n_trucks if vtype == VEH_TRUCK else n_bikes
        for vid in range(n_veh):
            if vtype == seed_vtype and vid == seed_vid:
                continue
            custs = get_route_customers_only(sol, vtype, vid)
            if len(custs) == 0:
                continue
            best_d = dist_matrix[seed_cust + 1, custs[0] + 1]
            best_idx = 0
            for ci in range(1, len(custs)):
                d = dist_matrix[seed_cust + 1, custs[ci] + 1]
                if d < best_d:
                    best_d = d
                    best_idx = ci
            nearest_cust = int(custs[best_idx])
            _, _, pos = get_customer_info(sol, nearest_cust)
            cand_dist[n_cand] = best_d
            cand_vtype[n_cand] = vtype
            cand_vid[n_cand] = vid
            cand_pos[n_cand] = pos
            n_cand += 1
    # Sort by distance
    order = np.argsort(cand_dist[:n_cand])
    out_vt = np.empty(n_cand, dtype=np.int32)
    out_vi = np.empty(n_cand, dtype=np.int32)
    out_po = np.empty(n_cand, dtype=np.int32)
    for i in range(n_cand):
        j = order[i]
        out_vt[i] = cand_vtype[j]
        out_vi[i] = cand_vid[j]
        out_po[i] = cand_pos[j]
    return out_vt, out_vi, out_po, n_cand
