"""Cross-layer operators — change assignment between truck/bike. All @njit.
Signature: (sol, customers_i64, dist_matrix_i64, vehicles_i64, seed, config_f64) -> bool
No restricted param — read customers[:, COL_RESTRICTED].
"""
import numpy as np
from numba import njit

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE,
    COL_X, COL_Y, COL_DEMAND, COL_RESTRICTED,
    VCOL_CAPACITY, SAT_CUST, SAT_BIKE,
    SOL_CUST_VTYPE, SOL_SATELLITES, SOL_BIKE_LOADS,
    SOL_META, META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
)
from src.solution.query import (
    get_assigned_customers, get_customer_info, get_route_customers_only,
)
from src.solution.route_ops import remove_stop, insert_stop
from src.solution.delta import find_best_insertion_all_routes, best_insertion_pos
from src.solution.satellite_ops import (
    remove_satellites_for_customer, remove_satellites_for_bike,
    add_satellite_event,
)
from src.solution._helpers import get_route_arrays, get_loads, get_distances
from src.alns.crosslayer_helpers import (
    replace_reload_node, estimate_transfer_g,
    remove_all_reload_at_node, score_satellite_candidate,
)


@njit(cache=True)
def _get_capacity(vehicles, vtype):
    """Get max capacity for vtype from vehicles array."""
    cap = np.int64(0)
    for v in range(len(vehicles)):
        if vehicles[v, 0] == vtype and vehicles[v, VCOL_CAPACITY] > cap:
            cap = vehicles[v, VCOL_CAPACITY]
    return cap


@njit(cache=True)
def swap_assignment(sol, customers, dist_matrix, vehicles, seed, config):
    """Move customer truck<->bike. No restricted param."""
    np.random.seed(seed)
    N = len(customers)
    assigned = get_assigned_customers(sol, N)
    cust_vtype = sol[SOL_CUST_VTYPE]
    bike_cap = _get_capacity(vehicles, VEH_BIKE)
    flexible = np.empty(len(assigned), dtype=np.int32)
    n_flex = 0
    for i in range(len(assigned)):
        c = int(assigned[i])
        if customers[c, COL_RESTRICTED] == 1:
            continue
        vt = int(cust_vtype[c])
        demand = customers[c, COL_DEMAND]
        if (vt == VEH_TRUCK and demand <= bike_cap) or vt == VEH_BIKE:
            flexible[n_flex] = c
            n_flex += 1
    if n_flex == 0:
        return False
    c = int(flexible[np.random.randint(0, n_flex)])
    vtype, vid, pos = get_customer_info(sol, c)
    if vtype == VEH_TRUCK:
        return _truck_to_bike(sol, c, vid, pos, dist_matrix, customers, vehicles)
    return _bike_to_truck(sol, c, vid, pos, dist_matrix, customers, vehicles)


@njit(cache=True)
def _truck_to_bike(sol, c, vid, pos, dist_matrix, customers, vehicles):
    remove_stop(sol, VEH_TRUCK, vid, pos, dist_matrix, customers)
    bv, bp, _ = find_best_insertion_all_routes(sol, VEH_BIKE, c, dist_matrix, customers, vehicles)
    if bv >= 0:
        insert_stop(sol, VEH_BIKE, bv, bp, c, ACT_DELIVER, dist_matrix, customers)
        return True
    L = int(get_route_arrays(sol, VEH_TRUCK)[2][vid])
    insert_stop(sol, VEH_TRUCK, vid, min(pos, L), c, ACT_DELIVER, dist_matrix, customers)
    return False


@njit(cache=True)
def _bike_to_truck(sol, c, vid, pos, dist_matrix, customers, vehicles):
    remove_stop(sol, VEH_BIKE, vid, pos, dist_matrix, customers)
    remove_satellites_for_customer(sol, c)
    bv, bp, _ = find_best_insertion_all_routes(sol, VEH_TRUCK, c, dist_matrix, customers, vehicles)
    if bv >= 0:
        insert_stop(sol, VEH_TRUCK, bv, bp, c, ACT_DELIVER, dist_matrix, customers)
        return True
    L = int(get_route_arrays(sol, VEH_BIKE)[2][vid])
    insert_stop(sol, VEH_BIKE, vid, min(pos, L), c, ACT_DELIVER, dist_matrix, customers)
    return False


@njit(cache=True)
def relocate_satellite(sol, customers, dist_matrix, vehicles, seed, config):
    """Move satellite to better node."""
    np.random.seed(seed)
    sats = sol[SOL_SATELLITES]
    n_sats = sol[SOL_META][META_N_SATELLITES]
    if n_sats == 0:
        return False
    N = len(customers)
    # Manual unique of sats[:n_sats, SAT_CUST]
    seen = np.zeros(N, dtype=np.bool_)
    unique_buf = np.empty(n_sats, dtype=np.int32)
    n_unique = 0
    for i in range(n_sats):
        v = np.int32(sats[i, SAT_CUST])
        if not seen[v]:
            seen[v] = True
            unique_buf[n_unique] = v
            n_unique += 1
    if n_unique == 0:
        return False
    old_sat = int(unique_buf[np.random.randint(0, n_unique)])
    assigned = get_assigned_customers(sol, N)
    # Compute distances from old_sat to all assigned
    n_assigned = len(assigned)
    dists = np.empty(n_assigned, dtype=np.int64)
    for i in range(n_assigned):
        dists[i] = dist_matrix[old_sat + 1, assigned[i] + 1]
    order = np.argsort(dists)
    top = min(10, n_assigned)
    best_new = np.int32(-1)
    best_score = np.int64(9999999999)
    for k in range(top):
        cand = int(assigned[order[k]])
        if cand == old_sat:
            continue
        score = score_satellite_candidate(sol, cand, old_sat, dist_matrix)
        if score < best_score:
            best_score = score
            best_new = np.int32(cand)
    if best_new == -1:
        return False
    # Update satellite customer references
    for i in range(n_sats):
        if sats[i, SAT_CUST] == old_sat:
            sats[i, SAT_CUST] = best_new
    replace_reload_node(sol, old_sat, best_new)
    return True


@njit(cache=True)
def create_new_satellite(sol, customers, dist_matrix, vehicles, seed, config):
    """Create satellite at center of long bike route. i64 coordinates."""
    bike_dists = get_distances(sol, VEH_BIKE)
    if bike_dists.max() == 0:
        return False
    meta = sol[SOL_META]
    worst_bike = int(np.argmax(bike_dists))
    bike_custs = get_route_customers_only(sol, VEH_BIKE, worst_bike)
    if len(bike_custs) == 0:
        return False
    N = len(customers)
    # Center of mass (i64 mean)
    sum_x = np.int64(0)
    sum_y = np.int64(0)
    for i in range(len(bike_custs)):
        sum_x += customers[bike_custs[i], COL_X]
        sum_y += customers[bike_custs[i], COL_Y]
    cx = sum_x // len(bike_custs)
    cy = sum_y // len(bike_custs)
    assigned = get_assigned_customers(sol, N)
    # Squared distance to center
    n_assigned = len(assigned)
    d2c = np.empty(n_assigned, dtype=np.int64)
    for i in range(n_assigned):
        a = assigned[i]
        dx = customers[a, COL_X] - cx
        dy = customers[a, COL_Y] - cy
        d2c[i] = dx * dx + dy * dy
    new_sat = int(assigned[np.argmin(d2c)])
    sats = sol[SOL_SATELLITES]
    n_sats = meta[META_N_SATELLITES]
    # Check if new_sat already a satellite
    for i in range(n_sats):
        if int(sats[i, SAT_CUST]) == new_sat:
            return False
    best_truck_vid = np.int32(-1)
    best_truck_pos = np.int32(-1)
    best_delta = np.int64(9999999999)
    for t in range(meta[META_N_TRUCKS]):
        pos, delta = best_insertion_pos(sol, VEH_TRUCK, t, new_sat, dist_matrix)
        if delta < best_delta:
            best_delta = delta
            best_truck_vid = np.int32(t)
            best_truck_pos = pos
    if best_truck_vid == -1:
        return False
    insert_stop(sol, VEH_TRUCK, best_truck_vid, best_truck_pos, new_sat,
                ACT_RELOAD, dist_matrix, customers)
    bike_pos, _ = best_insertion_pos(sol, VEH_BIKE, worst_bike, new_sat, dist_matrix)
    insert_stop(sol, VEH_BIKE, worst_bike, bike_pos, new_sat,
                ACT_RELOAD, dist_matrix, customers)
    transfer_g = estimate_transfer_g(sol, worst_bike, bike_pos, customers)
    add_satellite_event(sol, new_sat, worst_bike, best_truck_vid,
                        transfer_g, np.int64(0))
    return True


@njit(cache=True)
def remove_satellite_node(sol, customers, dist_matrix, vehicles, seed, config):
    """Remove satellite completely."""
    np.random.seed(seed)
    sats = sol[SOL_SATELLITES]
    n_sats = sol[SOL_META][META_N_SATELLITES]
    if n_sats == 0:
        return False
    N = len(customers)
    # Manual unique
    seen = np.zeros(N, dtype=np.bool_)
    unique_buf = np.empty(n_sats, dtype=np.int32)
    n_unique = 0
    for i in range(n_sats):
        v = np.int32(sats[i, SAT_CUST])
        if not seen[v]:
            seen[v] = True
            unique_buf[n_unique] = v
            n_unique += 1
    if n_unique == 0:
        return False
    target = int(unique_buf[np.random.randint(0, n_unique)])
    remove_satellites_for_customer(sol, target)
    remove_all_reload_at_node(sol, target, dist_matrix, customers)
    return True
