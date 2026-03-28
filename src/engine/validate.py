"""Constraint validation — all @njit. Flat violation arrays, manual loops."""
import numpy as np
from numba import njit
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, ST_CUST, ST_ACTION, ST_ARRIVE, ST_DEPART,
    ST_LOAD_AFT, SAT_CUST, SAT_BIKE, SAT_TRUCK, COL_TW_CLOSE, COL_RESTRICTED,
    VCOL_CAPACITY, MAX_VIOLATIONS, SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_TRUCK_LENGTHS, SOL_BIKE_LENGTHS,
    SOL_SATELLITES, SOL_META, META_N_TRUCKS, META_N_BIKES, META_N_SATELLITES,
)


@njit(cache=True)
def validate_delivery_uniqueness(truck_stops, truck_actions, truck_lengths,
                                  bike_stops, bike_actions, bike_lengths,
                                  n_trucks, n_bikes, n_customers):
    """Each customer delivered exactly once. Returns (unserved, n_unserved, n_dup)."""
    count = np.zeros(n_customers, dtype=np.int32)
    for i in range(n_trucks):
        for j in range(truck_lengths[i]):
            if truck_actions[i, j] == ACT_DELIVER and truck_stops[i, j] >= 0:
                c = truck_stops[i, j]
                if c < n_customers:
                    count[c] += 1
    for i in range(n_bikes):
        for j in range(bike_lengths[i]):
            if bike_actions[i, j] == ACT_DELIVER and bike_stops[i, j] >= 0:
                c = bike_stops[i, j]
                if c < n_customers:
                    count[c] += 1

    unserved = np.empty(n_customers, dtype=np.int32)
    n_unserved = np.int32(0)
    n_dup = np.int32(0)
    for i in range(n_customers):
        if count[i] == 0:
            unserved[n_unserved] = np.int32(i)
            n_unserved += 1
        elif count[i] > 1:
            n_dup += 1
    return unserved, n_unserved, n_dup


@njit(cache=True)
def validate_vehicle_restrictions(truck_stops, truck_actions, truck_lengths,
                                   n_trucks, customers):
    """Restricted customers must not be truck-delivered. Returns (vr_viol, n_vr)."""
    n_cust = len(customers)
    seen = np.zeros(n_cust, dtype=np.int8)
    vr_viol = np.empty(n_cust, dtype=np.int32)
    n_vr = np.int32(0)
    for i in range(n_trucks):
        for j in range(truck_lengths[i]):
            if truck_actions[i, j] == ACT_DELIVER:
                c = truck_stops[i, j]
                if c >= 0 and c < n_cust and customers[c, COL_RESTRICTED] == 1 and seen[c] == 0:
                    seen[c] = 1
                    vr_viol[n_vr] = c
                    n_vr += 1
    return vr_viol, n_vr


@njit(cache=True)
def validate_capacity_3d(sim_states, lengths, vehicles, vtype_offset, n_vehicles):
    """Load never > capacity. Returns (violations i64(MAX_VIOLATIONS,4), n_viol)."""
    violations = np.zeros((MAX_VIOLATIONS, 4), dtype=np.int64)
    n_viol = 0
    for i in range(n_vehicles):
        L = np.int32(lengths[i])
        if L == 0:
            continue
        cap = vehicles[vtype_offset + i, VCOL_CAPACITY]
        for j in range(L):
            load = sim_states[i, j, ST_LOAD_AFT]
            if load < 0 or load > cap:
                if n_viol < MAX_VIOLATIONS:
                    violations[n_viol, 0] = 0
                    violations[n_viol, 1] = i
                    violations[n_viol, 2] = j
                    violations[n_viol, 3] = load
                    n_viol += 1
    return violations, n_viol


@njit(cache=True)
def validate_time_windows_3d(sim_states, lengths, n_vehicles, customers, vtype_label):
    """DELIVER stops within TW. Returns (violations i64(MAX_VIOLATIONS,5), n_viol)."""
    violations = np.zeros((MAX_VIOLATIONS, 5), dtype=np.int64)
    n_viol = 0
    for vid in range(n_vehicles):
        L = np.int32(lengths[vid])
        for row_idx in range(L):
            if sim_states[vid, row_idx, ST_ACTION] != ACT_DELIVER:
                continue
            cust = np.int32(sim_states[vid, row_idx, ST_CUST])
            arrive = sim_states[vid, row_idx, ST_ARRIVE]
            tw_close = customers[cust, COL_TW_CLOSE]
            if arrive > tw_close:
                if n_viol < MAX_VIOLATIONS:
                    violations[n_viol, 0] = vtype_label
                    violations[n_viol, 1] = vid
                    violations[n_viol, 2] = cust
                    violations[n_viol, 3] = arrive
                    violations[n_viol, 4] = tw_close
                    n_viol += 1
    return violations, n_viol


@njit(cache=True)
def validate_sync_3d(truck_sim, truck_lengths, bike_sim, bike_lengths,
                     satellites, n_satellites, delta_t_s):
    """Sync validation. Returns (violations i64(MAX_VIOLATIONS,4), n_viol)."""
    violations = np.zeros((MAX_VIOLATIONS, 4), dtype=np.int64)
    n_viol = 0
    for s in range(n_satellites):
        cust = np.int32(satellites[s, SAT_CUST])
        bike_id = np.int32(satellites[s, SAT_BIKE])
        truck_id = np.int32(satellites[s, SAT_TRUCK])
        truck_depart = _find_depart_3d(truck_sim, truck_lengths, truck_id, cust, ACT_RELOAD)
        if truck_depart < 0:
            if n_viol < MAX_VIOLATIONS:
                violations[n_viol, 0] = s
                violations[n_viol, 1] = -1
                violations[n_viol, 2] = -1
                violations[n_viol, 3] = -1
                n_viol += 1
            continue
        bike_arrive = _find_arrive_3d(bike_sim, bike_lengths, bike_id, cust, ACT_RELOAD)
        if bike_arrive < 0:
            if n_viol < MAX_VIOLATIONS:
                violations[n_viol, 0] = s
                violations[n_viol, 1] = -1
                violations[n_viol, 2] = -1
                violations[n_viol, 3] = -1
                n_viol += 1
            continue
        gap = abs(truck_depart - bike_arrive)
        if gap > delta_t_s:
            if n_viol < MAX_VIOLATIONS:
                violations[n_viol, 0] = s
                violations[n_viol, 1] = truck_depart
                violations[n_viol, 2] = bike_arrive
                violations[n_viol, 3] = gap
                n_viol += 1
    return violations, n_viol


@njit(cache=True)
def _find_depart_3d(sim, lengths, vid, cust, action):
    if vid >= len(lengths):
        return np.int64(-1)
    L = np.int32(lengths[vid])
    for i in range(L):
        if np.int32(sim[vid, i, ST_CUST]) == cust and np.int32(sim[vid, i, ST_ACTION]) == action:
            return sim[vid, i, ST_DEPART]
    return np.int64(-1)


@njit(cache=True)
def _find_arrive_3d(sim, lengths, vid, cust, action):
    if vid >= len(lengths):
        return np.int64(-1)
    L = np.int32(lengths[vid])
    for i in range(L):
        if np.int32(sim[vid, i, ST_CUST]) == cust and np.int32(sim[vid, i, ST_ACTION]) == action:
            return sim[vid, i, ST_ARRIVE]
    return np.int64(-1)


@njit(cache=True)
def validate_all(sol, truck_sim, bike_sim, customers,
                 vehicles, n_customers, delta_t_s):
    """Run all validators. Fully @njit."""
    meta = sol[SOL_META]
    n_trucks = meta[META_N_TRUCKS]
    n_bikes = meta[META_N_BIKES]
    n_sats = meta[META_N_SATELLITES]

    truck_stops = sol[SOL_TRUCK_STOPS]
    truck_actions = sol[SOL_TRUCK_ACTIONS]
    bike_stops = sol[SOL_BIKE_STOPS]
    bike_actions = sol[SOL_BIKE_ACTIONS]
    truck_lengths = sol[SOL_TRUCK_LENGTHS]
    bike_lengths = sol[SOL_BIKE_LENGTHS]
    sats = sol[SOL_SATELLITES]

    unserved, n_unserved, n_dup = validate_delivery_uniqueness(
        truck_stops, truck_actions, truck_lengths,
        bike_stops, bike_actions, bike_lengths,
        n_trucks, n_bikes, n_customers)

    vr_viol, n_vr = validate_vehicle_restrictions(
        truck_stops, truck_actions, truck_lengths, n_trucks, customers)

    # Capacity
    t_cap, n_tc = validate_capacity_3d(truck_sim, truck_lengths, vehicles, 0, n_trucks)
    b_cap, n_bc = validate_capacity_3d(bike_sim, bike_lengths, vehicles, n_trucks, n_bikes)
    cap_viol = np.zeros((MAX_VIOLATIONS, 4), dtype=np.int64)
    for i in range(n_tc):
        cap_viol[i, 0] = 0
        cap_viol[i, 1] = t_cap[i, 1]
        cap_viol[i, 2] = t_cap[i, 2]
        cap_viol[i, 3] = t_cap[i, 3]
    for i in range(n_bc):
        idx = n_tc + i
        if idx < MAX_VIOLATIONS:
            cap_viol[idx, 0] = 1
            cap_viol[idx, 1] = b_cap[i, 1]
            cap_viol[idx, 2] = b_cap[i, 2]
            cap_viol[idx, 3] = b_cap[i, 3]
    n_cap = n_tc + n_bc

    # Time windows
    t_tw, n_ttw = validate_time_windows_3d(truck_sim, truck_lengths, n_trucks, customers, 0)
    b_tw, n_btw = validate_time_windows_3d(bike_sim, bike_lengths, n_bikes, customers, 1)
    tw_viol = np.zeros((MAX_VIOLATIONS, 5), dtype=np.int64)
    for i in range(n_ttw):
        for k in range(5):
            tw_viol[i, k] = t_tw[i, k]
    for i in range(n_btw):
        idx = n_ttw + i
        if idx < MAX_VIOLATIONS:
            for k in range(5):
                tw_viol[idx, k] = b_tw[i, k]
    n_tw = n_ttw + n_btw

    # Sync
    sync_viol, n_sync = validate_sync_3d(truck_sim, truck_lengths, bike_sim, bike_lengths,
                                          sats, n_sats, delta_t_s)

    valid = (n_unserved == 0 and n_dup == 0 and n_vr == 0
             and n_cap == 0 and n_tw == 0 and n_sync == 0)

    return (valid, unserved, n_unserved, n_dup,
            vr_viol, n_vr, cap_viol, n_cap, tw_viol, n_tw, sync_viol, n_sync)
