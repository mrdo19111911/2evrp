"""Constraint validation — full check after simulation."""
import numpy as np

from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD,
    ST_CUST, ST_ACTION, ST_ARRIVE, ST_DEPART, ST_LOAD_AFT,
    SAT_CUST, SAT_BIKE, SAT_TRUCK,
    COL_TW_CLOSE, VCOL_CAPACITY,
)


def validate_delivery_uniqueness(truck_stops, truck_actions, bike_stops, bike_actions, n_customers):
    """Each customer delivered exactly once. Returns (valid, unserved, duplicates)."""
    count = np.zeros(n_customers, dtype=np.int32)

    mask = (truck_actions == ACT_DELIVER) & (truck_stops >= 0)
    np.add.at(count, truck_stops[mask], 1)

    mask = (bike_actions == ACT_DELIVER) & (bike_stops >= 0)
    np.add.at(count, bike_stops[mask], 1)

    unserved = np.where(count == 0)[0].astype(np.int32)
    duplicates = np.where(count > 1)[0].astype(np.int32)
    valid = len(unserved) == 0 and len(duplicates) == 0
    return valid, unserved, duplicates


def validate_vehicle_restrictions(truck_stops, truck_actions, restricted):
    """Restricted customers not truck-delivered. Returns (valid, violations)."""
    mask = (truck_actions == ACT_DELIVER) & (truck_stops >= 0)
    truck_delivers = truck_stops[mask]
    violations = truck_delivers[restricted[truck_delivers] == 1]
    violations = np.unique(violations).astype(np.int32)
    valid = len(violations) == 0
    return valid, violations


def validate_capacity_all(truck_states, bike_states, vehicles):
    """Load never < 0 or > capacity. Returns (valid, violations)."""
    violations = []
    n_trucks = len(truck_states)

    for i, state in enumerate(truck_states):
        if len(state) == 0:
            continue
        cap = vehicles[i, VCOL_CAPACITY]
        load_aft = state[:, ST_LOAD_AFT]
        bad = np.where((load_aft < 0) | (load_aft > cap))[0]
        for j in bad:
            violations.append(("truck", i, int(j), float(load_aft[j])))

    for i, state in enumerate(bike_states):
        if len(state) == 0:
            continue
        cap = vehicles[n_trucks + i, VCOL_CAPACITY]
        load_aft = state[:, ST_LOAD_AFT]
        bad = np.where((load_aft < 0) | (load_aft > cap))[0]
        for j in bad:
            violations.append(("bike", i, int(j), float(load_aft[j])))

    return len(violations) == 0, violations


def validate_time_windows(truck_states, bike_states, customers):
    """DELIVER stops within TW. Returns (valid, violations)."""
    violations = []

    for vtype, states in [("truck", truck_states), ("bike", bike_states)]:
        for vid, state in enumerate(states):
            if len(state) == 0:
                continue
            for row_idx in range(len(state)):
                if int(state[row_idx, ST_ACTION]) != ACT_DELIVER:
                    continue
                cust = int(state[row_idx, ST_CUST])
                arrive = state[row_idx, ST_ARRIVE]
                tw_close = customers[cust, COL_TW_CLOSE]
                if arrive > tw_close:
                    violations.append((vtype, vid, cust, float(arrive), float(tw_close)))

    return len(violations) == 0, violations


def validate_sync(truck_states, bike_states, satellites, delta_t):
    """Truck RELOAD and bike RELOAD at same satellite node within +-delta_t.

    Truck action = RELOAD (hands off goods). Bike action = RELOAD (receives goods).

    Returns (valid, violations).
    """
    violations = []

    for s in range(len(satellites)):
        cust = int(satellites[s, SAT_CUST])
        bike_id = int(satellites[s, SAT_BIKE])
        truck_id = int(satellites[s, SAT_TRUCK])

        if truck_id >= len(truck_states) or bike_id >= len(bike_states):
            violations.append((s, "invalid_vehicle_id", {}))
            continue

        # Truck RELOAD at satellite real_node (hands off goods to bikes)
        t_state = truck_states[truck_id]
        t_mask = (t_state[:, ST_CUST] == cust) & (t_state[:, ST_ACTION] == ACT_RELOAD)
        if not np.any(t_mask):
            violations.append((s, "truck_not_at_node", {}))
            continue
        truck_depart = t_state[t_mask][0, ST_DEPART]

        # Bike RELOADs at satellite real_node
        b_state = bike_states[bike_id]
        b_mask = (b_state[:, ST_CUST] == cust) & (b_state[:, ST_ACTION] == ACT_RELOAD)
        if not np.any(b_mask):
            violations.append((s, "bike_not_at_node", {}))
            continue
        bike_arrive = b_state[b_mask][0, ST_ARRIVE]

        gap = abs(float(truck_depart) - float(bike_arrive))
        if gap > delta_t:
            violations.append((s, "sync_gap_too_large",
                {"truck_depart": float(truck_depart),
                 "bike_arrive": float(bike_arrive),
                 "gap": gap}))

    return len(violations) == 0, violations


def validate_all(truck_stops, truck_actions, bike_stops, bike_actions,
                 truck_states, bike_states, satellites, customers, restricted,
                 vehicles, n_customers, delta_t):
    """Run all validators. Returns (valid, report)."""
    du_valid, unserved, duplicates = validate_delivery_uniqueness(
        truck_stops, truck_actions, bike_stops, bike_actions, n_customers)
    vr_valid, vr_violations = validate_vehicle_restrictions(
        truck_stops, truck_actions, restricted)
    cap_valid, cap_violations = validate_capacity_all(truck_states, bike_states, vehicles)
    tw_valid, tw_violations = validate_time_windows(truck_states, bike_states, customers)
    sync_valid, sync_violations = validate_sync(truck_states, bike_states, satellites, delta_t)

    report = {
        "delivery_uniqueness": {"valid": du_valid, "unserved": unserved, "duplicates": duplicates},
        "vehicle_restrictions": {"valid": vr_valid, "violations": vr_violations},
        "capacity": {"valid": cap_valid, "violations": cap_violations},
        "time_windows": {"valid": tw_valid, "violations": tw_violations},
        "sync": {"valid": sync_valid, "violations": sync_violations},
    }
    valid = du_valid and vr_valid and cap_valid and tw_valid and sync_valid
    return valid, report
