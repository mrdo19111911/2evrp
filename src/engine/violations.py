"""Penalty computation from constraint violations."""
from src.data.constants import COL_DEMAND
from src.data.cost import (
    TRUCK_CAPACITY, BIKE_CAPACITY,
    TRUCK_FIXED_DAY, BIKE_FIXED_DAY,
    BIKE_TOTAL_KM, BIKE_DRIVER_HOUR, BIKE_DEPLOY_COST, BIKE_SPEED_URBAN,
    PENALTY_OVERLOAD_PER_PCT, PENALTY_UNSERVED_MULTIPLIER,
    SERVICE_BASE, SERVICE_PER_100KG,
)


def calc_unserved_penalty(customer_idx, customers, dist_matrix):
    """Dynamic penalty = 5x cheapest bike round trip cost."""
    d = dist_matrix[0, customer_idx + 1]
    distance = 2.0 * d
    travel_min = distance / BIKE_SPEED_URBAN * 60.0
    demand = customers[customer_idx, COL_DEMAND]
    service_min = SERVICE_BASE + SERVICE_PER_100KG * (demand / 100.0)
    total_min = travel_min + service_min

    cost = (distance * BIKE_TOTAL_KM
            + total_min / 60.0 * BIKE_DRIVER_HOUR
            + BIKE_DEPLOY_COST)
    return PENALTY_UNSERVED_MULTIPLIER * cost


def compute_penalties(report, penalty_weights, customers=None, dist_matrix=None):
    """Total penalty from violations. Returns (total, breakdown)."""
    breakdown = {}

    unserved = report["delivery_uniqueness"]["unserved"]
    if len(unserved) > 0 and customers is not None and dist_matrix is not None:
        breakdown["unserved"] = sum(
            calc_unserved_penalty(int(c), customers, dist_matrix) for c in unserved)
    else:
        breakdown["unserved"] = len(unserved) * penalty_weights["unserved"]

    breakdown["duplicate"] = len(report["delivery_uniqueness"]["duplicates"]) * penalty_weights["duplicate"]

    cap_penalty = 0.0
    for (vtype, vid, stop_idx, load) in report["capacity"]["violations"]:
        if vtype == "bike":
            cap, fixed = BIKE_CAPACITY, BIKE_FIXED_DAY
        else:
            cap, fixed = TRUCK_CAPACITY, TRUCK_FIXED_DAY
        overload_pct = max(0.0, (load - cap) / cap * 100.0)
        cap_penalty += overload_pct * fixed * PENALTY_OVERLOAD_PER_PCT
    breakdown["capacity"] = cap_penalty

    tw_violations = report["time_windows"]["violations"]
    tw_penalty = sum(arrive - tw_close for (_, _, _, arrive, tw_close) in tw_violations) * penalty_weights["time_window"]
    breakdown["time_window"] = tw_penalty

    breakdown["sync"] = len(report["sync"]["violations"]) * penalty_weights["sync"]
    breakdown["vehicle_restriction"] = len(report["vehicle_restrictions"]["violations"]) * penalty_weights["vehicle_restriction"]

    total = sum(breakdown.values())
    return total, breakdown
