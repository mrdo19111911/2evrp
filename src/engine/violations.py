"""Penalty computation from constraint violations. All i64. @njit."""
import numpy as np
from numba import njit

from src.data.constants import (
    COL_DEMAND, VCOL_CAPACITY, VCOL_COST_M, VCOL_SPEED,
    PW_UNSERVED, PW_DUPLICATE, PW_CAPACITY, PW_TIME_WINDOW,
    PW_SYNC, PW_VEHICLE_RESTRICTION,
)
from src.data.cost import (
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    TRUCK_FIXED_DAY, BIKE_FIXED_DAY,
    BIKE_COST_PER_M, BIKE_DRIVER_SEC, BIKE_DEPLOY_COST,
    BIKE_SPEED_US_PER_M,
    PENALTY_UNSERVED_MULTIPLIER,
    PENALTY_OVERLOAD_PER_PCT,
    PENALTY_OVERTIME,
    DAY_LENGTH,
)


@njit(cache=True)
def calc_unserved_penalty(customer_idx, customers, dist_matrix, max_dist_m, vehicles):
    """Distance-weighted penalty per unserved customer. All i64."""
    d = dist_matrix[0, customer_idx + 1]
    distance_m = np.int64(2) * d
    travel_s = distance_m * np.int64(BIKE_SPEED_US_PER_M) // np.int64(1_000_000)
    service_s = customers[customer_idx, 5]  # COL_SERVICE
    total_s = travel_s + service_s

    base_cost = (distance_m * np.int64(BIKE_COST_PER_M)
                 + total_s * np.int64(BIKE_DRIVER_SEC)
                 + np.int64(BIKE_DEPLOY_COST))

    # distance_multiplier: ratio^2 * 9 + 1, using integer approx
    # ratio = d / max_dist (0..1), ratio^2 * 9 + 1 in [1..10]
    # Use fixed-point: ratio_1000 = d * 1000 / max_dist
    safe_max = max(max_dist_m, np.int64(1))
    ratio_1000 = d * np.int64(1000) // safe_max
    # multiplier_1000 = ratio^2 * 9000 / 1000 + 1000
    multiplier_1000 = ratio_1000 * ratio_1000 * np.int64(9) // np.int64(1000) + np.int64(1000)

    penalty = np.int64(PENALTY_UNSERVED_MULTIPLIER) * base_cost * multiplier_1000 // np.int64(1000)
    return penalty


@njit(cache=True)
def compute_penalties(unserved, n_unserved, n_duplicates,
                      cap_viol, n_cap, tw_viol, n_tw,
                      sync_viol, n_sync, n_vr,
                      return_times, n_return_times,
                      penalty_w, customers, dist_matrix, vehicles):
    """Total penalty (i64). Returns (total, parts i64[7])."""
    parts = np.zeros(7, dtype=np.int64)

    # Unserved
    if n_unserved > 0:
        max_dist_m = np.int64(1)
        for j in range(1, dist_matrix.shape[1]):
            if dist_matrix[0, j] > max_dist_m:
                max_dist_m = dist_matrix[0, j]
        for i in range(n_unserved):
            parts[0] += calc_unserved_penalty(unserved[i], customers, dist_matrix, max_dist_m, vehicles)

    # Duplicates
    parts[1] = np.int64(n_duplicates) * penalty_w[PW_DUPLICATE]

    # Capacity
    for i in range(n_cap):
        vtype = cap_viol[i, 0]
        load = cap_viol[i, 3]
        if vtype == 1:
            cap = np.int64(BIKE_CAPACITY_G)
            fixed = np.int64(BIKE_FIXED_DAY)
        else:
            cap = np.int64(TRUCK_CAPACITY_G)
            fixed = np.int64(TRUCK_FIXED_DAY)
        if load > cap and cap > 0:
            overload_pct_x100 = (load - cap) * np.int64(10000) // cap
            parts[2] += overload_pct_x100 * fixed * np.int64(PENALTY_OVERLOAD_PER_PCT) // np.int64(10000)

    # Time windows
    for i in range(n_tw):
        arrive = tw_viol[i, 3]
        tw_close = tw_viol[i, 4]
        late_s = max(np.int64(0), arrive - tw_close)
        parts[3] += late_s * penalty_w[PW_TIME_WINDOW]

    # Sync
    parts[4] = np.int64(n_sync) * penalty_w[PW_SYNC]

    # Vehicle restriction
    parts[5] = np.int64(n_vr) * penalty_w[PW_VEHICLE_RESTRICTION]

    # Overtime
    day = np.int64(DAY_LENGTH)
    for i in range(n_return_times):
        overtime_s = max(np.int64(0), return_times[i] - day)
        if overtime_s > 0:
            parts[6] += overtime_s * np.int64(PENALTY_OVERTIME)

    total = np.int64(0)
    for i in range(7):
        total += parts[i]
    return total, parts
