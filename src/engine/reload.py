"""Truck/bike reload logic at satellite stops. All i64. @njit."""
import numpy as np
from numba import njit

from src.data.constants import SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME


@njit(cache=True)
def truck_reload_at_stop(customer_idx, truck_id, current_load_g, truck_arrive_s,
                         satellites, n_satellites, reload_service_s):
    """Truck serves N bikes. Returns (load_change_g, total_service_s, sync_wait_s). All i64."""
    n_match = 0
    for i in range(n_satellites):
        if satellites[i, SAT_CUST] == customer_idx and satellites[i, SAT_TRUCK] == truck_id:
            n_match += 1
    if n_match == 0:
        return np.int64(0), np.int64(0), np.int64(0)

    times = np.empty(n_match, dtype=np.int64)
    grams = np.empty(n_match, dtype=np.int64)
    k = 0
    for i in range(n_satellites):
        if satellites[i, SAT_CUST] == customer_idx and satellites[i, SAT_TRUCK] == truck_id:
            times[k] = satellites[i, SAT_TIME]
            grams[k] = satellites[i, SAT_KG]
            k += 1

    order = np.argsort(times)
    load_change = np.int64(0)
    for i in range(n_match):
        load_change -= grams[i]

    first_bike_time = times[order[0]]
    sync_wait = max(np.int64(0), first_bike_time - truck_arrive_s)
    cursor = truck_arrive_s + sync_wait

    for j in range(n_match):
        idx = order[j]
        transfer_start = max(cursor, times[idx])
        cursor = transfer_start + reload_service_s

    total_service = cursor - (truck_arrive_s + sync_wait)
    return load_change, total_service, sync_wait


@njit(cache=True)
def bike_reload_at_stop(customer_idx, bike_id, bike_arrive_s,
                        satellites, n_satellites, reload_service_s):
    """Bike receives from truck. Returns (load_change_g, service_s, sync_wait_s). All i64."""
    for i in range(n_satellites):
        if satellites[i, SAT_CUST] == customer_idx and satellites[i, SAT_BIKE] == bike_id:
            transfer_g = satellites[i, SAT_KG]
            planned_time = satellites[i, SAT_TIME]
            sync_wait = max(np.int64(0), planned_time - bike_arrive_s)
            return transfer_g, reload_service_s, sync_wait
    return np.int64(0), np.int64(0), np.int64(0)
