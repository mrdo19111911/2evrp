"""Truck/bike reload logic at satellite stops."""
import numpy as np

from src.data.constants import SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME


def truck_reload_at_stop(customer_idx, truck_id, current_load, truck_arrive_time,
                         satellites, reload_service_time=5.0):
    """Truck serves N bikes. Returns (load_change, total_service, sync_wait)."""
    mask = (satellites[:, SAT_CUST] == customer_idx) & (satellites[:, SAT_TRUCK] == truck_id)
    events = satellites[mask]
    if len(events) == 0:
        return 0.0, 0.0, 0.0

    events = events[np.argsort(events[:, SAT_TIME])]
    load_change = -events[:, SAT_KG].sum()

    first_bike_time = events[0, SAT_TIME]
    sync_wait = max(0.0, first_bike_time - truck_arrive_time)
    cursor = truck_arrive_time + sync_wait

    for j in range(len(events)):
        transfer_start = max(cursor, events[j, SAT_TIME])
        cursor = transfer_start + reload_service_time

    total_service = cursor - (truck_arrive_time + sync_wait)
    return load_change, total_service, sync_wait


def bike_reload_at_stop(customer_idx, bike_id, bike_arrive_time,
                        satellites, reload_service_time=5.0):
    """Bike receives from truck. Returns (load_change, service_time, sync_wait)."""
    mask = (satellites[:, SAT_CUST] == customer_idx) & (satellites[:, SAT_BIKE] == bike_id)
    events = satellites[mask]
    if len(events) == 0:
        return 0.0, 0.0, 0.0

    event = events[0]
    transfer_kg = event[SAT_KG]
    planned_time = event[SAT_TIME]
    sync_wait = max(0.0, planned_time - bike_arrive_time)
    return transfer_kg, reload_service_time, sync_wait
