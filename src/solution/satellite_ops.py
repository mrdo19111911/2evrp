"""Satellite (reload event) management."""
import numpy as np

from src.data.constants import SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_TIME


def add_satellite_event(sol, customer_idx, bike_id, truck_id, transfer_kg, planned_time):
    """Add reload event to satellites array."""
    new_row = np.array([[customer_idx, bike_id, truck_id, transfer_kg, planned_time]],
                       dtype=np.float64)
    sol["satellites"] = np.vstack([sol["satellites"], new_row])


def remove_satellite_event(sol, sat_idx=-1, customer_idx=-1, bike_id=-1, truck_id=-1):
    """Remove event by index or by match. Returns removed row."""
    if sat_idx >= 0:
        removed = sol["satellites"][sat_idx].copy()
        sol["satellites"] = np.delete(sol["satellites"], sat_idx, axis=0)
        return removed

    sats = sol["satellites"]
    mask = ((sats[:, SAT_CUST] == customer_idx) &
            (sats[:, SAT_BIKE] == bike_id) &
            (sats[:, SAT_TRUCK] == truck_id))
    idx = np.where(mask)[0]
    if len(idx) > 0:
        removed = sats[idx[0]].copy()
        sol["satellites"] = np.delete(sats, idx[0], axis=0)
        return removed
    return None


def remove_satellites_for_customer(sol, customer_idx):
    """Remove all events at a customer node. Returns count."""
    mask = sol["satellites"][:, SAT_CUST] != customer_idx
    removed_count = int(np.sum(~mask))
    sol["satellites"] = sol["satellites"][mask]
    return removed_count


def remove_satellites_for_bike(sol, bike_id):
    """Remove all events for a bike. Returns count."""
    mask = sol["satellites"][:, SAT_BIKE] != bike_id
    removed_count = int(np.sum(~mask))
    sol["satellites"] = sol["satellites"][mask]
    return removed_count


def update_satellite_time(sol, sat_idx, new_time):
    """Update planned_time for a satellite event."""
    sol["satellites"][sat_idx, SAT_TIME] = new_time
