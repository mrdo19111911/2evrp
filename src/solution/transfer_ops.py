"""Transfer event management (hub relay between truck and bike) — Data Model v2."""
import numpy as np

from src.data.constants import TR_HUB, TR_BIKE, TR_TRUCK, TR_KG, TR_CBM, TR_TIME


def add_transfer_event(sol, hub_loc, bike_id, truck_id, kg, cbm, time):
    """Append a transfer event to the transfers array."""
    new_row = np.array([[hub_loc, bike_id, truck_id, kg, cbm, time]],
                       dtype=np.float64)
    sol["transfers"] = np.vstack([sol["transfers"], new_row])


def remove_transfer_event(sol, idx=-1, hub_loc=-1, bike_id=-1, truck_id=-1):
    """Remove event by index or by match. Returns removed row or None."""
    if idx >= 0:
        removed = sol["transfers"][idx].copy()
        sol["transfers"] = np.delete(sol["transfers"], idx, axis=0)
        return removed

    tr = sol["transfers"]
    mask = np.ones(len(tr), dtype=bool)
    if hub_loc >= 0:
        mask &= tr[:, TR_HUB] == hub_loc
    if bike_id >= 0:
        mask &= tr[:, TR_BIKE] == bike_id
    if truck_id >= 0:
        mask &= tr[:, TR_TRUCK] == truck_id

    hits = np.where(mask)[0]
    if len(hits) > 0:
        removed = tr[hits[0]].copy()
        sol["transfers"] = np.delete(tr, hits[0], axis=0)
        return removed
    return None


def remove_transfers_for_hub(sol, hub_loc):
    """Remove all events at a hub. Returns count removed."""
    mask = sol["transfers"][:, TR_HUB] != hub_loc
    removed_count = int(np.sum(~mask))
    sol["transfers"] = sol["transfers"][mask]
    return removed_count


def remove_transfers_for_bike(sol, bike_id):
    """Remove all events for a bike. Returns count removed."""
    mask = sol["transfers"][:, TR_BIKE] != bike_id
    removed_count = int(np.sum(~mask))
    sol["transfers"] = sol["transfers"][mask]
    return removed_count


def update_transfer_time(sol, idx, new_time):
    """Update planned time for a transfer event."""
    sol["transfers"][idx, TR_TIME] = new_time
