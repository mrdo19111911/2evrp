"""Satellite (reload event) management — pre-allocated i64 array. @njit."""
import numpy as np
from numba import njit

from src.data.constants import (
    SAT_CUST, SAT_BIKE, SAT_TRUCK, SAT_KG, SAT_TIME,
    SOL_SATELLITES, SOL_META, META_N_SATELLITES,
)


@njit(cache=True)
def add_satellite_event(sol, customer_idx, bike_id, truck_id, transfer_g, planned_time_s):
    """Add reload event into pre-allocated i64 array."""
    sats = sol[SOL_SATELLITES]
    meta = sol[SOL_META]
    n = meta[META_N_SATELLITES]
    sats[n, 0] = customer_idx
    sats[n, 1] = bike_id
    sats[n, 2] = truck_id
    sats[n, 3] = transfer_g
    sats[n, 4] = planned_time_s
    meta[META_N_SATELLITES] = n + 1


@njit(cache=True)
def remove_satellite_by_idx(sol, sat_idx):
    """Remove event by index. Swap-with-last."""
    sats = sol[SOL_SATELLITES]
    meta = sol[SOL_META]
    n = meta[META_N_SATELLITES]
    if sat_idx < n - 1:
        for k in range(5):
            sats[sat_idx, k] = sats[n - 1, k]
    for k in range(5):
        sats[n - 1, k] = 0
    meta[META_N_SATELLITES] = n - 1


def remove_satellite_event(sol, sat_idx=None, customer_idx=None, bike_id=None, truck_id=None):
    """Remove event by index or by match. Python wrapper for optional args."""
    if sat_idx is not None:
        removed = sol[SOL_SATELLITES][sat_idx].copy()
        remove_satellite_by_idx(sol, sat_idx)
        return removed

    sats = sol[SOL_SATELLITES]
    n = sol[SOL_META][META_N_SATELLITES]
    for i in range(n):
        match = True
        if customer_idx is not None and sats[i, SAT_CUST] != customer_idx:
            match = False
        if bike_id is not None and sats[i, SAT_BIKE] != bike_id:
            match = False
        if truck_id is not None and sats[i, SAT_TRUCK] != truck_id:
            match = False
        if match:
            removed = sats[i].copy()
            remove_satellite_by_idx(sol, i)
            return removed
    return None


@njit(cache=True)
def remove_satellites_for_customer(sol, customer_idx):
    """Remove all events at a customer node. Returns count removed."""
    sats = sol[SOL_SATELLITES]
    meta = sol[SOL_META]
    n = meta[META_N_SATELLITES]
    removed_count = 0
    i = 0
    while i < n:
        if sats[i, SAT_CUST] == customer_idx:
            n -= 1
            if i < n:
                for k in range(5):
                    sats[i, k] = sats[n, k]
            for k in range(5):
                sats[n, k] = 0
            removed_count += 1
        else:
            i += 1
    meta[META_N_SATELLITES] = n
    return removed_count


@njit(cache=True)
def remove_satellites_for_bike(sol, bike_id):
    """Remove all events for a bike. Returns count removed."""
    sats = sol[SOL_SATELLITES]
    meta = sol[SOL_META]
    n = meta[META_N_SATELLITES]
    removed_count = 0
    i = 0
    while i < n:
        if sats[i, SAT_BIKE] == bike_id:
            n -= 1
            if i < n:
                for k in range(5):
                    sats[i, k] = sats[n, k]
            for k in range(5):
                sats[n, k] = 0
            removed_count += 1
        else:
            i += 1
    meta[META_N_SATELLITES] = n
    return removed_count


@njit(cache=True)
def update_satellite_time(sol, sat_idx, new_time_s):
    sol[SOL_SATELLITES][sat_idx, SAT_TIME] = new_time_s


@njit(cache=True)
def get_satellites(sol):
    """Return view of active satellite rows (i64)."""
    n = sol[SOL_META][META_N_SATELLITES]
    return sol[SOL_SATELLITES][:n]
