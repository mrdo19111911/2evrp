"""Tests for src/init/sync.py -- time synchronization.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.sync import synchronize_times
from src.data.constants import ACT_DELIVER, ACT_RELOAD, COL_DEMAND
from src.data.cost import (
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)


def _simple_dist_matrix(coords_with_depot):
    coords_f = coords_with_depot.astype(np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


# ---------------------------------------------------------------------------
# synchronize_times -- integration
# ---------------------------------------------------------------------------

def test_synchronize_times_basic(tiny_instance):
    """synchronize_times returns valid satellites ndarray."""
    data = tiny_instance
    customers = data["customers"]
    vehicles = data["vehicles"]
    depot = data["depot"]
    coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
    dm = _simple_dist_matrix(coords)

    # Build minimal truck route: C0 (deliver), C1 (reload)
    truck_sol = [
        {"stops": np.array([0, 1], dtype=np.int32),
         "actions": np.array([ACT_DELIVER, ACT_RELOAD], dtype=np.int8),
         "total_demand": 100000, "total_distance": 10000,
         "truck_id": 0},
    ]
    # Bike route: C2 (deliver), C1 (reload)
    bike_sol = [
        {"stops": np.array([2, 1], dtype=np.int32),
         "actions": np.array([ACT_DELIVER, ACT_RELOAD], dtype=np.int8),
         "total_demand": 10000, "total_distance": 5000,
         "bike_id": 0},
    ]

    result = synchronize_times(truck_sol, bike_sol, customers, dm, vehicles)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
