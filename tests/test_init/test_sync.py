"""Tests for src/init/sync.py — Data Model v2."""
import numpy as np
import pytest

from src.init.sync import synchronize_times
from src.data.constants import (
    ACT_DELIVER, ACT_PICKUP,
    ORD_LOC, ORD_QTY, ORD_UNIT_W, ORD_COLS,
    LOC_X, LOC_Y, LOC_UNLOAD_RATE, LOC_BASE_SVC, LOC_COLS,
    LTYPE_DEPOT, LTYPE_CUSTOMER,
    VEH_TYPE, VEH_CAP_KG, VEH_COLS, VTYPE_TRUCK, VTYPE_BIKE,
    TR_COLS,
)
from src.solution.structure import create_solution


def _make_locations(coords, depot_xy=(0.0, 0.0)):
    n = 1 + len(coords)
    locs = np.zeros((n, LOC_COLS), dtype=np.float64)
    locs[0] = [depot_xy[0], depot_xy[1], LTYPE_DEPOT, 99999.0, 20.0, 10.0]
    for i, (x, y) in enumerate(coords):
        locs[i + 1] = [x, y, LTYPE_CUSTOMER, 99999.0, 20.0, 5.0]
    return locs


def _build_dm(locations, speed=25.0):
    coords = locations[:, :2]
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dk = np.sqrt((diff ** 2).sum(axis=2))
    tm = dk / speed * 60.0
    return np.stack([dk, tm], axis=0)


# ---------------------------------------------------------------------------
# synchronize_times — integration
# ---------------------------------------------------------------------------

def test_synchronize_times_basic(tiny_data):
    """synchronize_times returns valid transfers array."""
    sol = create_solution(tiny_data["n_vehicles"], tiny_data["n_customers"],
                          n_sku=tiny_data["n_sku"])
    orders = tiny_data["orders"]
    vehicles = tiny_data["vehicles"]
    dm = tiny_data["dist_matrix"]
    loc_to_cust = tiny_data["loc_to_cust"]
    max_wt = tiny_data["max_working_time"]

    # Build a minimal route: truck delivers C0, pickup at C1 location
    sol["stops"][0, 0] = int(orders[0, ORD_LOC])
    sol["actions"][0, 0] = ACT_DELIVER
    sol["stops"][0, 1] = int(orders[1, ORD_LOC])
    sol["actions"][0, 1] = ACT_PICKUP
    sol["lengths"][0] = 2

    # Bike delivers C2, pickup at C1 location
    sol["stops"][1, 0] = int(orders[2, ORD_LOC])
    sol["actions"][1, 0] = ACT_DELIVER
    sol["stops"][1, 1] = int(orders[1, ORD_LOC])
    sol["actions"][1, 1] = ACT_PICKUP
    sol["lengths"][1] = 2

    locations = tiny_data["locations"]
    result = synchronize_times(sol, orders, locations, vehicles, dm, loc_to_cust, max_wt)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == TR_COLS
