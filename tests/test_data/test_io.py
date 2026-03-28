"""Tests for src/data/io.py -- load_instance, save_solution, load_solution.

API (i64):
  load_instance returns (customers_i64, depot_i64, vehicles_i64)
  customers: i64 (N, 7) [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]
  depot: i64 (2,) in meters
  vehicles: i64 (K, 4) [type, capacity_g, cost_per_m_vnd, speed_us_per_m]
"""
import json
import os

import numpy as np
import pytest

from src.data.io import load_instance, save_solution, load_solution
from src.data.constants import VCOL_TYPE, VEH_TRUCK, VEH_BIKE, CUST_COLS


# ---------------------------------------------------------------------------
# TC1: load grid_20x20.json -> shapes correct
# ---------------------------------------------------------------------------
GRID_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "grid_20x20.json")


@pytest.mark.skipif(not os.path.exists(GRID_PATH),
                    reason="grid_20x20.json not found")
def test_load_grid_20x20_shapes():
    """grid_20x20.json has 400 customers."""
    customers, depot, vehicles = load_instance(GRID_PATH)
    assert customers.shape[0] == 400
    assert customers.shape[1] == CUST_COLS  # 7 columns
    assert depot.shape == (2,)
    assert vehicles.ndim == 2


@pytest.mark.skipif(not os.path.exists(GRID_PATH),
                    reason="grid_20x20.json not found")
def test_load_grid_20x20_depot_value():
    """Depot should have 2 coordinates in meters (i64)."""
    customers, depot, vehicles = load_instance(GRID_PATH)
    assert depot.shape == (2,)
    assert depot.dtype == np.int64
    assert np.all(np.isfinite(depot))


@pytest.mark.skipif(not os.path.exists(GRID_PATH),
                    reason="grid_20x20.json not found")
def test_load_grid_20x20_dtypes():
    customers, depot, vehicles = load_instance(GRID_PATH)
    assert customers.dtype == np.int64
    assert vehicles.dtype == np.int64
    assert depot.dtype == np.int64


@pytest.mark.skipif(not os.path.exists(GRID_PATH),
                    reason="grid_20x20.json not found")
def test_load_grid_20x20_vehicles_split():
    """5 trucks (type 0) + 15 bikes (type 1) = 20 vehicles."""
    customers, depot, vehicles = load_instance(GRID_PATH)
    n_trucks = int((vehicles[:, VCOL_TYPE] == VEH_TRUCK).sum())
    n_bikes = int((vehicles[:, VCOL_TYPE] == VEH_BIKE).sum())
    assert n_trucks == 5
    assert n_bikes == 15


# ---------------------------------------------------------------------------
# TC2: save_solution -> load_solution round-trip
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    """Save a small solution, reload it, verify values match."""
    filepath = str(tmp_path / "sol_test.json")

    truck_stops = np.array([[0, 3, 1, -1], [0, 2, 4, 0]], dtype=np.int32)
    truck_actions = np.array([[0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.int32)
    bike_stops = np.array([[1, 2, -1]], dtype=np.int32)
    bike_actions = np.array([[0, 0, -1]], dtype=np.int32)
    satellites = np.array([[1, 0, 0, 20000, 1800]], dtype=np.int64)

    save_solution(filepath, truck_stops, truck_actions,
                  bike_stops, bike_actions, satellites)
    ts, ta, bs, ba, sats = load_solution(filepath)

    np.testing.assert_array_equal(ts, truck_stops)
    np.testing.assert_array_equal(ta, truck_actions)
    np.testing.assert_array_equal(bs, bike_stops)
    np.testing.assert_array_equal(ba, bike_actions)
    np.testing.assert_array_equal(sats, satellites)


def test_save_load_empty_routes(tmp_path):
    """Round-trip with empty routes."""
    filepath = str(tmp_path / "sol_empty.json")

    truck_stops = np.zeros((1, 3), dtype=np.int32)
    truck_actions = np.full((1, 3), -1, dtype=np.int32)
    bike_stops = np.zeros((1, 3), dtype=np.int32)
    bike_actions = np.full((1, 3), -1, dtype=np.int32)
    satellites = np.empty((0, 5), dtype=np.int64)

    save_solution(filepath, truck_stops, truck_actions,
                  bike_stops, bike_actions, satellites)
    ts, ta, bs, ba, sats = load_solution(filepath)

    assert ts.shape == (1, 3)
    assert sats.shape == (0, 5)


# ---------------------------------------------------------------------------
# TC3: load nonexistent file -> FileNotFoundError
# ---------------------------------------------------------------------------
def test_load_instance_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_instance("/tmp/nonexistent_abc123.json")


def test_load_solution_nonexistent():
    with pytest.raises(FileNotFoundError):
        load_solution("/tmp/nonexistent_abc123.json")


# ---------------------------------------------------------------------------
# TC4: invalid data -> ValueError (demand = 0 or negative)
# ---------------------------------------------------------------------------
def test_load_instance_zero_demand(tmp_path):
    """An instance with demand=0 for a customer should raise ValueError."""
    bad = {
        "depot": [0.0, 0.0],
        "customers": [[1.0, 1.0, 0.0, 0.0, 120.0, 5.0]],
        "restricted": [0],
        "vehicles": [[0.0, 2000.0, 12, 25.0]],
    }
    filepath = str(tmp_path / "bad_instance.json")
    with open(filepath, "w") as f:
        json.dump(bad, f)

    with pytest.raises(ValueError):
        load_instance(filepath)


def test_load_instance_negative_demand(tmp_path):
    """An instance with negative demand should raise ValueError."""
    bad = {
        "depot": [0.0, 0.0],
        "customers": [[1.0, 1.0, -5.0, 0.0, 120.0, 5.0]],
        "restricted": [0],
        "vehicles": [[0.0, 2000.0, 12, 25.0]],
    }
    filepath = str(tmp_path / "bad_neg.json")
    with open(filepath, "w") as f:
        json.dump(bad, f)

    with pytest.raises(ValueError):
        load_instance(filepath)
