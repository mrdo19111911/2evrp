"""Tests for src/data/io.py — load_instance, save_solution, load_solution (v2 API).

v2: load_instance returns a dict with keys: locations, orders, vehicles,
    dist_matrix, demand_per_sku, loc_to_cust, etc.
    save_solution/load_solution use a simple dict-based format.
"""
import json
import os

import numpy as np
import pytest

from src.data.io import load_instance, save_solution, load_solution
from src.data.constants import (
    VEH_TYPE, VTYPE_TRUCK, VTYPE_BIKE, ORD_UNIT_W,
)


# ---------------------------------------------------------------------------
# TC1: load grid_20x20.json -> shapes correct (v2 dict)
# ---------------------------------------------------------------------------
GRID_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "grid_20x20.json")


def test_load_grid_20x20_shapes():
    """grid_20x20.json has 400 customers, 20 vehicles."""
    data = load_instance(GRID_PATH)

    # locations: depot + 400 customers = 401 rows, 4 cols
    assert data["locations"].shape == (401, 4), f"locations shape {data['locations'].shape}"
    # orders: 400 customers, 7 cols
    assert data["orders"].shape == (400, 7), f"orders shape {data['orders'].shape}"
    # vehicles: 20 rows, 6 cols
    assert data["vehicles"].shape == (20, 6), f"vehicles shape {data['vehicles'].shape}"
    # dist_matrix: (2, 401, 401)
    assert data["dist_matrix"].shape == (2, 401, 401), f"dist_matrix shape {data['dist_matrix'].shape}"
    # legacy depot key
    assert data["depot"].shape == (2,), f"depot shape {data['depot'].shape}"


def test_load_grid_20x20_depot_value():
    """Depot should be at (50.0, 50.0)."""
    data = load_instance(GRID_PATH)
    np.testing.assert_array_equal(data["depot"], [50.0, 50.0])


def test_load_grid_20x20_dtypes():
    """Verify numpy dtypes are floating / integer as expected."""
    data = load_instance(GRID_PATH)
    assert np.issubdtype(data["locations"].dtype, np.floating)
    assert np.issubdtype(data["orders"].dtype, np.floating)
    assert np.issubdtype(data["vehicles"].dtype, np.floating)


def test_load_grid_20x20_vehicles_split():
    """5 trucks (type 0) + 15 bikes (type 1) = 20 vehicles."""
    data = load_instance(GRID_PATH)
    vehicles = data["vehicles"]
    n_trucks = int((vehicles[:, VEH_TYPE] == VTYPE_TRUCK).sum())
    n_bikes = int((vehicles[:, VEH_TYPE] == VTYPE_BIKE).sum())
    assert n_trucks == 5
    assert n_bikes == 15


def test_load_grid_20x20_restricted_count():
    """70 restricted customers in grid_20x20.json."""
    data = load_instance(GRID_PATH)
    assert int(data["restricted"].sum()) == 70


def test_load_grid_20x20_customer_values():
    """Spot-check first customer via legacy 'customers' key."""
    data = load_instance(GRID_PATH)
    c0 = data["customers"][0]
    np.testing.assert_array_almost_equal(
        c0, [0.0, 0.0, 400.0, 0.0, 120.0, 15.0]
    )


# ---------------------------------------------------------------------------
# TC2: save_solution -> load_solution round-trip (v2 dict format)
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    """Save a small solution dict, reload it, verify values match."""
    filepath = str(tmp_path / "sol_test.json")

    sol = {
        "stops": np.array([[0, 3, 1, -1], [0, 2, 4, 0]], dtype=np.int32),
        "actions": np.array([[0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.int32),
        "lengths": np.array([3, 4], dtype=np.int32),
    }
    save_solution(filepath, sol)
    loaded = load_solution(filepath)

    assert loaded["stops"] == sol["stops"].tolist()
    assert loaded["actions"] == sol["actions"].tolist()
    assert loaded["lengths"] == sol["lengths"].tolist()


def test_save_load_empty_routes(tmp_path):
    """Round-trip with empty routes."""
    filepath = str(tmp_path / "sol_empty.json")

    sol = {
        "stops": np.zeros((1, 3), dtype=np.int32),
        "actions": np.full((1, 3), -1, dtype=np.int8),
        "lengths": np.array([0], dtype=np.int32),
    }
    save_solution(filepath, sol)
    loaded = load_solution(filepath)

    assert len(loaded["stops"]) == 1
    assert len(loaded["stops"][0]) == 3


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
        "metadata": {"name": "bad", "n_customers": 1, "n_trucks": 1, "n_bikes": 0},
        "depot": [0.0, 0.0],
        "customers": [[1.0, 1.0, 0.0, 0.0, 120.0, 5.0]],
        "restricted": [0],
        "vehicles": [[0.0, 2000.0, 4.52, 25.0]],
    }
    filepath = str(tmp_path / "bad_instance.json")
    with open(filepath, "w") as f:
        json.dump(bad, f)

    with pytest.raises(ValueError):
        load_instance(filepath)


def test_load_instance_negative_demand(tmp_path):
    """An instance with negative demand should raise ValueError."""
    bad = {
        "metadata": {"name": "bad", "n_customers": 1, "n_trucks": 1, "n_bikes": 0},
        "depot": [0.0, 0.0],
        "customers": [[1.0, 1.0, -5.0, 0.0, 120.0, 5.0]],
        "restricted": [0],
        "vehicles": [[0.0, 2000.0, 4.52, 25.0]],
    }
    filepath = str(tmp_path / "bad_neg.json")
    with open(filepath, "w") as f:
        json.dump(bad, f)

    with pytest.raises(ValueError):
        load_instance(filepath)
