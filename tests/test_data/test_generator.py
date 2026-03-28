"""Tests for src/data/generator.py — Data Model v2.

Generator now returns a data dict with keys:
  locations, orders, vehicles, dist_matrix, allowed_bike, loc_to_cust, etc.
"""
import numpy as np
import pytest

from src.data.generator import generate_random_instance, generate_clustered_instance
from src.data.constants import (
    LOC_X, LOC_Y, LOC_COLS,
    ORD_LOC, ORD_QTY, ORD_UNIT_W, ORD_COLS,
    VEH_TYPE, VEH_COLS,
    VTYPE_TRUCK, VTYPE_BIKE,
)


# ===================================================================
# generate_random_instance
# ===================================================================

def test_random_instance_shapes():
    data = generate_random_instance(100, 3, 5, seed=42)
    assert data["orders"].shape[0] == 100
    assert data["orders"].shape[1] == ORD_COLS
    assert data["locations"].shape[1] == LOC_COLS
    assert data["vehicles"].shape == (8, VEH_COLS)  # 3 trucks + 5 bikes


def test_random_instance_vehicle_types():
    data = generate_random_instance(100, 3, 5, seed=42)
    veh = data["vehicles"]
    n_trucks = int((veh[:, VEH_TYPE] == VTYPE_TRUCK).sum())
    n_bikes = int((veh[:, VEH_TYPE] == VTYPE_BIKE).sum())
    assert n_trucks == 3
    assert n_bikes == 5


def test_random_instance_demand_positive():
    data = generate_random_instance(200, 2, 4, seed=99)
    orders = data["orders"]
    demands = orders[:, ORD_QTY] * orders[:, ORD_UNIT_W]
    assert demands.min() > 0.0


def test_random_instance_deterministic():
    d1 = generate_random_instance(50, 2, 3, seed=42)
    d2 = generate_random_instance(50, 2, 3, seed=42)
    np.testing.assert_array_equal(d1["orders"], d2["orders"])
    np.testing.assert_array_equal(d1["locations"], d2["locations"])
    np.testing.assert_array_equal(d1["vehicles"], d2["vehicles"])


def test_random_instance_different_seed():
    d1 = generate_random_instance(50, 2, 3, seed=42)
    d2 = generate_random_instance(50, 2, 3, seed=99)
    assert not np.array_equal(d1["orders"], d2["orders"])


def test_random_instance_coords_in_area():
    area = 50.0
    data = generate_random_instance(100, 2, 3, area_size=area, seed=42)
    locs = data["locations"]
    assert np.all(locs[:, LOC_X] >= 0.0)
    assert np.all(locs[:, LOC_X] <= area)
    assert np.all(locs[:, LOC_Y] >= 0.0)
    assert np.all(locs[:, LOC_Y] <= area)


def test_random_instance_dist_matrix():
    data = generate_random_instance(10, 1, 2, seed=42)
    dm = data["dist_matrix"]
    assert dm.ndim == 3
    assert dm.shape[0] == 2  # (DM_DIST, DM_TIME)
    n_locs = data["locations"].shape[0]
    assert dm.shape[1] == n_locs
    assert dm.shape[2] == n_locs


def test_random_instance_loc_to_cust():
    data = generate_random_instance(10, 1, 2, seed=42)
    ltc = data["loc_to_cust"]
    # Each customer maps to exactly one location
    custs_found = set(ltc[ltc >= 0].tolist())
    assert len(custs_found) == 10


# ===================================================================
# generate_clustered_instance
# ===================================================================

def test_clustered_instance_shapes():
    data = generate_clustered_instance(100, 3, 5, n_clusters=3, seed=42)
    assert data["orders"].shape[0] == 100
    assert data["vehicles"].shape == (8, VEH_COLS)


def test_clustered_instance_vehicle_types():
    data = generate_clustered_instance(100, 3, 5, n_clusters=3, seed=42)
    veh = data["vehicles"]
    n_trucks = int((veh[:, VEH_TYPE] == VTYPE_TRUCK).sum())
    n_bikes = int((veh[:, VEH_TYPE] == VTYPE_BIKE).sum())
    assert n_trucks == 3
    assert n_bikes == 5


def test_clustered_instance_demand_positive():
    data = generate_clustered_instance(100, 2, 4, n_clusters=4, seed=42)
    orders = data["orders"]
    demands = orders[:, ORD_QTY] * orders[:, ORD_UNIT_W]
    assert np.all(demands > 0.0)


def test_clustered_instance_deterministic():
    d1 = generate_clustered_instance(80, 2, 4, n_clusters=3, seed=42)
    d2 = generate_clustered_instance(80, 2, 4, n_clusters=3, seed=42)
    np.testing.assert_array_equal(d1["orders"], d2["orders"])
    np.testing.assert_array_equal(d1["locations"], d2["locations"])


def test_clustered_instance_coords_in_area():
    area = 50.0
    data = generate_clustered_instance(100, 2, 3, n_clusters=5, area_size=area, seed=42)
    locs = data["locations"]
    assert np.all(locs[:, LOC_X] >= 0.0)
    assert np.all(locs[:, LOC_X] <= area)


def test_clustered_instance_dtypes():
    data = generate_clustered_instance(50, 1, 2, n_clusters=2, seed=42)
    assert np.issubdtype(data["orders"].dtype, np.floating)
    assert np.issubdtype(data["locations"].dtype, np.floating)
    assert np.issubdtype(data["vehicles"].dtype, np.floating)
