"""Tests for src/data/generator.py -- instance generation.

API (i64):
  generate_random_instance returns (customers_i64, depot_i64, vehicles_i64)
  customers: i64 (N, 7) [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]
  depot: i64 (2,) in meters
  vehicles: i64 (K, 4) [type, capacity_g, cost_per_m_vnd, speed_us_per_m]
"""
import numpy as np
import pytest

from src.data.generator import generate_random_instance, generate_clustered_instance
from src.data.constants import (
    COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
    COL_RESTRICTED, CUST_COLS,
    VCOL_TYPE, VEH_TRUCK, VEH_BIKE,
)


# ===================================================================
# generate_random_instance
# ===================================================================

def test_random_instance_shapes():
    customers, depot, vehicles = generate_random_instance(100, 3, 5, seed=42)
    assert customers.shape == (100, CUST_COLS)  # 7 columns
    assert depot.shape == (2,)
    assert vehicles.shape == (8, 4)  # 3 trucks + 5 bikes


def test_random_instance_dtypes():
    customers, depot, vehicles = generate_random_instance(10, 1, 1, seed=42)
    assert customers.dtype == np.int64
    assert depot.dtype == np.int64
    assert vehicles.dtype == np.int64


def test_random_instance_vehicle_types():
    customers, depot, vehicles = generate_random_instance(100, 3, 5, seed=42)
    n_trucks = int((vehicles[:, VCOL_TYPE] == VEH_TRUCK).sum())
    n_bikes = int((vehicles[:, VCOL_TYPE] == VEH_BIKE).sum())
    assert n_trucks == 3
    assert n_bikes == 5


def test_random_instance_demand_positive():
    customers, _, _ = generate_random_instance(200, 2, 4, seed=99)
    assert customers[:, COL_DEMAND].min() > 0


def test_random_instance_deterministic():
    c1, d1, v1 = generate_random_instance(50, 2, 3, seed=42)
    c2, d2, v2 = generate_random_instance(50, 2, 3, seed=42)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(v1, v2)


def test_random_instance_different_seed():
    c1, _, _ = generate_random_instance(50, 2, 3, seed=42)
    c2, _, _ = generate_random_instance(50, 2, 3, seed=99)
    assert not np.array_equal(c1, c2)


def test_random_instance_coords_in_area():
    area_m = 50000  # meters
    customers, _, _ = generate_random_instance(
        100, 2, 3, area_size_m=area_m, seed=42)
    assert np.all(customers[:, COL_X] >= 0)
    assert np.all(customers[:, COL_X] <= area_m)
    assert np.all(customers[:, COL_Y] >= 0)
    assert np.all(customers[:, COL_Y] <= area_m)


def test_random_instance_restricted_column():
    customers, _, _ = generate_random_instance(100, 2, 3, seed=42)
    restricted = customers[:, COL_RESTRICTED]
    unique = np.unique(restricted)
    assert set(unique.tolist()) <= {0, 1}


# ===================================================================
# generate_clustered_instance
# ===================================================================

def test_clustered_instance_shapes():
    customers, depot, vehicles = generate_clustered_instance(
        100, 3, 5, n_clusters=3, seed=42)
    assert customers.shape == (100, CUST_COLS)
    assert vehicles.shape == (8, 4)


def test_clustered_instance_vehicle_types():
    _, _, vehicles = generate_clustered_instance(100, 3, 5, n_clusters=3, seed=42)
    n_trucks = int((vehicles[:, VCOL_TYPE] == VEH_TRUCK).sum())
    n_bikes = int((vehicles[:, VCOL_TYPE] == VEH_BIKE).sum())
    assert n_trucks == 3
    assert n_bikes == 5


def test_clustered_instance_demand_positive():
    customers, _, _ = generate_clustered_instance(100, 2, 4, n_clusters=4, seed=42)
    assert np.all(customers[:, COL_DEMAND] > 0)


def test_clustered_instance_deterministic():
    c1, d1, v1 = generate_clustered_instance(80, 2, 4, n_clusters=3, seed=42)
    c2, d2, v2 = generate_clustered_instance(80, 2, 4, n_clusters=3, seed=42)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(d1, d2)


def test_clustered_instance_coords_in_area():
    area_m = 50000
    customers, _, _ = generate_clustered_instance(
        100, 2, 3, n_clusters=5, area_size_m=area_m, seed=42)
    assert np.all(customers[:, COL_X] >= 0)
    assert np.all(customers[:, COL_X] <= area_m)


def test_clustered_instance_dtypes():
    customers, depot, vehicles = generate_clustered_instance(
        50, 1, 2, n_clusters=2, seed=42)
    assert customers.dtype == np.int64
    assert vehicles.dtype == np.int64
    assert depot.dtype == np.int64
