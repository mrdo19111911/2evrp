"""Shared fixtures: tiny instances for all tests. ALL i64 per INTERFACE.md."""
import sys
import numpy as np
import pytest

# Increase thread stack size for Numba LLVM compilation
import threading
try:
    threading.stack_size(32 * 1024 * 1024)  # 32MB
except (ValueError, threading.ThreadError):
    pass

from src.data.constants import COL_X, COL_Y
from src.data.cost import (
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)


def _make_dist_matrix_i64(depot, customers):
    """Euclidean distance matrix (N+1, N+1), i64 meters. Row/col 0 = depot."""
    depot_xy = depot.reshape(1, 2).astype(np.float64)
    cust_xy = customers[:, [COL_X, COL_Y]].astype(np.float64)
    coords = np.vstack([depot_xy, cust_xy])
    diff = coords[:, None, :] - coords[None, :, :]
    dist_f = np.sqrt((diff ** 2).sum(axis=2))
    return np.round(dist_f).astype(np.int64)


def _make_vehicles_i64(n_trucks, n_bikes):
    """Vehicles array i64 (K, 4): [type, capacity_g, cost_per_m, speed_us_per_m]."""
    rows = []
    for _ in range(n_trucks):
        rows.append([0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M])
    for _ in range(n_bikes):
        rows.append([1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M])
    return np.array(rows, dtype=np.int64)


@pytest.fixture
def tiny_instance():
    """5 customers, 1 truck, 2 bikes. Simplest testable instance.

    Layout (10x10 km = 10000x10000 m):
        C2(10kg,restricted)
        |
    C0(100kg) --- Depot --- C1(20kg)
        |
    C3(8kg)      C4(5kg)

    Depot at (5000, 5000) meters.
    All units: meters, grams, seconds.
    """
    # i64 (N, 7): [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]
    customers = np.array([
        [    0, 5000, 100000,    0, 7200,  900, 0],  # C0: heavy (100kg)
        [10000, 5000,  20000, 3600, 10800, 300, 0],  # C1: normal
        [ 5000, 10000, 10000,    0, 14400, 300, 1],  # C2: bike only
        [ 2000,     0,  8000, 1800, 9000,  300, 0],  # C3: normal
        [ 8000,     0,  5000,    0, 28800, 300, 0],  # C4: normal
    ], dtype=np.int64)

    depot = np.array([5000, 5000], dtype=np.int64)

    n_trucks, n_bikes = 1, 2
    vehicles = _make_vehicles_i64(n_trucks, n_bikes)

    return {
        "customers": customers,
        "depot": depot,
        "vehicles": vehicles,
        "n_trucks": n_trucks,
        "n_bikes": n_bikes,
        "n_customers": 5,
    }


@pytest.fixture
def tiny_dist_matrix(tiny_instance):
    """Precomputed distance matrix for tiny_instance. (6x6) i64 meters."""
    return _make_dist_matrix_i64(tiny_instance["depot"], tiny_instance["customers"])


@pytest.fixture
def medium_instance():
    """20 customers, 2 trucks, 5 bikes. For operator / integration tests.

    Grid 5x4, spacing 10km (10000m). Some heavy, some restricted.
    All units: meters, grams, seconds.
    """
    rng = np.random.default_rng(42)
    n = 20
    gx = np.arange(5).repeat(4)
    gy = np.tile(np.arange(4), 5)
    x = gx * 10000  # meters
    y = gy * 10000

    demands_kg = rng.choice([5, 10, 15, 30, 80, 200], size=n,
                            p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    demands_g = demands_kg * 1000

    tw_open_min = rng.uniform(0, 200, n)
    tw_close_min = tw_open_min + rng.uniform(60, 180, n)
    tw_close_min = np.minimum(tw_close_min, 480.0)
    tw_open_s = np.round(tw_open_min * 60).astype(np.int64)
    tw_close_s = np.round(tw_close_min * 60).astype(np.int64)

    service_s = np.where(demands_g > 60000, 900, 300).astype(np.int64)

    restricted = np.zeros(n, dtype=np.int64)
    restricted[[3, 7, 11, 15]] = 1

    customers = np.column_stack([
        x, y, demands_g, tw_open_s, tw_close_s, service_s, restricted
    ]).astype(np.int64)

    depot = np.array([20000, 15000], dtype=np.int64)

    n_trucks, n_bikes = 2, 5
    vehicles = _make_vehicles_i64(n_trucks, n_bikes)

    return {
        "customers": customers,
        "depot": depot,
        "vehicles": vehicles,
        "n_trucks": n_trucks,
        "n_bikes": n_bikes,
        "n_customers": n,
    }


@pytest.fixture
def medium_dist_matrix(medium_instance):
    """Distance matrix for medium_instance. i64 meters."""
    return _make_dist_matrix_i64(medium_instance["depot"], medium_instance["customers"])
