"""Tests for src/data/distance.py -- compute_dist_matrix.

API (i64): compute_dist_matrix(depot_i64, customers_i64) -> i64 (N+1, N+1) meters.
Index 0 = depot, indices 1..N = customers.
depot: i64 (2,) in meters.
customers: i64 (N, 7) [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted].
"""
import numpy as np
import pytest

from src.data.distance import compute_dist_matrix
from src.data.constants import CUST_COLS


def _make_customers_i64(xy_list):
    """Helper: create i64 (N, 7) customers from xy pairs (meters)."""
    n = len(xy_list)
    out = np.zeros((n, CUST_COLS), dtype=np.int64)
    for i, (x, y) in enumerate(xy_list):
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = 10000   # 10 kg demand
        out[i, 3] = 0       # tw_open
        out[i, 4] = 28800   # tw_close (8h)
        out[i, 5] = 600     # service_s
        out[i, 6] = 0       # unrestricted
    return out


# ---------------------------------------------------------------------------
# TC1: depot(0,0), 1 customer(3000,4000) -> dist[0,1] = 5000m (3-4-5)
# ---------------------------------------------------------------------------
def test_single_customer_3_4_5():
    depot = np.array([0, 0], dtype=np.int64)
    customers = _make_customers_i64([(3000, 4000)])
    dm = compute_dist_matrix(depot, customers)
    assert dm.shape == (2, 2)
    assert dm.dtype == np.int64
    assert dm[0, 1] == 5000
    assert dm[1, 0] == 5000


# ---------------------------------------------------------------------------
# TC2: symmetry -- dist[i,j] == dist[j,i]
# ---------------------------------------------------------------------------
def test_symmetry():
    depot = np.array([5000, 5000], dtype=np.int64)
    customers = _make_customers_i64([
        (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
    ])
    dm = compute_dist_matrix(depot, customers)
    n = dm.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            assert dm[i, j] == dm[j, i], (
                f"Asymmetry at ({i},{j}): {dm[i,j]} != {dm[j,i]}")


# ---------------------------------------------------------------------------
# TC3: diagonal is all zeros
# ---------------------------------------------------------------------------
def test_diagonal_zero():
    depot = np.array([5000, 5000], dtype=np.int64)
    customers = _make_customers_i64([
        (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
    ])
    dm = compute_dist_matrix(depot, customers)
    for i in range(dm.shape[0]):
        assert dm[i, i] == 0


# ---------------------------------------------------------------------------
# TC4: triangle inequality (with integer rounding tolerance of 1m)
# ---------------------------------------------------------------------------
def test_triangle_inequality():
    depot = np.array([5000, 5000], dtype=np.int64)
    customers = _make_customers_i64([
        (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
    ])
    dm = compute_dist_matrix(depot, customers)
    n = dm.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert dm[i, k] <= dm[i, j] + dm[j, k] + 1


# ---------------------------------------------------------------------------
# TC5: shape = (N+1, N+1); 5 customers -> (6, 6)
# ---------------------------------------------------------------------------
def test_shape_five_customers():
    depot = np.array([5000, 5000], dtype=np.int64)
    customers = _make_customers_i64([
        (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
    ])
    dm = compute_dist_matrix(depot, customers)
    assert dm.shape == (6, 6)
    assert dm.dtype == np.int64


# ---------------------------------------------------------------------------
# TC6: concrete depot-to-customer distances
# ---------------------------------------------------------------------------
def test_depot_to_customer_distances():
    """Depot (5000,5000) in meters:
      C0 (0,5000):     dist = 5000
      C1 (10000,5000):  dist = 5000
      C2 (5000,10000):  dist = 5000
      C3 (2000,0):      dist = sqrt(9e6+25e6) = sqrt(34e6) ~ 5831
      C4 (8000,0):      dist = sqrt(9e6+25e6) = sqrt(34e6) ~ 5831
    """
    depot = np.array([5000, 5000], dtype=np.int64)
    customers = _make_customers_i64([
        (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
    ])
    dm = compute_dist_matrix(depot, customers)
    assert dm[0, 1] == 5000    # depot -> C0
    assert dm[0, 2] == 5000    # depot -> C1
    assert dm[0, 3] == 5000    # depot -> C2
    expected_c3 = round(np.sqrt(3000**2 + 5000**2))
    assert dm[0, 4] == expected_c3  # depot -> C3
    assert dm[0, 5] == expected_c3  # depot -> C4
