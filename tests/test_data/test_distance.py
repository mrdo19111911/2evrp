"""Tests for src/data/distance.py — compute_dist_matrix (v2 API).

v2: compute_dist_matrix(locations) -> (2, N_loc, N_loc) tensor.
     [DM_DIST] = km, [DM_TIME] = minutes.
"""
import numpy as np
import pytest

from src.data.distance import compute_dist_matrix, DEFAULT_SPEED
from src.data.constants import DM_DIST, DM_TIME, LOC_COLS, LTYPE_DEPOT, LTYPE_CUSTOMER


def _make_locations(depot_xy, customer_rows):
    """Build v2 locations array from depot coords and customer (x,y,...) rows."""
    n = 1 + len(customer_rows)
    locs = np.zeros((n, LOC_COLS), dtype=np.float64)
    locs[0, :2] = depot_xy
    locs[0, 2] = LTYPE_DEPOT
    locs[0, 3] = 99999.0
    for i, row in enumerate(customer_rows):
        locs[i + 1, 0] = row[0]
        locs[i + 1, 1] = row[1]
        locs[i + 1, 2] = LTYPE_CUSTOMER
        locs[i + 1, 3] = 99999.0
    return locs


# ---------------------------------------------------------------------------
# TC1: depot(0,0), 1 customer(3,4) -> dist[DM_DIST,0,1] = 5.0  (3-4-5)
# ---------------------------------------------------------------------------
def test_single_customer_3_4_5():
    locs = _make_locations([0.0, 0.0], [[3.0, 4.0]])
    dm = compute_dist_matrix(locs)
    assert dm.shape == (2, 2, 2)
    assert dm[DM_DIST, 0, 1] == pytest.approx(5.0)
    assert dm[DM_DIST, 1, 0] == pytest.approx(5.0)
    # time check
    expected_time = 5.0 / DEFAULT_SPEED * 60.0
    assert dm[DM_TIME, 0, 1] == pytest.approx(expected_time)


# ---------------------------------------------------------------------------
# TC2: symmetry — dist[DM_DIST,i,j] == dist[DM_DIST,j,i]
# ---------------------------------------------------------------------------
def test_symmetry(tiny_data):
    dm = tiny_data["dist_matrix"]
    n = dm.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            assert dm[DM_DIST, i, j] == pytest.approx(dm[DM_DIST, j, i]), (
                f"Asymmetry at ({i},{j}): {dm[DM_DIST,i,j]} != {dm[DM_DIST,j,i]}"
            )


# ---------------------------------------------------------------------------
# TC3: diagonal is all zeros
# ---------------------------------------------------------------------------
def test_diagonal_zero(tiny_data):
    dm = tiny_data["dist_matrix"]
    for i in range(dm.shape[1]):
        assert dm[DM_DIST, i, i] == pytest.approx(0.0), f"diag[{i}] = {dm[DM_DIST,i,i]}"
        assert dm[DM_TIME, i, i] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TC4: triangle inequality on DM_DIST layer
# ---------------------------------------------------------------------------
def test_triangle_inequality(tiny_data):
    dm = tiny_data["dist_matrix"][DM_DIST]
    n = dm.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert dm[i, k] <= dm[i, j] + dm[j, k] + 1e-9, (
                    f"Triangle inequality violated: "
                    f"d[{i},{k}]={dm[i,k]} > d[{i},{j}]={dm[i,j]} + d[{j},{k}]={dm[j,k]}"
                )


# ---------------------------------------------------------------------------
# TC5: shape = (2, N_loc, N_loc); tiny has 6 locations -> (2, 6, 6)
# ---------------------------------------------------------------------------
def test_shape_tiny(tiny_data):
    dm = tiny_data["dist_matrix"]
    assert dm.shape == (2, 6, 6)
    assert dm.dtype == np.float64 or np.issubdtype(dm.dtype, np.floating)


# ---------------------------------------------------------------------------
# Concrete value: compute_dist_matrix matches fixture's precomputed matrix
# ---------------------------------------------------------------------------
def test_matches_precomputed_tiny(tiny_data, tiny_dist_matrix):
    dm = compute_dist_matrix(tiny_data["locations"])
    np.testing.assert_allclose(dm, tiny_dist_matrix, atol=1e-12)


def test_depot_to_customer_distances(tiny_data):
    """Verify concrete depot-to-customer distances.

    Depot (5,5):
      C0 (0,5):  dist = 5.0
      C1 (10,5): dist = 5.0
      C2 (5,10): dist = 5.0
      C3 (2,0):  dist = sqrt(9+25) = sqrt(34) ~ 5.831
      C4 (8,0):  dist = sqrt(9+25) = sqrt(34) ~ 5.831
    """
    dm = tiny_data["dist_matrix"]
    assert dm[DM_DIST, 0, 1] == pytest.approx(5.0)           # depot -> C0
    assert dm[DM_DIST, 0, 2] == pytest.approx(5.0)           # depot -> C1
    assert dm[DM_DIST, 0, 3] == pytest.approx(5.0)           # depot -> C2
    assert dm[DM_DIST, 0, 4] == pytest.approx(np.sqrt(34.0)) # depot -> C3
    assert dm[DM_DIST, 0, 5] == pytest.approx(np.sqrt(34.0)) # depot -> C4


def test_medium_data_shape(medium_data):
    """medium_data has 20 customers -> (2, 21, 21)."""
    dm = medium_data["dist_matrix"]
    assert dm.shape == (2, 21, 21)
