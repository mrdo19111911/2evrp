"""QA tests for src/data/distance.py -- distance matrix properties validation.

API (i64): compute_dist_matrix(depot_i64, customers_i64) -> i64 (N+1, N+1) meters.
All coordinates and distances are integers in meters.
"""
import numpy as np
import pytest
from src.data.distance import compute_dist_matrix
from src.data.constants import CUST_COLS


def _make_cust(xy_list):
    """Helper: create i64 (N, 7) customers from xy pairs (meters)."""
    n = len(xy_list)
    out = np.zeros((n, CUST_COLS), dtype=np.int64)
    for i, (x, y) in enumerate(xy_list):
        out[i, 0] = x
        out[i, 1] = y
        out[i, 2] = 10000   # demand_g
        out[i, 3] = 0       # tw_open_s
        out[i, 4] = 28800   # tw_close_s
        out[i, 5] = 600     # service_s
    return out


class TestDistanceMatrixSymmetry:
    """Verify distance matrix is symmetric."""

    def test_symmetry_single_customer(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 1] == dm[1, 0]

    def test_symmetry_three_customers(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000), (5000, 0), (0, 5000)])
        dm = compute_dist_matrix(depot, customers)
        n = dm.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert dm[i, j] == dm[j, i]

    def test_symmetry_large(self):
        rng = np.random.default_rng(123)
        depot = np.array([50000, 50000], dtype=np.int64)
        xy = rng.integers(0, 100000, (50, 2))
        customers = _make_cust(xy.tolist())
        dm = compute_dist_matrix(depot, customers)
        n = dm.shape[0]
        for _ in range(500):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i != j:
                assert dm[i, j] == dm[j, i]


class TestDistanceMatrixDiagonal:
    """Verify diagonal elements are zero."""

    def test_diagonal_zero_single_customer(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 0] == 0
        assert dm[1, 1] == 0

    def test_diagonal_zero_five_customers(self):
        depot = np.array([5000, 5000], dtype=np.int64)
        customers = _make_cust([
            (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
        ])
        dm = compute_dist_matrix(depot, customers)
        for i in range(dm.shape[0]):
            assert dm[i, i] == 0

    def test_diagonal_zero_large(self):
        rng = np.random.default_rng(456)
        depot = np.array([50000, 50000], dtype=np.int64)
        xy = rng.integers(0, 100000, (100, 2))
        customers = _make_cust(xy.tolist())
        dm = compute_dist_matrix(depot, customers)
        for i in range(dm.shape[0]):
            assert dm[i, i] == 0


class TestDistanceMatrixTriangleInequality:
    """Verify triangle inequality (with 1m integer rounding tolerance)."""

    def test_triangle_inequality_three_customers(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000), (5000, 0), (0, 5000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 3] <= dm[0, 1] + dm[1, 3] + 1

    def test_triangle_inequality_all_triples(self):
        depot = np.array([5000, 5000], dtype=np.int64)
        customers = _make_cust([
            (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
        ])
        dm = compute_dist_matrix(depot, customers)
        n = dm.shape[0]
        violations = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if dm[i, k] > dm[i, j] + dm[j, k] + 1:
                        violations.append((i, j, k))
        assert len(violations) == 0

    def test_triangle_inequality_random_large(self):
        rng = np.random.default_rng(789)
        depot = np.array([50000, 50000], dtype=np.int64)
        xy = rng.integers(0, 100000, (50, 2))
        customers = _make_cust(xy.tolist())
        dm = compute_dist_matrix(depot, customers)
        n = dm.shape[0]
        for _ in range(1000):
            i, j, k = rng.integers(0, n, 3)
            assert dm[i, k] <= dm[i, j] + dm[j, k] + 1


class TestDistanceMatrixShape:
    """Verify output shape is (N+1, N+1)."""

    def test_shape_single_customer(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm.shape == (2, 2)

    def test_shape_five_customers(self):
        depot = np.array([5000, 5000], dtype=np.int64)
        customers = _make_cust([
            (0, 5000), (10000, 5000), (5000, 10000), (2000, 0), (8000, 0)
        ])
        dm = compute_dist_matrix(depot, customers)
        assert dm.shape == (6, 6)
        assert dm.ndim == 2

    def test_shape_fifty_customers(self):
        rng = np.random.default_rng(111)
        depot = np.array([50000, 50000], dtype=np.int64)
        xy = rng.integers(0, 100000, (50, 2))
        customers = _make_cust(xy.tolist())
        dm = compute_dist_matrix(depot, customers)
        assert dm.shape == (51, 51)

    def test_dtype_int64(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000), (5000, 0)])
        dm = compute_dist_matrix(depot, customers)
        assert dm.dtype == np.int64


class TestDistanceMatrixConcrete:
    """Verify specific computed distances match expected values."""

    def test_3_4_5_triangle(self):
        """Depot(0,0), Customer(3000m, 4000m) -> distance = 5000m."""
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 1] == 5000

    def test_right_angle_distances(self):
        """Axis-aligned distances in meters."""
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(10000, 0), (0, 10000), (10000, 10000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 1] == 10000
        assert dm[0, 2] == 10000
        assert dm[0, 3] == round(np.sqrt(2) * 10000)

    def test_collinear_points(self):
        """Collinear points on x-axis in meters."""
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(5000, 0), (10000, 0)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 1] == 5000
        assert dm[0, 2] == 10000
        assert dm[1, 2] == 5000


class TestDistanceMatrixEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_depot_at_origin(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(1000, 1000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 1] == round(np.sqrt(2) * 1000)

    def test_depot_at_large_coordinates(self):
        """Depot(1000000,1000000), C(1003000,1004000) -> 5000m."""
        depot = np.array([1000000, 1000000], dtype=np.int64)
        customers = _make_cust([(1003000, 1004000)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 1] == 5000

    def test_all_customers_at_depot(self):
        depot = np.array([5000, 5000], dtype=np.int64)
        customers = _make_cust([(5000, 5000), (5000, 5000)])
        dm = compute_dist_matrix(depot, customers)
        for i in range(dm.shape[0]):
            for j in range(dm.shape[0]):
                assert dm[i, j] == 0


class TestDistanceMatrixReproducibility:
    """Verify reproducibility and consistency."""

    def test_deterministic_output(self):
        depot = np.array([5000, 5000], dtype=np.int64)
        customers = _make_cust([(0, 5000), (10000, 5000), (5000, 10000)])
        dm1 = compute_dist_matrix(depot, customers)
        dm2 = compute_dist_matrix(depot, customers)
        np.testing.assert_array_equal(dm1, dm2)

    def test_depot_index_zero(self):
        depot = np.array([0, 0], dtype=np.int64)
        customers = _make_cust([(3000, 4000), (5000, 0)])
        dm = compute_dist_matrix(depot, customers)
        assert dm[0, 0] == 0
        assert dm[0, 1] == 5000
        assert dm[0, 2] == 5000
