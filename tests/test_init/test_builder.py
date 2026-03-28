"""Tests for src/init/builder.py -- initial solution orchestrator.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.builder import build_initial_solution
from src.data.constants import ACT_DELIVER, SOL_TRUCK_STOPS, SOL_TRUCK_LENGTHS


def _make_dist_matrix(depot, customers):
    """Euclidean i64 distance matrix from depot + customers."""
    depot_xy = depot.reshape(1, 2).astype(np.float64)
    cust_xy = customers[:, :2].astype(np.float64)
    coords = np.vstack([depot_xy, cust_xy])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


# ---------------------------------------------------------------------------
# build_initial_solution (integration)
# ---------------------------------------------------------------------------
class TestBuildInitialSolution:
    def test_returns_valid_tuple(self, tiny_instance):
        data = tiny_instance
        dm = _make_dist_matrix(data["depot"], data["customers"])
        sol = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=42)
        assert isinstance(sol, tuple)

    def test_has_solution_arrays(self, tiny_instance):
        data = tiny_instance
        dm = _make_dist_matrix(data["depot"], data["customers"])
        sol = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=42)
        assert sol[SOL_TRUCK_STOPS] is not None
        assert sol[SOL_TRUCK_LENGTHS] is not None

    def test_deterministic_with_same_seed(self, tiny_instance):
        data = tiny_instance
        dm = _make_dist_matrix(data["depot"], data["customers"])
        sol1 = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=42)
        sol2 = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=42)
        for i in range(len(sol1)):
            np.testing.assert_array_equal(sol1[i], sol2[i],
                                          err_msg=f"Mismatch at index {i}")

    def test_different_seed_may_differ(self, tiny_instance):
        data = tiny_instance
        dm = _make_dist_matrix(data["depot"], data["customers"])
        sol1 = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=42)
        sol2 = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=999)
        assert isinstance(sol1, tuple)
        assert isinstance(sol2, tuple)

    def test_medium_instance(self, medium_instance):
        data = medium_instance
        dm = _make_dist_matrix(data["depot"], data["customers"])
        sol = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=42)
        assert isinstance(sol, tuple)
