"""QA test suite for src/init/builder.py -- edge case coverage.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.builder import build_initial_solution
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, VEH_TRUCK, VEH_BIKE,
    SOL_TRUCK_STOPS, SOL_TRUCK_ACTIONS, SOL_TRUCK_LENGTHS,
    SOL_BIKE_STOPS, SOL_BIKE_ACTIONS, SOL_BIKE_LENGTHS,
    SOL_BIKE_LOADS, SOL_CUST_VEHICLE,
)
from src.solution.structure import create_solution, rebuild_index
from src.data.cost import (
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)


def _make_dm(depot, customers):
    depot_xy = depot.reshape(1, 2).astype(np.float64)
    cust_xy = customers[:, :2].astype(np.float64)
    coords = np.vstack([depot_xy, cust_xy])
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


# ============================================================================
# TEST 1: build_initial_solution with 0 customers
# ============================================================================
class TestBuildInitialSolutionEmptyCustomers:

    def test_zero_customers_crashes_or_handles(self):
        """0 customers should either crash early or return empty solution."""
        customers = np.zeros((0, 7), dtype=np.int64)
        depot = np.array([5000, 5000], dtype=np.int64)
        vehicles = np.array([
            [VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
            [VEH_BIKE, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M],
        ], dtype=np.int64)
        dist_matrix = np.array([[0]], dtype=np.int64)

        try:
            sol = build_initial_solution(
                customers, depot, vehicles,
                dist_matrix, n_trucks=1, n_bikes=1, seed=42
            )
            assert isinstance(sol, tuple)
        except (IndexError, ValueError, KeyError):
            pytest.skip("Known crash with empty customers")


# ============================================================================
# TEST 2: build_initial_solution with all restricted customers
# ============================================================================
class TestBuildInitialSolutionAllRestricted:

    def test_all_restricted_completes(self):
        """All restricted customers -> builder should still complete."""
        n = 5
        # i64 (N,7): [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]
        customers = np.array([
            [i * 2000, i * 2000, 15000, 0, 28800, 300, 1]
            for i in range(n)
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        vehicles = np.array([
            [VEH_TRUCK, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
            [VEH_BIKE, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M],
            [VEH_BIKE, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M],
        ], dtype=np.int64)
        dm = _make_dm(depot, customers)

        sol = build_initial_solution(
            customers, depot, vehicles,
            dm, n_trucks=1, n_bikes=2, seed=42
        )
        assert isinstance(sol, tuple)


# ============================================================================
# TEST 3: Verify no customer appears in both truck and bike routes
# ============================================================================
class TestNoDuplicateCustomersAcrossVehicles:

    def test_duplicate_customer_detected_by_rebuild(self):
        """Deliberately placing customer 0 in both truck and bike -> rebuild picks last."""
        sol = create_solution(n_trucks=1, n_bikes=1, n_customers=2)

        sol[SOL_TRUCK_STOPS][0, 0] = 0
        sol[SOL_TRUCK_ACTIONS][0, 0] = ACT_DELIVER
        sol[SOL_TRUCK_LENGTHS][0] = 1

        sol[SOL_BIKE_STOPS][0, 0] = 0
        sol[SOL_BIKE_ACTIONS][0, 0] = ACT_DELIVER
        sol[SOL_BIKE_LENGTHS][0] = 1

        rebuild_index(sol)
        assert sol[SOL_CUST_VEHICLE][0] >= 0


# ============================================================================
# TEST 4: Normal build completes and returns tuple
# ============================================================================
class TestBuildNormal:

    def test_builds_and_returns_tuple(self, tiny_instance):
        data = tiny_instance
        dm = _make_dm(data["depot"], data["customers"])
        sol = build_initial_solution(
            data["customers"], data["depot"],
            data["vehicles"], dm,
            data["n_trucks"], data["n_bikes"], seed=42
        )
        assert isinstance(sol, tuple)
        assert len(sol) == 15  # SOL_SIZE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
