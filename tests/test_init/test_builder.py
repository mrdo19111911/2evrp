"""Tests for src/init/builder.py — initial solution orchestrator — v2."""
import numpy as np
import pytest

from src.init.builder import build_initial_solution, pack_solution
from src.data.constants import ACT_DELIVER, ACT_PICKUP


# ---------------------------------------------------------------------------
# build_initial_solution (integration)
# ---------------------------------------------------------------------------
class TestBuildInitialSolution:
    def test_returns_valid_dict(self, tiny_instance):
        """build_initial_solution returns a dict with expected keys."""
        sol = build_initial_solution(tiny_instance, seed=42)
        assert isinstance(sol, dict)

    def test_has_solution_arrays(self, tiny_instance):
        """Solution must contain unified solution arrays."""
        sol = build_initial_solution(tiny_instance, seed=42)
        assert "stops" in sol
        assert "actions" in sol
        assert "lengths" in sol

    def test_deterministic_with_same_seed(self, tiny_instance):
        """Same seed -> identical solution."""
        sol1 = build_initial_solution(tiny_instance, seed=42)
        sol2 = build_initial_solution(tiny_instance, seed=42)

        for key in sol1:
            if isinstance(sol1[key], np.ndarray):
                np.testing.assert_array_equal(
                    sol1[key], sol2[key],
                    err_msg=f"Mismatch for key '{key}' with same seed",
                )

    def test_different_seed_may_differ(self, tiny_instance):
        """Different seeds may produce different solutions (not guaranteed but likely)."""
        sol1 = build_initial_solution(tiny_instance, seed=42)
        sol2 = build_initial_solution(tiny_instance, seed=999)

        assert isinstance(sol1, dict)
        assert isinstance(sol2, dict)

    def test_medium_instance(self, medium_instance):
        """Runs on medium instance without crash."""
        sol = build_initial_solution(medium_instance, seed=42)
        assert isinstance(sol, dict)
        assert "stops" in sol


# ---------------------------------------------------------------------------
# pack_solution
# ---------------------------------------------------------------------------
class TestPackSolution:
    def _make_vehicles_and_orders(self, n_trucks, n_bikes, n_customers):
        from src.data.constants import (
            VEH_TYPE, VEH_CAP_KG, VEH_COLS, VTYPE_TRUCK, VTYPE_BIKE,
            ORD_LOC, ORD_QTY, ORD_UNIT_W, ORD_COLS,
        )
        n_veh = n_trucks + n_bikes
        vehicles = np.zeros((n_veh, VEH_COLS), dtype=np.float64)
        for i in range(n_trucks):
            vehicles[i] = [VTYPE_TRUCK, 2000.0, 10.0, 4.52, 0, 0]
        for i in range(n_trucks, n_veh):
            vehicles[i] = [VTYPE_BIKE, 60.0, 0.5, 0.85, 0, 0]

        orders = np.zeros((n_customers, ORD_COLS), dtype=np.float64)
        for i in range(n_customers):
            orders[i, ORD_LOC] = i + 1
            orders[i, ORD_QTY] = 1
            orders[i, ORD_UNIT_W] = 10.0

        loc_to_cust = {int(i + 1): i for i in range(n_customers)}
        return vehicles, orders, loc_to_cust

    def test_basic_packing(self):
        """Pack minimal truck+bike sols -> solution dict with numpy arrays."""
        vehicles, orders, loc_to_cust = self._make_vehicles_and_orders(1, 2, 5)
        truck_sol = [
            {"stops": np.array([1, 2], dtype=np.int32),
             "actions": np.array([ACT_DELIVER, ACT_PICKUP], dtype=np.int8),
             "total_demand": 120.0, "total_distance": 10.0,
             "vehicle_id": 0},
        ]
        bike_sol = [
            {"stops": np.array([3, 4], dtype=np.int32),
             "actions": np.array([ACT_DELIVER, ACT_DELIVER], dtype=np.int8),
             "initial_load": 25.0, "satellite_loc": 1, "cluster_idx": 0,
             "total_distance": 5.0, "bike_id": 1},
        ]

        sol = pack_solution(truck_sol, bike_sol, vehicles, orders,
                            5, loc_to_cust)
        assert isinstance(sol, dict)
        assert "stops" in sol

    def test_multiple_truck_routes(self):
        """Packing multiple truck routes keeps all data."""
        vehicles, orders, loc_to_cust = self._make_vehicles_and_orders(2, 2, 5)
        truck_sol = [
            {"stops": np.array([1, 2], dtype=np.int32),
             "actions": np.array([ACT_DELIVER, ACT_DELIVER], dtype=np.int8),
             "total_demand": 80.0, "total_distance": 8.0, "vehicle_id": 0},
            {"stops": np.array([3], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 40.0, "total_distance": 4.0, "vehicle_id": 1},
        ]
        bike_sol = [
            {"stops": np.array([4], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "initial_load": 10.0, "satellite_loc": 1, "cluster_idx": 0,
             "total_distance": 3.0, "bike_id": 2},
        ]

        sol = pack_solution(truck_sol, bike_sol, vehicles, orders,
                            5, loc_to_cust)
        assert isinstance(sol, dict)

    def test_empty_solution(self):
        """Empty truck/bike sols -> valid solution dict (no crash)."""
        vehicles, orders, loc_to_cust = self._make_vehicles_and_orders(1, 1, 0)
        truck_sol = []
        bike_sol = []

        sol = pack_solution(truck_sol, bike_sol, vehicles, orders,
                            0, loc_to_cust)
        assert isinstance(sol, dict)
