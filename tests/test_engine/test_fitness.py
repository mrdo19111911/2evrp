"""Tests for src/engine/fitness.py — objective, penalty, fitness."""
import numpy as np
import pytest

from src.engine.fitness import compute_fitness, evaluate_solution
from src.engine.route_cost import compute_route_cost, compute_makespan
from src.engine.violations import compute_penalties
from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE,
    ST_COLS,
)
from src.data.cost import (
    TRUCK_TOTAL_KM, TRUCK_DRIVER_HOUR, TRUCK_FIXED_DAY, TRUCK_DEPLOY_COST,
    BIKE_TOTAL_KM, BIKE_DRIVER_HOUR, BIKE_FIXED_DAY, BIKE_DEPLOY_COST,
    RELOAD_HANDLING_COST,
    PENALTY_UNSERVED,
)


# ---------------------------------------------------------------------------
# compute_route_cost
# ---------------------------------------------------------------------------

class TestComputeRouteCostTruck:
    """Truck: 50 km, 3 h (180 min), 2 reloads. Uses current cost constants."""

    def test_truck_cost(self):
        cost, breakdown = compute_route_cost(
            total_distance=50.0,
            total_time=180.0,
            n_reloads=2,
            vtype=VEH_TRUCK,
        )

        exp_dist = 50 * TRUCK_TOTAL_KM
        exp_time = (180 / 60) * TRUCK_DRIVER_HOUR
        exp_reload = 2 * RELOAD_HANDLING_COST
        exp_total = exp_dist + exp_time + TRUCK_FIXED_DAY + TRUCK_DEPLOY_COST + exp_reload

        assert breakdown["distance_cost"] == pytest.approx(exp_dist, abs=1.0)
        assert breakdown["time_cost"] == pytest.approx(exp_time, abs=1.0)
        assert breakdown["fixed_cost"] == pytest.approx(TRUCK_FIXED_DAY, abs=1.0)
        assert breakdown["deploy_cost"] == pytest.approx(TRUCK_DEPLOY_COST, abs=1.0)
        assert breakdown["reload_cost"] == pytest.approx(exp_reload, abs=1.0)
        assert cost == pytest.approx(exp_total, abs=1.0)


class TestComputeRouteCostBike:
    """Bike: 30 km, 4 h (240 min), 1 reload. Uses current cost constants."""

    def test_bike_cost(self):
        cost, breakdown = compute_route_cost(
            total_distance=30.0,
            total_time=240.0,
            n_reloads=1,
            vtype=VEH_BIKE,
        )

        exp_dist = 30 * BIKE_TOTAL_KM
        exp_time = (240 / 60) * BIKE_DRIVER_HOUR
        exp_reload = 1 * RELOAD_HANDLING_COST
        exp_total = exp_dist + exp_time + BIKE_FIXED_DAY + BIKE_DEPLOY_COST + exp_reload

        assert breakdown["distance_cost"] == pytest.approx(exp_dist, abs=1.0)
        assert breakdown["time_cost"] == pytest.approx(exp_time, abs=1.0)
        assert breakdown["fixed_cost"] == pytest.approx(BIKE_FIXED_DAY, abs=1.0)
        assert breakdown["deploy_cost"] == pytest.approx(BIKE_DEPLOY_COST, abs=1.0)
        assert breakdown["reload_cost"] == pytest.approx(exp_reload, abs=1.0)
        assert cost == pytest.approx(exp_total, abs=1.0)


class TestComputeRouteCostZero:
    """Empty route: 0 km, 0 min, 0 reloads -> still pays fixed cost."""

    def test_zero_distance_truck(self):
        cost, breakdown = compute_route_cost(
            total_distance=0.0,
            total_time=0.0,
            n_reloads=0,
            vtype=VEH_TRUCK,
        )

        assert breakdown["distance_cost"] == pytest.approx(0.0, abs=1e-9)
        assert breakdown["time_cost"] == pytest.approx(0.0, abs=1e-9)
        assert breakdown["fixed_cost"] == pytest.approx(TRUCK_FIXED_DAY, abs=1.0)
        assert breakdown["deploy_cost"] == pytest.approx(TRUCK_DEPLOY_COST, abs=1.0)
        assert breakdown["reload_cost"] == pytest.approx(0.0, abs=1e-9)
        assert cost == pytest.approx(TRUCK_FIXED_DAY + TRUCK_DEPLOY_COST, abs=1.0)


# ---------------------------------------------------------------------------
# compute_makespan
# ---------------------------------------------------------------------------

class TestComputeMakespan:
    """Latest vehicle return time across all vehicles."""

    def test_trucks_and_bikes(self):
        truck_return_times = np.array([100.0, 200.0])
        bike_return_times = np.array([150.0])
        makespan = compute_makespan(truck_return_times, bike_return_times)
        assert makespan == pytest.approx(200.0, abs=1e-9)

    def test_bike_is_latest(self):
        truck_return_times = np.array([100.0])
        bike_return_times = np.array([300.0, 50.0])
        makespan = compute_makespan(truck_return_times, bike_return_times)
        assert makespan == pytest.approx(300.0, abs=1e-9)

    def test_all_zeros(self):
        truck_return_times = np.array([0.0, 0.0])
        bike_return_times = np.array([0.0])
        makespan = compute_makespan(truck_return_times, bike_return_times)
        assert makespan == pytest.approx(0.0, abs=1e-9)

    def test_single_truck_no_bikes(self):
        truck_return_times = np.array([180.0])
        bike_return_times = np.array([], dtype=np.float64)
        makespan = compute_makespan(truck_return_times, bike_return_times)
        assert makespan == pytest.approx(180.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_penalties
# ---------------------------------------------------------------------------

class TestComputePenalties:
    """Penalty calculation from validation report."""

    @staticmethod
    def _clean_report():
        """Report with no violations."""
        return {
            "delivery_uniqueness": {"valid": True, "unserved": [], "duplicates": []},
            "vehicle_restrictions": {"valid": True, "violations": []},
            "capacity": {"valid": True, "violations": []},
            "time_windows": {"valid": True, "violations": []},
            "sync": {"valid": True, "violations": []},
        }

    @staticmethod
    def _default_weights():
        return {
            "unserved": PENALTY_UNSERVED,       # 500_000
            "duplicate": PENALTY_UNSERVED,       # 500_000
            "capacity": 50_000.0,
            "time_window": 10_000.0,             # per minute
            "sync": 100_000.0,
            "vehicle_restriction": 200_000.0,
        }

    def test_no_violations(self):
        report = self._clean_report()
        weights = self._default_weights()
        total, breakdown = compute_penalties(report, weights)

        assert total == pytest.approx(0.0, abs=1e-9)
        for v in breakdown.values():
            assert v == pytest.approx(0.0, abs=1e-9)

    def test_two_unserved(self):
        """2 unserved customers -> 2 * PENALTY_UNSERVED = 1_000_000."""
        report = self._clean_report()
        report["delivery_uniqueness"]["valid"] = False
        report["delivery_uniqueness"]["unserved"] = [2, 4]
        weights = self._default_weights()

        total, breakdown = compute_penalties(report, weights)

        expected_unserved = 2 * PENALTY_UNSERVED  # 2 * 500_000 = 1_000_000
        assert breakdown["unserved"] == pytest.approx(expected_unserved, abs=1.0)
        assert total >= expected_unserved

    def test_mixed_violations(self):
        """Multiple violation types at once."""
        report = self._clean_report()
        report["delivery_uniqueness"]["valid"] = False
        report["delivery_uniqueness"]["unserved"] = [3]             # 1 unserved
        report["delivery_uniqueness"]["duplicates"] = [1]           # 1 duplicate
        report["vehicle_restrictions"]["valid"] = False
        report["vehicle_restrictions"]["violations"] = [2]          # 1 restriction
        report["sync"]["valid"] = False
        report["sync"]["violations"] = [(0, "bike_too_early", {})] # 1 sync fail

        weights = self._default_weights()
        total, breakdown = compute_penalties(report, weights)

        expected = (
            1 * 500_000    # unserved
            + 1 * 500_000  # duplicate
            + 1 * 200_000  # vehicle restriction
            + 1 * 100_000  # sync
        )
        assert total == pytest.approx(expected, abs=1.0)


# ---------------------------------------------------------------------------
# compute_fitness
# ---------------------------------------------------------------------------

class TestComputeFitness:
    """Pure VND sum: cost + sync_cost + penalty."""

    def test_basic_fitness(self):
        """cost=1000, sync=0, penalty=0 -> 1000."""
        fitness = compute_fitness(
            total_cost=1000.0,
            sync_cost=0.0,
            total_penalty=0.0,
        )
        assert fitness == pytest.approx(1000.0, abs=1e-9)

    def test_penalty_only(self):
        """cost=0, sync=0, penalty=5000 -> 5000."""
        fitness = compute_fitness(
            total_cost=0.0,
            sync_cost=0.0,
            total_penalty=5000.0,
        )
        assert fitness == pytest.approx(5000.0, abs=1e-9)

    def test_all_nonzero(self):
        """cost=10000, sync=500, penalty=2000 -> 12500."""
        fitness = compute_fitness(
            total_cost=10_000.0,
            sync_cost=500.0,
            total_penalty=2_000.0,
        )
        assert fitness == pytest.approx(12_500.0, abs=1e-9)

    def test_sync_cost_included(self):
        """Sync cost is summed into fitness."""
        fitness = compute_fitness(
            total_cost=100.0,
            sync_cost=200.0,
            total_penalty=300.0,
        )
        assert fitness == pytest.approx(600.0, abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate_solution — integration
# ---------------------------------------------------------------------------

class TestEvaluateSolution:
    """Integration test: simulate -> validate -> fitness, using tiny_instance."""

    def test_simple_valid_solution(self, tiny_instance, tiny_dist_matrix):
        """Truck delivers C0, C1. Bike0 delivers C2, C3. Bike1 delivers C4.

        All customers served, no restrictions violated, no reloads needed.
        """
        L = 4

        truck_stops = np.full((1, L), -1, dtype=np.int32)
        truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
        truck_stops[0, 0] = 0;  truck_actions[0, 0] = ACT_DELIVER  # C0
        truck_stops[0, 1] = 1;  truck_actions[0, 1] = ACT_DELIVER  # C1

        bike_stops = np.full((2, L), -1, dtype=np.int32)
        bike_actions = np.full((2, L), ACT_PAD, dtype=np.int8)
        bike_stops[0, 0] = 2;  bike_actions[0, 0] = ACT_DELIVER   # C2 (restricted=bike only)
        bike_stops[0, 1] = 3;  bike_actions[0, 1] = ACT_DELIVER   # C3
        bike_stops[1, 0] = 4;  bike_actions[1, 0] = ACT_DELIVER   # C4

        satellites = np.empty((0, 5), dtype=np.float64)

        penalty_weights = {
            "unserved": 500_000.0,
            "duplicate": 500_000.0,
            "capacity": 50_000.0,
            "time_window": 10_000.0,
            "sync": 100_000.0,
            "vehicle_restriction": 200_000.0,
        }

        result = evaluate_solution(
            sol={
                "truck_stops": truck_stops,
                "truck_actions": truck_actions,
                "bike_stops": bike_stops,
                "bike_actions": bike_actions,
                "satellites": satellites,
            },
            customers=tiny_instance["customers"],
            restricted=tiny_instance["restricted"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            delta_t=15.0,
            penalty_weights=penalty_weights,
        )

        assert result["feasible"] is True
        assert result["total_penalty"] == pytest.approx(0.0, abs=1.0)
        assert result["sync_cost"] == pytest.approx(0.0, abs=1.0)
        assert result["fitness"] > 0.0
        assert result["cost"] > 0.0
        assert result["makespan"] > 0.0
        # fitness = cost + sync_cost + penalty (pure VND)
        assert result["fitness"] == pytest.approx(
            result["cost"] + result["sync_cost"] + result["total_penalty"], abs=1.0)

    def test_deterministic(self, tiny_instance, tiny_dist_matrix):
        """Same input -> same output (deterministic)."""
        L = 3

        truck_stops = np.full((1, L), -1, dtype=np.int32)
        truck_actions = np.full((1, L), ACT_PAD, dtype=np.int8)
        truck_stops[0, 0] = 0;  truck_actions[0, 0] = ACT_DELIVER
        truck_stops[0, 1] = 1;  truck_actions[0, 1] = ACT_DELIVER

        bike_stops = np.full((2, L), -1, dtype=np.int32)
        bike_actions = np.full((2, L), ACT_PAD, dtype=np.int8)
        bike_stops[0, 0] = 2;  bike_actions[0, 0] = ACT_DELIVER
        bike_stops[0, 1] = 3;  bike_actions[0, 1] = ACT_DELIVER
        bike_stops[1, 0] = 4;  bike_actions[1, 0] = ACT_DELIVER

        satellites = np.empty((0, 5), dtype=np.float64)

        penalty_weights = {
            "unserved": 500_000.0,
            "duplicate": 500_000.0,
            "capacity": 50_000.0,
            "time_window": 10_000.0,
            "sync": 100_000.0,
            "vehicle_restriction": 200_000.0,
        }

        kwargs = dict(
            sol={
                "truck_stops": truck_stops,
                "truck_actions": truck_actions,
                "bike_stops": bike_stops,
                "bike_actions": bike_actions,
                "satellites": satellites,
            },
            customers=tiny_instance["customers"],
            restricted=tiny_instance["restricted"],
            vehicles=tiny_instance["vehicles"],
            dist_matrix=tiny_dist_matrix,
            delta_t=15.0,
            penalty_weights=penalty_weights,
        )

        r1 = evaluate_solution(**kwargs)
        r2 = evaluate_solution(**kwargs)

        assert r1["fitness"] == pytest.approx(r2["fitness"], abs=1e-9)
        assert r1["cost"] == pytest.approx(r2["cost"], abs=1e-9)
        assert r1["makespan"] == pytest.approx(r2["makespan"], abs=1e-9)
