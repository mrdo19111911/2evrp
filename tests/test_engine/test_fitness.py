"""Tests for src/engine/fitness.py -- objective, penalty, fitness. All i64.

API:
- compute_route_cost(total_dist_m, total_time_s, n_reloads, vtype, total_wait_s, vehicles, vid) -> i64
- evaluate_solution(sol, customers, vehicles, dist_matrix, delta_t_s, penalty_w) -> i64[EV_SIZE]
- penalty_w is i64 (PW_SIZE,)
"""
import numpy as np
import pytest

from src.engine.fitness import compute_fitness, evaluate_solution
from src.engine.route_cost import compute_route_cost, compute_makespan
from src.engine.violations import compute_penalties
from src.data.constants import (
    ACT_DELIVER, ACT_PAD, VEH_TRUCK, VEH_BIKE, ST_COLS, PW_SIZE,
    EV_FITNESS, EV_COST, EV_SYNC_COST, EV_MAKESPAN,
    EV_TOTAL_PENALTY, EV_FEASIBLE,
)
from src.data.cost import (
    TRUCK_COST_PER_M, TRUCK_DRIVER_SEC, TRUCK_FIXED_DAY, TRUCK_DEPLOY_COST,
    BIKE_COST_PER_M, BIKE_DRIVER_SEC, BIKE_FIXED_DAY, BIKE_DEPLOY_COST,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    RELOAD_HANDLING_COST,
)
from tests.test_engine.conftest import make_sol_from_routes, make_penalty_weights


def _make_vehicles(n_trucks=1, n_bikes=1):
    """Build vehicles array i64 (K,4)."""
    rows = []
    for _ in range(n_trucks):
        rows.append([0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M])
    for _ in range(n_bikes):
        rows.append([1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M])
    return np.array(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# compute_route_cost
# ---------------------------------------------------------------------------

class TestComputeRouteCostTruck:
    """Truck: 50000m, 10800s, 2 reloads."""

    def test_truck_cost(self):
        vehicles = _make_vehicles(1, 0)
        cost = compute_route_cost(50000, 10800, 2, VEH_TRUCK, 0, vehicles, 0)

        exp_dist = 50000 * TRUCK_COST_PER_M
        exp_time = 10800 * TRUCK_DRIVER_SEC
        exp_reload = 2 * RELOAD_HANDLING_COST
        exp_total = exp_dist + exp_time + TRUCK_FIXED_DAY + TRUCK_DEPLOY_COST + exp_reload
        assert cost == exp_total


class TestComputeRouteCostBike:
    """Bike: 30000m, 14400s, 1 reload."""

    def test_bike_cost(self):
        vehicles = _make_vehicles(0, 1)
        cost = compute_route_cost(30000, 14400, 1, VEH_BIKE, 0, vehicles, 0)

        exp_dist = 30000 * BIKE_COST_PER_M
        exp_time = 14400 * BIKE_DRIVER_SEC
        exp_reload = 1 * RELOAD_HANDLING_COST
        exp_total = exp_dist + exp_time + BIKE_FIXED_DAY + BIKE_DEPLOY_COST + exp_reload
        assert cost == exp_total


class TestComputeRouteCostZero:
    """Empty route: 0m, 0s -> still pays fixed cost."""

    def test_zero_distance_truck(self):
        vehicles = _make_vehicles(1, 0)
        cost = compute_route_cost(0, 0, 0, VEH_TRUCK, 0, vehicles, 0)
        assert cost == TRUCK_FIXED_DAY + TRUCK_DEPLOY_COST


# ---------------------------------------------------------------------------
# compute_makespan
# ---------------------------------------------------------------------------

class TestComputeMakespan:
    def test_trucks_and_bikes(self):
        assert compute_makespan(np.array([100, 200], dtype=np.int64),
                                np.array([150], dtype=np.int64)) == 200

    def test_bike_is_latest(self):
        assert compute_makespan(np.array([100], dtype=np.int64),
                                np.array([300, 50], dtype=np.int64)) == 300

    def test_all_zeros(self):
        assert compute_makespan(np.array([0, 0], dtype=np.int64),
                                np.array([0], dtype=np.int64)) == 0

    def test_single_truck_no_bikes(self):
        assert compute_makespan(np.array([10800], dtype=np.int64),
                                np.array([], dtype=np.int64)) == 10800


# ---------------------------------------------------------------------------
# compute_penalties
# ---------------------------------------------------------------------------

class TestComputePenalties:

    def test_no_violations(self):
        pw = make_penalty_weights()
        unserved = np.array([], dtype=np.int32)
        cap_viol = np.zeros((0, 4), dtype=np.int64)
        tw_viol = np.zeros((0, 5), dtype=np.int64)
        sync_viol = np.zeros((0, 4), dtype=np.int64)
        return_times = np.array([], dtype=np.int64)

        dummy_custs = np.zeros((1, 7), dtype=np.int64)
        dummy_dm = np.zeros((2, 2), dtype=np.int64)
        dummy_vehs = _make_vehicles(1, 0)
        total, parts = compute_penalties(
            unserved, 0, 0,
            cap_viol, 0, tw_viol, 0,
            sync_viol, 0, 0,
            return_times, 0, pw, dummy_custs, dummy_dm, dummy_vehs,
        )
        assert total == 0
        assert np.all(parts == 0)

    def test_two_unserved_penalty(self):
        pw = make_penalty_weights()
        unserved = np.array([0, 1], dtype=np.int32)
        custs = np.zeros((5, 7), dtype=np.int64)
        custs[:, 2] = 50000  # demand 50kg
        custs[:, 5] = 300    # service
        dm = np.ones((6, 6), dtype=np.int64) * 5000
        np.fill_diagonal(dm, 0)
        vehs = _make_vehicles(1, 1)

        total, parts = compute_penalties(
            unserved, 2, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            np.array([], dtype=np.int64), 0, pw, custs, dm, vehs,
        )
        assert parts[0] > 0
        assert total > 0


# ---------------------------------------------------------------------------
# compute_fitness
# ---------------------------------------------------------------------------

class TestComputeFitness:
    def test_basic_fitness(self):
        assert compute_fitness(1000, 0, 0) == 1000

    def test_penalty_only(self):
        assert compute_fitness(0, 0, 5000) == 5000

    def test_all_nonzero(self):
        assert compute_fitness(10000, 500, 2000) == 12500

    def test_sync_cost_included(self):
        assert compute_fitness(100, 200, 300) == 600


# ---------------------------------------------------------------------------
# evaluate_solution -- integration
# ---------------------------------------------------------------------------

class TestEvaluateSolution:

    def test_simple_valid_solution(self, tiny_instance, tiny_dist_matrix):
        """Truck delivers C0, C1. Bike0 delivers C2, C3. Bike1 delivers C4."""
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[
                [(2, ACT_DELIVER), (3, ACT_DELIVER)],
                [(4, ACT_DELIVER)],
            ],
        )
        pw = make_penalty_weights()

        ev = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )

        assert ev[EV_FEASIBLE] == 1
        assert ev[EV_TOTAL_PENALTY] == 0
        assert ev[EV_SYNC_COST] == 0
        assert ev[EV_COST] > 0
        assert ev[EV_MAKESPAN] > 0

    def test_deterministic(self, tiny_instance, tiny_dist_matrix):
        sol = make_sol_from_routes(
            n_trucks=1, n_bikes=2, n_customers=5,
            truck_routes=[[(0, ACT_DELIVER), (1, ACT_DELIVER)]],
            bike_routes=[
                [(2, ACT_DELIVER), (3, ACT_DELIVER)],
                [(4, ACT_DELIVER)],
            ],
        )
        pw = make_penalty_weights()

        ev1 = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )
        ev2 = evaluate_solution(
            sol, tiny_instance["customers"],
            tiny_instance["vehicles"], tiny_dist_matrix, 900, pw,
        )

        assert ev1[EV_FITNESS] == ev2[EV_FITNESS]
        assert ev1[EV_COST] == ev2[EV_COST]
        assert ev1[EV_MAKESPAN] == ev2[EV_MAKESPAN]
