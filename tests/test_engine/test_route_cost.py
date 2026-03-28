"""Tests for src/engine/route_cost.py -- cost computation and makespan. All i64.

API: compute_route_cost(total_dist_m, total_time_s, n_reloads, vtype, total_wait_s, vehicles, vid) -> i64
     count_reloads_3d(sim_states, vid, length)
     total_wait_time_3d(sim_states, vid, length)
"""
import numpy as np
import pytest

from src.engine.route_cost import (
    compute_route_cost, compute_makespan, count_reloads_3d, total_wait_time_3d,
)
from src.data.constants import (
    ACT_RELOAD, ACT_DELIVER, ST_ACTION, ST_WAIT, ST_COLS, VEH_TRUCK, VEH_BIKE,
)
from src.data.cost import (
    TRUCK_COST_PER_M, TRUCK_DRIVER_SEC, TRUCK_FIXED_DAY, TRUCK_DEPLOY_COST,
    TRUCK_WAIT_COST_SEC,
    BIKE_COST_PER_M, BIKE_DRIVER_SEC, BIKE_FIXED_DAY, BIKE_DEPLOY_COST,
    BIKE_WAIT_COST_SEC,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    RELOAD_HANDLING_COST,
)


def _make_vehicles(n_trucks=1, n_bikes=0):
    rows = []
    for _ in range(n_trucks):
        rows.append([0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M])
    for _ in range(n_bikes):
        rows.append([1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M])
    return np.array(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# TEST 1: empty state
# ---------------------------------------------------------------------------

class TestRouteCostEmptyState:

    def test_empty_truck_route_pays_fixed_costs(self):
        vehicles = _make_vehicles(1, 0)
        cost = compute_route_cost(0, 0, 0, VEH_TRUCK, 0, vehicles, 0)
        expected = TRUCK_FIXED_DAY + TRUCK_DEPLOY_COST
        assert cost == expected

    def test_empty_bike_route_pays_fixed_costs(self):
        vehicles = _make_vehicles(0, 1)
        cost = compute_route_cost(0, 0, 0, VEH_BIKE, 0, vehicles, 0)
        expected = BIKE_FIXED_DAY + BIKE_DEPLOY_COST
        assert cost == expected


# ---------------------------------------------------------------------------
# TEST 2: distance cost
# ---------------------------------------------------------------------------

class TestDistanceCostFormula:

    def test_truck_distance_cost_50km(self):
        vehicles = _make_vehicles(1, 0)
        cost = compute_route_cost(50000, 0, 0, VEH_TRUCK, 0, vehicles, 0)
        exp_dist = 50000 * TRUCK_COST_PER_M
        assert cost == exp_dist + TRUCK_FIXED_DAY + TRUCK_DEPLOY_COST

    def test_bike_distance_cost_30km(self):
        vehicles = _make_vehicles(0, 1)
        cost = compute_route_cost(30000, 0, 0, VEH_BIKE, 0, vehicles, 0)
        exp_dist = 30000 * BIKE_COST_PER_M
        assert cost == exp_dist + BIKE_FIXED_DAY + BIKE_DEPLOY_COST


# ---------------------------------------------------------------------------
# TEST 3: wait cost non-negative
# ---------------------------------------------------------------------------

class TestWaitCostNonNegative:

    def test_truck_wait_cost_zero(self):
        vehicles = _make_vehicles(1, 0)
        cost_no_wait = compute_route_cost(50000, 10800, 0, VEH_TRUCK, 0, vehicles, 0)
        cost_with_wait = compute_route_cost(50000, 10800, 0, VEH_TRUCK, 1800, vehicles, 0)
        wait_cost = cost_with_wait - cost_no_wait
        assert wait_cost == 1800 * TRUCK_WAIT_COST_SEC
        assert wait_cost >= 0

    def test_bike_wait_cost_900s(self):
        vehicles = _make_vehicles(0, 1)
        cost_no_wait = compute_route_cost(30000, 7200, 0, VEH_BIKE, 0, vehicles, 0)
        cost_with_wait = compute_route_cost(30000, 7200, 0, VEH_BIKE, 900, vehicles, 0)
        wait_cost = cost_with_wait - cost_no_wait
        assert wait_cost == 900 * BIKE_WAIT_COST_SEC


# ---------------------------------------------------------------------------
# TEST 4: makespan empty
# ---------------------------------------------------------------------------

class TestMakespanAllEmpty:
    def test_makespan_both_arrays_empty(self):
        truck_rt = np.array([], dtype=np.int64)
        bike_rt = np.array([], dtype=np.int64)
        assert compute_makespan(truck_rt, bike_rt) == 0


# ---------------------------------------------------------------------------
# TEST 5: makespan mixed
# ---------------------------------------------------------------------------

class TestMakespanMixedVehicles:
    def test_makespan_truck_max(self):
        assert compute_makespan(np.array([100, 200], dtype=np.int64),
                                np.array([150], dtype=np.int64)) == 200

    def test_makespan_bike_max(self):
        assert compute_makespan(np.array([100], dtype=np.int64),
                                np.array([150, 300, 50], dtype=np.int64)) == 300

    def test_makespan_all_zeros(self):
        assert compute_makespan(np.array([0, 0], dtype=np.int64),
                                np.array([0], dtype=np.int64)) == 0

    def test_makespan_only_trucks(self):
        assert compute_makespan(np.array([100, 250, 50], dtype=np.int64),
                                np.array([], dtype=np.int64)) == 250

    def test_makespan_only_bikes(self):
        assert compute_makespan(np.array([], dtype=np.int64),
                                np.array([150, 300, 75], dtype=np.int64)) == 300


# ---------------------------------------------------------------------------
# TEST 6: reload cost
# ---------------------------------------------------------------------------

class TestReloadCostFormula:
    def test_reload_cost_zero_reloads(self):
        vehicles = _make_vehicles(1, 0)
        c0 = compute_route_cost(50000, 10800, 0, VEH_TRUCK, 0, vehicles, 0)
        c1 = compute_route_cost(50000, 10800, 1, VEH_TRUCK, 0, vehicles, 0)
        assert c1 - c0 == RELOAD_HANDLING_COST

    def test_reload_cost_five_reloads(self):
        vehicles = _make_vehicles(1, 0)
        c0 = compute_route_cost(50000, 10800, 0, VEH_TRUCK, 0, vehicles, 0)
        c5 = compute_route_cost(50000, 10800, 5, VEH_TRUCK, 0, vehicles, 0)
        assert c5 - c0 == 5 * RELOAD_HANDLING_COST


# ---------------------------------------------------------------------------
# SUPPLEMENTARY: count_reloads_3d and total_wait_time_3d
# ---------------------------------------------------------------------------

class TestCountReloads3d:

    def test_count_reloads_empty(self):
        sim = np.zeros((1, 5, ST_COLS), dtype=np.int64)
        assert count_reloads_3d(sim, 0, 0) == 0

    def test_count_reloads_no_reloads(self):
        sim = np.zeros((1, 3, ST_COLS), dtype=np.int64)
        sim[0, :, ST_ACTION] = ACT_DELIVER
        assert count_reloads_3d(sim, 0, 3) == 0

    def test_count_reloads_two_reloads(self):
        sim = np.zeros((1, 5, ST_COLS), dtype=np.int64)
        sim[0, :, ST_ACTION] = ACT_DELIVER
        sim[0, 1, ST_ACTION] = ACT_RELOAD
        sim[0, 3, ST_ACTION] = ACT_RELOAD
        assert count_reloads_3d(sim, 0, 5) == 2


class TestTotalWaitTime3d:

    def test_total_wait_time_empty(self):
        sim = np.zeros((1, 5, ST_COLS), dtype=np.int64)
        assert total_wait_time_3d(sim, 0, 0) == 0

    def test_total_wait_time_no_wait(self):
        sim = np.zeros((1, 3, ST_COLS), dtype=np.int64)
        assert total_wait_time_3d(sim, 0, 3) == 0

    def test_total_wait_time_sum(self):
        sim = np.zeros((1, 3, ST_COLS), dtype=np.int64)
        sim[0, 0, ST_WAIT] = 500
        sim[0, 1, ST_WAIT] = 1000
        sim[0, 2, ST_WAIT] = 300
        assert total_wait_time_3d(sim, 0, 3) == 1800
