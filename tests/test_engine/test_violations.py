"""QA tests for src/engine/violations.py -- penalty computation. All i64.

API: compute_penalties(unserved, n_unserved, n_duplicates,
                       cap_viol, n_cap, tw_viol, n_tw,
                       sync_viol, n_sync, n_vr,
                       return_times, n_return_times,
                       penalty_w, customers, dist_matrix, vehicles)
     -> (i64 total, i64[7] parts)
     parts: [unserved, duplicate, capacity, tw, sync, restriction, overtime]
"""
import numpy as np
import pytest

from src.engine.violations import calc_unserved_penalty, compute_penalties
from src.data.cost import (
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    TRUCK_FIXED_DAY, BIKE_FIXED_DAY,
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    PENALTY_OVERLOAD_PER_PCT,
    PENALTY_OVERTIME,
    DAY_LENGTH,
    BIKE_DRIVER_SEC, BIKE_DEPLOY_COST,
    PENALTY_UNSERVED_MULTIPLIER,
)
from src.data.constants import COL_DEMAND, PW_SIZE
from tests.test_engine.conftest import make_penalty_weights


def _make_vehicles(n_trucks=1, n_bikes=1):
    rows = []
    for _ in range(n_trucks):
        rows.append([0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M])
    for _ in range(n_bikes):
        rows.append([1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M])
    return np.array(rows, dtype=np.int64)


_dc = np.zeros((1, 7), dtype=np.int64)
_dd = np.ones((2, 2), dtype=np.int64) * 1000
np.fill_diagonal(_dd, 0)
_dv = _make_vehicles(1, 1)


def _empty_viols():
    """Return empty violation arrays for compute_penalties."""
    return (
        np.array([], dtype=np.int32),     # unserved
        0,                                  # n_unserved
        0,                                  # n_duplicates
        np.zeros((0, 4), dtype=np.int64),  # cap_viol
        0,                                  # n_cap
        np.zeros((0, 5), dtype=np.int64),  # tw_viol
        0,                                  # n_tw
        np.zeros((0, 4), dtype=np.int64),  # sync_viol
        0,                                  # n_sync
        0,                                  # n_vr
        np.array([], dtype=np.int64),      # return_times
        0,                                  # n_return_times
    )


# ---------------------------------------------------------------------------
# No violations
# ---------------------------------------------------------------------------

class TestComputePenaltiesNoViolations:
    def test_empty_report_returns_zero(self):
        pw = make_penalty_weights()
        args = _empty_viols()
        total, parts = compute_penalties(*args, pw, _dc, _dd, _dv)
        assert total == 0
        assert np.all(parts == 0)

    def test_parts_has_7_elements(self):
        pw = make_penalty_weights()
        args = _empty_viols()
        total, parts = compute_penalties(*args, pw, _dc, _dd, _dv)
        assert len(parts) == 7

    def test_duplicates_penalty(self):
        pw = make_penalty_weights()
        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 2,
            np.zeros((0, 4), dtype=np.int64), 0,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            np.array([], dtype=np.int64), 0,
            pw, _dc, _dd, _dv,
        )
        expected = 2 * 500_000
        assert total == expected
        assert total == parts.sum()


# ---------------------------------------------------------------------------
# Unserved penalty
# ---------------------------------------------------------------------------

class TestUnservedPenalty:
    def test_unserved_penalty_far_customer_larger_than_near(self, tiny_instance, tiny_dist_matrix):
        customers = tiny_instance["customers"]
        vehicles = tiny_instance["vehicles"]
        max_dist = int(tiny_dist_matrix[0, 1:].max())
        penalty_c0 = calc_unserved_penalty(0, customers, tiny_dist_matrix, max_dist, vehicles)
        penalty_c1 = calc_unserved_penalty(1, customers, tiny_dist_matrix, max_dist, vehicles)
        # C0 is at (0,5000), C1 at (10000,5000). From depot (5000,5000):
        # dist to C0 = 5000, dist to C1 = 5000. Both same distance so similar penalty.
        # But service times differ: C0=900s, C1=300s -> C0 penalty > C1
        assert penalty_c0 > penalty_c1

    def test_unserved_penalty_in_compute_penalties(self):
        customers = np.array([
            [0, 5000, 100000, 0, 7200, 900, 0],
            [10000, 5000, 20000, 3600, 10800, 300, 0],
        ], dtype=np.int64)
        dm = np.array([
            [0, 5000, 5000],
            [5000, 0, 10000],
            [5000, 10000, 0],
        ], dtype=np.int64)
        vehs = _make_vehicles(1, 1)

        pw = make_penalty_weights()
        unserved = np.array([0], dtype=np.int32)
        max_dist = int(dm[0, 1:].max())

        total, parts = compute_penalties(
            unserved, 1, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            np.array([], dtype=np.int64), 0,
            pw, customers, dm, vehs,
        )
        expected_unserved = calc_unserved_penalty(0, customers, dm, max_dist, vehs)
        assert parts[0] == expected_unserved
        assert total == expected_unserved


# ---------------------------------------------------------------------------
# Capacity penalty
# ---------------------------------------------------------------------------

class TestCapacityPenalty:
    def test_no_capacity_violations_zero_penalty(self):
        pw = make_penalty_weights()
        args = _empty_viols()
        total, parts = compute_penalties(*args, pw, _dc, _dd, _dv)
        assert parts[2] == 0

    def test_bike_capacity_penalty_10pct_overload(self):
        pw = make_penalty_weights()
        load = BIKE_CAPACITY_G + BIKE_CAPACITY_G // 10  # 10% over = 66000
        cap_viol = np.array([[1, 0, 5, load]], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            cap_viol, 1,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            np.array([], dtype=np.int64), 0,
            pw, _dc, _dd, _dv,
        )
        # overload_pct_x100 = (66000-60000)*10000/60000 = 1000
        # penalty = 1000 * BIKE_FIXED_DAY * PENALTY_OVERLOAD_PER_PCT / 10000
        overload_pct_x100 = (load - BIKE_CAPACITY_G) * 10000 // BIKE_CAPACITY_G
        expected = overload_pct_x100 * BIKE_FIXED_DAY * PENALTY_OVERLOAD_PER_PCT // 10000
        assert parts[2] == expected

    def test_truck_capacity_penalty_5pct_overload(self):
        pw = make_penalty_weights()
        load = TRUCK_CAPACITY_G + TRUCK_CAPACITY_G // 20  # 5% over = 2100000
        cap_viol = np.array([[0, 0, 3, load]], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            cap_viol, 1,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            np.array([], dtype=np.int64), 0,
            pw, _dc, _dd, _dv,
        )
        overload_pct_x100 = (load - TRUCK_CAPACITY_G) * 10000 // TRUCK_CAPACITY_G
        expected = overload_pct_x100 * TRUCK_FIXED_DAY * PENALTY_OVERLOAD_PER_PCT // 10000
        assert parts[2] == expected


# ---------------------------------------------------------------------------
# Overtime penalty
# ---------------------------------------------------------------------------

class TestOvertimePenalty:
    def test_no_overtime_returns_zero(self):
        pw = make_penalty_weights()
        return_times = np.array([100, 200, 300], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            return_times, 3,
            pw, _dc, _dd, _dv,
        )
        assert parts[6] == 0

    def test_overtime_600s_one_vehicle(self):
        pw = make_penalty_weights()
        return_times = np.array([DAY_LENGTH + 600], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            return_times, 1,
            pw, _dc, _dd, _dv,
        )
        expected = 600 * PENALTY_OVERTIME
        assert parts[6] == expected

    def test_overtime_accumulates_across_vehicles(self):
        pw = make_penalty_weights()
        return_times = np.array([
            DAY_LENGTH + 300, DAY_LENGTH + 600, DAY_LENGTH + 900,
        ], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            return_times, 3,
            pw, _dc, _dd, _dv,
        )
        expected = (300 + 600 + 900) * PENALTY_OVERTIME
        assert parts[6] == expected

    def test_overtime_exactly_at_daylength_no_penalty(self):
        pw = make_penalty_weights()
        return_times = np.array([DAY_LENGTH], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            np.zeros((0, 5), dtype=np.int64), 0,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            return_times, 1,
            pw, _dc, _dd, _dv,
        )
        assert parts[6] == 0


# ---------------------------------------------------------------------------
# Combined violations
# ---------------------------------------------------------------------------

class TestCombinedViolations:
    def test_all_violation_types_sum_to_total(self):
        pw = make_penalty_weights()

        bike_load = BIKE_CAPACITY_G + BIKE_CAPACITY_G // 10
        cap_viol = np.array([[1, 0, 5, bike_load]], dtype=np.int64)
        tw_viol = np.array([[0, 0, 2, 30000, 25000]], dtype=np.int64)  # 5000s late
        sync_viol = np.array([[0, 6000, 12000, 6000]], dtype=np.int64)
        return_times = np.array([DAY_LENGTH + 1200], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 1,  # 1 duplicate
            cap_viol, 1,
            tw_viol, 1,
            sync_viol, 1,
            2,  # n_vr = 2
            return_times, 1,
            pw, _dc, _dd, _dv,
        )

        overload_pct_x100 = (bike_load - BIKE_CAPACITY_G) * 10000 // BIKE_CAPACITY_G
        expected = (
            500_000 +  # duplicate
            overload_pct_x100 * BIKE_FIXED_DAY * PENALTY_OVERLOAD_PER_PCT // 10000 +
            5000 * 10_000 +  # time_window
            1 * 100_000 +    # sync
            2 * 200_000 +    # vehicle_restriction
            1200 * PENALTY_OVERTIME  # overtime
        )
        assert total == expected
        assert total == parts.sum()


# ---------------------------------------------------------------------------
# Bug detection
# ---------------------------------------------------------------------------

class TestBugDetection:
    def test_time_window_violation_late_arrival(self):
        """TW: arrive=24000, tw_close=21000 -> 3000s late."""
        pw = make_penalty_weights()
        tw_viol = np.array([[0, 0, 2, 24000, 21000]], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            tw_viol, 1,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            np.array([], dtype=np.int64), 0,
            pw, _dc, _dd, _dv,
        )
        expected = 3000 * 10_000
        assert parts[3] == expected

    def test_early_arrival_produces_zero_penalty(self):
        """arrive=18000, tw_close=24000 -> max(0, -6000)=0."""
        pw = make_penalty_weights()
        tw_viol = np.array([[0, 0, 2, 18000, 24000]], dtype=np.int64)

        total, parts = compute_penalties(
            np.array([], dtype=np.int32), 0, 0,
            np.zeros((0, 4), dtype=np.int64), 0,
            tw_viol, 1,
            np.zeros((0, 4), dtype=np.int64), 0, 0,
            np.array([], dtype=np.int64), 0,
            pw, _dc, _dd, _dv,
        )
        assert parts[3] == 0
