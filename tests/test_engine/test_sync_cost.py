"""QA tests for src/engine/sync_cost.py -- truck-bike sync at satellites. All i64.

API: compute_sync_cost(truck_sim, truck_lengths, bike_sim, bike_lengths,
                       satellites, n_satellites, delta_t_s) -> i64 scalar
     _find_depart_3d(sim, lengths, vid, cust) -> i64
     _find_arrive_3d(sim, lengths, vid, cust) -> i64
"""
import numpy as np
import pytest

from src.engine.sync_cost import compute_sync_cost, _find_depart_3d, _find_arrive_3d
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD,
    ST_ACTION, ST_CUST, ST_DEPART, ST_ARRIVE,
    ST_COLS, MAX_SATELLITES,
)
from src.data.cost import SYNC_DELTA_T, PENALTY_MISSING_RELOAD, PENALTY_SYNC_GAP_SEC


def _make_3d_sim(states_2d_list, max_len=10):
    """Build 3D i64 sim array from list of 2D state arrays."""
    n_vehicles = len(states_2d_list)
    sim = np.zeros((n_vehicles, max_len, ST_COLS), dtype=np.int64)
    lengths = np.zeros(n_vehicles, dtype=np.int32)
    for i, s in enumerate(states_2d_list):
        L = len(s)
        lengths[i] = L
        if L > 0:
            sim[i, :L, :] = s
    return sim, lengths


def _make_sats(rows_list):
    """Build satellites array i64 from list of (cust, bike, truck, grams, time_s)."""
    sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)
    for i, row in enumerate(rows_list):
        sats[i] = row
    return sats, len(rows_list)


class TestZeroSatellites:

    def test_empty_satellites(self):
        truck_sim, truck_lengths = _make_3d_sim([np.array([]).reshape(0, ST_COLS)])
        bike_sim, bike_lengths = _make_3d_sim([np.array([]).reshape(0, ST_COLS)])
        sats = np.zeros((MAX_SATELLITES, 5), dtype=np.int64)

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, 0, SYNC_DELTA_T,
        )
        assert total == 0


class TestPerfectSync:

    def test_truck_and_bike_same_time(self):
        truck_state = np.array([
            [10, ACT_DELIVER, 3000, 0, 3000, 300, 3300, 1000000, 900000, 1],
            [5, ACT_RELOAD, 5400, 600, 6000, 300, 6000, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [10, ACT_DELIVER, 2700, 0, 2700, 300, 3000, 50000, 40000, 1],
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == 0


class TestSyncGapExceedsDelta:

    def test_gap_exceeds_delta_t(self):
        # truck departs at 7200, bike arrives at 5400 -> gap=1800 > 900
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 1200, 7200, 300, 7200, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 5400, 0, 5400, 0, 5400, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 5400)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )

        expected = PENALTY_SYNC_GAP_SEC * (1800 - SYNC_DELTA_T)
        assert total == expected

    def test_gap_exactly_at_delta_t_boundary(self):
        # gap = SYNC_DELTA_T = 900 -> no penalty
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 900, 6900, 300, 6900, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == 0


class TestMissingTruckAtSatellite:

    def test_truck_not_found(self):
        truck_state = np.array([
            [10, ACT_DELIVER, 3000, 0, 3000, 300, 3300, 1000000, 900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == PENALTY_MISSING_RELOAD

    def test_truck_state_empty(self):
        truck_sim, truck_lengths = _make_3d_sim([np.array([]).reshape(0, ST_COLS)])
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == PENALTY_MISSING_RELOAD


class TestMissingBikeAtSatellite:

    def test_bike_not_found(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 1200, 7200, 300, 7200, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [10, ACT_DELIVER, 3000, 0, 3000, 300, 3300, 50000, 40000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 5400)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == PENALTY_MISSING_RELOAD

    def test_bike_state_empty(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 1200, 7200, 300, 7200, 2000000, 1900000, 1],
        ], dtype=np.int64)
        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([np.array([]).reshape(0, ST_COLS)])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 5400)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == PENALTY_MISSING_RELOAD


class TestMultipleSatellites:

    def test_three_satellites_mixed(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 300, 6000, 2000000, 1900000, 1],
            [7, ACT_RELOAD, 9000, 0, 9000, 300, 9000, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
            [7, ACT_RELOAD, 7200, 0, 7200, 0, 7200, 60000, 60000, 1],
            [9, ACT_RELOAD, 12000, 0, 12000, 0, 12000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([
            (5, 0, 0, 100000, 6000),    # perfect sync
            (7, 0, 0, 150000, 7200),    # gap=1800 > 900
            (9, 0, 0, 999000, 12000),   # missing truck
        ])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )

        expected_gap_cost = PENALTY_SYNC_GAP_SEC * (1800 - SYNC_DELTA_T)
        expected_total = expected_gap_cost + PENALTY_MISSING_RELOAD
        assert total == expected_total


class TestNegativeGap:

    def test_bike_early_arrival(self):
        # truck departs at 7200, bike arrives at 6000 -> gap=1200 > 900
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 1200, 7200, 300, 7200, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 120000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        expected = PENALTY_SYNC_GAP_SEC * (1200 - SYNC_DELTA_T)
        assert total == expected


class TestHelperFunctions:

    def test_find_depart_3d_basic(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 1200, 7200, 300, 7200, 2000000, 1900000, 1],
        ], dtype=np.int64)
        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        depart = _find_depart_3d(truck_sim, truck_lengths, vid=0, cust=5)
        assert depart == 7200

    def test_find_depart_3d_not_found(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 1200, 7200, 300, 7200, 2000000, 1900000, 1],
        ], dtype=np.int64)
        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        depart = _find_depart_3d(truck_sim, truck_lengths, vid=0, cust=999)
        assert depart == -1

    def test_find_arrive_3d_basic(self):
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        arrive = _find_arrive_3d(bike_sim, bike_lengths, vid=0, cust=5)
        assert arrive == 6000

    def test_find_arrive_3d_not_found(self):
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        arrive = _find_arrive_3d(bike_sim, bike_lengths, vid=0, cust=999)
        assert arrive == -1


class TestEdgeCases:

    def test_very_large_gap(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 60000, 66000, 300, 66000, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 66000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        expected = PENALTY_SYNC_GAP_SEC * (60000 - SYNC_DELTA_T)
        assert total == expected

    def test_zero_delta_t(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 60, 6060, 300, 6060, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 6060, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, delta_t_s=0,
        )
        expected = PENALTY_SYNC_GAP_SEC * 60
        assert total == expected


class TestTruckDeliverNotReload:

    def test_truck_deliver_not_reload(self):
        truck_state = np.array([
            [5, ACT_DELIVER, 6000, 1200, 7200, 300, 7200, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 120000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == PENALTY_MISSING_RELOAD

    def test_bike_deliver_not_reload(self):
        truck_state = np.array([
            [5, ACT_RELOAD, 6000, 0, 6000, 300, 6000, 2000000, 1900000, 1],
        ], dtype=np.int64)
        bike_state = np.array([
            [5, ACT_DELIVER, 6000, 0, 6000, 0, 6000, 60000, 60000, 1],
        ], dtype=np.int64)

        truck_sim, truck_lengths = _make_3d_sim([truck_state])
        bike_sim, bike_lengths = _make_3d_sim([bike_state])
        sats, n_sats = _make_sats([(5, 0, 0, 100000, 6000)])

        total = compute_sync_cost(
            truck_sim, truck_lengths, bike_sim, bike_lengths,
            sats, n_sats, SYNC_DELTA_T,
        )
        assert total == PENALTY_MISSING_RELOAD
