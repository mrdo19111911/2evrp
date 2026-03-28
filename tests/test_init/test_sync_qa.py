"""QA Tests for src/init/sync.py -- Time synchronization.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.sync import (
    synchronize_times,
    match_reload_events,
    adjust_sync_times,
    estimate_all_arrival_times,
)
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_DEMAND, SAT_CUST, SAT_BIKE, SAT_TRUCK,
    SAT_KG, SAT_TIME,
)
from src.data.cost import (
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_COST_PER_M, BIKE_COST_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
    SYNC_DELTA_T,
)


def _simple_dist_matrix(coords_with_depot):
    coords_f = coords_with_depot.astype(np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _make_vehicles():
    return np.array([
        [0, TRUCK_CAPACITY_G, TRUCK_COST_PER_M, TRUCK_SPEED_US_PER_M],
        [1, BIKE_CAPACITY_G, BIKE_COST_PER_M, BIKE_SPEED_US_PER_M],
    ], dtype=np.int64)


# ---------------------------------------------------------------------------
# Test 1: synchronize_times with 0 satellites
# ---------------------------------------------------------------------------
class TestSynchronizeTimesZeroSatellites:

    def test_synchronize_times_no_reload_stops(self):
        coords = np.array([
            [0, 0], [5000, 0], [10000, 0],
        ], dtype=np.int64)
        dist_matrix = _simple_dist_matrix(coords)

        # customers i64 (N,7) -- note: indices in dist_matrix are offset by 1
        # customer 0 is at dist_matrix[1], customer 1 at dist_matrix[2]
        customers = np.array([
            [5000, 0, 50000, 0, 28800, 300, 0],
            [10000, 0, 30000, 0, 28800, 300, 0],
        ], dtype=np.int64)

        truck_sol = [{
            "stops": np.array([0, 1], dtype=np.int32),
            "actions": np.array([ACT_DELIVER, ACT_DELIVER], dtype=np.int8),
        }]
        bike_sol = [{
            "stops": np.array([0], dtype=np.int32),
            "actions": np.array([ACT_DELIVER], dtype=np.int8),
        }]
        vehicles = _make_vehicles()

        result = synchronize_times(truck_sol, bike_sol, customers,
                                    dist_matrix, vehicles)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 0
        assert result.shape[1] == 5


# ---------------------------------------------------------------------------
# Test 2: match_reload_events with no RELOAD stops on truck
# ---------------------------------------------------------------------------
class TestMatchReloadEventsNoTruckReload:

    def test_match_reload_no_truck_reload_stops(self):
        coords = np.array([
            [0, 0], [5000, 0], [10000, 0], [15000, 0],
        ], dtype=np.int64)
        dist_matrix = _simple_dist_matrix(coords)

        customers = np.array([
            [5000, 0, 50000, 0, 28800, 300, 0],
            [10000, 0, 0, 0, 28800, 300, 0],
            [15000, 0, 30000, 0, 28800, 300, 0],
        ], dtype=np.int64)

        truck_times = [np.array([720, 1440], dtype=np.int64)]
        truck_sol = [{
            "stops": np.array([0, 2], dtype=np.int32),
            "actions": np.array([ACT_DELIVER, ACT_DELIVER], dtype=np.int8),
        }]

        bike_times = [np.array([1440, 2160], dtype=np.int64)]
        bike_sol = [{
            "stops": np.array([1, 2], dtype=np.int32),
            "actions": np.array([ACT_RELOAD, ACT_DELIVER], dtype=np.int8),
            "satellite_node": 1,
        }]

        result = match_reload_events(truck_sol, bike_sol,
                                      truck_times, bike_times, customers)
        assert result.shape[1] == 5

    def test_match_reload_empty_truck_map(self):
        customers = np.array([
            [0, 0, 0, 0, 28800, 600, 0],
            [5000, 0, 50000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        truck_times = []
        truck_sol = []
        bike_times = [np.array([1440, 2160], dtype=np.int64)]
        bike_sol = [{
            "stops": np.array([1], dtype=np.int32),
            "actions": np.array([ACT_RELOAD], dtype=np.int8),
            "satellite_node": 1,
        }]
        result = match_reload_events(truck_sol, bike_sol,
                                      truck_times, bike_times, customers)
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# Test 3: adjust_sync_times convergence
# ---------------------------------------------------------------------------
class TestAdjustSyncTimesConvergence:

    def test_adjust_sync_times_empty_satellites(self):
        satellites = np.zeros((0, 5), dtype=np.int64)
        truck_times = [np.array([], dtype=np.int64)]
        bike_times = [np.array([], dtype=np.int64)]
        result = adjust_sync_times(satellites, truck_times, bike_times,
                                    SYNC_DELTA_T)
        assert result.shape == (0, 5)

    def test_adjust_sync_times_stabilizes(self):
        satellites = np.array([
            [1, 0, 0, 50000, 600],
            [2, 0, 0, 40000, 900],
            [3, 0, 0, 30000, 1200],
        ], dtype=np.int64)
        truck_times = [np.array([300, 720, 1080], dtype=np.int64)]
        bike_times = [np.array([480, 840, 1320], dtype=np.int64)]

        prev_result = satellites.copy()
        converged = False
        for iteration in range(5):
            result = adjust_sync_times(prev_result, truck_times,
                                        bike_times, SYNC_DELTA_T)
            if np.array_equal(result, prev_result):
                converged = True
                break
            prev_result = result

        assert converged
        assert result.shape == satellites.shape
        assert np.all(result[:, SAT_TIME] >= 0)

    def test_adjust_sync_times_preserves_non_time_columns(self):
        satellites = np.array([
            [10, 1, 0, 100000, 1500],
            [20, 1, 0, 80000, 2100],
        ], dtype=np.int64)
        truck_times = [np.array([1200, 1800], dtype=np.int64)]
        bike_times = [np.array([1320, 2160], dtype=np.int64)]
        result = adjust_sync_times(satellites, truck_times, bike_times,
                                    SYNC_DELTA_T)
        np.testing.assert_array_equal(result[:, :SAT_TIME],
                                      satellites[:, :SAT_TIME])

    def test_adjust_sync_times_negative_times_clamped_to_zero(self):
        satellites = np.array([
            [1, 0, 0, 50000, -300],
            [2, 0, 0, 40000, 600],
        ], dtype=np.int64)
        truck_times = [np.array([0, 480], dtype=np.int64)]
        bike_times = [np.array([0, 720], dtype=np.int64)]
        result = adjust_sync_times(satellites, truck_times, bike_times,
                                    SYNC_DELTA_T)
        assert result[0, SAT_TIME] == 0
        assert result[1, SAT_TIME] == 600


# ---------------------------------------------------------------------------
# Test 4: estimate_all_arrival_times edge cases
# ---------------------------------------------------------------------------
class TestEstimateArrivalTimes:

    def test_estimate_all_arrival_times_empty_routes(self):
        dist_matrix = np.zeros((1, 1), dtype=np.int64)
        result = estimate_all_arrival_times([], dist_matrix,
                                            TRUCK_SPEED_US_PER_M)
        assert result == []

    def test_estimate_all_arrival_times_single_stop(self):
        coords = np.array([
            [0, 0], [5000, 0],
        ], dtype=np.int64)
        dist_matrix = _simple_dist_matrix(coords)
        customers = np.array([
            [5000, 0, 50000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        routes = [{
            "stops": np.array([0], dtype=np.int32),
            "actions": np.array([ACT_DELIVER], dtype=np.int8),
        }]
        result = estimate_all_arrival_times(routes, dist_matrix,
                                            TRUCK_SPEED_US_PER_M, customers)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] > 0


# ---------------------------------------------------------------------------
# Integration: Full sync flow
# ---------------------------------------------------------------------------
class TestSyncIntegration:

    def test_full_sync_with_reload_events(self):
        coords = np.array([
            [0, 0], [5000, 0], [10000, 0], [15000, 0],
        ], dtype=np.int64)
        dist_matrix = _simple_dist_matrix(coords)

        customers = np.array([
            [5000, 0, 50000, 0, 28800, 300, 0],
            [10000, 0, 0, 0, 28800, 0, 0],
            [15000, 0, 30000, 0, 28800, 300, 0],
        ], dtype=np.int64)

        truck_sol = [{
            "stops": np.array([0, 1], dtype=np.int32),
            "actions": np.array([ACT_DELIVER, ACT_RELOAD], dtype=np.int8),
        }]
        bike_sol = [{
            "stops": np.array([1, 2], dtype=np.int32),
            "actions": np.array([ACT_RELOAD, ACT_DELIVER], dtype=np.int8),
            "satellite_node": 1,
        }]
        vehicles = _make_vehicles()

        result = synchronize_times(truck_sol, bike_sol, customers,
                                    dist_matrix, vehicles)
        assert result.shape[1] == 5
        if result.shape[0] > 0:
            assert result[0, SAT_CUST] == 1
            assert result[0, SAT_BIKE] == 0
            assert result[0, SAT_TRUCK] == 0
            assert result[0, SAT_TIME] >= 0
