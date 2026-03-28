"""QA tests for src/init/constraints.py -- Route feasibility checks.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.data.constants import (
    COL_DEMAND, COL_TW_CLOSE, ACT_DELIVER, ACT_RELOAD,
)
from src.data.cost import (
    DAY_LENGTH, RELOAD_SERVICE_TIME,
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)
from src.init.constraints import (
    check_trip_capacity,
    check_trip_time_window,
    check_trip_distance,
    check_trip_feasibility,
)


# ============================================================================
# FIXTURE: Minimal test data (all i64)
# ============================================================================

@pytest.fixture
def mock_customers():
    """Mock i64 (N,7): [x_m, y_m, demand_g, tw_open_s, tw_close_s, service_s, restricted]."""
    return np.array([
        [0, 5000, 0, 0, 7200, 600, 0],          # C0: no demand
        [10000, 5000, 20000, 3600, 10800, 300, 0],  # C1: 20kg
        [5000, 10000, 100000, 0, 14400, 900, 0],  # C2: 100kg
        [2000, 0, 10000, 1800, 9000, 300, 0],     # C3: 10kg (was -10, invalid)
        [8000, 0, 60000, 0, 28800, 300, 0],       # C4: 60kg (bike capacity)
    ], dtype=np.int64)


@pytest.fixture
def mock_dist_matrix():
    """Mock i64 (6,6) distance matrix (depot + 5 customers)."""
    return np.array([
        [0, 5000, 5000, 7071, 3606, 3606],
        [5000, 0, 10000, 14142, 5000, 10000],
        [5000, 10000, 0, 7071, 10000, 5000],
        [7071, 14142, 7071, 0, 10000, 6000],
        [3606, 5000, 10000, 10000, 0, 6000],
        [3606, 10000, 5000, 6000, 6000, 0],
    ], dtype=np.int64)


# ============================================================================
# TEST: check_trip_capacity
# ============================================================================

class TestCheckTripCapacity:

    def test_empty_route_capacity_ok(self, mock_customers):
        stops = np.array([], dtype=np.int32)
        actions = np.array([], dtype=np.int8)
        feasible, max_load = check_trip_capacity(stops, actions,
                                                  mock_customers,
                                                  TRUCK_CAPACITY_G)
        assert feasible is True
        assert max_load == 0

    def test_single_delivery_under_capacity(self, mock_customers):
        stops = np.array([1], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        feasible, max_load = check_trip_capacity(stops, actions,
                                                  mock_customers,
                                                  TRUCK_CAPACITY_G)
        assert feasible is True
        assert max_load == 20000

    def test_delivery_exceeds_capacity(self, mock_customers):
        stops = np.array([2], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        feasible, max_load = check_trip_capacity(stops, actions,
                                                  mock_customers,
                                                  BIKE_CAPACITY_G)
        assert feasible is False
        assert max_load == 100000

    def test_reload_resets_load(self, mock_customers):
        stops = np.array([1, 1, 1], dtype=np.int32)
        actions = np.array([ACT_DELIVER, ACT_RELOAD, ACT_DELIVER], dtype=np.int8)
        feasible, max_load = check_trip_capacity(stops, actions,
                                                  mock_customers,
                                                  BIKE_CAPACITY_G)
        assert feasible is True
        assert max_load == 20000


# ============================================================================
# TEST: check_trip_time_window
# ============================================================================

class TestCheckTripTimeWindow:

    def test_empty_route_time_window_ok(self, mock_customers,
                                         mock_dist_matrix):
        stops = np.array([], dtype=np.int32)
        actions = np.array([], dtype=np.int8)
        feasible, n_violations, total_time = check_trip_time_window(
            stops, actions, mock_customers, mock_dist_matrix,
            speed_us=TRUCK_SPEED_US_PER_M, start_node=0)
        assert feasible is True
        assert n_violations == 0
        assert total_time == 0

    def test_single_stop_within_window(self, mock_customers,
                                        mock_dist_matrix):
        stops = np.array([1], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        feasible, n_violations, total_time = check_trip_time_window(
            stops, actions, mock_customers, mock_dist_matrix,
            speed_us=TRUCK_SPEED_US_PER_M, start_node=0)
        assert n_violations == 0


# ============================================================================
# TEST: check_trip_distance
# ============================================================================

class TestCheckTripDistance:

    def test_empty_route_distance_zero(self, mock_dist_matrix):
        stops = np.array([], dtype=np.int32)
        distance = check_trip_distance(stops, mock_dist_matrix, start_node=0)
        assert distance == 0

    def test_single_customer_round_trip(self, mock_dist_matrix):
        stops = np.array([1], dtype=np.int32)
        distance = check_trip_distance(stops, mock_dist_matrix, start_node=0)
        # depot->C1 + C1->depot = 5000 + 5000 = 10000 meters
        assert distance == 10000

    def test_two_customer_route(self, mock_dist_matrix):
        stops = np.array([0, 1], dtype=np.int32)
        distance = check_trip_distance(stops, mock_dist_matrix, start_node=0)
        # depot->C0(5000) + C0->C1(10000) + C1->depot(5000) = 20000
        assert distance == 20000


# ============================================================================
# TEST: check_trip_feasibility (combined)
# ============================================================================

class TestCheckTripFeasibility:

    def test_empty_route_feasibility(self, mock_customers, mock_dist_matrix):
        stops = np.array([], dtype=np.int32)
        actions = np.array([], dtype=np.int8)
        feasible, max_load, n_tw, total_time, distance = \
            check_trip_feasibility(
                stops, actions, mock_customers, mock_dist_matrix,
                TRUCK_CAPACITY_G, TRUCK_SPEED_US_PER_M, start_node=0)
        assert feasible is True
        assert max_load == 0
        assert n_tw == 0
        assert total_time == 0
        assert distance == 0

    def test_route_exceeding_capacity_infeasible(self, mock_customers,
                                                   mock_dist_matrix):
        stops = np.array([2], dtype=np.int32)
        actions = np.array([ACT_DELIVER], dtype=np.int8)
        feasible, max_load, n_tw, total_time, distance = \
            check_trip_feasibility(
                stops, actions, mock_customers, mock_dist_matrix,
                BIKE_CAPACITY_G, BIKE_SPEED_US_PER_M, start_node=0)
        assert feasible is False
        assert max_load > BIKE_CAPACITY_G
