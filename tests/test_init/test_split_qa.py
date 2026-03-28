"""QA tests for src/init/split.py edge cases and boundary conditions.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.split import (
    split_to_trips,
    group_trips_to_trucks,
    _compute_route_distance,
    _estimate_trip_time,
)
from src.init.split_bike import split_bike_gt, _split_gt_segments
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE,
)
from src.data.cost import (
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M, DAY_LENGTH,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

def _simple_dist_matrix(coords_with_depot):
    coords_f = coords_with_depot.astype(np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _make_stop(node, stype, demand):
    return {"node": node, "type": stype, "demand": demand}


def _make_customers_i64(n, demand_g=10000):
    """Create i64 (N,7) customers equally spaced on a line."""
    customers = np.zeros((n, 7), dtype=np.int64)
    for i in range(n):
        customers[i, 0] = i * 1000
        customers[i, 2] = demand_g
        customers[i, 4] = 28800  # tw_close_s
        customers[i, 5] = 300    # service_s
    return customers


# ---------------------------------------------------------------------------
# TEST: split_to_trips WITH HUGE CUSTOMER (demand > capacity)
# ---------------------------------------------------------------------------
class TestSplitToTripsHugeCustomer:

    def test_huge_customer_demand_exceeds_capacity_is_skipped(self):
        customers = np.array([
            [1000, 0, 5000000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [_make_stop(0, "deliver", 5000000)]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 0

    def test_huge_first_customer_then_normal_customers(self):
        customers = np.array([
            [1000, 0, 5000000, 0, 28800, 300, 0],
            [2000, 0, 100000, 0, 28800, 300, 0],
            [3000, 0, 100000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 5000000),
            _make_stop(1, "deliver", 100000),
            _make_stop(2, "deliver", 100000),
        ]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 1
        route_nodes = set(routes[0]["stops"].tolist())
        assert route_nodes == {1, 2}

    def test_huge_customer_in_middle_of_sequence(self):
        customers = np.array([
            [1000, 0, 100000, 0, 28800, 300, 0],
            [2000, 0, 5000000, 0, 28800, 300, 0],
            [3000, 0, 100000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 100000),
            _make_stop(1, "deliver", 5000000),
            _make_stop(2, "deliver", 100000),
        ]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 2
        all_nodes = set()
        for r in routes:
            all_nodes.update(r["stops"].tolist())
        assert all_nodes == {0, 2}


# ---------------------------------------------------------------------------
# TEST: _split_gt_segments WITH EMPTY DELIVERS (only reloads)
# ---------------------------------------------------------------------------
class TestSplitGtSegmentsNoDelivers:

    def test_gt_with_three_reload_nodes_only(self):
        giant_tour = [
            _make_stop(0, "reload", 0),
            _make_stop(1, "reload", 0),
            _make_stop(2, "reload", 0),
        ]
        segments = _split_gt_segments(giant_tour)
        assert len(segments) >= 0
        for reload_node, delivers in segments:
            assert len(delivers) == 0

    def test_gt_with_single_reload_no_delivers(self):
        giant_tour = [_make_stop(0, "reload", 0)]
        segments = _split_gt_segments(giant_tour)
        if len(segments) > 0:
            reload_node, delivers = segments[0]
            assert len(delivers) == 0


# ---------------------------------------------------------------------------
# TEST: split_bike_gt WITH ZERO BIKES
# ---------------------------------------------------------------------------
class TestSplitBikeGtZeroBikes:

    def test_split_bike_gt_with_zero_bikes(self):
        customers = _make_customers_i64(3, demand_g=10000)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "reload", 0),
            _make_stop(1, "deliver", 10000),
            _make_stop(2, "deliver", 10000),
        ]
        routes = split_bike_gt(giant_tour, customers, dm,
                               capacity=BIKE_CAPACITY_G,
                               speed_us=BIKE_SPEED_US_PER_M, n_bikes=0)
        assert len(routes) == 0


# ---------------------------------------------------------------------------
# TEST: group_trips_to_trucks WITH ZERO TRIPS
# ---------------------------------------------------------------------------
class TestGroupTripsToTrucksZeroTrips:

    def test_group_trips_with_zero_trips(self):
        customers = _make_customers_i64(2)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        result = group_trips_to_trucks([], n_trucks=3,
                                        customers=customers, dist_matrix=dm)
        assert len(result) == 0

    def test_group_trips_zero_trucks_with_trips(self):
        customers = np.array([
            [1000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        trips = [{
            "stops": np.array([0], dtype=np.int32),
            "actions": np.array([ACT_DELIVER], dtype=np.int8),
            "total_demand": 10000,
            "total_distance": 2000,
        }]
        result = group_trips_to_trucks(trips, n_trucks=0,
                                        customers=customers, dist_matrix=dm)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TEST: empty GT edge cases
# ---------------------------------------------------------------------------
class TestEmptyGiantTour:

    def test_split_to_trips_empty_gt(self):
        customers = _make_customers_i64(1)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        routes = split_to_trips([], customers, dm, capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 0

    def test_split_bike_gt_empty_gt(self):
        customers = _make_customers_i64(1)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        routes = split_bike_gt([], customers, dm,
                               capacity=BIKE_CAPACITY_G,
                               speed_us=BIKE_SPEED_US_PER_M, n_bikes=2)
        assert len(routes) == 0


# ---------------------------------------------------------------------------
# TEST: trip time exceeds DAY_LENGTH
# ---------------------------------------------------------------------------
class TestTripTimeExceedsDAYLength:

    def test_split_to_trips_single_trip_exceeds_daylength(self):
        """Very far customer -> trip time > DAY_LENGTH -> skipped."""
        customers = np.array([
            [1000000, 0, 50000, 0, 36000, 300, 0],  # 1000km from depot
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [_make_stop(0, "deliver", 50000)]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 0


# ---------------------------------------------------------------------------
# BOUNDARY: Capacity exactly at limit
# ---------------------------------------------------------------------------
class TestCapacityBoundary:

    def test_split_to_trips_demand_exactly_at_capacity(self):
        customers = np.array([
            [1000, 0, 100000, 0, 28800, 300, 0],
            [2000, 0, 100000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 100000),
            _make_stop(1, "deliver", 100000),
        ]
        routes = split_to_trips(giant_tour, customers, dm, capacity=200000)
        assert len(routes) == 1
        route_nodes = set(routes[0]["stops"].tolist())
        assert route_nodes == {0, 1}

    def test_split_to_trips_demand_exceeds_capacity_by_one(self):
        customers = np.array([
            [1000, 0, 100000, 0, 28800, 300, 0],
            [2000, 0, 100000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 100000),
            _make_stop(1, "deliver", 100000),
        ]
        routes = split_to_trips(giant_tour, customers, dm, capacity=199999)
        assert len(routes) >= 2


# ---------------------------------------------------------------------------
# SANITY: Array types and shapes
# ---------------------------------------------------------------------------
class TestArrayTypes:

    def test_split_to_trips_returns_int32_stops(self):
        customers = np.array([
            [1000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [_make_stop(0, "deliver", 10000)]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 1
        assert routes[0]["stops"].dtype == np.int32
        assert routes[0]["actions"].dtype == np.int8
