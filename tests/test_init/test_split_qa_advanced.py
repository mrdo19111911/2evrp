"""Advanced QA tests for split.py - catch bugs with i64 interface.

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
    TRUCK_SPEED_US_PER_M, BIKE_SPEED_US_PER_M,
    DAY_LENGTH, RELOAD_SERVICE_TIME,
    TRUCK_CAPACITY_G, BIKE_CAPACITY_G,
)


def _simple_dist_matrix(coords_with_depot):
    coords_f = coords_with_depot.astype(np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _make_stop(node, stype, demand):
    return {"node": node, "type": stype, "demand": demand}


# ---------------------------------------------------------------------------
# TEST: Route dict structure and keys
# ---------------------------------------------------------------------------
class TestRouteStructure:

    def test_split_to_trips_route_has_all_keys(self):
        customers = np.array([
            [1000, 0, 50000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [_make_stop(0, "deliver", 50000)]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 1
        required_keys = {"stops", "actions", "total_demand", "total_distance"}
        assert required_keys.issubset(set(routes[0].keys()))

    def test_split_bike_gt_route_has_all_keys(self):
        customers = np.array([
            [0, 0, 0, 0, 28800, 300, 0],
            [1000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "reload", 0),
            _make_stop(1, "deliver", 10000),
        ]
        routes = split_bike_gt(giant_tour, customers, dm,
                               capacity=BIKE_CAPACITY_G,
                               speed_us=BIKE_SPEED_US_PER_M, n_bikes=1)
        if len(routes) > 0:
            required_keys = {"stops", "actions", "total_demand",
                             "total_distance", "bike_id"}
            assert required_keys.issubset(set(routes[0].keys()))

    def test_group_trips_to_trucks_route_has_truck_id(self):
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
        result = group_trips_to_trucks(trips, n_trucks=2,
                                        customers=customers, dist_matrix=dm)
        assert len(result) == 1
        assert "truck_id" in result[0]


# ---------------------------------------------------------------------------
# TEST: Actions array consistency
# ---------------------------------------------------------------------------
class TestActionSequence:

    def test_split_to_trips_actions_match_stops_length(self):
        customers = np.array([
            [1000, 0, 10000, 0, 28800, 300, 0],
            [2000, 0, 10000, 0, 28800, 300, 0],
            [3000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 10000),
            _make_stop(1, "deliver", 10000),
            _make_stop(2, "deliver", 10000),
        ]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        for i, route in enumerate(routes):
            assert len(route["stops"]) == len(route["actions"])

    def test_split_to_trips_all_actions_are_valid(self):
        customers = np.array([
            [1000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [_make_stop(0, "deliver", 10000)]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        for route in routes:
            for action in route["actions"]:
                assert action in (ACT_DELIVER, ACT_RELOAD)


# ---------------------------------------------------------------------------
# TEST: Distance calculations
# ---------------------------------------------------------------------------
class TestDistanceCalculation:

    def test_compute_route_distance_single_stop(self):
        depot = np.array([0, 0], dtype=np.int64)
        stop = np.array([3000, 4000], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), stop.reshape(1, 2)])
        dm = _simple_dist_matrix(coords)
        stops = np.array([0], dtype=np.int32)
        dist = _compute_route_distance(stops, dm)
        # Distance should be ~5000 + ~5000 = ~10000 (3-4-5 triangle, meters)
        expected = 2 * 5000
        assert abs(dist - expected) < 10

    def test_compute_route_distance_empty_route(self):
        dm = np.zeros((1, 1), dtype=np.int64)
        stops = np.array([], dtype=np.int32)
        dist = _compute_route_distance(stops, dm)
        assert dist == 0

    def test_split_to_trips_distance_is_positive(self):
        customers = np.array([
            [1000, 0, 10000, 0, 28800, 300, 0],
            [2000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 10000),
            _make_stop(1, "deliver", 10000),
        ]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        for route in routes:
            assert route["total_distance"] >= 0


# ---------------------------------------------------------------------------
# TEST: Bike GT multi-segment logic
# ---------------------------------------------------------------------------
class TestBikeGtMultiSegment:

    def test_split_bike_gt_two_satellite_segments(self):
        customers = np.array([
            [0, 0, 0, 0, 28800, 300, 0],
            [1000, 0, 5000, 0, 28800, 300, 0],
            [2000, 0, 5000, 0, 28800, 300, 0],
            [3000, 0, 0, 0, 28800, 300, 0],
            [4000, 0, 5000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "reload", 0),
            _make_stop(1, "deliver", 5000),
            _make_stop(2, "deliver", 5000),
            _make_stop(3, "reload", 0),
            _make_stop(4, "deliver", 5000),
        ]
        routes = split_bike_gt(giant_tour, customers, dm,
                               capacity=BIKE_CAPACITY_G,
                               speed_us=BIKE_SPEED_US_PER_M, n_bikes=2)
        assert len(routes) > 0


# ---------------------------------------------------------------------------
# TEST: Demand accumulation
# ---------------------------------------------------------------------------
class TestDemandAccumulation:

    def test_split_to_trips_demand_accumulates_correctly(self):
        customers = np.array([
            [1000, 0, 15000, 0, 28800, 300, 0],
            [2000, 0, 25000, 0, 28800, 300, 0],
            [3000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 15000),
            _make_stop(1, "deliver", 25000),
            _make_stop(2, "deliver", 10000),
        ]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 1
        expected_demand = 15000 + 25000 + 10000
        assert routes[0]["total_demand"] == expected_demand


# ---------------------------------------------------------------------------
# TEST: Trip time calculation
# ---------------------------------------------------------------------------
class TestTripTimeEstimate:

    def test_estimate_trip_time_zero_stops(self):
        customers = np.array([[0, 0, 0, 0, 28800, 300, 0]], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        time = _estimate_trip_time(np.array([], dtype=np.int32), customers, dm)
        assert time == 0

    def test_estimate_trip_time_positive(self):
        customers = np.array([
            [1000, 0, 10000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        time = _estimate_trip_time(np.array([0], dtype=np.int32),
                                    customers, dm)
        assert time > 0
