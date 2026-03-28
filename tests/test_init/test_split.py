"""Tests for src/init/split.py -- split giant tour into truck routes.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.split import (
    split_to_trips,
    group_trips_to_trucks,
)
from src.data.constants import (
    ACT_DELIVER, ACT_RELOAD, COL_X, COL_Y, COL_DEMAND,
    COL_TW_OPEN, COL_TW_CLOSE,
)
from src.data.cost import TRUCK_SPEED_US_PER_M, TRUCK_CAPACITY_G


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _simple_dist_matrix(coords_with_depot):
    coords_f = coords_with_depot.astype(np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _make_stop(node, stype, demand):
    return {"node": node, "type": stype, "demand": demand,
            "cluster_idx": -1}


def _make_customers_i64(n, demand_g=10000, tw_close_s=28800):
    """Create i64 (N, 7) customers on a line."""
    customers = np.zeros((n, 7), dtype=np.int64)
    for i in range(n):
        customers[i, 0] = (i + 1) * 1000  # x_m
        customers[i, 2] = demand_g
        customers[i, 4] = tw_close_s
        customers[i, 5] = 300  # service_s
    return customers


# ---------------------------------------------------------------------------
# split_to_trips: low demand -> 1 route
# ---------------------------------------------------------------------------
class TestSplitLowDemand:
    def test_three_stops_under_capacity_one_route(self):
        customers = np.array([
            [1000, 0, 30000, 0, 28800, 300, 0],
            [2000, 0, 30000, 0, 28800, 300, 0],
            [3000, 0, 40000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 30000),
            _make_stop(1, "deliver", 30000),
            _make_stop(2, "deliver", 40000),
        ]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 1
        assert routes[0]["total_demand"] == 100000


# ---------------------------------------------------------------------------
# split_to_trips: high demand -> multiple routes
# ---------------------------------------------------------------------------
class TestSplitHighDemand:
    def test_five_stops_three_times_capacity(self):
        customers = np.array([
            [1000, 0, 50000, 0, 28800, 300, 0],
            [2000, 0, 50000, 0, 28800, 300, 0],
            [3000, 0, 40000, 0, 28800, 300, 0],
            [4000, 0, 60000, 0, 28800, 300, 0],
            [5000, 0, 50000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 50000),
            _make_stop(1, "deliver", 50000),
            _make_stop(2, "deliver", 40000),
            _make_stop(3, "deliver", 60000),
            _make_stop(4, "deliver", 50000),
        ]
        routes = split_to_trips(giant_tour, customers, dm, capacity=100000)
        assert len(routes) >= 3

    def test_each_route_under_capacity(self):
        customers = np.array([
            [1000, 0, 50000, 0, 28800, 300, 0],
            [2000, 0, 50000, 0, 28800, 300, 0],
            [3000, 0, 40000, 0, 28800, 300, 0],
            [4000, 0, 60000, 0, 28800, 300, 0],
            [5000, 0, 50000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 50000),
            _make_stop(1, "deliver", 50000),
            _make_stop(2, "deliver", 40000),
            _make_stop(3, "deliver", 60000),
            _make_stop(4, "deliver", 50000),
        ]
        routes = split_to_trips(giant_tour, customers, dm, capacity=100000)
        for r in routes:
            assert r["total_demand"] <= 100000

    def test_all_stops_covered(self):
        customers = np.array([
            [1000, 0, 30000, 0, 28800, 300, 0],
            [2000, 0, 30000, 0, 28800, 300, 0],
            [3000, 0, 40000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [
            _make_stop(0, "deliver", 30000),
            _make_stop(1, "deliver", 30000),
            _make_stop(2, "deliver", 40000),
        ]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        all_nodes = set()
        for r in routes:
            for node in r["stops"]:
                all_nodes.add(int(node))
        assert all_nodes == {0, 1, 2}

    def test_route_dict_structure(self):
        customers = np.array([
            [1000, 0, 30000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        giant_tour = [_make_stop(0, "deliver", 30000)]
        routes = split_to_trips(giant_tour, customers, dm,
                                capacity=TRUCK_CAPACITY_G)
        assert len(routes) == 1
        r = routes[0]
        assert "stops" in r
        assert "actions" in r
        assert "total_demand" in r
        assert "total_distance" in r


# ---------------------------------------------------------------------------
# group_trips_to_trucks
# ---------------------------------------------------------------------------
class TestGroupTripsToTrucks:
    def test_5_trips_to_3_trucks(self):
        customers = _make_customers_i64(5)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        trips = [
            {"stops": np.array([i], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 10000, "total_distance": 2000}
            for i in range(5)
        ]
        result = group_trips_to_trucks(trips, n_trucks=3,
                                        customers=customers, dist_matrix=dm)
        assert len(result) <= 3
        tids = [r["truck_id"] for r in result]
        assert all(0 <= t < 3 for t in tids)

    def test_fewer_trips_than_trucks(self):
        dm = np.zeros((3, 3), dtype=np.int64)
        customers = _make_customers_i64(2)
        trips = [
            {"stops": np.array([0], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 10000, "total_distance": 2000},
            {"stops": np.array([1], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 20000, "total_distance": 4000},
        ]
        result = group_trips_to_trucks(trips, n_trucks=3,
                                        customers=customers, dist_matrix=dm)
        assert len(result) == 2
