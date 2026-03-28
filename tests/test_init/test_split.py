"""Tests for src/init/split.py — split giant tour into truck routes."""
import numpy as np
import pytest

from src.init.split import (
    split_to_trips,
    backtrack_split,
    group_trips_to_trucks,
)
from src.data.constants import ACT_DELIVER, ACT_RELOAD, COL_X, COL_Y, COL_DEMAND, COL_TW_OPEN, COL_TW_CLOSE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _simple_dist_matrix(coords_with_depot):
    """Euclidean distance matrix from (N+1, 2) coords (index 0 = depot)."""
    n = len(coords_with_depot)
    diff = coords_with_depot[:, None, :] - coords_with_depot[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _make_stop(node, stype, demand, cluster_idx=-1):
    return {"node": node, "type": stype, "demand": demand,
            "cluster_idx": cluster_idx}


def _make_cluster(members, big, bike, customers):
    members_arr = np.array(members, dtype=np.int32)
    return {
        "members": members_arr,
        "big_nodes": np.array(big, dtype=np.int32),
        "bike_nodes": np.array(bike, dtype=np.int32),
        "restricted": np.array([], dtype=np.int32),
        "total_demand": float(customers[members_arr, 2].sum()),
        "centroid": customers[members_arr, :2].mean(axis=0),
    }


# ---------------------------------------------------------------------------
# split_to_trips: low demand -> 1 route
# ---------------------------------------------------------------------------
class TestSplitLowDemand:
    def test_three_stops_under_capacity_one_route(self):
        """3 stops, total demand=100 < capacity=2000 -> 1 route."""
        customers = np.array([
            [1.0, 0.0, 30.0, 0.0, 120.0, 5.0],   # 0
            [2.0, 0.0, 30.0, 0.0, 120.0, 5.0],   # 1
            [3.0, 0.0, 40.0, 0.0, 120.0, 5.0],   # 2
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        giant_tour = [
            _make_stop(0, "deliver", 30.0),
            _make_stop(1, "deliver", 30.0),
            _make_stop(2, "deliver", 40.0),
        ]
        clusters = [_make_cluster([0, 1, 2], [0, 1, 2], [], customers)]

        routes = split_to_trips(
            giant_tour, customers, dm,
            capacity=2000.0,
        )
        assert len(routes) == 1
        assert routes[0]["total_demand"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# split_to_trips: high demand -> 3 routes
# ---------------------------------------------------------------------------
class TestSplitHighDemand:
    def test_five_stops_three_times_capacity(self):
        """5 stops, total demand ~ 3x capacity -> at least 3 routes.

        Capacity=100. Demands: [50, 50, 40, 60, 50] = 250 ~ 2.5x.
        Expect 3 routes (ceil(250/100)=3).
        """
        customers = np.array([
            [1.0, 0.0, 50.0, 0.0, 120.0, 5.0],  # 0
            [2.0, 0.0, 50.0, 0.0, 120.0, 5.0],  # 1
            [3.0, 0.0, 40.0, 0.0, 120.0, 5.0],  # 2
            [4.0, 0.0, 60.0, 0.0, 120.0, 5.0],  # 3
            [5.0, 0.0, 50.0, 0.0, 120.0, 5.0],  # 4
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        giant_tour = [
            _make_stop(0, "deliver", 50.0),
            _make_stop(1, "deliver", 50.0),
            _make_stop(2, "deliver", 40.0),
            _make_stop(3, "deliver", 60.0),
            _make_stop(4, "deliver", 50.0),
        ]
        clusters = [_make_cluster([0, 1, 2, 3, 4],
                                  [0, 1, 2, 3, 4], [], customers)]

        routes = split_to_trips(
            giant_tour, customers, dm,
            capacity=100.0,
        )
        assert len(routes) >= 3

    def test_each_route_under_capacity(self):
        """Every route must have total_demand <= capacity."""
        customers = np.array([
            [1.0, 0.0, 50.0, 0.0, 120.0, 5.0],
            [2.0, 0.0, 50.0, 0.0, 120.0, 5.0],
            [3.0, 0.0, 40.0, 0.0, 120.0, 5.0],
            [4.0, 0.0, 60.0, 0.0, 120.0, 5.0],
            [5.0, 0.0, 50.0, 0.0, 120.0, 5.0],
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        giant_tour = [
            _make_stop(0, "deliver", 50.0),
            _make_stop(1, "deliver", 50.0),
            _make_stop(2, "deliver", 40.0),
            _make_stop(3, "deliver", 60.0),
            _make_stop(4, "deliver", 50.0),
        ]
        clusters = [_make_cluster([0, 1, 2, 3, 4],
                                  [0, 1, 2, 3, 4], [], customers)]

        routes = split_to_trips(
            giant_tour, customers, dm,
            capacity=100.0,
        )
        for i, r in enumerate(routes):
            assert r["total_demand"] <= 100.0 + 1e-9, (
                f"Route {i} demand {r['total_demand']} exceeds capacity 100"
            )

    def test_all_stops_covered(self):
        """Union of all route stops == all giant tour nodes."""
        customers = np.array([
            [1.0, 0.0, 30.0, 0.0, 120.0, 5.0],
            [2.0, 0.0, 30.0, 0.0, 120.0, 5.0],
            [3.0, 0.0, 40.0, 0.0, 120.0, 5.0],
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        giant_tour = [
            _make_stop(0, "deliver", 30.0),
            _make_stop(1, "deliver", 30.0),
            _make_stop(2, "deliver", 40.0),
        ]
        clusters = [_make_cluster([0, 1, 2], [0, 1, 2], [], customers)]

        routes = split_to_trips(
            giant_tour, customers, dm,
            capacity=2000.0,
        )
        all_nodes = set()
        for r in routes:
            for node in r["stops"]:
                all_nodes.add(int(node))
        expected_nodes = {0, 1, 2}
        assert all_nodes == expected_nodes

    def test_route_dict_structure(self):
        """Each route dict has stops, actions, total_demand, total_distance."""
        customers = np.array([
            [1.0, 0.0, 30.0, 0.0, 120.0, 5.0],
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        giant_tour = [_make_stop(0, "deliver", 30.0)]
        clusters = [_make_cluster([0], [0], [], customers)]

        routes = split_to_trips(
            giant_tour, customers, dm,
            capacity=2000.0,
        )
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
        """5 short trips, 3 trucks -> multi-trip, all nodes present."""
        customers_coords = np.array([
            [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0],
        ])
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers_coords])
        dm = _simple_dist_matrix(coords)

        trips = [
            {"stops": np.array([0], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 10.0, "total_distance": 2.0},
            {"stops": np.array([1], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 15.0, "total_distance": 4.0},
            {"stops": np.array([2], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 20.0, "total_distance": 6.0},
            {"stops": np.array([3], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 25.0, "total_distance": 8.0},
            {"stops": np.array([4], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 30.0, "total_distance": 10.0},
        ]

        customers = np.zeros((5, 6), dtype=np.float64)
        for i in range(5):
            customers[i, COL_X] = float(i + 1)
            customers[i, COL_DEMAND] = trips[i]["total_demand"]
            customers[i, COL_TW_CLOSE] = 480.0

        result = group_trips_to_trucks(trips, n_trucks=3,
                                        customers=customers, dist_matrix=dm)

        # All nodes present
        all_nodes = set()
        for r in result:
            for node in r["stops"]:
                all_nodes.add(int(node))
        assert all_nodes == {0, 1, 2, 3, 4}

        # Each route has a truck_id
        tids = [r["truck_id"] for r in result]
        assert all(0 <= t < 3 for t in tids)

    def test_fewer_trips_than_trucks(self):
        """2 trips, 3 trucks -> each trip gets own truck."""
        dm = np.zeros((3, 3), dtype=np.float64)
        customers = np.zeros((2, 6), dtype=np.float64)
        customers[:, COL_TW_CLOSE] = 480.0
        trips = [
            {"stops": np.array([0], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 10.0, "total_distance": 2.0},
            {"stops": np.array([1], dtype=np.int32),
             "actions": np.array([ACT_DELIVER], dtype=np.int8),
             "total_demand": 20.0, "total_distance": 4.0},
        ]
        result = group_trips_to_trucks(trips, n_trucks=3,
                                        customers=customers, dist_matrix=dm)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# backtrack_split
# ---------------------------------------------------------------------------
class TestBacktrackSplit:
    def test_pred_array_two_routes(self):
        """pred = [-1, 0, 0, 2] with n=3 -> 2 routes: [0,1] and [2]."""
        giant_tour = [
            _make_stop(10, "deliver", 50.0),
            _make_stop(11, "deliver", 50.0),
            _make_stop(12, "deliver", 50.0),
        ]
        pred = np.array([-1, 0, 0, 2], dtype=np.int64)

        routes = backtrack_split(pred, 3, giant_tour)
        assert len(routes) == 2

        # Route boundaries: one route has nodes [10, 11], other has [12]
        route_node_sets = [set(r["stops"].tolist()) for r in routes]
        assert {10, 11} in route_node_sets or {12} in route_node_sets

    def test_single_route(self):
        """pred = [-1, 0] with n=1 -> 1 route with 1 stop."""
        giant_tour = [_make_stop(5, "deliver", 30.0)]
        pred = np.array([-1, 0], dtype=np.int64)

        routes = backtrack_split(pred, 1, giant_tour)
        assert len(routes) == 1
        assert 5 in routes[0]["stops"]
