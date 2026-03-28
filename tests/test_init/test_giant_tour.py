"""Tests for src/init/giant_tour.py -- giant tour construction.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.giant_tour import (
    build_giant_tour,
    find_cluster_satellite,
    cw_savings_order,
    build_bike_giant_tour,
)
from src.data.constants import COL_DEMAND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cluster(members, big_nodes, bike_nodes, customers):
    """Build a cluster dict from member lists."""
    members_arr = np.array(members, dtype=np.int32)
    big_arr = np.array(big_nodes, dtype=np.int32)
    bike_arr = np.array(bike_nodes, dtype=np.int32)
    rest_arr = np.array([], dtype=np.int32)
    coords = customers[members_arr, :2].astype(np.float64)
    return {
        "members": members_arr,
        "big_nodes": big_arr,
        "bike_nodes": bike_arr,
        "restricted": rest_arr,
        "total_demand": int(customers[members_arr, COL_DEMAND].sum()),
        "centroid": coords.mean(axis=0),
    }


def _simple_dist_matrix(coords_with_depot):
    """Euclidean i64 distance matrix from (N+1, 2) coords (index 0 = depot)."""
    coords_f = coords_with_depot.astype(np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


# ---------------------------------------------------------------------------
# find_cluster_satellite
# ---------------------------------------------------------------------------
class TestFindClusterSatellite:
    def test_returns_closest_to_centroid(self):
        customers = np.array([
            [0, 0, 20000, 0, 7200, 300, 0],
            [1000, 0, 20000, 0, 7200, 300, 0],
            [10000, 0, 20000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([5000, 5000], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        cluster = _make_cluster([0, 1, 2], [], [0, 1, 2], customers)
        sat = find_cluster_satellite(cluster, customers, dm)
        assert sat == 1

    def test_single_node_cluster(self):
        customers = np.array([
            [3000, 4000, 15000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        cluster = _make_cluster([0], [], [0], customers)
        sat = find_cluster_satellite(cluster, customers, dm)
        assert sat == 0


# ---------------------------------------------------------------------------
# cw_savings_order
# ---------------------------------------------------------------------------
class TestNearestNeighborOrder:
    def test_four_stops_ordered(self):
        customers_coords = np.array([
            [1000, 0], [5000, 0], [2000, 0], [3000, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers_coords])
        dm = _simple_dist_matrix(coords)
        stops = [
            {"node": 0, "type": "deliver", "demand": 10000, "cluster_idx": -1},
            {"node": 1, "type": "deliver", "demand": 10000, "cluster_idx": -1},
            {"node": 2, "type": "satellite", "demand": 20000, "cluster_idx": 0},
            {"node": 3, "type": "deliver", "demand": 10000, "cluster_idx": -1},
        ]
        ordered = cw_savings_order(stops, dm)
        ordered_nodes = [s["node"] for s in ordered]
        assert set(ordered_nodes) == {0, 1, 2, 3}
        assert len(ordered_nodes) == 4

    def test_preserves_stop_metadata(self):
        customers_coords = np.array([[1000, 0], [2000, 0]], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers_coords])
        dm = _simple_dist_matrix(coords)
        stops = [
            {"node": 1, "type": "satellite", "demand": 50000, "cluster_idx": 3},
            {"node": 0, "type": "deliver", "demand": 100000, "cluster_idx": -1},
        ]
        ordered = cw_savings_order(stops, dm)
        assert len(ordered) == 2
        nodes_map = {s["node"]: s for s in ordered}
        assert nodes_map[0]["type"] == "deliver"
        assert nodes_map[0]["demand"] == 100000
        assert nodes_map[1]["type"] == "satellite"
        assert nodes_map[1]["cluster_idx"] == 3


# ---------------------------------------------------------------------------
# build_giant_tour
# ---------------------------------------------------------------------------
class TestBuildGiantTour:
    def test_cluster_with_big_and_bike_nodes(self):
        customers = np.array([
            [0, 0, 100000, 0, 7200, 300, 0],
            [1000, 0, 20000, 0, 7200, 300, 0],
            [2000, 0, 15000, 0, 7200, 300, 0],
            [10000, 0, 10000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([5000, 5000], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        cluster0 = _make_cluster([0, 1, 2], [0], [1, 2], customers)
        cluster1 = _make_cluster([3], [], [3], customers)
        tour = build_giant_tour([cluster0, cluster1], customers, depot, dm)
        types = [s["type"] for s in tour]
        assert "deliver" in types
        assert types.count("satellite") == 2
        deliver_nodes = [s["node"] for s in tour if s["type"] == "deliver"]
        assert 0 in deliver_nodes

    def test_cluster_only_big_nodes(self):
        customers = np.array([
            [0, 0, 100000, 0, 7200, 300, 0],
            [5000, 0, 200000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        cluster = _make_cluster([0, 1], [0, 1], [], customers)
        tour = build_giant_tour([cluster], customers, depot, dm)
        types = [s["type"] for s in tour]
        assert all(t == "deliver" for t in types)
        assert len(tour) == 2

    def test_tour_stop_dict_structure(self):
        customers = np.array([
            [0, 0, 50000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([5000, 5000], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        cluster = _make_cluster([0], [], [0], customers)
        tour = build_giant_tour([cluster], customers, depot, dm)
        for stop in tour:
            assert "node" in stop
            assert "type" in stop
            assert "demand" in stop
            assert "cluster_idx" in stop


# ---------------------------------------------------------------------------
# build_bike_giant_tour
# ---------------------------------------------------------------------------
class TestBuildBikeGiantTour:

    def test_empty_truck_gt_empty_bike_customers(self):
        customers = np.zeros((0, 7), dtype=np.int64)
        dm = np.zeros((1, 1), dtype=np.int64)
        result = build_bike_giant_tour([], [], customers, dm)
        assert result == []

    def test_single_reload_zero_bike_customers(self):
        customers = np.array([
            [0, 0, 100000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        truck_gt = [
            {"node": 0, "type": "deliver", "demand": 100000, "cluster_idx": -1}
        ]
        result = build_bike_giant_tour(truck_gt, [], customers, dm)
        assert len(result) == 1
        assert result[0]["node"] == 0
        assert result[0]["type"] == "reload"
        assert result[0]["demand"] == 0

    def test_cheapest_insertion_all_same_location(self):
        customers = np.array([
            [1000, 0, 10000, 0, 7200, 300, 0],
            [1000, 0, 20000, 0, 7200, 300, 0],
            [1000, 0, 15000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        truck_gt = [
            {"node": 0, "type": "deliver", "demand": 100000, "cluster_idx": -1}
        ]
        result = build_bike_giant_tour(truck_gt, [1, 2], customers, dm)
        nodes = [s["node"] for s in result]
        assert set(nodes) == {0, 1, 2}
        assert result[0]["type"] == "reload"

    def test_bike_customer_demands_preserved(self):
        customers = np.array([
            [0, 0, 100000, 0, 7200, 300, 0],
            [5000, 0, 23000, 0, 7200, 300, 0],
            [10000, 0, 44000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        truck_gt = [
            {"node": 0, "type": "deliver", "demand": 100000, "cluster_idx": -1}
        ]
        result = build_bike_giant_tour(truck_gt, [1, 2], customers, dm)
        bikes = [s for s in result if s["node"] in [1, 2]]
        for b in bikes:
            assert b["demand"] == int(customers[b["node"], COL_DEMAND])

    def test_empty_truck_gt_with_bike_customers(self):
        customers = np.array([
            [5000, 0, 15000, 0, 7200, 300, 0],
            [10000, 0, 20000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        result = build_bike_giant_tour([], [0, 1], customers, dm)
        assert len(result) == 2
        assert all(s["type"] == "deliver" for s in result)
        nodes = [s["node"] for s in result]
        assert set(nodes) == {0, 1}


# ---------------------------------------------------------------------------
# cw_savings_order edge cases
# ---------------------------------------------------------------------------
class TestCWSavingsOrderEdgeCases:

    def test_single_node(self):
        customers_coords = np.array([[5000, 0]], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers_coords])
        dm = _simple_dist_matrix(coords)
        stops = [
            {"node": 0, "type": "deliver", "demand": 50000, "cluster_idx": -1}
        ]
        result = cw_savings_order(stops, dm)
        assert len(result) == 1
        assert result[0]["node"] == 0

    def test_two_nodes(self):
        customers_coords = np.array([[1000, 0], [5000, 0]], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers_coords])
        dm = _simple_dist_matrix(coords)
        stops = [
            {"node": 0, "type": "deliver", "demand": 10000, "cluster_idx": -1},
            {"node": 1, "type": "deliver", "demand": 20000, "cluster_idx": -1},
        ]
        result = cw_savings_order(stops, dm)
        assert len(result) == 2
        nodes = [s["node"] for s in result]
        assert set(nodes) == {0, 1}

    def test_empty_stops(self):
        dm = np.zeros((1, 1), dtype=np.int64)
        result = cw_savings_order([], dm)
        assert result == []


# ---------------------------------------------------------------------------
# find_cluster_satellite edge cases
# ---------------------------------------------------------------------------
class TestFindClusterSatelliteEdgeCase:

    def test_single_member_cluster(self):
        customers = np.array([
            [7000, 3000, 50000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        cluster_dict = {
            "members": np.array([0], dtype=np.int32),
            "centroid": customers[0, :2].astype(np.float64),
        }
        sat = find_cluster_satellite(cluster_dict, customers, dm)
        assert sat == 0

    def test_two_equidistant_nodes(self):
        customers = np.array([
            [1000, 0, 10000, 0, 7200, 300, 0],
            [-1000, 0, 20000, 0, 7200, 300, 0],
        ], dtype=np.int64)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        centroid = np.array([0.0, 0.0])
        cluster_dict = {
            "members": np.array([0, 1], dtype=np.int32),
            "centroid": centroid,
        }
        sat = find_cluster_satellite(cluster_dict, customers, dm)
        assert sat in {0, 1}
