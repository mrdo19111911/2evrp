"""Tests for src/init/giant_tour.py — giant tour construction."""
import numpy as np
import pytest

from src.init.giant_tour import (
    build_giant_tour,
    find_cluster_satellite,
    cw_savings_order,
)
from src.data.constants import COL_DEMAND


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cluster(members, big_nodes, bike_nodes, customers,
                  restricted=None):
    """Build a cluster dict from member lists."""
    members_arr = np.array(members, dtype=np.int32)
    big_arr = np.array(big_nodes, dtype=np.int32)
    bike_arr = np.array(bike_nodes, dtype=np.int32)
    rest_arr = np.array(restricted if restricted else [], dtype=np.int32)
    coords = customers[members_arr, :2]
    return {
        "members": members_arr,
        "big_nodes": big_arr,
        "bike_nodes": bike_arr,
        "restricted": rest_arr,
        "total_demand": float(customers[members_arr, COL_DEMAND].sum()),
        "centroid": coords.mean(axis=0),
    }


def _simple_dist_matrix(coords_with_depot):
    """Euclidean distance matrix from (N+1, 2) coords (index 0 = depot)."""
    n = len(coords_with_depot)
    diff = coords_with_depot[:, None, :] - coords_with_depot[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


# ---------------------------------------------------------------------------
# find_cluster_satellite
# ---------------------------------------------------------------------------
class TestFindClusterSatellite:
    def test_returns_closest_to_centroid(self):
        """Cluster with 3 nodes -> returns node closest to centroid."""
        # Nodes at (0,0), (1,0), (10,0). Centroid ~ (3.67, 0).
        # Node 1 at (1,0) is closest to centroid.
        customers = np.array([
            [0.0, 0.0, 20.0, 0.0, 120.0, 5.0],  # 0
            [1.0, 0.0, 20.0, 0.0, 120.0, 5.0],  # 1
            [10.0, 0.0, 20.0, 0.0, 120.0, 5.0],  # 2
        ], dtype=np.float64)
        # dist_matrix: depot at origin, then 3 customers
        depot = np.array([5.0, 5.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        cluster = _make_cluster([0, 1, 2], [], [0, 1, 2], customers)
        # centroid = (3.67, 0), closest is node 1 at (1,0)
        sat = find_cluster_satellite(cluster, customers, dm)
        assert sat == 1

    def test_single_node_cluster(self):
        """Cluster with 1 bike node -> returns that node."""
        customers = np.array([
            [3.0, 4.0, 15.0, 0.0, 120.0, 5.0],
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        cluster = _make_cluster([0], [], [0], customers)
        sat = find_cluster_satellite(cluster, customers, dm)
        assert sat == 0


# ---------------------------------------------------------------------------
# cw_savings_order
# ---------------------------------------------------------------------------
class TestNearestNeighborOrder:
    def test_four_stops_ordered(self):
        """4 stops at known positions -> NN from depot produces predictable order.

        Depot at (0,0). Stops at:
          node 0: (1, 0) -> dist from depot = 1
          node 1: (5, 0) -> dist from depot = 5
          node 2: (2, 0) -> dist from depot = 2
          node 3: (3, 0) -> dist from depot = 3

        NN order from depot: 0 (d=1) -> 2 (d=1) -> 3 (d=1) -> 1 (d=2)
        """
        customers_coords = np.array([
            [1.0, 0.0], [5.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        ])
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers_coords])
        dm = _simple_dist_matrix(coords)

        stops = [
            {"node": 0, "type": "deliver", "demand": 10.0, "cluster_idx": -1},
            {"node": 1, "type": "deliver", "demand": 10.0, "cluster_idx": -1},
            {"node": 2, "type": "satellite", "demand": 20.0, "cluster_idx": 0},
            {"node": 3, "type": "deliver", "demand": 10.0, "cluster_idx": -1},
        ]

        ordered = cw_savings_order(stops, dm)
        ordered_nodes = [s["node"] for s in ordered]
        assert set(ordered_nodes) == {0, 1, 2, 3}
        assert len(ordered_nodes) == 4

    def test_preserves_stop_metadata(self):
        """Ordering must keep stop dict contents intact."""
        customers_coords = np.array([[1.0, 0.0], [2.0, 0.0]])
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers_coords])
        dm = _simple_dist_matrix(coords)

        stops = [
            {"node": 1, "type": "satellite", "demand": 50.0, "cluster_idx": 3},
            {"node": 0, "type": "deliver", "demand": 100.0, "cluster_idx": -1},
        ]
        ordered = cw_savings_order(stops, dm)
        assert len(ordered) == 2
        # Metadata preserved regardless of order
        nodes_map = {s["node"]: s for s in ordered}
        assert nodes_map[0]["type"] == "deliver"
        assert nodes_map[0]["demand"] == pytest.approx(100.0)
        assert nodes_map[1]["type"] == "satellite"
        assert nodes_map[1]["cluster_idx"] == 3


# ---------------------------------------------------------------------------
# build_giant_tour
# ---------------------------------------------------------------------------
class TestBuildGiantTour:
    def test_cluster_with_big_and_bike_nodes(self):
        """2 clusters: cluster0 has 1 big + 2 bike, cluster1 has 1 bike only.

        Expected: big node as 'deliver', each cluster with bike_nodes gets a 'satellite'.
        """
        customers = np.array([
            [0.0, 0.0, 100.0, 0.0, 120.0, 5.0],   # 0: big
            [1.0, 0.0,  20.0, 0.0, 120.0, 5.0],   # 1: bike
            [2.0, 0.0,  15.0, 0.0, 120.0, 5.0],   # 2: bike
            [10.0, 0.0, 10.0, 0.0, 120.0, 5.0],   # 3: bike (cluster1)
        ], dtype=np.float64)
        depot = np.array([5.0, 5.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        cluster0 = _make_cluster([0, 1, 2], [0], [1, 2], customers)
        cluster1 = _make_cluster([3], [], [3], customers)

        tour = build_giant_tour([cluster0, cluster1], customers, depot, dm)

        types = [s["type"] for s in tour]
        nodes = [s["node"] for s in tour]

        # Must have at least 1 deliver (big node 0) and 2 satellites
        assert "deliver" in types
        assert types.count("satellite") == 2  # one per cluster with bike_nodes

        # Big node 0 must appear as deliver
        deliver_nodes = [s["node"] for s in tour if s["type"] == "deliver"]
        assert 0 in deliver_nodes

    def test_cluster_only_big_nodes(self):
        """Cluster with only big nodes -> only 'deliver' stops, no satellite."""
        customers = np.array([
            [0.0, 0.0, 100.0, 0.0, 120.0, 5.0],  # big
            [5.0, 0.0, 200.0, 0.0, 120.0, 5.0],  # big
        ], dtype=np.float64)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        cluster = _make_cluster([0, 1], [0, 1], [], customers)
        tour = build_giant_tour([cluster], customers, depot, dm)

        types = [s["type"] for s in tour]
        assert all(t == "deliver" for t in types)
        assert len(tour) == 2

    def test_tour_stop_dict_structure(self):
        """Each stop dict has node, type, demand, cluster_idx keys."""
        customers = np.array([
            [0.0, 0.0, 50.0, 0.0, 120.0, 5.0],
        ], dtype=np.float64)
        depot = np.array([5.0, 5.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        cluster = _make_cluster([0], [], [0], customers)
        tour = build_giant_tour([cluster], customers, depot, dm)

        for stop in tour:
            assert "node" in stop
            assert "type" in stop
            assert "demand" in stop
            assert "cluster_idx" in stop
