"""Tests for bike route construction: greedy NN + satellite reload."""
import numpy as np
import pytest

from src.init.bike_routing import build_bike_routes
from src.init.bike_split import build_cluster_bike_route
from src.data.constants import ACT_DELIVER, ACT_RELOAD


def _simple_dist_matrix(coords_with_depot):
    n = len(coords_with_depot)
    diff = coords_with_depot[:, None, :] - coords_with_depot[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def _make_customers(n, demand=10.0, tw_close=480.0):
    customers = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        customers[i, 0] = float(i + 1)
        customers[i, 2] = demand
        customers[i, 4] = tw_close
    return customers


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


class TestBuildClusterBikeRoute:
    def test_small_cluster_no_reload(self):
        """3 customers x 10kg = 30kg < 60kg -> no reload needed."""
        customers = _make_customers(3)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        bike_nodes = np.array([0, 1, 2], dtype=np.int32)
        reload_dm = [0]

        stops, actions, served = build_cluster_bike_route(
            bike_nodes, reload_dm, customers, dm)

        assert served == {0, 1, 2}
        assert ACT_RELOAD not in actions

    def test_needs_reload(self):
        """5 customers x 20kg = 100kg > 60kg -> reload needed."""
        customers = _make_customers(5, demand=20.0)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        bike_nodes = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        # Satellite at node 2 (dm index 3)
        reload_dm = [0, 3]

        stops, actions, served = build_cluster_bike_route(
            bike_nodes, reload_dm, customers, dm)

        assert len(served) == 5
        assert ACT_RELOAD in actions

    def test_all_served(self):
        """All customers served when time allows."""
        customers = _make_customers(6, demand=10.0)
        depot = np.array([0.0, 0.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)
        bike_nodes = np.arange(6, dtype=np.int32)
        reload_dm = [0, 3]

        stops, actions, served = build_cluster_bike_route(
            bike_nodes, reload_dm, customers, dm)

        assert served == set(range(6))


class TestBuildBikeRoutes:
    def test_all_bike_customers_served(self):
        """End-to-end: all bike customers from clusters are delivered."""
        customers = np.array([
            [0.0, 0.0, 100.0, 0.0, 480.0, 5.0],  # 0: big
            [1.0, 0.0,  20.0, 0.0, 480.0, 5.0],  # 1: bike
            [2.0, 0.0,  15.0, 0.0, 480.0, 5.0],  # 2: bike
            [10.0, 0.0, 10.0, 0.0, 480.0, 5.0],  # 3: bike
        ], dtype=np.float64)
        depot = np.array([5.0, 5.0])
        coords = np.vstack([depot, customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        cluster0 = _make_cluster([0, 1, 2], [0], [1, 2], customers)
        cluster1 = _make_cluster([3], [], [3], customers)
        giant_tour = [{"node": 0, "type": "deliver", "demand": 100.0, "cluster_idx": -1}]

        bike_sol = build_bike_routes(
            [cluster0, cluster1], [], customers, dm,
            n_bikes=2, bike_capacity=60.0, giant_tour=giant_tour)

        all_delivered = set()
        for r in bike_sol:
            if r.get("bike_id", -1) < 0:
                continue
            delivered = r["stops"][r["actions"] == ACT_DELIVER]
            all_delivered.update(delivered.tolist())
        assert {1, 2, 3}.issubset(all_delivered)
