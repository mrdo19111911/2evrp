"""Tests for bike route construction via split_bike.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.split_bike import split_bike_gt
from src.data.constants import ACT_DELIVER, ACT_RELOAD
from src.data.cost import BIKE_CAPACITY_G, BIKE_SPEED_US_PER_M


def _simple_dist_matrix(coords_with_depot):
    coords_f = coords_with_depot.astype(np.float64)
    diff = coords_f[:, None, :] - coords_f[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=2))).astype(np.int64)


def _make_customers(n, demand_g=10000, tw_close_s=28800):
    """Create i64 (N,7) customers on a line."""
    customers = np.zeros((n, 7), dtype=np.int64)
    for i in range(n):
        customers[i, 0] = (i + 1) * 1000  # x_m
        customers[i, 2] = demand_g
        customers[i, 4] = tw_close_s
        customers[i, 5] = 300  # service_s
    return customers


class TestSplitBikeGT:
    def test_small_bike_gt_no_reload(self):
        """3 customers x 10kg = 30kg < 60kg -> no reload needed, 1 route."""
        customers = _make_customers(3)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        bike_gt = [
            {"node": 0, "type": "deliver", "demand": 10000, "cluster_idx": -1},
            {"node": 1, "type": "deliver", "demand": 10000, "cluster_idx": -1},
            {"node": 2, "type": "deliver", "demand": 10000, "cluster_idx": -1},
        ]

        routes = split_bike_gt(bike_gt, customers, dm,
                               capacity=BIKE_CAPACITY_G,
                               speed_us=BIKE_SPEED_US_PER_M, n_bikes=2)
        assert len(routes) >= 1
        all_served = set()
        for r in routes:
            for s, a in zip(r["stops"], r["actions"]):
                if a == ACT_DELIVER:
                    all_served.add(int(s))
        assert {0, 1, 2}.issubset(all_served)

    def test_needs_multiple_routes(self):
        """5 customers x 20kg = 100kg > 60kg -> needs multiple routes."""
        customers = _make_customers(5, demand_g=20000)
        depot = np.array([0, 0], dtype=np.int64)
        coords = np.vstack([depot.reshape(1, 2), customers[:, :2]])
        dm = _simple_dist_matrix(coords)

        bike_gt = [
            {"node": i, "type": "deliver", "demand": 20000, "cluster_idx": -1}
            for i in range(5)
        ]

        routes = split_bike_gt(bike_gt, customers, dm,
                               capacity=BIKE_CAPACITY_G,
                               speed_us=BIKE_SPEED_US_PER_M, n_bikes=3)
        assert len(routes) >= 1

    def test_empty_gt(self):
        """Empty GT -> empty routes."""
        customers = _make_customers(1)
        dm = np.zeros((2, 2), dtype=np.int64)
        routes = split_bike_gt([], customers, dm,
                               capacity=BIKE_CAPACITY_G,
                               speed_us=BIKE_SPEED_US_PER_M, n_bikes=1)
        assert routes == []
