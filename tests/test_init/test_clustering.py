"""Tests for src/init/clustering.py — Data Model v2."""
import numpy as np
import pytest

from src.init.clustering import (
    estimate_n_clusters,
    kmeans,
    cluster_customers,
    group_by_label,
    rebalance_clusters,
    make_single_cluster,
)
from src.data.constants import (
    ORD_LOC, ORD_QTY, ORD_UNIT_W, ORD_COLS,
    LOC_X, LOC_Y, LOC_COLS, LTYPE_DEPOT, LTYPE_CUSTOMER,
    VEH_CAP_KG,
)


def _demand_kg(orders, idx):
    return orders[idx, ORD_QTY] * orders[idx, ORD_UNIT_W]


def _make_orders(demands, loc_start=1):
    """Build minimal orders array from demand list."""
    n = len(demands)
    orders = np.zeros((n, ORD_COLS), dtype=np.float64)
    for i, d in enumerate(demands):
        orders[i, ORD_LOC] = loc_start + i
        orders[i, ORD_QTY] = max(1, int(d / 5.0))
        orders[i, ORD_UNIT_W] = d / max(1, orders[i, ORD_QTY])
    return orders


def _make_locations(coords, depot_xy=(5.0, 5.0)):
    """Build locations array from coords list (customer coords only)."""
    n_locs = 1 + len(coords)
    locs = np.zeros((n_locs, LOC_COLS), dtype=np.float64)
    locs[0] = [depot_xy[0], depot_xy[1], LTYPE_DEPOT, 99999.0, 100.0, 10.0]
    for i, (x, y) in enumerate(coords):
        locs[i + 1] = [x, y, LTYPE_CUSTOMER, 99999.0, 20.0, 5.0]
    return locs


# ---------------------------------------------------------------------------
# estimate_n_clusters
# ---------------------------------------------------------------------------

def test_estimate_n_clusters_basic():
    """total_demand ~ 13750, capacity=2000 -> k ~ ceil(13750/2000) = 7."""
    demands = [1500, 1200, 1000, 2000, 1500, 1800, 1250, 1000, 1500, 1000]
    orders = _make_orders(demands)
    k = estimate_n_clusters(orders, capacity=2000.0)
    assert k >= 1
    assert isinstance(k, (int, np.integer))


def test_estimate_small_demand():
    orders = _make_orders([30, 40, 30])
    k = estimate_n_clusters(orders, capacity=2000.0)
    assert k >= 1


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------

def test_kmeans_two_clusters():
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [10.0, 10.0], [11.0, 10.0], [10.0, 11.0],
    ], dtype=np.float64)
    rng = np.random.default_rng(42)
    labels = kmeans(coords, k=2, rng=rng)
    assert labels.shape == (6,)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_kmeans_deterministic():
    coords = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 0.0],
        [10.0, 10.0], [11.0, 11.0], [12.0, 10.0],
    ], dtype=np.float64)
    a = kmeans(coords, k=2, rng=np.random.default_rng(99))
    b = kmeans(coords, k=2, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)


def test_kmeans_k_equals_n():
    coords = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float64)
    labels = kmeans(coords, k=3, rng=np.random.default_rng(42))
    assert len(np.unique(labels)) == 3


# ---------------------------------------------------------------------------
# cluster_customers
# ---------------------------------------------------------------------------

def test_cluster_every_customer_assigned(tiny_data):
    """Each customer in exactly one cluster."""
    rng = np.random.default_rng(42)
    clusters = cluster_customers(
        tiny_data["orders"], tiny_data["locations"],
        tiny_data["allowed_bike"],
        bike_capacity=60.0, truck_capacity=2000.0, rng=rng,
    )
    all_members = np.concatenate([c["members"] for c in clusters])
    all_sorted = np.sort(all_members)
    expected = np.arange(tiny_data["n_customers"], dtype=all_sorted.dtype)
    np.testing.assert_array_equal(all_sorted, expected)


def test_cluster_dict_fields(tiny_data):
    rng = np.random.default_rng(42)
    clusters = cluster_customers(
        tiny_data["orders"], tiny_data["locations"],
        tiny_data["allowed_bike"],
        bike_capacity=60.0, truck_capacity=2000.0, rng=rng,
    )
    required = {"members", "big_nodes", "bike_nodes", "total_demand", "centroid"}
    for c in clusters:
        assert required.issubset(c.keys())


# ---------------------------------------------------------------------------
# group_by_label
# ---------------------------------------------------------------------------

def test_group_basic():
    labels = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    demands = [10.0, 20.0, 30.0, 40.0, 50.0]
    orders = _make_orders(demands)
    coords = [(0, 0), (1, 0), (5, 5), (6, 5), (0, 1)]
    locations = _make_locations(coords)
    allowed_bike = np.array([1, 1, 1, 1, 1], dtype=np.int8)

    clusters = group_by_label(labels, orders, locations, allowed_bike,
                              bike_capacity=60.0)
    assert len(clusters) == 2


def test_group_centroid():
    labels = np.array([0, 0], dtype=np.int32)
    orders = _make_orders([10.0, 20.0])
    locations = _make_locations([(2.0, 4.0), (6.0, 8.0)])
    allowed_bike = np.array([1, 1], dtype=np.int8)

    clusters = group_by_label(labels, orders, locations, allowed_bike,
                              bike_capacity=60.0)
    np.testing.assert_allclose(clusters[0]["centroid"], [4.0, 6.0])


# ---------------------------------------------------------------------------
# rebalance_clusters
# ---------------------------------------------------------------------------

def test_rebalance_no_change():
    demands = [200, 150, 150]
    orders = _make_orders(demands)
    locations = _make_locations([(0, 0), (1, 0), (2, 0)])
    members = np.array([0, 1, 2], dtype=np.int32)
    cluster = {
        "members": members,
        "big_nodes": np.array([], dtype=np.int32),
        "bike_nodes": members.copy(),
        "total_demand": 500.0,
        "centroid": np.array([1.0, 0.0]),
    }
    result = rebalance_clusters([cluster], orders, locations,
                                bike_capacity=60.0, truck_capacity=2000.0)
    assert len(result) >= 1
    all_m = np.sort(np.concatenate([c["members"] for c in result]))
    np.testing.assert_array_equal(all_m, [0, 1, 2])


# ---------------------------------------------------------------------------
# make_single_cluster
# ---------------------------------------------------------------------------

def test_make_single_cluster():
    orders = _make_orders([150.0])
    locations = _make_locations([(7.0, 3.0)])
    c = make_single_cluster(0, orders, locations, bike_capacity=60.0)
    assert list(c["members"]) == [0]
    assert c["total_demand"] == pytest.approx(150.0, rel=0.1)
