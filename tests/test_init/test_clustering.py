"""Tests for src/init/clustering.py.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.clustering import (
    estimate_n_clusters,
    kmeans,
    cluster_customers,
    group_by_label,
    make_single_cluster,
)
from src.data.constants import COL_X, COL_Y, COL_DEMAND


def _make_customers(demands_g, coords_m):
    """Build i64 (N,7) customers from demand (grams) and coord (meters) lists."""
    n = len(demands_g)
    customers = np.zeros((n, 7), dtype=np.int64)
    for i in range(n):
        customers[i, COL_X] = coords_m[i][0]
        customers[i, COL_Y] = coords_m[i][1]
        customers[i, COL_DEMAND] = demands_g[i]
        customers[i, 4] = 28800  # tw_close_s (8h)
        customers[i, 5] = 300    # service_s
    return customers


# ---------------------------------------------------------------------------
# estimate_n_clusters
# ---------------------------------------------------------------------------

def test_estimate_n_clusters_basic():
    """total_demand ~ 13750g, capacity=2000000g -> k=1."""
    demands = [1500000, 1200000, 1000000, 2000000, 1500000,
               1800000, 1250000, 1000000, 1500000, 1000000]
    coords = [(i * 1000, 0) for i in range(len(demands))]
    customers = _make_customers(demands, coords)
    k = estimate_n_clusters(customers, 3, capacity=2000000)
    assert k >= 1
    assert isinstance(k, (int, np.integer))


def test_estimate_small_demand():
    customers = _make_customers([30000, 40000, 30000],
                                [(0, 0), (1000, 0), (2000, 0)])
    k = estimate_n_clusters(customers, 1, capacity=2000000)
    assert k >= 1


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------

def test_kmeans_two_clusters():
    coords = np.array([
        [0, 0], [1000, 0], [0, 1000],
        [10000, 10000], [11000, 10000], [10000, 11000],
    ], dtype=np.int64)
    rng = np.random.default_rng(42)
    labels = kmeans(coords, k=2, rng=rng)
    assert labels.shape == (6,)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_kmeans_deterministic():
    coords = np.array([
        [0, 0], [1000, 1000], [2000, 0],
        [10000, 10000], [11000, 11000], [12000, 10000],
    ], dtype=np.int64)
    a = kmeans(coords, k=2, rng=np.random.default_rng(99))
    b = kmeans(coords, k=2, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)


def test_kmeans_k_equals_n():
    coords = np.array([[0, 0], [10000, 10000], [20000, 20000]], dtype=np.int64)
    labels = kmeans(coords, k=3, rng=np.random.default_rng(42))
    assert len(np.unique(labels)) == 3


# ---------------------------------------------------------------------------
# cluster_customers
# ---------------------------------------------------------------------------

def test_cluster_every_customer_assigned(tiny_instance):
    """Each customer in exactly one cluster."""
    data = tiny_instance
    rng = np.random.default_rng(42)
    dm = np.zeros((data["n_customers"] + 1, data["n_customers"] + 1),
                  dtype=np.int64)
    clusters = cluster_customers(data["customers"], dm, rng)
    all_members = np.concatenate([c["members"] for c in clusters])
    all_sorted = np.sort(all_members)
    expected = np.arange(data["n_customers"], dtype=all_sorted.dtype)
    np.testing.assert_array_equal(all_sorted, expected)


def test_cluster_dict_fields(tiny_instance):
    data = tiny_instance
    rng = np.random.default_rng(42)
    dm = np.zeros((data["n_customers"] + 1, data["n_customers"] + 1),
                  dtype=np.int64)
    clusters = cluster_customers(data["customers"], dm, rng)
    required = {"members", "big_nodes", "bike_nodes", "total_demand", "centroid"}
    for c in clusters:
        assert required.issubset(c.keys())


# ---------------------------------------------------------------------------
# group_by_label
# ---------------------------------------------------------------------------

def test_group_basic():
    labels = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    demands = [10000, 20000, 30000, 40000, 50000]
    coords = [(0, 0), (1000, 0), (5000, 5000), (6000, 5000), (0, 1000)]
    customers = _make_customers(demands, coords)
    clusters = group_by_label(labels, customers)
    assert len(clusters) == 2


def test_group_centroid():
    labels = np.array([0, 0], dtype=np.int32)
    customers = _make_customers([10000, 20000], [(2000, 4000), (6000, 8000)])
    clusters = group_by_label(labels, customers)
    np.testing.assert_allclose(clusters[0]["centroid"], [4000.0, 6000.0])


# ---------------------------------------------------------------------------
# make_single_cluster
# ---------------------------------------------------------------------------

def test_make_single_cluster():
    customers = _make_customers([150000], [(7000, 3000)])
    c = make_single_cluster(0, customers)
    assert list(c["members"]) == [0]
    assert c["total_demand"] == 150000
