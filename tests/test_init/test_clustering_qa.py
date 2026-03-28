"""QA tests for src/init/clustering.py -- boundary & edge cases.

All units: meters (i64), seconds (i64), grams (i64).
"""
import numpy as np
import pytest

from src.init.clustering import (
    estimate_n_clusters,
    kmeans,
    group_by_label,
    cluster_customers,
    make_single_cluster,
    split_cluster,
    rebalance_clusters,
)
from src.data.constants import COL_X, COL_Y, COL_DEMAND
from src.data.cost import BIKE_CAPACITY_G, TRUCK_CAPACITY_G


# ============================================================
# TEST 1: Single Customer in Cluster
# ============================================================
class TestSingleCustomer:

    def test_make_single_cluster_light(self):
        """1 customer, demand < BIKE_CAPACITY_G (60000g) -> bike_nodes."""
        customers = np.array([[10000, 20000, 45000, 0, 28800, 300, 0]],
                             dtype=np.int64)
        cluster = make_single_cluster(0, customers)
        assert len(cluster["members"]) == 1
        assert len(cluster["big_nodes"]) == 0
        assert len(cluster["bike_nodes"]) == 1
        assert cluster["total_demand"] == 45000
        np.testing.assert_allclose(cluster["centroid"], [10000.0, 20000.0])

    def test_make_single_cluster_heavy(self):
        """1 customer, demand > BIKE_CAPACITY_G (60000g) -> big_nodes."""
        customers = np.array([[5000, 5000, 100000, 0, 28800, 600, 0]],
                             dtype=np.int64)
        cluster = make_single_cluster(0, customers)
        assert len(cluster["members"]) == 1
        assert len(cluster["big_nodes"]) == 1
        assert len(cluster["bike_nodes"]) == 0
        assert cluster["total_demand"] == 100000

    def test_make_single_cluster_restricted(self):
        """1 customer, restricted=1 -> in 'restricted' key."""
        customers = np.array([[0, 0, 30000, 0, 28800, 300, 1]],
                             dtype=np.int64)
        cluster = make_single_cluster(0, customers)
        assert len(cluster["restricted"]) == 1
        assert cluster["restricted"][0] == 0


# ============================================================
# TEST 2: All Customers at Same Location
# ============================================================
class TestAllCustomersSameLocation:

    def test_kmeans_all_same_location(self):
        coords = np.array([[5000, 5000]] * 5, dtype=np.int64)
        rng = np.random.default_rng(42)
        labels = kmeans(coords, k=2, rng=rng, max_iter=50)
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 2
        assert (labels == labels[0]).all()

    def test_cluster_customers_all_same_location(self):
        customers = np.array([
            [5000, 5000, 20000, 0, 28800, 300, 0],
            [5000, 5000, 30000, 0, 28800, 300, 0],
            [5000, 5000, 15000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        dist_matrix = np.zeros((4, 4), dtype=np.int64)
        rng = np.random.default_rng(42)
        clusters = cluster_customers(customers, dist_matrix, rng)
        assert len(clusters) == 1
        assert clusters[0]["total_demand"] == 65000
        np.testing.assert_allclose(clusters[0]["centroid"], [5000.0, 5000.0])


# ============================================================
# TEST 3: Rebalance with Demand > TRUCK_CAPACITY_G
# ============================================================
class TestRebalanceHeavyDemand:

    def test_rebalance_single_heavy_customer(self):
        """1 customer with demand > TRUCK_CAPACITY_G -> own cluster."""
        customers = np.array(
            [[0, 0, 3000000, 0, 28800, 1800, 0]], dtype=np.int64
        )
        cluster = {
            "members": np.array([0], dtype=np.int32),
            "big_nodes": np.array([0], dtype=np.int32),
            "bike_nodes": np.array([], dtype=np.int32),
            "restricted": np.array([], dtype=np.int32),
            "total_demand": 3000000,
            "centroid": np.array([0.0, 0.0]),
        }
        result = rebalance_clusters([cluster], customers, TRUCK_CAPACITY_G)
        assert len(result) == 1
        assert result[0]["total_demand"] == 3000000

    def test_rebalance_all_customers_heavy(self):
        """All demand > half of TRUCK_CAPACITY_G -> each own cluster."""
        customers = np.array([
            [0, 0, 1200000, 0, 28800, 1200, 0],
            [5000, 5000, 1100000, 0, 28800, 1200, 0],
        ], dtype=np.int64)
        cluster = {
            "members": np.array([0, 1], dtype=np.int32),
            "big_nodes": np.array([0, 1], dtype=np.int32),
            "bike_nodes": np.array([], dtype=np.int32),
            "restricted": np.array([], dtype=np.int32),
            "total_demand": 2300000,
            "centroid": np.array([2500.0, 2500.0]),
        }
        result = rebalance_clusters([cluster], customers, TRUCK_CAPACITY_G)
        assert len(result) == 2
        assert result[0]["total_demand"] == 1200000
        assert result[1]["total_demand"] == 1100000

    def test_rebalance_mixed_heavy_light(self):
        customers = np.array([
            [0, 0, 1500000, 0, 28800, 1200, 0],
            [2000, 2000, 100000, 0, 28800, 300, 0],
            [4000, 4000, 200000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        cluster = {
            "members": np.array([0, 1, 2], dtype=np.int32),
            "big_nodes": np.array([0], dtype=np.int32),
            "bike_nodes": np.array([1, 2], dtype=np.int32),
            "restricted": np.array([], dtype=np.int32),
            "total_demand": 1800000,
            "centroid": np.array([2000.0, 2000.0]),
        }
        result = rebalance_clusters([cluster], customers, TRUCK_CAPACITY_G)
        heavy_found = any(c["total_demand"] == 1500000 for c in result)
        assert heavy_found
        assert len(result) >= 2


# ============================================================
# TEST 4: Boundary -- BIKE_CAPACITY_G (60000g)
# ============================================================
class TestBikeBoundary:

    def test_customer_exactly_60kg_light(self):
        """demand == BIKE_CAPACITY_G (60000g) -> bike-capable (not >)."""
        customers = np.array([[0, 0, 60000, 0, 28800, 300, 0]],
                             dtype=np.int64)
        cluster = make_single_cluster(0, customers)
        assert len(cluster["bike_nodes"]) == 1
        assert len(cluster["big_nodes"]) == 0

    def test_customer_60001g_heavy(self):
        """demand == 60001g -> big_nodes."""
        customers = np.array([[0, 0, 60001, 0, 28800, 300, 0]],
                             dtype=np.int64)
        cluster = make_single_cluster(0, customers)
        assert len(cluster["big_nodes"]) == 1
        assert len(cluster["bike_nodes"]) == 0

    def test_customer_59999g_light(self):
        customers = np.array([[0, 0, 59999, 0, 28800, 300, 0]],
                             dtype=np.int64)
        cluster = make_single_cluster(0, customers)
        assert len(cluster["bike_nodes"]) == 1
        assert len(cluster["big_nodes"]) == 0

    def test_group_by_label_boundary_mixed(self):
        customers = np.array([
            [0, 0, 59000, 0, 28800, 300, 0],
            [1000, 1000, 60000, 0, 28800, 300, 0],
            [2000, 2000, 60001, 0, 28800, 300, 0],
            [3000, 3000, 61000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        labels = np.array([0, 0, 0, 0], dtype=np.int32)
        clusters = group_by_label(labels, customers)
        assert len(clusters) == 1
        assert len(clusters[0]["bike_nodes"]) == 2
        assert len(clusters[0]["big_nodes"]) == 2


# ============================================================
# TEST 5: Estimate N Clusters
# ============================================================
class TestEstimateNClusters:

    def test_estimate_n_clusters_simple(self):
        """Total demand 2000000g / capacity 2000000g = 1 cluster."""
        customers = np.array([[0, 0, 1000000, 0, 0, 0, 0],
                              [1000, 1000, 1000000, 0, 0, 0, 0]],
                             dtype=np.int64)
        n = estimate_n_clusters(customers, n_trucks=1, capacity=2000000)
        assert n == 1

    def test_estimate_n_clusters_ceil_rounding(self):
        """Total demand 2100000g / capacity 2000000g -> ceil = 2."""
        customers = np.array([[0, 0, 1050000, 0, 0, 0, 0],
                              [1000, 1000, 1050000, 0, 0, 0, 0]],
                             dtype=np.int64)
        n = estimate_n_clusters(customers, n_trucks=2, capacity=2000000)
        assert n == 2

    def test_estimate_n_clusters_zero_demand(self):
        customers = np.array([[0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        n = estimate_n_clusters(customers, n_trucks=1, capacity=2000000)
        assert n == 1


# ============================================================
# TEST 6: Split Cluster Edge Cases
# ============================================================
class TestSplitCluster:

    def test_split_cluster_k_exceeds_size(self):
        customers = np.array([
            [0, 0, 50000, 0, 28800, 300, 0],
            [5000, 5000, 40000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        members = np.array([0, 1], dtype=np.int32)
        result = split_cluster(members, customers, k=3)
        assert len(result) == 1
        assert len(result[0]["members"]) == 2

    def test_split_cluster_exactly_k_members(self):
        customers = np.array([
            [0, 0, 50000, 0, 28800, 300, 0],
            [5000, 5000, 40000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        members = np.array([0, 1], dtype=np.int32)
        result = split_cluster(members, customers, k=2)
        assert len(result) == 1


# ============================================================
# TEST 7: KMeans Convergence
# ============================================================
class TestKMeansBasic:

    def test_kmeans_single_cluster(self):
        coords = np.array([[0, 0], [1000, 1000], [2000, 2000]], dtype=np.int64)
        rng = np.random.default_rng(42)
        labels = kmeans(coords, k=1, rng=rng)
        assert (labels == 0).all()

    def test_kmeans_well_separated_points(self):
        coords = np.array([
            [0, 0], [100, 100], [10000, 10000], [10100, 10100],
        ], dtype=np.int64)
        rng = np.random.default_rng(42)
        labels = kmeans(coords, k=2, rng=rng, max_iter=50)
        unique = np.unique(labels)
        assert len(unique) == 2


# ============================================================
# INTEGRATION: Full Clustering Pipeline
# ============================================================
class TestClusteringIntegration:

    def test_cluster_then_rebalance(self):
        customers = np.array([
            [0, 0, 1500000, 0, 28800, 1200, 0],
            [2000, 2000, 100000, 0, 28800, 300, 0],
            [4000, 4000, 200000, 0, 28800, 300, 0],
            [6000, 6000, 150000, 0, 28800, 300, 0],
            [8000, 8000, 80000, 0, 28800, 300, 0],
        ], dtype=np.int64)
        dist_matrix = np.zeros((6, 6), dtype=np.int64)
        rng = np.random.default_rng(42)

        clusters = cluster_customers(customers, dist_matrix, rng)
        rebalanced = rebalance_clusters(clusters, customers, TRUCK_CAPACITY_G)

        half_cap = TRUCK_CAPACITY_G // 2
        for cluster in rebalanced:
            if len(cluster["members"]) > 1:
                assert cluster["total_demand"] <= half_cap + 1


# ============================================================
# SUSPECTED BUGS
# ============================================================
class TestSuspectedBugs:

    def test_kmeans_choice_without_replace_potential_issue(self):
        """kmeans with k > n raises ValueError."""
        coords = np.array([[0, 0], [1000, 1000]], dtype=np.int64)
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            kmeans(coords, k=3, rng=rng)
