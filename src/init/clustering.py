"""Customer clustering for initial solution."""
import numpy as np
from math import ceil

from ..data.constants import COL_X, COL_Y, COL_DEMAND
from ..data.cost import BIKE_CAPACITY


def estimate_n_clusters(customers, n_trucks, capacity):
    """Estimate k from total demand / capacity."""
    total_demand = customers[:, COL_DEMAND].sum()
    return max(1, int(ceil(total_demand / capacity)))


def kmeans(coords, k, rng, max_iter=50):
    """Simple k-means, pure numpy. Returns labels ndarray."""
    n = len(coords)
    # Initialize centroids by random choice
    indices = rng.choice(n, size=k, replace=False)
    centroids = coords[indices].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # Assign each point to nearest centroid
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centroids
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = coords[mask].mean(axis=0)

    return labels


def group_by_label(labels, customers, restricted):
    """Labels -> list of cluster dicts with members, big_nodes, bike_nodes."""
    unique_labels = np.unique(labels)
    clusters = []
    for lbl in unique_labels:
        members = np.where(labels == lbl)[0].astype(np.int32)
        demands = customers[members, COL_DEMAND]
        big_mask = demands > BIKE_CAPACITY
        rest_mask = restricted[members] == 1
        # big_nodes: demand > BIKE_CAPACITY (truck must serve, regardless of restriction)
        big_nodes = members[big_mask]
        # bike_nodes: demand <= BIKE_CAPACITY (restricted customers still bike-compatible)
        bike_nodes = members[~big_mask]
        centroid = customers[members][:, [COL_X, COL_Y]].mean(axis=0)
        clusters.append({
            "members": members,
            "big_nodes": big_nodes,
            "bike_nodes": bike_nodes,
            "restricted": members[rest_mask],
            "total_demand": float(demands.sum()),
            "centroid": centroid,
        })
    return clusters


def cluster_customers(customers, restricted, dist_matrix, rng):
    """K-means clustering. Returns list of cluster dicts."""
    n_trucks = max(1, len(customers) // 5)  # rough estimate
    coords = customers[:, [COL_X, COL_Y]]
    n_clusters = estimate_n_clusters(customers, n_trucks, 2000.0)
    n_clusters = min(n_clusters, len(customers))
    labels = kmeans(coords, n_clusters, rng)
    return group_by_label(labels, customers, restricted)


def make_single_cluster(customer_idx, customers, restricted=None):
    """1 customer -> 1 cluster dict."""
    c = customer_idx
    demand = float(customers[c, COL_DEMAND])
    members = np.array([c], dtype=np.int32)
    is_big = demand > BIKE_CAPACITY
    is_rest = restricted is not None and restricted[c] == 1
    return {
        "members": members,
        "big_nodes": members if is_big else np.array([], dtype=np.int32),
        "bike_nodes": np.array([], dtype=np.int32) if is_big else members,
        "restricted": members if is_rest else np.array([], dtype=np.int32),
        "total_demand": demand,
        "centroid": customers[c, [COL_X, COL_Y]].copy(),
    }


def split_cluster(members, customers, restricted=None, k=2):
    """Split 1 cluster into k sub-clusters."""
    if len(members) <= k:
        # Can't split further, return as single cluster
        demands = customers[members, COL_DEMAND]
        big_mask = demands > BIKE_CAPACITY
        rest_mask = restricted[members] == 1 if restricted is not None else np.zeros(len(members), dtype=bool)
        centroid = customers[members][:, [COL_X, COL_Y]].mean(axis=0)
        return [{
            "members": members,
            "big_nodes": members[big_mask],
            "bike_nodes": members[~big_mask],
            "restricted": members[rest_mask],
            "total_demand": float(demands.sum()),
            "centroid": centroid,
        }]

    coords = customers[members][:, [COL_X, COL_Y]]
    rng = np.random.default_rng(0)
    labels = kmeans(coords, k, rng)
    result = []
    for lbl in np.unique(labels):
        sub_members = members[labels == lbl]
        demands = customers[sub_members, COL_DEMAND]
        big_mask = demands > BIKE_CAPACITY
        rest_mask = restricted[sub_members] == 1 if restricted is not None else np.zeros(len(sub_members), dtype=bool)
        centroid = customers[sub_members][:, [COL_X, COL_Y]].mean(axis=0)
        result.append({
            "members": sub_members,
            "big_nodes": sub_members[big_mask],
            "bike_nodes": sub_members[~big_mask],
            "restricted": sub_members[rest_mask],
            "total_demand": float(demands.sum()),
            "centroid": centroid,
        })
    return result


def rebalance_clusters(clusters, customers, truck_capacity):
    """Split heavy clusters (>capacity/2). Keep single heavy customers intact."""
    result = []
    half_cap = truck_capacity / 2.0

    for cluster in clusters:
        if cluster["total_demand"] <= half_cap:
            result.append(cluster)
            continue

        heavy_mask = customers[cluster["members"], COL_DEMAND] > half_cap
        heavy = cluster["members"][heavy_mask]

        if len(heavy) > 0 and len(cluster["members"]) == len(heavy):
            # All heavy -> each gets own cluster
            for c in heavy:
                result.append(make_single_cluster(c, customers))
            continue

        if len(heavy) > 0:
            for c in heavy:
                result.append(make_single_cluster(c, customers))
            remaining = np.setdiff1d(cluster["members"], heavy)
        else:
            remaining = cluster["members"]

        sub_clusters = split_cluster(remaining, customers, k=2)
        result.extend(sub_clusters)

    return result
